"""Tools: functions an agent may call while it answers.

A tool is described once, as a :class:`Tool`: a name, a description and a JSON
Schema of its arguments, derived from a plain function by :func:`tool`. Where a
tool runs is a :class:`ToolSource`. Bare ``Tool`` objects run in-process
(:class:`LocalTools`); :class:`xmas.tools.mcp.MCPTools` runs them on an MCP
server. A MAS resolves ``Agent.tools`` names against its sources when it is
built, so a misspelled name fails before any model is called.

An agent with tools runs a bounded loop in its node: the model may request tool
calls, each result goes back as a ``role=tool`` message, and after
``max_tool_rounds`` such turns one last turn with the tools withheld forces a
text answer. Every call is kept as a :class:`~xmas.trace.ToolCall` on the
agent's record. A tool that fails is data (the model sees the error and may try
again); a source that refuses the caller ends the run.

The same registry serves steps (version 2): a :class:`~xmas.nodes.Step` names a
tool in ``uses`` and :func:`run_step` runs it with values from the graph. A tool
reads its parameters (:attr:`Tool.reads`) and writes :attr:`Tool.writes`.
"""

from __future__ import annotations

import inspect
import json
import re
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import BaseModel, ValidationError

from xmas.errors import BadFormat, BudgetExceeded, Stopped, XmasError
from xmas.schemas import RETRY_NOTE
from xmas.tools.schema import (
    arguments_from,
    arguments_model,
    describe_validation_error,
    function_name,
    json_schema,
    split_docstring,
    valid_tool_name,
)
from xmas.trace import TokenUsage, ToolCall
from xmas.values import TEXT, ValueType, check_value, parse_type, type_of_hint

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.agent import Agent
    from xmas.budget import RunMeter, VisitMeter
    from xmas.llms.meta import CallMeta, SupportsMeta


def _return_type(fn: Callable[..., Any] | None) -> ValueType:
    """Return the type a callable's single write has: text for a ``str`` return hint or a remote tool, else json."""
    if fn is None:
        return TEXT
    try:
        hint = typing.get_type_hints(fn, include_extras=True).get("return")
    except Exception:  # an unresolvable hint, or no hints at all, says nothing about the type
        hint = None
    return type_of_hint(hint)


@dataclass(frozen=True, init=False)
class Tool:
    """A function an agent may call, described once.

    Attributes:
        name: Unique tool name, what ``Agent.tools`` refers to.
        description: What the model is told the tool does (the docstring summary).
        parameters: JSON Schema of the arguments object.
        fn: The implementation for a local tool, ``None`` for a remote one (called
            through its source).
        writes: What the tool writes when it runs as a step, ``{write name:
            type}``. It may be given as type words; ``None`` is one write named
            like the tool: text when ``fn``'s return hint is ``str`` or the tool
            is remote, json otherwise.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None
    arguments_model: type[BaseModel] | None = field(compare=False, repr=False)
    writes: dict[str, ValueType]

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any] | None = None,
        arguments_model: type[BaseModel] | None = None,
        writes: Mapping[str, ValueType | str] | None = None,
    ) -> None:
        """Reject names the OpenAI ``tools`` payload would refuse, and read the write types."""
        if not valid_tool_name(name):
            raise ValueError(
                f"Tool name {name!r} must match the OpenAI rule: letters, digits, _ and -, at most 64 characters."
            )
        given = {name: _return_type(fn)} if writes is None else writes
        if not given:
            raise ValueError(f"Tool {name!r} needs at least one write.")
        typed = {key: vtype if isinstance(vtype, ValueType) else parse_type(vtype) for key, vtype in given.items()}
        values = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "fn": fn,
            "arguments_model": arguments_model,
            "writes": typed,
        }
        for key, value in values.items():
            object.__setattr__(self, key, value)  # frozen

    @property
    def reads(self) -> dict[str, ValueType]:
        """The type each parameter reads when the tool runs as a step: text for a ``string``, else json."""
        properties = self.parameters.get("properties") or {}
        return {name: type_of_hint(schema) for name, schema in properties.items()}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call ``fn`` directly, so a decorated function stays a function; a remote tool raises ``TypeError``."""
        if self.fn is None:
            raise TypeError(f"Tool {self.name!r} is remote; call it through its source.")
        return self.fn(*args, **kwargs)

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return model-supplied arguments as keyword arguments for ``fn``, or raise ``XmasError``.

        A tool built by :func:`tool` validates (and coerces) with its pydantic
        model. A tool built from a foreign schema only checks required and
        unknown keys.
        """
        if not isinstance(arguments, dict):
            raise XmasError(f"arguments must be an object, got {type(arguments).__name__}")
        if self.arguments_model is None:
            props = self.parameters.get("properties", {})
            missing = [key for key in self.parameters.get("required", []) if key not in arguments]
            if missing:
                raise XmasError(f"missing required argument(s): {', '.join(missing)}")
            if self.parameters.get("additionalProperties") is False:
                unexpected = [key for key in arguments if key not in props]
                if unexpected:
                    raise XmasError(f"unexpected argument(s): {', '.join(unexpected)}")
            return dict(arguments)
        try:
            instance = self.arguments_model.model_validate(arguments)
        except ValidationError as exc:
            raise XmasError(f"invalid arguments for {self.name}: {describe_validation_error(exc)}") from None
        return arguments_from(instance)

    def to_openai(self) -> dict[str, Any]:
        """Return the entry for an OpenAI chat-completions ``tools`` list."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the model asked for.

    Attributes:
        id: The provider's call id, echoed back as ``tool_call_id``.
        name: The tool name.
        arguments: The raw JSON arguments, as the model sent them.
    """

    id: str
    name: str
    arguments: str


class ToolSource:
    """Where a MAS resolves tool names.

    :class:`xmas.tools.mcp.MCPTools` is one; bare local Tools are wrapped in a
    :class:`LocalTools`.
    """

    def tools(self) -> list[Tool]:
        """Every tool this source offers. May do I/O once; the source caches it."""
        raise NotImplementedError

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return the text the model sees.

        Raises:
            XmasError: The tool failed, and the message is fed back to the model;
                or, with ``status`` 401, 403 or 429, the source refused the caller
                and the run ends.
        """
        raise NotImplementedError


class LocalTools(ToolSource):
    """A ToolSource over in-process Tools; the MAS wraps bare Tool objects in one.

    ``call`` validates, then runs ``fn(**kwargs)``. Any exception becomes an
    :class:`~xmas.errors.XmasError`. A ``str`` result is returned as is, anything
    else as JSON (``default=str``), the same rule as serving-xmas's in-process executor.
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        """Wrap local tools; ``ValueError`` if one has no ``fn`` or two share a name."""
        self.by_name: dict[str, Tool] = {}
        for item in tools:
            if item.fn is None:
                raise ValueError(f"Tool {item.name!r} has no local implementation.")
            if item.name in self.by_name and self.by_name[item.name] is not item:
                raise ValueError(f"Two local tools are named {item.name!r}.")
            self.by_name[item.name] = item

    def tools(self) -> list[Tool]:
        """Return the wrapped tools."""
        return list(self.by_name.values())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Validate the arguments, run the function, and return its result as text.

        Raises:
            XmasError: If the tool is unknown, the arguments are invalid, or the
                function raised. ``BudgetExceeded`` and ``Stopped`` pass through
                unwrapped, since they end the run.
        """
        spec = self.by_name.get(name)
        if spec is None:
            raise XmasError(f"unknown tool {name!r}")
        kwargs = spec.validate(arguments)
        try:
            result = spec(**kwargs)
        except (BudgetExceeded, Stopped):
            raise
        except Exception as exc:
            raise XmasError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        """Name the wrapped tools."""
        return f"LocalTools({sorted(self.by_name)!r})"


ToolIndex = dict[str, tuple[Tool, ToolSource]]
"""Tools by name, each with the source that runs it."""


@overload
def tool(fn: Callable[..., Any], /, *, name: str | None = None, writes: Mapping[str, str] | None = None) -> Tool: ...
@overload
def tool(
    *, name: str | None = None, writes: Mapping[str, str] | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None, /, *, name: str | None = None, writes: Mapping[str, str] | None = None
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Declare a function as a tool.

    Usable bare (``@tool``) or with keywords (``@tool(name="add")``,
    ``@tool(writes={"counts": "json"})``).

    Args:
        fn: A function with a type hint on every parameter and a docstring whose
            summary becomes the description. ``*args``, ``**kwargs`` and
            positional-only parameters are rejected. Google-style ``Args:``
            entries become the parameter descriptions.
        name: Tool name. Defaults to ``fn.__name__``.
        writes: What the tool writes as a step, ``{write name: type}``. Defaults
            to one write named like the tool: text when the return hint is
            ``str``, else json. Several writes need ``fn`` to return a mapping
            with exactly those keys.

    Returns:
        The :class:`Tool`, or a decorator producing one when ``fn`` is not given.

    Raises:
        ValueError: If the function cannot be described (missing hint or
            docstring, unschemable type, bad name).
    """

    def describe(func: Callable[..., Any]) -> Tool:
        doc = inspect.getdoc(func)
        if not doc or not doc.strip():
            raise ValueError(f"{function_name(func)}: a tool needs a docstring")
        summary, descriptions = split_docstring(doc)
        if not summary:
            raise ValueError(f"{function_name(func)}: the docstring has no summary paragraph")
        tool_name = name or getattr(func, "__name__", None)
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError(f"cannot derive a tool name for {func!r}; pass name=")
        model = arguments_model(func, descriptions)
        return Tool(tool_name, summary, json_schema(model, func), fn=func, arguments_model=model, writes=writes)

    return describe if fn is None else describe(fn)


def index_tools(items: Sequence[ToolSource | Tool]) -> ToolIndex:
    """Index every tool offered by a MAS's ``tools=`` list by name.

    Bare Tools are grouped into one :class:`LocalTools`. The same object listed
    twice counts once; two different offers of one name are a ``ValueError``,
    with no shadowing and no priority order.
    """
    local: list[Tool] = []
    sources: list[ToolSource] = []
    for item in items:
        if isinstance(item, Tool):
            if not any(seen is item for seen in local):
                local.append(item)
        elif isinstance(item, ToolSource):
            if not any(seen is item for seen in sources):
                sources.append(item)
        else:
            raise TypeError(f"tools=[...] takes Tool objects and tool sources, not {type(item).__name__}.")
    if local:
        sources.insert(0, LocalTools(local))
    index: ToolIndex = {}
    for source in sources:
        for spec in source.tools():
            if spec.name in index:
                raise ValueError(f"Tool {spec.name!r} is offered by both {index[spec.name][1]!r} and {source!r}.")
            index[spec.name] = (spec, source)
    return index


def select_tools(index: ToolIndex, agent: Agent, llm: BaseLLM | None) -> ToolIndex:
    """Return the tools one agent may call, in its order; ``ValueError`` for an unknown name or a tool-less model."""
    unknown = [name for name in agent.tools if name not in index]
    if unknown:
        raise ValueError(f"Agent {agent.name!r} uses unknown tools {unknown}; available: {sorted(index)}.")
    if llm is not None and not hasattr(llm, "get_tool_response"):
        raise ValueError(
            f"Agent {agent.name!r} uses tools, but its model {type(llm).__name__} has no get_tool_response."
        )
    return {name: index[name] for name in agent.tools}


def run_step(spec: Tool, source: ToolSource, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a callable as a step and return ``{write: value}``, keyed by its own write names.

    A local tool's ``fn`` is called here, not through :meth:`LocalTools.call`,
    which would turn its result into text, and what it raises passes unchanged;
    a remote tool runs through ``source.call``, and a json write is the JSON its
    text holds. One write takes the whole result; several need a mapping with
    exactly those keys.

    Raises:
        XmasError: If the arguments do not match the parameters, or a remote
            tool failed or was refused (``status`` 401, 403 or 429).
        BadFormat: If the result does not fit the writes; it names the callable.
    """
    writes = spec.writes
    kwargs = spec.validate(arguments)
    if spec.fn is not None:
        result = spec.fn(**kwargs)
    else:
        text = source.call(spec.name, kwargs)
        if len(writes) == 1 and next(iter(writes.values())) == TEXT:
            result = text
        else:
            try:
                result = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BadFormat(spec.name, (f"the result is not JSON: {exc}",)) from exc
    if len(writes) == 1:
        produced = {next(iter(writes)): result}
    elif isinstance(result, Mapping) and set(result) == set(writes):
        produced = dict(result)
    else:
        got = sorted(result) if isinstance(result, Mapping) else type(result).__name__
        raise BadFormat(spec.name, (f"expected a mapping with keys {sorted(writes)}, got {got}",))
    problems = [
        f"{name}: {problem}"
        for name, value in produced.items()
        if (problem := check_value(value, writes[name])) is not None
    ]
    if problems:
        raise BadFormat(spec.name, ("; ".join(problems),))
    return produced


_INTEGER = re.compile(r"-?[0-9]+")
_NUMBER = re.compile(r"-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")


def _numbers_from_strings(arguments: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Read top-level numeric strings as numbers where the schema types a property integer or number.

    ``null`` may be allowed beside the number. Only plain decimal literals are
    converted (``"-1.5"`` for ``number`` only), and a number-typed ``"20"``
    becomes ``20``, not ``20.0``.
    """
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return arguments
    out = dict(arguments)
    for key, value in arguments.items():
        schema = properties.get(key)
        declared = schema.get("type") if isinstance(schema, dict) else None
        listed = [declared] if isinstance(declared, str) else declared if isinstance(declared, list) else []
        types = {item for item in listed if isinstance(item, str)} - {"null"}
        if not isinstance(value, str) or len(types) != 1 or not types <= {"integer", "number"}:
            continue
        if _INTEGER.fullmatch(value):
            out[key] = int(value)
        elif types == {"number"} and _NUMBER.fullmatch(value):
            out[key] = float(value)
    return out


def _turn_usage(
    count: Callable[[str], float], messages: list[dict[str, Any]], text: str, requests: Sequence[ToolCallRequest]
) -> TokenUsage:
    """Estimate one tool turn's tokens with ``count``: every message's content in, the text and call arguments out."""
    return TokenUsage(
        input_tokens=sum(int(count(str(item.get("content") or ""))) for item in messages),
        output_tokens=int(count(text)) + sum(int(count(request.arguments)) for request in requests),
    )


def run_tool_loop(
    llm: Any,
    rendered: str,
    tools: ToolIndex,
    max_rounds: int,
    system_prompt: str,
    count_tokens: Callable[[str], float],
    meter: RunMeter | VisitMeter,
    check: Callable[[str], str | None] | None = None,
) -> tuple[str, tuple[ToolCall, ...], TokenUsage, CallMeta | None]:
    """Answer one rendered prompt, letting the model call tools.

    At most ``max_rounds`` turns may request tools; calls within a turn run in
    order. If the model is still asking after that, one more turn with the
    tools withheld forces a text answer. With ``check``, a final answer it
    names a problem with stays in the conversation, and one more turn, tools
    withheld, asks again with :data:`~xmas.schemas.RETRY_NOTE`; that answer is
    returned unchecked. Every turn is checked against ``meter`` before it starts.

    Returns:
        The final text, the calls made, the approximate usage over all turns
        (``count_tokens``), and the turns' combined metadata, or ``None`` when
        the model gives none (see :func:`xmas.llms.meta.supports_meta`).

    Raises:
        BudgetExceeded: If the meter refuses a turn.
        EmptyResponse: If a turn came back empty after its retries.
        XmasError: If a tool source refused the caller (``status`` 401, 403 or 429).
    """
    from xmas.llms.meta import supports_meta  # here, not at the top: xmas.llms imports this module

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rendered},
    ]
    specs = [spec for spec, _ in tools.values()]
    calls: list[ToolCall] = []
    usage = TokenUsage()
    meta: CallMeta | None = None
    with_meta = cast("SupportsMeta", llm) if supports_meta(llm) else None

    def ask(offered: list[Tool]) -> tuple[tuple[str, list[ToolCallRequest]], CallMeta | None]:
        if with_meta is None:
            return llm.get_tool_response(messages, offered), None
        text, requests, call_meta = with_meta.get_tool_response_meta(messages, offered)
        return (text, requests), call_meta

    def turn(offered: list[Tool]) -> tuple[str, list[ToolCallRequest]]:
        nonlocal usage, meta
        (text, requests), turn_meta = meter.metered(lambda: ask(offered))
        if turn_meta is not None:
            meta = turn_meta if meta is None else meta.combine(turn_meta)
        usage = usage + _turn_usage(count_tokens, messages, text, requests)
        return text, requests

    def checked(text: str) -> str:
        problem = None if check is None else check(text)
        if problem is None:
            return text
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": RETRY_NOTE.format(problem=problem)})
        return turn([])[0]

    for _ in range(max_rounds):
        text, requests = turn(specs)
        if not requests:
            return checked(text), tuple(calls), usage, meta
        wired = [
            {"id": r.id, "type": "function", "function": {"name": r.name, "arguments": r.arguments}} for r in requests
        ]
        messages.append({"role": "assistant", "content": text or None, "tool_calls": wired})
        for request in requests:
            call = execute_tool_call(tools, request)
            calls.append(call)
            messages.append({"role": "tool", "tool_call_id": request.id, "content": call.result})
    text, _ = turn([])
    return checked(text), tuple(calls), usage, meta


def execute_tool_call(tools: ToolIndex, request: ToolCallRequest) -> ToolCall:
    """Run one requested call; every failure but a refusal becomes a failed ToolCall.

    An XmasError with ``status`` 401, 403 or 429 is a refusal: it propagates and ends the run.
    So do :class:`~xmas.errors.BudgetExceeded` and :class:`~xmas.errors.Stopped`,
    which end a run wherever they are raised.
    """
    try:
        arguments = json.loads(request.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(request.name, {"raw": request.arguments}, f"invalid JSON arguments: {exc}", failed=True)
    if not isinstance(arguments, dict):
        return ToolCall(request.name, {"raw": request.arguments}, "invalid JSON arguments: not an object", failed=True)
    entry = tools.get(request.name)
    if entry is None:
        return ToolCall(request.name, arguments, f"unknown tool; available: {', '.join(tools)}", failed=True)
    spec, source = entry
    try:
        result = source.call(spec.name, arguments)
    except XmasError as exc:
        if exc.status in (401, 403, 429) or isinstance(exc, BudgetExceeded | Stopped):
            raise
        return ToolCall(request.name, arguments, str(exc), failed=True)
    return ToolCall(request.name, arguments, result)
