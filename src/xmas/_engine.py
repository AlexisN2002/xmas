"""Private execution engine.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data. Keeping the dependency behind this wall is what makes
replacing it later a contained change.

Each run gets its own :class:`~xmas.budget.RunMeter`, handed to the nodes in the
run config under :data:`_RUN_KEY`. Nodes read it back to check the budget before
each call and to keep every finished agent's record, so a failed run still
returns what it did.

A version 2 system (``mas.version == 2``) compiles by its own rules, and every
version 1 path stays as it was: values are typed (text or json) and render by
:func:`~xmas.values.render_value`; steps run callables; a json agent answers in
its answer form, with one retry; and a node fed by several edges is a barrier
join that runs once, after all of them, where version 1 defers.
"""

from __future__ import annotations

import dataclasses
import operator
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypedDict, cast

from langgraph.config import get_config
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from xmas import tools as _tools
from xmas.agent import Agent
from xmas.budget import RunMeter
from xmas.errors import BadFormat, EmptyResponse, ErrorKind, classify
from xmas.graph import END, back_edges, join_nodes, terminal_agent
from xmas.llms.meta import SupportsMeta, supports_meta
from xmas.prompts import safe_format
from xmas.schemas import RETRY_NOTE, AnswerForm, FormError, compile_form
from xmas.trace import AgentRecord, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, parse_type, render_value

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from promptolution.llms.base_llm import BaseLLM

    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS
    from xmas.nodes import Step


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on.

A cyclic topology is a retry loop, and something has to decide when to stop. The
agent that closes the loop decides, by emitting this marker in its response, which
mirrors the seed engine's ``<continue_iter>`` convention. An agent that never
emits it turns its cycle into a single pass, so a mutated topology that grows a
cycle stays runnable without anyone rewriting a prompt for it.
"""

_RUN_KEY = "xmas_run"
"""Key of the run's :class:`~xmas.budget.RunMeter` in ``config["configurable"]``."""

MAX_NODE_VISITS = 3
"""How often one agent may run in a single query before its back edges close.

The second stop condition, and the one that holds when a prompt emits the marker
unconditionally. Matches the seed's ``stop_condition_max_retries``. The graph-wide
``max_steps`` still caps total super-steps on top of this.
"""


def _merge_counts(current: dict[str, int], incoming: dict[str, int]) -> dict[str, int]:
    """Add per-agent visit counts, so a looping agent accumulates its own.

    Args:
        current: Counts so far.
        incoming: Counts from the agents that just finished.

    Returns:
        The summed mapping.
    """
    merged = dict(current)
    for name, count in incoming.items():
        merged[name] = merged.get(name, 0) + count
    return merged


def _merge_values(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge one node's writes into the shared value map.

    Agents writing concurrently each contribute their own key, so a plain merge
    is well defined. Keeping every variable in one reduced channel, rather than
    building a channel per variable, means the state schema is fixed and typed
    no matter what variables a topology happens to use.

    Args:
        current: Values written so far.
        incoming: Values from the agent that just finished.

    Returns:
        The merged mapping.
    """
    return {**current, **incoming}


class MASState(TypedDict, total=False):
    """The state a run threads through the graph.

    Attributes:
        values: Every state variable, including ``query``, keyed by name: a
            ``str`` for text, plain JSON data for json (version 2). Version 1
            writes only ``str``.
        prompts: The prompt set for this run, keyed by agent name. Prompts travel
            in the state rather than being baked into the compiled graph, so one
            compilation serves every candidate an optimizer proposes.
        records: Accumulated per-agent records, in completion order.
        visits: How often each agent has run, which is what bounds a retry loop.
    """

    values: Annotated[dict[str, Any], _merge_values]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    visits: Annotated[dict[str, int], _merge_counts]


class CompiledGraph(Protocol):
    """The executable produced by :func:`compile_mas`.

    Declared as a protocol so that nothing outside this module needs to name a
    langgraph type, including the tests. The parameters are deliberately loose:
    the point is to keep the dependency behind this wall, and pinning the exact
    overloaded signature of whatever backend is in use would defeat that by
    tying the protocol to one backend's spelling.
    """

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run one state to completion and return the final state."""
        ...

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        """Run several states, concurrently where possible."""
        ...


_Body = Callable[[MASState, RunMeter | None], AgentRecord]
"""What a version 1 agent does in one visit: read the state, call its model, return its record."""

_NodeBody = Callable[[MASState, RunMeter | None], tuple[AgentRecord, dict[str, Any]]]
"""What a node does in one visit: its record, and the values it writes."""


def _current_meter(mas: MAS) -> RunMeter | None:
    """Return the meter of the run in progress, if it belongs to ``mas``.

    A graph invoked inside a node without its own config inherits the outer
    run's, so the owner check keeps a nested run of another MAS off this meter.

    Args:
        mas: The system whose node asks.

    Returns:
        The meter, or ``None`` outside a run, for a foreign object, or when the
        meter belongs to another MAS.
    """
    try:
        config = get_config()
    except RuntimeError:  # outside a run
        return None
    meter = (config.get("configurable") or {}).get(_RUN_KEY)
    if not isinstance(meter, RunMeter) or meter.owner is not mas:
        return None
    return meter


def _generate(llm: BaseLLM, rendered: str) -> tuple[str, CallMeta | None]:
    """Make a plain agent's one model call.

    Args:
        llm: The agent's model.
        rendered: The rendered prompt.

    Returns:
        The answer, and the call's metadata when the model opted in to
        ``full_response``, else ``None`` from today's ``get_response`` call.
    """
    if supports_meta(llm):
        return cast("SupportsMeta", llm).generate_with_meta(rendered)
    return llm.get_response([rendered])[0], None


def _call(llm: BaseLLM, rendered: str, meter: RunMeter | None) -> tuple[str, CallMeta | None]:
    """Make one metered model call: check the budget, call, count what it cost.

    Raises:
        BudgetExceeded: If the meter refuses the call.
        EmptyResponse: If the answer was empty; the meter counts its paid
            attempts first.
    """
    if meter is not None:
        meter.before_call()
    try:
        response, meta = _generate(llm, rendered)
    except EmptyResponse as error:  # its attempts were paid: the meter counts them, then the run fails
        if meter is not None:
            meter.after_call(error.meta)
        raise
    if meter is not None:
        meter.after_call(meta)
    return response, meta


def _plain_body(agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float]) -> _Body:
    """Build what a plain agent does in one visit: render, check the budget, call, record.

    Args:
        agent: The agent.
        llm: Its model.
        count_tokens: Token counter for the model.

    Returns:
        The agent's body, for :func:`_wrap`.
    """

    def body(state: MASState, meter: RunMeter | None) -> AgentRecord:
        prompts = state.get("prompts") or {}
        template = prompts.get(agent.name, agent.prompt)

        values = state.get("values") or {}
        inputs = {name: str(values.get(name) or "") for name in agent.inputs}
        rendered = safe_format(template, inputs)
        response, meta = _call(llm, rendered, meter)

        return AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            usage=TokenUsage(
                input_tokens=int(count_tokens(rendered)),
                output_tokens=int(count_tokens(response)),
            ),
            model=agent.model,
            raw_response=response,
            **(meta.record_fields() if meta is not None else {}),
        )

    return body


def _wrap(mas: MAS, name: str, body: _NodeBody) -> Callable[[MASState], dict[str, Any]]:
    """Turn a node's body into the node LangGraph runs.

    The node times the body, sets ``latency_s`` on its record, keeps the record
    on the run's meter, and writes the body's values into the state.

    Args:
        mas: The system, whose run meter the node reads.
        name: The node's name.
        body: What the node does in one visit.

    Returns:
        A function mapping the incoming state to this node's state updates.
    """

    def node(state: MASState) -> dict[str, Any]:
        meter = _current_meter(mas)
        if meter is not None:
            meter.start()  # the run's wall clock starts with its first agent, not with the batch
        started = time.perf_counter()
        record, writes = body(state, meter)
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started)
        if meter is not None:
            meter.keep(record)
        return {"values": writes, "records": [record], "visits": {name: 1}}

    return node


def _make_node(mas: MAS, agent: Agent, count_tokens: Callable[[str], float]) -> Callable[[MASState], dict[str, Any]]:
    """Build the callable LangGraph runs for one agent.

    Args:
        mas: The system the agent belongs to, used to resolve its model.
        agent: The agent to wrap.
        count_tokens: Token counter for the agent's model.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """
    llm: BaseLLM = mas.resolve_llm(agent)
    # an agent with tools runs the tool loop; every other agent makes one call
    body = _tools.tool_body(mas, agent, llm, count_tokens) if agent.tools else _plain_body(agent, llm, count_tokens)

    def visit(state: MASState, meter: RunMeter | None) -> tuple[AgentRecord, dict[str, Any]]:
        record = body(state, meter)
        return record, {agent.output: record.raw_response}

    return _wrap(mas, agent.name, visit)


# ------------------------------------------------------------------ version 2


def _is_v2(mas: MAS) -> bool:
    """Whether ``mas`` is a version 2 system."""
    return getattr(mas, "version", 1) == 2


def _steps(mas: MAS) -> Iterable[Step]:
    """Return the system's steps, held as a sequence or by name."""
    steps = getattr(mas, "steps", ())
    return steps.values() if isinstance(steps, dict) else steps


def _types(mas: MAS) -> dict[str, ValueType]:
    """Return the type of every value, from ``mas.info.types``, words read as types."""
    types = getattr(getattr(mas, "info", None), "types", None) or {}
    return {name: vtype if isinstance(vtype, ValueType) else parse_type(vtype) for name, vtype in types.items()}


def _usage(count_tokens: Callable[[str], float], sent: str, received: str) -> TokenUsage:
    """Estimate one call's usage."""
    return TokenUsage(input_tokens=int(count_tokens(sent)), output_tokens=int(count_tokens(received)))


def _combine(first: CallMeta | None, second: CallMeta | None) -> CallMeta | None:
    """Combine two calls' metadata; a model that gives none gives none for either."""
    return second if first is None else first if second is None else first.combine(second)


def _answer_plain(
    node: str,
    llm: BaseLLM,
    rendered: str,
    form: AnswerForm,
    count_tokens: Callable[[str], float],
    meter: RunMeter | None,
) -> tuple[str, Any, tuple[Rejected, ...], TokenUsage, CallMeta | None]:
    """Ask for an answer in its form, once more with :data:`~xmas.schemas.RETRY_NOTE` if it does not fit.

    Returns:
        The accepted text, its value, the rejected answer (if any), and usage
        and metadata summed over both calls.

    Raises:
        BadFormat: If the retry does not fit either; it carries both problems
            and both calls' metadata. The meter has counted both calls.
    """
    response, meta = _call(llm, rendered, meter)
    usage = _usage(count_tokens, rendered, response)
    try:
        return response, form.parse(response), (), usage, meta
    except FormError as first:
        retry = f"{rendered}\n\n{RETRY_NOTE.format(problem=first.problem)}"
        again, again_meta = _call(llm, retry, meter)
        usage, meta = usage + _usage(count_tokens, retry, again), _combine(meta, again_meta)
        try:
            value = form.parse(again)
        except FormError as second:
            raise BadFormat(node, (first.problem, second.problem), meta) from second
        return again, value, (Rejected(response, first.problem),), usage, meta


def _agent_body(
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float], types: dict[str, ValueType]
) -> _NodeBody:
    """Build what a version 2 agent does in one visit.

    Inputs render by their types. A json agent's prompt ends with its form's
    format block, and its answer is parsed into the value it writes: a plain
    agent retries in a second call (:func:`_answer_plain`), an agent with tools
    in one more turn of its conversation, tools withheld.

    Args:
        mas: The system, which resolves the agent's tools.
        agent: The agent.
        llm: Its model.
        count_tokens: Token counter for the model.
        types: The type of every value.

    Returns:
        The agent's body, for :func:`_wrap`.
    """
    form = compile_form(agent.schema) if parse_type(agent.output_type) == JSON else None
    tools = mas.resolve_tools(agent) if agent.tools else None
    if tools is not None and not isinstance(llm, _tools.SupportsToolCalls):  # resolve_tools checked it; this narrows
        raise ValueError(f"Agent {agent.name!r} uses tools, but its model cannot call them.")

    def body(state: MASState, meter: RunMeter | None) -> tuple[AgentRecord, dict[str, Any]]:
        template = (state.get("prompts") or {}).get(agent.name, agent.prompt)
        values = state.get("values") or {}
        inputs = {name: render_value(values.get(name), types.get(name, TEXT)) for name in agent.inputs}
        rendered = safe_format(template, inputs)
        if form is not None:
            rendered = f"{rendered}\n\n{form.format_block()}"

        calls: tuple[ToolCall, ...] = ()
        rejected: tuple[Rejected, ...] = ()
        if tools is not None:
            refused: list[Rejected] = []

            def check(text: str) -> str | None:
                try:
                    cast("AnswerForm", form).parse(text)
                except FormError as error:
                    refused.append(Rejected(text, error.problem))
                    return error.problem
                return None

            response, calls, usage, meta = _tools.run_tool_loop(
                cast("_tools.SupportsToolCalls", llm),
                rendered,
                tools,
                mas.max_tool_rounds,
                DEFAULT_SYS_PROMPT,
                count_tokens,
                meter=meter,
                check=None if form is None else check,
            )
            value: Any = response
            rejected = tuple(refused)
            if form is not None:
                try:
                    value = form.parse(response)
                except FormError as error:
                    raise BadFormat(agent.name, (*(r.error for r in refused), error.problem), meta) from error
        elif form is not None:
            response, value, rejected, usage, meta = _answer_plain(agent.name, llm, rendered, form, count_tokens, meter)
        else:
            response, meta = _call(llm, rendered, meter)
            value, usage = response, _usage(count_tokens, rendered, response)

        record = AgentRecord(
            agent=agent.name,
            step=-1,
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: render_value(value, TEXT if form is None else JSON)},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
            rejected=rejected,
            **(meta.record_fields() if meta is not None else {}),
        )
        return record, {agent.output: value}

    return body


def _step_body(mas: MAS, step: Step, types: dict[str, ValueType]) -> _NodeBody:
    """Build what a step does in one visit: read its parameters, run its callable, write.

    The meter's wall and cost limits are checked first; a step makes no model
    call, so it counts none. A parameter whose value is not in the state is
    left out and takes its default.

    Args:
        mas: The system, whose registry resolves the step's callable.
        step: The step.
        types: The type of every value, for the record's inputs.

    Returns:
        The step's body, for :func:`_wrap`.
    """
    spec, source = cast("Any", mas).resolve_callable(step)
    write_types: dict[str, ValueType] = spec.writes

    def body(state: MASState, meter: RunMeter | None) -> tuple[AgentRecord, dict[str, Any]]:
        if meter is not None:
            meter.check_boundary()
        values = state.get("values") or {}
        arguments: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for parameter in spec.reads:
            name = step.reads.get(parameter, parameter)
            inputs[name] = render_value(values.get(name), types.get(name, TEXT))
            if name in values:
                arguments[parameter] = values[name]
        try:
            produced = _tools.run_step(spec, source, arguments)
        except BadFormat as error:
            raise BadFormat(step.name, error.problems, error.meta) from error
        writes = {step.writes.get(write, write): value for write, value in produced.items()}
        outputs = {
            step.writes.get(write, write): render_value(value, write_types[write]) for write, value in produced.items()
        }
        record = AgentRecord(
            agent=step.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs=inputs,
            outputs=outputs,
            raw_response=next(iter(outputs.values())) if len(outputs) == 1 else render_value(writes, JSON),
            kind="step",
        )
        return record, writes

    return body


def _compile_v2(mas: MAS) -> CompiledGraph:
    """Compile a version 2 system: agents and steps, and a barrier join wherever several edges arrive.

    A barrier (``add_edge([...], node)``) runs its node once, after every
    source has run; nothing is deferred. Phase 2a has no conditional edges, so
    every edge in ``mas.graph`` is plain and every node fed by several is a join.
    """
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    types = _types(mas)
    counters: dict[int, Callable[[str], float]] = {}
    for agent in mas.agents.values():
        llm = mas.resolve_llm(agent)
        if id(llm) not in counters:
            counters[id(llm)] = get_token_counter(llm)
        body = _agent_body(mas, agent, llm, counters[id(llm)], types)
        builder.add_node(agent.name, _wrap(mas, agent.name, body))  # type: ignore[call-overload]
    for step in _steps(mas):
        builder.add_node(step.name, _wrap(mas, step.name, _step_body(mas, step, types)))  # type: ignore[call-overload]

    for target in mas.graph.nodes:
        sources = sorted(mas.graph.predecessors(target))
        if len(sources) > 1:
            builder.add_edge(sources, target)
        elif sources:
            builder.add_edge(sources[0], target)
    return builder.compile()


def _v2_output(mas: MAS, records: list[AgentRecord]) -> str:
    """Return the rendered write of the terminal node that ran last, or ``""`` if none ran.

    A terminal with several writes gives them as one JSON object.
    """
    terminals = set(mas.graph.predecessors(END))
    ran = [record for record in records if record.agent in terminals]
    if not ran:
        return ""
    outputs = ran[-1].outputs
    return next(iter(outputs.values())) if len(outputs) == 1 else ran[-1].raw_response


def _check_query_input(mas: MAS) -> None:
    """Refuse to run a version 2 system on a query when its inputs are not exactly ``{query: text}``.

    Raises:
        ValueError: If they are not.
    """
    if not _is_v2(mas):
        return
    inputs = {name: str(vtype) for name, vtype in (getattr(mas, "inputs", None) or {"query": TEXT}).items()}
    if inputs != {"query": str(TEXT)}:
        raise ValueError(
            f"This system's inputs are {inputs}, not {{'query': 'text'}}, so it does not run on a query: "
            "it runs as a box, or with run_state in phase 3."
        )


def _make_router(
    agent: Agent,
    back_targets: list[str],
    forward_targets: list[str],
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for an agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says, because a prompt that always emits the marker is a
    thing an optimizer will eventually propose: the agent's own visit count, and
    the graph-wide recursion limit behind it.

    Args:
        agent: The agent whose out-edges are being routed.
        back_targets: Targets reached by a back edge, taken when looping.
        forward_targets: Targets reached otherwise, taken when moving on.

    Returns:
        A function mapping state to the node names to run next.
    """
    onward = forward_targets or [END]

    def route(state: MASState) -> list[str]:
        values = state.get("values") or {}
        visits = (state.get("visits") or {}).get(agent.name, 0)
        looping = CONTINUE_MARKER in str(values.get(agent.output) or "")
        return back_targets if looping and visits < MAX_NODE_VISITS else onward

    return route


def compile_mas(mas: MAS) -> CompiledGraph:
    """Compile a MAS topology into an executable graph.

    Agents identified by :func:`xmas.graph.join_nodes` are marked deferred, so a
    node fed by predecessors at unequal depth runs exactly once per query rather
    than once per wave of arriving inputs.

    Cycle-closing edges compile to conditional branches rather than to plain
    edges. A static edge set cannot express a loop, because the edge back and the
    edge onward would both fire on every pass and the run would spin until it hit
    the recursion limit. Routing the two exclusively, on the looping agent's own
    output, is what makes a cyclic topology terminate with an answer.

    Args:
        mas: The system to compile.

    A version 2 system compiles by :func:`_compile_v2` instead.

    Returns:
        The compiled graph.
    """
    if _is_v2(mas):
        return _compile_v2(mas)
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    joins = join_nodes(mas.graph)
    back = back_edges(mas.graph)

    counters: dict[int, Callable[[str], float]] = {}
    for name, agent in mas.agents.items():
        llm = mas.resolve_llm(agent)
        if id(llm) not in counters:
            counters[id(llm)] = get_token_counter(llm)
        # langgraph's add_node overloads bind their node type to a protocol that a
        # typing.TypedDict class does not satisfy under mypy, even though passing a
        # TypedDict schema is the documented usage. Runtime behaviour is covered by
        # the join-semantics and quickstart tests.
        builder.add_node(  # type: ignore[call-overload]
            name,
            _make_node(mas, agent, counters[id(llm)]),
            defer=name in joins,
        )

    for source in mas.graph.nodes:
        targets = sorted(mas.graph.successors(source))
        back_targets = [target for target in targets if (source, target) in back]
        if not back_targets:
            for target in targets:
                builder.add_edge(source, target)
            continue
        forward_targets = [target for target in targets if (source, target) not in back]
        router = _make_router(mas.agents[source], back_targets, forward_targets)
        builder.add_conditional_edges(source, router, sorted({*back_targets, *forward_targets, END}))

    return builder.compile()


def _to_trace(mas: MAS, query: str, result: dict[str, Any], meter: RunMeter | None = None) -> Trace:  # noqa: ARG001 (meter: for phase 2b's health)
    """Assemble a trace from a finished run's state.

    A version 2 trace's output is the rendered write of the terminal that ran.
    """
    records: list[AgentRecord] = list(result.get("records") or [])
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    if _is_v2(mas):
        return Trace(query=query, output=_v2_output(mas, records), records=numbered)
    values: dict[str, Any] = result.get("values") or {}
    output_var = mas.agents[terminal_agent(mas.graph)].output
    return Trace(query=query, output=str(values.get(output_var) or ""), records=numbered)


def _kind_of(error: BaseException) -> ErrorKind:
    """Return the kind of the error that ended a run.

    Hitting ``max_steps`` is a step budget running out, so it is ``BUDGET``;
    anything else is :func:`xmas.errors.classify`'s answer.
    """
    if isinstance(error, GraphRecursionError):
        return ErrorKind.BUDGET
    return classify(error)


def _failed_trace(mas: MAS, query: str, error: BaseException, meter: RunMeter) -> Trace:  # noqa: ARG001 (mas: same shape as _to_trace)
    """Assemble a failed run's trace from the records its meter kept.

    Args:
        mas: The system that ran.
        query: The input.
        error: What ended the run.
        meter: The run's meter, holding the finished agents' records in
            wall-clock order.

    Returns:
        The trace, with ``error`` as ``"Type: message"`` and its kind.
    """
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(meter.records))
    return Trace(
        query=query,
        output="",
        records=numbered,
        error=f"{type(error).__name__}: {error}",
        error_kind=_kind_of(error),
    )


def _initial_state(query: str, prompts: dict[str, str]) -> MASState:
    """Build the state a run starts from."""
    return MASState(values={"query": query}, prompts=dict(prompts), records=[], visits={})


def _run_config(mas: MAS, meter: RunMeter) -> dict[str, Any]:
    """Build one run's config: the step cap, and the meter the nodes read."""
    return {"recursion_limit": mas.max_steps, "configurable": {_RUN_KEY: meter}}


def run_one(mas: MAS, query: str, prompts: dict[str, str]) -> Trace:
    """Execute a compiled MAS against one input with a given prompt set.

    Args:
        mas: The system to run.
        query: The input.
        prompts: Prompt text per agent name, overriding what the agents hold.

    Returns:
        The trace of the run. A failure is recorded in ``Trace.error`` and
        ``Trace.error_kind`` rather than raised, so one bad rollout does not lose
        a whole batch, and the trace keeps the records of the agents that
        finished.

    Raises:
        ValueError: If a version 2 system's inputs are not exactly ``{query: text}``.
    """
    _check_query_input(mas)
    meter = RunMeter(mas.budget, owner=mas)
    try:
        result: dict[str, Any] = mas.compiled.invoke(_initial_state(query, prompts), config=_run_config(mas, meter))
    except Exception as error:  # a failed rollout is data, not a crash
        return _failed_trace(mas, query, error, meter)
    return _to_trace(mas, query, result, meter)


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Execute a MAS over parallel lists of inputs and prompt sets.

    Every item runs once, with its own meter and budget. An item that fails
    becomes a failed trace; the others are unaffected.

    Args:
        mas: The system to run.
        queries: The inputs.
        prompts: Prompt sets, one per input.

    Returns:
        One trace per input, in the order given.

    Raises:
        ValueError: If a version 2 system's inputs are not exactly ``{query: text}``.
    """
    _check_query_input(mas)
    if not queries:
        return []

    states = [_initial_state(query, prompt_set) for query, prompt_set in zip(queries, prompts, strict=True)]
    meters = [RunMeter(mas.budget, owner=mas) for _ in states]
    configs = [_run_config(mas, meter) for meter in meters]
    results: list[Any] = mas.compiled.batch(states, config=configs, return_exceptions=True)
    return [
        _failed_trace(mas, query, result, meter)
        if isinstance(result, BaseException)
        else _to_trace(mas, query, result, meter)
        for query, result, meter in zip(queries, results, meters, strict=True)
    ]
