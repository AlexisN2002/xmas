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
its answer form, with one retry; a node with conditional or error edges routes
by a :class:`_Decision` after each visit; a join is a barrier that runs once per
pass, after all its groups, where version 1 defers; and a box runs a whole
inner system as one node, on the outer run's meter.
"""

from __future__ import annotations

import dataclasses
import operator
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Protocol, TypedDict, cast

from langgraph.config import get_config
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from xmas import tools as _tools
from xmas.agent import Agent
from xmas.budget import RunMeter
from xmas.check import step_maps
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import BadFormat, EmptyResponse, ErrorKind, classify
from xmas.graph import END, Join, back_edges, join_nodes, terminal_agent
from xmas.health import Health, summarize
from xmas.llms.meta import SupportsMeta, supports_meta
from xmas.nodes import Box, Step
from xmas.predicates import Context, Predicate, parse_predicate
from xmas.prompts import safe_format
from xmas.schemas import RETRY_NOTE, AnswerForm, FormError, compile_form
from xmas.spec import SPEC_VERSION_2
from xmas.trace import AgentRecord, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, parse_type, render_value

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.check import SystemInfo
    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS


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


_RUN_ENDS = frozenset({ErrorKind.BUDGET, ErrorKind.STOPPED})
"""Error kinds that end the run itself: never routed, and no failed record, since no node failed."""


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
        routes: Where each deciding node's latest visit sent the run (version
            2), keyed by node; absent until a node decides.
    """

    values: Annotated[dict[str, Any], _merge_values]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    visits: Annotated[dict[str, int], _merge_counts]
    routes: Annotated[dict[str, list[str]], _merge_values]


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
    if not isinstance(meter, RunMeter) or not meter.owns(mas):
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


def _wrap(
    mas: MAS, name: str, body: _NodeBody, decision: _Decision | None = None
) -> Callable[[MASState], dict[str, Any]]:
    """Turn a node's body into the node LangGraph runs.

    The node times the body, sets ``latency_s`` on its record, keeps the record
    on the run's meter, and writes the body's values into the state. A node
    with a decision then writes where it goes in ``routes``.

    A version 2 body that raises gets a failed record (:func:`_failed_record`),
    unless the error ends the run (:data:`_RUN_ENDS`). When an error route
    matches the error's kind, the visit writes nothing and takes the route;
    otherwise the record is kept on the meter for the failed run's trace and
    the error goes on. Version 1 raises as it always did.

    Args:
        mas: The system, whose run meter the node reads.
        name: The node's name.
        body: What the node does in one visit.
        decision: The node's routing, for a version 2 node with conditional or
            error edges.

    Returns:
        A function mapping the incoming state to this node's state updates.
    """

    def node(state: MASState) -> dict[str, Any]:
        meter = _current_meter(mas)
        if meter is not None:
            meter.start()  # the run's wall clock starts with its first agent, not with the batch
        started = time.perf_counter()
        route: list[str] | None = None
        try:
            record, writes = body(state, meter)
        except Exception as error:
            kind = _kind_of(error)
            if mas.version != SPEC_VERSION_2 or kind in _RUN_ENDS:
                raise
            target = None if decision is None else decision.on_error(kind)
            record = _failed_record(mas, name, error, kind, time.perf_counter() - started, target)
            if meter is not None:
                meter.keep(record, mas)
            if target is None:
                raise
            return {"records": [record], "visits": {name: 1}, "routes": {name: [target]}}
        if decision is not None:
            route, capped = decision.route(_context(mas, name, state, record, writes, meter))
            record = dataclasses.replace(record, route=tuple(route), capped=capped)
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started)
        if meter is not None:
            meter.keep(record, mas)
        update: dict[str, Any] = {"values": writes, "records": [record], "visits": {name: 1}}
        if route is not None:
            update["routes"] = {name: route}
        return update

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
        inputs = {name: render_value(values.get(name), types[name]) for name in agent.inputs}
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
    spec, source = mas.resolve_callable(step)
    read_map, write_map = step_maps(step, spec)

    def body(state: MASState, meter: RunMeter | None) -> tuple[AgentRecord, dict[str, Any]]:
        if meter is not None:
            meter.check_boundary()
        values = state.get("values") or {}
        arguments: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for parameter, name in read_map.items():
            inputs[name] = render_value(values.get(name), types[name])
            if name in values:
                arguments[parameter] = values[name]
        try:
            produced = _tools.run_step(spec, source, arguments)
        except BadFormat as error:
            raise BadFormat(step.name, error.problems, error.meta) from error
        writes = {write_map[write]: value for write, value in produced.items()}
        outputs = {name: render_value(value, types[name]) for name, value in writes.items()}
        record = AgentRecord(
            agent=step.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs=inputs,
            outputs=outputs,
            raw_response=_raw(outputs, writes),
            kind="step",
        )
        return record, writes

    return body


def _raw(outputs: dict[str, str], writes: dict[str, Any]) -> str:
    """Return a node's raw response: its one rendered write, or all its writes as one JSON object."""
    return next(iter(outputs.values())) if len(outputs) == 1 else render_value(writes, JSON)


def _box_body(box: Box, types: dict[str, ValueType]) -> _NodeBody:
    """Build what a box does in one visit: run its inner system from the mapped values.

    The inner system runs on the outer run's meter, which adopts it, so its
    calls count against the outer budget. Its failure raises out of the box, so
    the box's own error edges route by the inner error's kind.

    Args:
        box: The box.
        types: The type of every outer value.

    Returns:
        The box's body, for :func:`_wrap`.
    """
    inner: MAS = box.system

    def body(state: MASState, meter: RunMeter | None) -> tuple[AgentRecord, dict[str, Any]]:
        if meter is None:  # the graph was invoked outside a run: the inner system counts on its own meter
            meter = RunMeter(inner.budget, owner=inner)
        meter.check_boundary()
        meter.adopt(inner)
        values = state.get("values") or {}
        given = {inside: values[outside] for inside, outside in box.reads.items() if outside in values}
        trace, produced = _run_values(inner, given, inner.prompts(), meter)
        writes = {outside: produced.get(inside) for inside, outside in box.writes.items()}
        outputs = {name: render_value(value, types[name]) for name, value in writes.items()}
        costs = [record.cost_usd for record in trace.records if record.cost_usd is not None]
        tokens = [record.tokens for record in trace.records if record.tokens is not None]
        record = AgentRecord(
            agent=box.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs={outside: render_value(values.get(outside), types[outside]) for outside in box.reads.values()},
            outputs=outputs,
            usage=trace.usage,
            raw_response=_raw(outputs, writes),
            cost_usd=sum(costs) if costs else None,
            tokens=sum(tokens[1:], tokens[0]) if tokens else None,
            kind="box",
            inner=trace,
        )
        return record, writes

    return body


@dataclasses.dataclass(frozen=True)
class _Decision:
    """Where a version 2 node with conditional or error edges goes after a visit.

    Attributes:
        plain: Targets taken after every visit that did not fail.
        choices: The ordered when-group: ``when``, ``while`` and ``switch``
            edges, each with its predicate (a switch compares its path with its
            case). The first that holds is taken.
        otherwise: The ``else`` target, taken when no choice is.
        errors: The error edges, in order.
    """

    plain: tuple[str, ...]
    choices: tuple[tuple[Edge, Predicate], ...]
    otherwise: str | None
    errors: tuple[Edge, ...]

    @classmethod
    def of(cls, edges: list[Edge]) -> _Decision:
        """Build the decision of one node from its out-edges, in declared order."""
        choices = []
        for edge in edges:
            if edge.kind is EdgeKind.SWITCH:
                path = ("path", tuple(str(edge.switch).split(".")))
                compare = ("cmp", "==", path, ("lit", edge.case))
                choices.append((edge, Predicate(f"{edge.switch} == {edge.case!r}", compare)))
            elif edge.kind in (EdgeKind.WHEN, EdgeKind.WHILE):
                choices.append((edge, parse_predicate(str(edge.when))))
        return cls(
            plain=tuple(edge.target for edge in edges if edge.kind is EdgeKind.PLAIN),
            choices=tuple(choices),
            otherwise=next((edge.target for edge in edges if edge.kind is EdgeKind.ELSE), None),
            errors=tuple(edge for edge in edges if edge.kind is EdgeKind.ERROR),
        )

    def route(self, ctx: Context) -> tuple[list[str], bool]:
        """Return the targets after a visit that did not fail, and whether a ``while`` cap was hit.

        A ``while`` edge that holds while its target has run ``max_visits``
        times is skipped and marks the visit capped; the loop's exit is then
        the ``else`` target, since the other choices were written for the case
        where the loop's condition does not hold.
        """
        exit_ = [*self.plain, *([self.otherwise] if self.otherwise else [])]
        for edge, predicate in self.choices:
            if not predicate.evaluate(ctx):
                continue
            if edge.kind is EdgeKind.WHILE and ctx.visits.get(edge.target, 0) >= cast("int", edge.max_visits):
                return exit_, True
            return [*self.plain, edge.target], False
        return exit_, False

    def on_error(self, kind: ErrorKind) -> str | None:
        """Return the target of the first error edge routing ``kind``; :data:`_RUN_ENDS` are never routed."""
        if kind in _RUN_ENDS:
            return None
        for edge in self.errors:
            if ANY_ERROR in edge.errors or kind.value in edge.errors:
                return edge.target
        return None


def _context(
    mas: MAS, name: str, state: MASState, record: AgentRecord, writes: dict[str, Any], meter: RunMeter | None
) -> Context:
    """Return what a node's predicates see: its fresh writes, its visit counted, the spend so far."""
    visits = dict(state.get("visits") or {})
    visits[name] = visits.get(name, 0) + 1
    if meter is not None:
        cost: float | None = meter.cost_usd
    else:
        cost = sum(r.cost_usd for r in [*(state.get("records") or []), record] if r.cost_usd is not None)

    def call(function: str, args: tuple[Any, ...]) -> bool:
        spec, source = mas.resolve_callable(Step(function, function))
        produced = _tools.run_step(spec, source, dict(zip(spec.reads, args, strict=True)))
        return cast("bool", next(iter(produced.values())))

    return Context(
        values={**(state.get("values") or {}), **writes},
        visits=visits,
        node=name,
        cost_usd=cost,
        elapsed_s=None if meter is None else meter.elapsed_s,
        call=call,
    )


def _failed_record(
    mas: MAS, name: str, error: BaseException, kind: ErrorKind, latency_s: float, target: str | None
) -> AgentRecord:
    """Return the record of a version 2 visit that failed: no outputs, the paid attempts' cost, the route taken."""
    meta: CallMeta | None = getattr(error, "meta", None)  # EmptyResponse and BadFormat carry their attempts
    paid = {**(meta.record_fields() if meta is not None else {}), "error_kind": kind}
    return AgentRecord(
        agent=name,
        step=-1,
        prompt="",
        rendered_prompt="",
        inputs={},
        outputs={},
        latency_s=latency_s,
        kind="agent" if name in mas.agents else "step" if name in mas.steps else "box",
        error=f"{type(error).__name__}: {error}",
        route=None if target is None else (target,),
        **paid,
    )


def _out_edges(mas: MAS) -> dict[str, list[Edge]]:
    """Return each node's out-edges: the declared ones in order, then the implicit start and end wiring."""
    out: dict[str, list[Edge]] = {}
    for edge in mas.edges:
        out.setdefault(edge.source, []).append(edge)
    declared = {(edge.source, edge.target) for edge in mas.edges}
    for source, target in mas.graph.edges:
        if (source, target) not in declared:
            out.setdefault(source, []).append(Edge(source, target))
    return out


def _relay(state: MASState) -> dict[str, Any]:  # noqa: ARG001 (a node's signature)
    """Pass a group's arrival on to a join's barrier; writes nothing."""
    return {}


def _add_joins(
    builder: StateGraph[MASState, Any, MASState, MASState], joins: Mapping[str, Join]
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    """Add one barrier per join: it waits for each of the join's groups once per pass.

    A group without a relay (one plain edge from a node without error edges)
    is waited on at its node. Any other group gets the no-op relay node
    :func:`~xmas.graph.classify_joins` named (``__in__.<join>.<k>``: LangGraph
    reserves ``:`` in node names) that its sources route to. A merge needs
    nothing: its edges stay plain, and it runs on whichever arrives.

    Args:
        builder: The graph being built.
        joins: ``mas.info.joins``.

    Returns:
        The (source, join) pairs the barrier waits on directly, and the relay
        each other (source, join) pair goes to.
    """
    direct: set[tuple[str, str]] = set()
    relays: dict[tuple[str, str], str] = {}
    for node, join in joins.items():
        if join.kind != "join":
            continue
        starts = []
        for group, relay in zip(join.groups, join.relays, strict=True):
            if relay is None:
                direct.add((group[0], node))
                starts.append(group[0])
                continue
            builder.add_node(relay, _relay)
            relays.update(((member, node), relay) for member in group)
            starts.append(relay)
        builder.add_edge(starts, node)
    return direct, relays


def _router(name: str, targets: dict[str, str]) -> Callable[[MASState], list[str]]:
    """Build the router of a deciding node: it reads the route its wrapper wrote, mapped to relays."""

    def route(state: MASState) -> list[str]:
        chosen = (state.get("routes") or {}).get(name, [])
        return [targets[target] for target in chosen if target in targets]

    return route


def _compile_v2(mas: MAS) -> CompiledGraph:
    """Compile a version 2 system: agents, steps and boxes, decisions, merges and joins.

    A node with conditional or error edges gets a :class:`_Decision`; its
    wrapper evaluates it after each visit and writes ``routes[node]``, which
    the node's conditional edge only reads. A join is a barrier
    (:func:`_add_joins`) that runs its node once per pass; nothing is
    deferred. Every other edge is plain.
    """
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    info = cast("SystemInfo", mas.info)  # check_system set it when the system was built
    types = info.types
    out = _out_edges(mas)
    decisions = {
        source: _Decision.of(edges) for source, edges in out.items() if any(e.kind is not EdgeKind.PLAIN for e in edges)
    }
    bodies: dict[str, _NodeBody] = {}
    counters: dict[int, Callable[[str], float]] = {}
    for agent in mas.agents.values():
        llm = mas.resolve_llm(agent)
        if id(llm) not in counters:
            counters[id(llm)] = get_token_counter(llm)
        bodies[agent.name] = _agent_body(mas, agent, llm, counters[id(llm)], types)
    for step in mas.steps.values():
        bodies[step.name] = _step_body(mas, step, types)
    for box in mas.boxes.values():
        bodies[box.name] = _box_body(box, types)
    for name, body in bodies.items():
        builder.add_node(name, _wrap(mas, name, body, decisions.get(name)))  # type: ignore[arg-type]

    direct, relays = _add_joins(builder, info.joins)
    for source, edges in out.items():
        targets = {
            edge.target: relays.get((source, edge.target), edge.target)
            for edge in edges
            if (source, edge.target) not in direct
        }
        if source in decisions and targets:
            builder.add_conditional_edges(source, _router(source, targets), sorted(set(targets.values())))
        elif source not in decisions:
            for target in targets.values():
                builder.add_edge(source, target)
    return builder.compile()


def _v2_output(mas: MAS, records: list[AgentRecord]) -> str:
    """Return the rendered write of the terminal node that ran last, or ``""`` if none ran.

    A terminal with several writes gives them as one JSON object.
    """
    terminals = set(mas.graph.predecessors(END))
    ran = [record for record in records if record.agent in terminals and record.error is None]
    if not ran:
        return ""
    outputs = ran[-1].outputs
    return next(iter(outputs.values())) if len(outputs) == 1 else ran[-1].raw_response


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
    if mas.version == SPEC_VERSION_2:
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
        builder.add_node(
            name,
            _make_node(mas, agent, counters[id(llm)]),  # type: ignore[arg-type]
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


def _health(mas: MAS, records: tuple[AgentRecord, ...], meter: RunMeter | None) -> Health:
    """Summarize a run's health with the system's settings and its ``while`` edges."""
    loops = [(edge.source, edge.target) for edge in mas.edges if edge.kind is EdgeKind.WHILE]
    return summarize(records, meter=meter, settings=mas.health, while_edges=loops)


def _to_trace(mas: MAS, query: str, result: dict[str, Any], meter: RunMeter | None = None) -> Trace:
    """Assemble a trace from a finished run's state, its health summarized.

    A version 2 trace's output is the rendered write of the terminal that ran.
    Without a meter the health has no budget entry.
    """
    records: list[AgentRecord] = list(result.get("records") or [])
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    health = _health(mas, numbered, meter)
    if mas.version == SPEC_VERSION_2:
        return Trace(query=query, output=_v2_output(mas, records), records=numbered, health=health)
    values: dict[str, Any] = result.get("values") or {}
    output_var = mas.agents[terminal_agent(mas.graph)].output
    return Trace(query=query, output=str(values.get(output_var) or ""), records=numbered, health=health)


def _kind_of(error: BaseException) -> ErrorKind:
    """Return the kind of the error that ended a run.

    Hitting ``max_steps`` is a step budget running out, so it is ``BUDGET``;
    anything else is :func:`xmas.errors.classify`'s answer.
    """
    if isinstance(error, GraphRecursionError):
        return ErrorKind.BUDGET
    return classify(error)


def _failed_trace(mas: MAS, query: str, error: BaseException, meter: RunMeter) -> Trace:
    """Assemble a failed run's trace from the records its meter kept, its health summarized.

    A version 2 run's records include the failed visit's own.

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
        health=_health(mas, numbered, meter),
    )


def _initial_state(query: str, prompts: dict[str, str]) -> MASState:
    """Build the state a run starts from."""
    return MASState(values={"query": query}, prompts=dict(prompts), records=[], visits={})


def _run_values(
    mas: MAS, values: dict[str, Any], prompts: dict[str, str], meter: RunMeter
) -> tuple[Trace, dict[str, Any]]:
    """Run a version 2 system from given values on a given meter, as a box runs its inner system.

    Args:
        mas: The system to run.
        values: Its inputs, by name.
        prompts: Prompt text per agent name.
        meter: The meter to count on; it owns or has adopted ``mas``.

    Returns:
        The run's trace (its query is the inputs as one JSON object; its health
        has no budget entry, which belongs to the meter's owner), and the
        run's final values.

    Raises:
        Exception: Whatever ended the run; nothing is caught.
    """
    state = MASState(values=dict(values), prompts=dict(prompts), records=[], visits={})
    result: dict[str, Any] = mas.compiled.invoke(state, config=_run_config(mas, meter))
    return _to_trace(mas, render_value(values, JSON), result), dict(result.get("values") or {})


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
    """
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
    """
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
