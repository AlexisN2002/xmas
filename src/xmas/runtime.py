"""Compile a MAS and run it.

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

from xmas.agent import Agent
from xmas.budget import RunMeter, VisitMeter
from xmas.check import step_maps
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import BadFormat, BudgetExceeded, ErrorKind, classify
from xmas.graph import END, Join, back_edges, join_nodes, terminal_agent
from xmas.health import Health, summarize
from xmas.llms.meta import SupportsMeta, supports_meta
from xmas.nodes import Box, Step
from xmas.predicates import Context, Predicate, parse_predicate
from xmas.prompts import safe_format
from xmas.schemas import RETRY_NOTE, AnswerForm, FormError, compile_form
from xmas.spec import SPEC_VERSION_2
from xmas.tools import run_step, run_tool_loop
from xmas.trace import AgentRecord, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, parse_type, render_value

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.check import SystemInfo
    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on."""

_RUN_KEY = "xmas_run"
"""Key of the run's :class:`~xmas.budget.RunMeter` in ``config["configurable"]``."""

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


class MASState(TypedDict, total=False):
    """The state a run threads through the graph.

    Attributes:
        values: Every state variable, including ``query``, keyed by name: a
            ``str`` for text, plain JSON data for json (version 2). Version 1
            writes only ``str``. Nodes writing concurrently each contribute
            their own key, so the reducer is a plain dict union.
        prompts: The prompt set for this run, keyed by agent name. Prompts travel
            in the state rather than being baked into the compiled graph, so one
            compilation serves every candidate an optimizer proposes.
        records: Accumulated per-agent records, in completion order.
        visits: How often each node has run, which bounds a version 1 retry
            loop and is what a version 2 ``while`` edge and ``visits(x)`` read.
        routes: Where each deciding node's latest visit sent the run (version
            2), keyed by node; absent until a node decides.
    """

    values: Annotated[dict[str, Any], operator.or_]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    visits: Annotated[dict[str, int], _merge_counts]
    routes: Annotated[dict[str, list[str]], operator.or_]


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


_NodeBody = Callable[[MASState, VisitMeter], tuple[AgentRecord, dict[str, Any]]]
"""What a node does in one visit, on its share of the run's meter: its record, and the values it writes."""


def _current_meter(mas: MAS) -> RunMeter:
    """Return the run's meter.

    A graph invoked without a run config, or inheriting another system's (a
    nested run inside a node), counts on a meter of its own.
    """
    meter = get_config().get("configurable", {}).get(_RUN_KEY)
    return meter if isinstance(meter, RunMeter) and meter.owns(mas) else RunMeter(mas.budget, owner=mas)


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


def _call(llm: BaseLLM, rendered: str, meter: VisitMeter, *, batch_error: bool = False) -> tuple[str, CallMeta | None]:
    """Make one metered model call.

    With ``batch_error`` (a version 1 agent) a failed call raises as
    promptolution's batch call does, ``RuntimeError: LLM batch failed: ...``,
    whichever path the model takes; :func:`~xmas.errors.classify` reads the
    kind from the cause.
    """
    try:
        return meter.metered(lambda: _generate(llm, rendered))
    except Exception as error:
        if not batch_error or not supports_meta(llm) or isinstance(error, BudgetExceeded):
            raise  # promptolution's get_response wrapped it already, or the meter refused the call
        raise RuntimeError(f"LLM batch failed: {error}") from error


def _wrap(
    mas: MAS, name: str, body: _NodeBody, decision: _Decision | None = None
) -> Callable[[MASState], dict[str, Any]]:
    """Turn a node's body into the node LangGraph runs.

    The node times the body, sets ``latency_s`` on its record, keeps the record
    on the run's meter, and writes the body's values into the state. A node
    with a decision then writes where it goes in ``routes``.

    A version 2 body that raises, or a decision that raises, gets a failed
    record (:func:`_failed_record`) holding every paid call of the visit,
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
        meter.start()  # the run's wall clock starts with its first agent, not with the batch
        started = time.perf_counter()
        visit = VisitMeter(meter)
        route: list[str] | None = None
        try:
            record, writes = body(state, visit)
            if decision is not None:
                route, capped = decision.route(_context(mas, name, state, writes, meter))
                record = dataclasses.replace(record, route=tuple(route), capped=capped)
        except Exception as error:
            kind = _kind_of(error)
            if mas.version != SPEC_VERSION_2 or kind in _RUN_ENDS:
                raise
            target = None if decision is None else decision.on_error(kind)
            record = _failed_record(mas, name, error, kind, time.perf_counter() - started, target, visit)
            meter.keep(record, mas)
            if target is None:
                raise
            return {"records": [record], "visits": {name: 1}, "routes": {name: [target]}}
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started)
        meter.keep(record, mas)
        update: dict[str, Any] = {"values": writes, "records": [record], "visits": {name: 1}}
        if route is not None:
            update["routes"] = {name: route}
        return update

    return node


def agent_node(
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float]
) -> Callable[[MASState], dict[str, Any]]:
    """Build the callable LangGraph runs for one version 1 agent.

    An agent with tools answers through :func:`xmas.tools.run_tool_loop`; every
    other agent makes one model call. Each visit starts fresh. :func:`_wrap`
    times the visit, keeps its record on the run's meter and writes the answer.

    Args:
        mas: The system the agent belongs to, which resolves its tools.
        agent: The agent to wrap.
        llm: Its model.
        count_tokens: Token counter for the model.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """
    tools = mas.resolve_tools(agent) if agent.tools else {}

    def visit(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        template = (state.get("prompts") or {}).get(agent.name, agent.prompt)
        values = state.get("values") or {}
        inputs = {name: str(values.get(name) or "") for name in agent.inputs}
        rendered = safe_format(template, inputs)
        calls: tuple[ToolCall, ...] = ()
        if agent.tools and hasattr(llm, "get_tool_response"):  # resolve_tools checked the model can call tools
            response, calls, usage, meta = run_tool_loop(
                llm, rendered, tools, mas.max_tool_rounds, DEFAULT_SYS_PROMPT, count_tokens, meter=meter
            )
        else:
            response, meta = _call(llm, rendered, meter, batch_error=True)
            usage = _usage(count_tokens, rendered, response)
        record = AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
            prompt=template,
            rendered_prompt=rendered,
            inputs=inputs,
            outputs={agent.output: response},
            tool_calls=calls,
            usage=usage,
            model=agent.model,
            raw_response=response,
            **(meta.record_fields() if meta is not None else {}),
        )
        return record, {agent.output: response}

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
    meter: VisitMeter,
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

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        template = (state.get("prompts") or {}).get(agent.name, agent.prompt)
        values = state.get("values") or {}
        inputs = {name: render_value(values.get(name), types[name], present=name in values) for name in agent.inputs}
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

            response, calls, usage, meta = run_tool_loop(
                llm,
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

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        meter.check_boundary()
        values = state.get("values") or {}
        arguments: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for parameter, name in read_map.items():
            inputs[name] = render_value(values.get(name), types[name], present=name in values)
            if name in values:
                arguments[parameter] = values[name]
        try:
            produced = run_step(spec, source, arguments)
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
            cost_usd=0.0,  # a step makes no model call
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
    the box's own error edges route by the inner error's kind; the visit keeps
    the inner trace so far for the box's failed record.

    Args:
        box: The box.
        types: The type of every outer value.

    Returns:
        The box's body, for :func:`_wrap`.
    """
    inner: MAS = box.system

    def body(state: MASState, visit: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        run = visit.run
        run.check_boundary()
        run.adopt(inner)
        values = state.get("values") or {}
        given = {inside: values[outside] for inside, outside in box.reads.items() if outside in values}
        kept = len(run.records_of(inner))
        try:
            trace, produced = _run_values(inner, given, inner.prompts(), run)
        except Exception as error:
            visit.inner = _failed_trace(inner, render_value(given, JSON), error, run, kept)
            raise
        writes = {outside: produced.get(inside) for inside, outside in box.writes.items()}
        outputs = {
            outside: render_value(writes[outside], types[outside], present=inside in produced)
            for inside, outside in box.writes.items()
        }
        record = AgentRecord(
            agent=box.name,
            step=-1,
            prompt="",
            rendered_prompt="",
            inputs={
                outside: render_value(values.get(outside), types[outside], present=outside in values)
                for outside in box.reads.values()
            },
            outputs=outputs,
            usage=trace.usage,
            raw_response=_raw(outputs, writes),
            kind="box",
            inner=trace,
            **_paid_inside(trace),
        )
        return record, writes

    return body


def _paid_inside(trace: Trace) -> dict[str, Any]:
    """Return a box record's cost and tokens: what its inner records report, summed; ``None`` when none does."""
    costs = [record.cost_usd for record in trace.records if record.cost_usd is not None]
    tokens = [record.tokens for record in trace.records if record.tokens is not None]
    return {"cost_usd": sum(costs) if costs else None, "tokens": sum(tokens[1:], tokens[0]) if tokens else None}


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


def _context(mas: MAS, name: str, state: MASState, writes: dict[str, Any], meter: RunMeter) -> Context:
    """Return what a node's predicates see: its fresh writes, its visit counted, the spend so far."""
    visits = dict(state.get("visits") or {})
    visits[name] = visits.get(name, 0) + 1

    def call(function: str, args: tuple[Any, ...]) -> bool:
        spec, source = mas.resolve_callable(Step(function, function))
        produced = run_step(spec, source, dict(zip(spec.reads, args, strict=True)))
        return cast("bool", next(iter(produced.values())))

    return Context(
        values={**(state.get("values") or {}), **writes},
        visits=visits,
        node=name,
        cost_usd=meter.cost_usd,
        elapsed_s=meter.elapsed_s,
        call=call,
    )


def _failed_record(
    mas: MAS,
    name: str,
    error: BaseException,
    kind: ErrorKind,
    latency_s: float,
    target: str | None,
    visit: VisitMeter,
) -> AgentRecord:
    """Return the record of a version 2 visit that failed: no outputs, the route taken, and what it paid.

    The cost and tokens are those of every call the visit made (outside a run,
    the attempts the error carries); a step's are ``0``, and a box's those of
    its inner trace, which the record holds.
    """
    node = mas.nodes[name]
    inner = visit.inner
    meta = visit.meta
    paid: dict[str, Any] = meta.record_fields() if meta is not None else {}
    if isinstance(node, Step):
        paid["cost_usd"] = 0.0
    elif inner is not None:
        paid.update(_paid_inside(inner))
    paid["error_kind"] = kind
    return AgentRecord(
        agent=name,
        step=-1,
        prompt="",
        rendered_prompt="",
        inputs={},
        outputs={},
        latency_s=latency_s,
        kind="agent" if isinstance(node, Agent) else "step" if isinstance(node, Step) else "box",
        error=f"{type(error).__name__}: {error}",
        route=None if target is None else (target,),
        inner=inner,
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
    for agent in mas.agents.values():
        llm = mas.resolve_llm(agent)
        bodies[agent.name] = _agent_body(mas, agent, llm, get_token_counter(llm), types)
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


def cycle_router(
    mas: MAS, agent: Agent, back_targets: list[str], forward_targets: list[str]
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for a version 1 agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says: the agent's own visit count against ``mas.max_visits``,
    and the graph-wide recursion limit behind it.

    Args:
        mas: The system, which bounds the visits.
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
        return back_targets if looping and visits < mas.max_visits else onward

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

    A version 2 system compiles by :func:`_compile_v2` instead.

    Args:
        mas: The system to compile.

    Returns:
        The compiled graph.
    """
    if mas.version == SPEC_VERSION_2:
        return _compile_v2(mas)
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    joins = join_nodes(mas.graph)
    back = back_edges(mas.graph)

    for name, agent in mas.agents.items():
        llm = mas.resolve_llm(agent)
        # mypy rejects a TypedDict schema here; langgraph documents it as the usage.
        builder.add_node(
            name,
            agent_node(mas, agent, llm, get_token_counter(llm)),  # type: ignore[arg-type]
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
        router = cycle_router(mas, mas.agents[source], back_targets, forward_targets)
        builder.add_conditional_edges(source, router, back_targets + forward_targets + [END])

    return builder.compile()


def _health(mas: MAS, records: tuple[AgentRecord, ...], meter: RunMeter) -> Health:
    """Summarize a run's health; the budget entry is the meter owner's alone."""
    values = mas.info.types if mas.info is not None else {"query", *(a.output for a in mas.agents.values())}
    own = meter if mas is meter.owner else None
    return summarize(records, meter=own, settings=mas.health, while_edges=mas.while_edges, values=values)


def trace_from_state(mas: MAS, query: str, result: Any, meter: RunMeter) -> Trace:
    """Assemble a trace from a finished run's state, or from the error that ended it."""
    if isinstance(result, BaseException):
        return _failed_trace(mas, query, result, meter)
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


def _failed_trace(mas: MAS, query: str, error: BaseException, meter: RunMeter, start: int = 0) -> Trace:
    """Assemble a failed run's trace from the records its meter kept, its health summarized.

    A version 2 run's records include the failed visit's own.

    Args:
        mas: The system that ran: the meter's owner, or a system it adopted.
        query: The input.
        error: What ended the run.
        meter: The run's meter, holding the finished agents' records in
            wall-clock order.
        start: How many of ``mas``'s kept records belong to earlier runs on
            the meter (an adopted box system's earlier visits), left out.

    Returns:
        The trace, with ``error`` as ``"Type: message"`` and its kind.
    """
    kept = meter.records_of(mas)[start:]
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(kept))
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
    return trace_from_state(mas, render_value(values, JSON), result, meter), dict(result.get("values") or {})


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
        ``Trace.error_kind`` rather than raised, and the trace keeps the records
        of the agents that finished.
    """
    return run_many(mas, [query], [prompts])[0]


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Execute a MAS over parallel lists of inputs and prompt sets.

    Every item runs once, with its own meter and budget. An item that fails
    becomes a failed trace; the others are unaffected.

    Args:
        mas: The system to run.
        queries: The inputs.
        prompts: Prompt sets, one per input.

    Returns:
        One trace per input, in the order given. A failed rollout is data, not a
        crash: its trace carries the error and the rest of the batch is kept.
    """
    if not queries:
        return []
    states = [_initial_state(query, prompt_set) for query, prompt_set in zip(queries, prompts, strict=True)]
    meters = [RunMeter(mas.budget, owner=mas) for _ in states]
    configs = [_run_config(mas, meter) for meter in meters]
    results: list[Any] = mas.compiled.batch(states, config=configs, return_exceptions=True)
    return [
        trace_from_state(mas, query, result, meter)
        for query, result, meter in zip(queries, results, meters, strict=True)
    ]
