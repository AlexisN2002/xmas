"""Compile a MAS and run it.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data.

Each run gets its own :class:`~xmas.budget.RunMeter`, handed to the nodes in the
run config under :data:`_RUN_KEY`. Nodes check the budget on it before each call
and keep every finished visit's record there, so a failed run still returns
what it did.

In a version 2 system values are typed (text or json) and render by
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
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langgraph.config import get_config
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from promptolution.utils.templates import DEFAULT_SYS_PROMPT
from promptolution.utils.token_counter import get_token_counter

from xmas.agent import Agent
from xmas.budget import RunMeter, VisitMeter
from xmas.check import step_maps
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import RUN_ENDS, BadFormat, BudgetExceeded, ErrorKind, classify
from xmas.graph import END, back_edges, join_nodes
from xmas.health import summarize
from xmas.llms.meta import SupportsMeta, supports_meta
from xmas.nodes import Box, Step
from xmas.predicates import Context, Predicate
from xmas.prompts import safe_format
from xmas.routes import Join
from xmas.schemas import RETRY_NOTE, AnswerForm, FormError, compile_form
from xmas.tools import run_step, run_tool_loop
from xmas.trace import AgentRecord, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, parse_type, render_value

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on."""

_RUN_KEY = "xmas_run"
"""Key of the run's :class:`~xmas.budget.RunMeter` in ``config["configurable"]``."""


class MASState(TypedDict):
    """The state a run threads through the graph.

    Attributes:
        values: Every value, ``query`` included, keyed by name: a ``str`` for
            text, plain JSON data for json. Nodes writing concurrently each
            contribute their own key, so the reducer is a plain dict union.
        prompts: The run's prompt set, keyed by agent name. Prompts travel in
            the state, so one compilation serves every candidate an optimizer
            proposes.
        records: One record per visit, in completion order, so they also count
            the visits.
        routes: Where each deciding node's latest visit sent the run.
    """

    values: Annotated[dict[str, Any], operator.or_]
    prompts: dict[str, str]
    records: Annotated[list[AgentRecord], operator.add]
    routes: Annotated[dict[str, list[str]], operator.or_]


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
    """Make a plain agent's one model call; the metadata is ``None`` unless the model opted in to ``full_response``."""
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

    The node times the body, keeps its record on the run's meter and writes its
    values; a node with a decision also writes where it goes in ``routes``.

    In version 2 a visit that raises gets a failed record (:func:`_failed_record`)
    holding every paid call, unless the error ends the run (:data:`~xmas.errors.RUN_ENDS`).
    When an error edge routes the error's kind, the visit writes nothing and
    takes it; otherwise the error goes on. Version 1 raises.
    """

    def node(state: MASState) -> dict[str, Any]:
        meter = _current_meter(mas)
        meter.start()  # the run's wall clock starts with its first agent, not with the batch
        started = time.perf_counter()
        visit = VisitMeter(meter)
        update: dict[str, Any] = {}
        try:
            record, update["values"] = body(state, visit)
            if decision is not None:
                route, capped = decision.route(_context(mas, name, state, update["values"], meter))
                record = dataclasses.replace(record, route=tuple(route), capped=capped)
                update["routes"] = {name: route}
        except Exception as error:
            kind = _kind_of(error)
            if mas.version != 2 or kind in RUN_ENDS:
                raise
            target = None if decision is None else decision.on_error(kind)
            record = _failed_record(mas, name, error, kind, time.perf_counter() - started, target, visit)
            meter.keep(record, mas)
            if target is None:
                raise
            return {"records": [record], "routes": {name: [target]}}
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started)
        meter.keep(record, mas)
        return {**update, "records": [record]}

    return node


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

    Returns the accepted text, its value, the rejected answer (if any), and
    usage and metadata summed over both calls. A retry that does not fit
    either raises :class:`~xmas.errors.BadFormat` with both problems.
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
    mas: MAS, agent: Agent, llm: BaseLLM, count_tokens: Callable[[str], float], types: Mapping[str, ValueType]
) -> _NodeBody:
    """Build what an agent does in one visit.

    Inputs render by their types (text when ``types`` does not name them, as
    in version 1). A json agent's prompt ends with its form's format block, and
    its answer is parsed into the value it writes: a plain agent retries in a
    second call (:func:`_answer_plain`), an agent with tools in one more turn of
    its conversation, tools withheld.
    """
    form = compile_form(agent.schema) if parse_type(agent.output_type) == JSON else None
    tools = mas.resolve_tools(agent) if agent.tools else None
    batch_error = mas.version != 2

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        template = state["prompts"].get(agent.name, agent.prompt)
        values = state["values"]
        inputs = {
            name: render_value(values.get(name), types.get(name, TEXT), present=name in values) for name in agent.inputs
        }
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
            response, meta = _call(llm, rendered, meter, batch_error=batch_error)
            value, usage = response, _usage(count_tokens, rendered, response)

        record = AgentRecord(
            agent=agent.name,
            step=-1,  # renumbered once the run finishes and the order is known
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


def _step_body(mas: MAS, step: Step, types: Mapping[str, ValueType]) -> _NodeBody:
    """Build what a step does in one visit: read its parameters, run its callable, write.

    The meter's wall and cost limits are checked first; a step makes no model
    call, so it counts none. A parameter whose value is not in the state is
    left out and takes its default.
    """
    spec, source = mas.resolve_callable(step)
    read_map, write_map = step_maps(step, spec)

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        meter.check_boundary()
        values = state["values"]
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


def _box_body(box: Box, types: Mapping[str, ValueType]) -> _NodeBody:
    """Build what a box does in one visit: run its inner system from the mapped values.

    The inner system runs on the outer run's meter, which adopts it, so its
    calls count against the outer budget. Its failure raises out of the box, so
    the box's own error edges route by the inner error's kind; the visit keeps
    the inner trace so far for the box's failed record.
    """
    inner: MAS = box.system

    def body(state: MASState, visit: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        run = visit.run
        run.check_boundary()
        run.adopt(inner)
        values = state["values"]
        given = {inside: values[outside] for inside, outside in box.reads.items() if outside in values}
        query = render_value(given, JSON)
        kept = len(run.records_of(inner))
        try:
            result = inner.compiled.invoke(_initial_state(given, inner.prompts()), config=_run_config(inner, run))
        except Exception as error:
            visit.inner = _failed_trace(inner, query, error, run, kept)
            raise
        trace, produced = trace_from_state(inner, query, result, run), result["values"]
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
    def of(cls, edges: list[Edge], predicates: Mapping[Edge, Predicate]) -> _Decision:
        """Build the decision of one node from its out-edges, in declared order, and the check's predicates."""
        choices = []
        for edge in edges:
            if edge.kind is EdgeKind.SWITCH:
                compare = ("cmp", "==", predicates[edge].tree, ("lit", edge.case))
                choices.append((edge, Predicate(f"{edge.switch} == {edge.case!r}", compare)))
            elif edge.kind in (EdgeKind.WHEN, EdgeKind.WHILE):
                choices.append((edge, predicates[edge]))
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
        """Return the target of the first error edge routing ``kind``."""
        for edge in self.errors:
            if ANY_ERROR in edge.errors or kind.value in edge.errors:
                return edge.target
        return None


def _context(mas: MAS, name: str, state: MASState, writes: dict[str, Any], meter: RunMeter) -> Context:
    """Return what a node's predicates see: its fresh writes, its visit counted, the spend so far."""
    visits = Counter(record.agent for record in state["records"])
    visits[name] += 1

    def call(function: str, args: tuple[Any, ...]) -> bool:
        spec, source = mas.resolve_callable(Step(function, function))
        produced = run_step(spec, source, dict(zip(spec.reads, args, strict=True)))
        return cast("bool", next(iter(produced.values())))

    return Context(
        values={**state["values"], **writes},
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

    The cost and tokens are those of every call the visit made; a step's are
    ``0``, and a box's those of its inner trace, which the record holds.
    """
    node = mas.nodes[name]
    meta = visit.meta
    paid: dict[str, Any] = meta.record_fields() if meta is not None else {}
    if isinstance(node, Step):
        paid["cost_usd"] = 0.0
    elif visit.inner is not None:
        paid.update(_paid_inside(visit.inner))
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
        inner=visit.inner,
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
    is waited on at its node. Any other group gets the no-op relay node its
    :class:`~xmas.routes.Join` names, which its sources route to. A merge needs
    nothing: its edges stay plain, and it runs on whichever arrives.

    Returns the (source, join) pairs the barrier waits on directly, and the
    relay each other (source, join) pair goes to.
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
        return [targets[target] for target in state["routes"].get(name, []) if target in targets]

    return route


def cycle_router(
    mas: MAS, agent: Agent, back_targets: list[str], forward_targets: list[str]
) -> Callable[[MASState], list[str]]:
    """Build the branch decision for a version 1 agent that closes a cycle.

    The agent decides by what it wrote: emitting :data:`CONTINUE_MARKER` takes the
    back edge, anything else moves on. Two guards make the loop finite regardless
    of what the model says: the agent's own visit count against ``mas.max_visits``,
    and the graph-wide recursion limit behind it.
    """
    onward = forward_targets or [END]

    def route(state: MASState) -> list[str]:
        visits = sum(1 for record in state["records"] if record.agent == agent.name)
        looping = CONTINUE_MARKER in str(state["values"].get(agent.output) or "")
        return back_targets if looping and visits < mas.max_visits else onward

    return route


def _wire_v1(builder: StateGraph[MASState, Any, MASState, MASState], mas: MAS) -> None:
    """Wire a version 1 graph: plain edges, and a :func:`cycle_router` for each agent that closes a cycle.

    A static edge set cannot express a loop: the edge back and the edge onward
    would both fire on every pass. Routing the two exclusively, on the looping
    agent's own output, is what makes a cyclic topology end with an answer.
    """
    back = back_edges(mas.graph)
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


def _wire_v2(
    builder: StateGraph[MASState, Any, MASState, MASState], mas: MAS, out: dict[str, list[Edge]], deciding: set[str]
) -> None:
    """Wire a version 2 graph: a deciding node routes by what its wrapper wrote, joins are barriers, the rest plain."""
    direct, relays = _add_joins(builder, mas._checked().joins)
    for source, edges in out.items():
        targets = {
            edge.target: relays.get((source, edge.target), edge.target)
            for edge in edges
            if (source, edge.target) not in direct
        }
        if source not in deciding:
            for target in targets.values():
                builder.add_edge(source, target)
        elif targets:
            builder.add_conditional_edges(source, _router(source, targets), sorted(set(targets.values())))


def compile_mas(mas: MAS) -> Any:
    """Compile a MAS topology into an executable graph.

    In version 1, agents identified by :func:`xmas.graph.join_nodes` are
    deferred, so a node fed by predecessors at unequal depth runs once per
    query rather than once per wave of arriving inputs. In version 2 a node
    with conditional or error edges gets a :class:`_Decision`, and a join is a
    barrier that runs its node once per pass.
    """
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(MASState)
    types = mas.info.types if mas.info is not None else {}
    bodies: dict[str, _NodeBody] = {}
    for agent in mas.agents.values():
        llm = mas.resolve_llm(agent)
        bodies[agent.name] = _agent_body(mas, agent, llm, get_token_counter(llm), types)
    for step in mas.steps.values():
        bodies[step.name] = _step_body(mas, step, types)
    for box in mas.boxes.values():
        bodies[box.name] = _box_body(box, types)

    out = _out_edges(mas) if mas.version == 2 else {}
    decisions = {
        source: _Decision.of(edges, mas._checked().predicates)
        for source, edges in out.items()
        if any(e.kind is not EdgeKind.PLAIN for e in edges)
    }
    deferred = join_nodes(mas.graph) if mas.version != 2 else set()
    for name, body in bodies.items():
        # mypy rejects a TypedDict schema here; langgraph documents it as the usage.
        builder.add_node(name, _wrap(mas, name, body, decisions.get(name)), defer=name in deferred)  # type: ignore[arg-type]
    if mas.version == 2:
        _wire_v2(builder, mas, out, set(decisions))
    else:
        _wire_v1(builder, mas)
    return builder.compile()


def _output(mas: MAS, records: list[AgentRecord]) -> str:
    """Return the rendered write of the terminal node that ran last, or ``""`` if none ran.

    A terminal with several writes gives them as one JSON object.
    """
    terminals = set(mas.graph.predecessors(END))
    ran = [record for record in records if record.agent in terminals and record.error is None]
    if not ran:
        return ""
    outputs = ran[-1].outputs
    return next(iter(outputs.values())) if len(outputs) == 1 else ran[-1].raw_response


def _kind_of(error: BaseException) -> ErrorKind:
    """Return the kind of the error that ended a run; hitting ``max_steps`` is ``BUDGET``."""
    if isinstance(error, GraphRecursionError):
        return ErrorKind.BUDGET
    return classify(error)


def _trace(mas: MAS, query: str, records: list[AgentRecord], meter: RunMeter, **ending: Any) -> Trace:
    """Number a run's records and summarize its health; the budget entry is the meter owner's alone."""
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    values = mas.info.types if mas.info is not None else {"query", *(a.output for a in mas.agents.values())}
    health = summarize(
        numbered,
        meter=meter if mas is meter.owner else None,
        settings=mas.health,
        while_edges=mas.while_edges,
        values=values,
    )
    return Trace(query=query, records=numbered, health=health, **ending)


def trace_from_state(mas: MAS, query: str, result: Any, meter: RunMeter) -> Trace:
    """Assemble a trace from a finished run's state, or from the error that ended it."""
    if isinstance(result, BaseException):
        return _failed_trace(mas, query, result, meter)
    return _trace(mas, query, result["records"], meter, output=_output(mas, result["records"]))


def _failed_trace(mas: MAS, query: str, error: BaseException, meter: RunMeter, start: int = 0) -> Trace:
    """Assemble a failed run's trace from the records its meter kept for ``mas``.

    ``start`` skips the records of earlier runs on the meter (an adopted box
    system's earlier visits).
    """
    return _trace(
        mas,
        query,
        meter.records_of(mas)[start:],
        meter,
        output="",
        error=f"{type(error).__name__}: {error}",
        error_kind=_kind_of(error),
    )


def _initial_state(query: str | Mapping[str, Any], prompts: Mapping[str, str]) -> MASState:
    """Build the state a run starts from: the query, or a box's inner values."""
    values = {"query": query} if isinstance(query, str) else dict(query)
    return MASState(values=values, prompts=dict(prompts), records=[], routes={})


def _run_config(mas: MAS, meter: RunMeter) -> dict[str, Any]:
    """Build one run's config: the step cap, and the meter the nodes read."""
    return {"recursion_limit": mas.max_steps, "configurable": {_RUN_KEY: meter}}


def run_one(mas: MAS, query: str, prompts: dict[str, str]) -> Trace:
    """Run a MAS on one input with a given prompt set; a failure is recorded in the trace, not raised."""
    return run_many(mas, [query], [prompts])[0]


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Run a MAS over parallel lists of inputs and prompt sets.

    Every item runs once, with its own meter and budget. An item that fails
    becomes a failed trace that keeps its finished records; the others are
    unaffected.
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
