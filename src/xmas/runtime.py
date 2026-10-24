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

A ``for each`` chain compiles to three nodes around its head ``h``:
``__each__.h.in`` starts a pass and fans the list out with one ``Send`` per
item, ``__each__.h`` runs the chain's own system on one item, and
``__each__.h.out`` gathers each write into a list aligned to the items. Every
record a chain's item makes carries the item, and records keep the order of the
items, however the items finish.
"""

from __future__ import annotations

import dataclasses
import operator
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp, GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.types import Send
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
from xmas.trace import AgentRecord, Event, Rejected, TokenUsage, ToolCall, Trace
from xmas.values import JSON, TEXT, ValueType, check_value, parse_type, render_value

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

    from xmas.chains import Chain
    from xmas.llms.meta import CallMeta
    from xmas.mas import MAS


CONTINUE_MARKER = "<continue>"
"""Marker an agent emits to take a back edge instead of moving on."""

_RUN_KEY = "xmas_run"
"""Key of the run's :class:`~xmas.budget.RunMeter` in ``config["configurable"]``."""

_ITEM_KEY = "xmas_item"
"""Key of the chain item a run works on, ``(index, key)``, in ``config["configurable"]``."""


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


def _merge_items(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge chain updates: an entry with ``n`` starts the chain's pass afresh, ``rows`` add items to it."""
    merged = dict(left)
    for head, part in right.items():
        if "n" in part:
            merged[head] = dict(part)
        else:
            entry = merged.get(head, {})
            merged[head] = {**entry, "rows": {**entry.get("rows", {}), **part["rows"]}}
    return merged


class _ChainState(MASState):
    """The state of a system with chains.

    Attributes:
        items: Each chain's current pass by head: ``n`` items, the pass's
            number, and one row per finished item, ``{"values": ...}`` or
            ``{"error": ..., "kind": ...}``.
    """

    items: Annotated[dict[str, Any], _merge_items]


_NodeBody = Callable[[MASState, VisitMeter], tuple[AgentRecord, dict[str, Any]]]
"""What a node does in one visit, on its share of the run's meter: its record, and the values it writes."""


def _current_meter(mas: MAS) -> RunMeter:
    """Return the run's meter.

    A graph invoked without a run config, or inheriting another system's (a
    nested run inside a node), counts on a meter of its own.
    """
    meter = get_config().get("configurable", {}).get(_RUN_KEY)
    return meter if isinstance(meter, RunMeter) and meter.owns(mas) else RunMeter(mas.budget, owner=mas)


def _item() -> tuple[int, str | None] | None:
    """Return the chain item the running node works on, ``(index, key)``, or ``None`` outside a chain."""
    item: tuple[int, str | None] | None = get_config().get("configurable", {}).get(_ITEM_KEY)
    return item


def _chains(mas: MAS) -> dict[str, Chain]:
    """Return the system's ``for each`` chains by head; none in version 1."""
    return mas.info.chains if mas.info is not None else {}


def _run_value(meter: RunMeter, mas: MAS) -> dict[str, Any]:
    """Return the reserved value ``run``: what the run has spent so far, and the visits finished in ``mas``.

    ``cost_usd`` is ``None`` while a call's cost is unknown; ``records`` counts
    chain items' visits too.
    """
    return {
        "cost_usd": None if meter.uncosted_calls else meter.cost_usd,
        "elapsed_s": meter.elapsed_s,
        "calls": meter.calls,
        "records": len(meter.records_of(mas)),
    }


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
        if not batch_error or not supports_meta(llm) or isinstance(error, BudgetExceeded | GraphBubbleUp):
            raise  # promptolution's get_response wrapped it already, the meter refused the call, or LangGraph signals
        raise RuntimeError(f"LLM batch failed: {error}") from error


def _wrap(
    mas: MAS, name: str, body: _NodeBody, decision: _Decision | None = None, *, writes_error: bool = False
) -> Callable[[MASState], dict[str, Any]]:
    """Turn a node's body into the node LangGraph runs.

    The node times the body, keeps its record on the run's meter and writes its
    values; a node with a decision also writes where it goes in ``routes``. It
    emits a ``start`` and an ``end`` event on the meter.

    In version 2 every record carries its visit number (among this node's
    records in the state, so per item inside a chain), the chain item it worked
    on, and the retries it used: a kind the node's ``retry`` names runs the body
    again, on the same visit meter, while attempts are left. A visit that raises
    gets a failed record (:func:`_failed_record`) holding every paid call,
    unless the error ends the run (:data:`~xmas.errors.RUN_ENDS`). When an
    error edge routes the error's kind, the visit takes it and writes only
    ``error`` (``{kind, message, node}``, when ``writes_error``); otherwise the
    error goes on. Version 1 raises.
    """
    retry = mas.retry.get(name, {})

    def node(state: MASState) -> dict[str, Any]:
        meter = _current_meter(mas)
        meter.start()  # the run's wall clock starts with its first agent, not with the batch
        index, key = _item() or (None, None)
        here: dict[str, Any] = {
            "visit": 1 + sum(record.agent == name for record in state["records"]),
            "item": index,
            "item_key": key,
        }
        meter.emit("start", name, **here)
        started = time.perf_counter()
        visit = VisitMeter(meter)
        update: dict[str, Any] = {}
        used: Counter[str] = Counter()
        try:
            while True:
                try:
                    record, update["values"] = body(state, visit)
                    break
                except GraphBubbleUp:
                    raise
                except Exception as error:
                    again = _kind_of(error).value
                    if used[again] >= retry.get(again, 0):
                        raise
                    used[again] += 1
            if used and record.kind == "agent":  # its cost is that of every attempt
                paid = visit.meta
                record = dataclasses.replace(
                    record, **paid.record_fields() if paid else {"cost_usd": None, "tokens": None}
                )
            if decision is not None:
                route, capped = decision.route(_context(mas, name, state, update["values"], meter))
                record = dataclasses.replace(record, route=tuple(route), capped=capped)
                update["routes"] = {name: route}
        except GraphBubbleUp:  # LangGraph's own signals, such as an interrupt, pass untouched
            raise
        except Exception as error:
            kind = _kind_of(error)
            if mas.version != 2 or kind in RUN_ENDS:
                meter.emit("end", name, status="failed", **here)
                raise
            target = None if decision is None else decision.on_error(kind)
            record = _failed_record(mas, name, error, kind, time.perf_counter() - started, target, visit)
            record = dataclasses.replace(record, **here, retries=used.total())
            meter.keep(record, mas)
            meter.emit("end", name, status="failed" if target is None else "routed", cost_usd=record.cost_usd, **here)
            if target is None:
                raise
            routed: dict[str, Any] = {"records": [record], "routes": {name: [target]}}
            if writes_error:
                routed["values"] = {"error": {"kind": kind.value, "message": record.error, "node": name}}
            return routed
        stamps = {**here, "retries": used.total()} if mas.version == 2 else {}
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started, **stamps)
        meter.keep(record, mas)
        meter.emit("end", name, status="ok", cost_usd=record.cost_usd, **here)
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
    reads = agent.reads
    reads_run = mas.version == 2 and "run" in reads.values()

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        template = state["prompts"].get(agent.name, agent.prompt)
        values = {**state["values"], "run": _run_value(meter.run, mas)} if reads_run else state["values"]
        inputs = {
            placeholder: render_value(values.get(name), types.get(name, TEXT), present=name in values)
            for placeholder, name in reads.items()
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
    reads_run = "run" in read_map.values()

    def body(state: MASState, meter: VisitMeter) -> tuple[AgentRecord, dict[str, Any]]:
        meter.check_boundary()
        values = {**state["values"], "run": _run_value(meter.run, mas)} if reads_run else state["values"]
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
    calls count against the outer budget, and on the chain item the box works
    on, if any. Its failure raises out of the box, so the box's own error edges
    route by the inner error's kind; the visit keeps the inner trace so far for
    the box's failed record.
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
            config = _run_config(inner, run, item=_item())
            result = inner.compiled.invoke(_initial_state(given, inner.prompts()), config=config)
        except GraphBubbleUp:
            raise
        except Exception as error:
            item = _item()  # a box in a chain: its siblings' inner records sit on the same list
            visit.inner = _failed_trace(inner, query, error, run, kept, None if item is None else item[0])
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
    """Return a box record's cost and tokens: its inner records' summed, ``None`` when any of them is unknown."""
    tokens = [record.tokens for record in trace.records]
    known = cast("list[TokenUsage]", tokens)
    return {"cost_usd": trace.cost_usd, "tokens": None if not tokens or None in tokens else sum(known[1:], known[0])}


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
    """Return what a node's predicates see: its fresh writes, its visit counted, the spend so far.

    A chain's head has run as often as the chain has fanned out.
    """
    visits = Counter(record.agent for record in state["records"])
    visits[name] += 1
    passes = cast("_ChainState", state).get("items", {})
    for head in _chains(mas):
        visits[head] = passes.get(head, {}).get("pass", 0)

    def call(function: str, args: tuple[Any, ...]) -> bool:
        spec, source = mas.resolve_callable(Step(function, function))
        produced = run_step(spec, source, dict(zip(spec.reads, args, strict=True)))
        return cast("bool", next(iter(produced.values())))

    return Context(
        values={"run": _run_value(meter, mas), **state["values"], **writes},
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
    """Return each node's out-edges: the declared ones in order, then the implicit start and end wiring.

    A chain's members are left out: the head stands for the chain, its exits
    leaving from it.
    """
    info = mas._checked()
    edges, flow = info.edges, info.flow
    out: dict[str, list[Edge]] = {}
    for edge in edges:
        out.setdefault(edge.source, []).append(edge)
    declared = {(edge.source, edge.target) for edge in edges}
    for source, target in flow.edges:
        if (source, target) not in declared:
            out.setdefault(source, []).append(Edge(source, target))
    return out


def _relay(state: MASState) -> dict[str, Any]:  # noqa: ARG001 (a node's signature)
    """Pass a group's arrival on to a join's barrier; writes nothing."""
    return {}


def _entry(name: str, chains: Mapping[str, Chain]) -> str:
    """Return the node an edge into ``name`` lands on: a chain head's ``in`` node, else ``name``."""
    return f"__each__.{name}.in" if name in chains else name


def _exit(name: str, chains: Mapping[str, Chain]) -> str:
    """Return the node an edge out of ``name`` leaves from: a chain head's ``out`` node, else ``name``."""
    return f"__each__.{name}.out" if name in chains else name


def _add_joins(
    builder: StateGraph[MASState, Any, MASState, MASState], joins: Mapping[str, Join], chains: Mapping[str, Chain]
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    """Add one barrier per join: it waits for each of the join's groups once per pass.

    A group without a relay (one plain edge from a node without error edges)
    is waited on at its node. Any other group gets the no-op relay node its
    :class:`~xmas.routes.Join` names, which its sources route to. A merge needs
    nothing: its edges stay plain, and it runs on whichever arrives. Nor does a
    join at :data:`~xmas.graph.END`: the run ends once every branch has.

    Returns the (source, join) pairs the barrier waits on directly, and the
    relay each other (source, join) pair goes to.
    """
    direct: set[tuple[str, str]] = set()
    relays: dict[tuple[str, str], str] = {}
    for node, join in joins.items():
        if join.kind != "join" or node == END:
            continue
        starts = []
        for group, relay in zip(join.groups, join.relays, strict=True):
            if relay is None:
                direct.add((group[0], node))
                starts.append(_exit(group[0], chains))
                continue
            builder.add_node(relay, _relay)
            relays.update(((member, node), relay) for member in group)
            starts.append(relay)
        builder.add_edge(starts, _entry(node, chains))
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
    """Wire a version 2 graph: a deciding node routes by what its wrapper wrote, joins are barriers, the rest plain.

    Each chain becomes its three nodes (:func:`_add_chain`); edges into its head
    land on the first, edges out of it leave from the last.
    """
    chains = _chains(mas)
    for chain in chains.values():
        _add_chain(builder, mas, chain)
    direct, relays = _add_joins(builder, mas._checked().joins, chains)
    for source, edges in out.items():
        targets = {
            edge.target: relays.get((source, edge.target), _entry(edge.target, chains))
            for edge in edges
            if (source, edge.target) not in direct
        }
        if source not in deciding:
            for target in targets.values():
                builder.add_edge(_exit(source, chains), target)
        else:
            builder.add_conditional_edges(source, _router(source, targets), sorted(set(targets.values())))


def _items(chain: Chain, values: Mapping[str, Any]) -> list[tuple[Any, str | None]]:
    """Return a chain's items with their keys, from the list at its path.

    A missing path or ``None`` is no item. Anything else that is not a list,
    and with ``key`` an item without a scalar key or a key given twice, raises
    ``ValueError`` naming the path, before any item runs; so does a step of the
    path through a value that is no mapping.
    """
    each = chain.each
    found, path = values.get(each.root), each.root
    for field in each.fields:
        if found is None:
            break
        if not isinstance(found, dict):
            raise ValueError(f"for each {each}: {path} holds {type(found).__name__}, not a mapping with {field!r}")
        found, path = found.get(field), f"{path}.{field}"
    if found is None:
        return []
    if not isinstance(found, list):
        raise ValueError(f"for each {each}: {each.over} holds {type(found).__name__}, not a list")
    if each.key is None:
        return [(item, None) for item in found]
    keys = []
    for index, item in enumerate(found):
        key = item.get(each.key) if isinstance(item, dict) else None
        if isinstance(key, bool) or not isinstance(key, str | int | float):
            raise ValueError(f"for each {each}: item {index} of {each.over} has no key {each.key!r} (a text or number)")
        keys.append(str(key))
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"for each {each}: {each.over} repeats the key {each.key!r}: {repeated}")
    return list(zip(found, keys, strict=True))


def _cost(records: list[AgentRecord]) -> float | None:
    """Return the records' costs summed, ``None`` when there is none or any is unknown."""
    costs = [record.cost_usd for record in records]
    return None if not costs or None in costs else sum(cast("list[float]", costs))


def _add_chain(builder: StateGraph[MASState, Any, MASState, MASState], mas: MAS, chain: Chain) -> None:
    """Add a chain's three nodes: a pass starts and fans out, each item runs, the writes are gathered.

    An item runs ``chain.system`` on the item and the outside values it reads,
    on the run's meter, which keeps its records with ``mas``'s, at most ``at
    once`` items at a time. An item whose run fails becomes a row with its
    error, and ``None`` in every gathered list, unless the error ends the run
    (:data:`~xmas.errors.RUN_ENDS`) or is ``money``, which every other item
    would meet too.
    """
    head, each, system = chain.head, chain.each, chain.system
    start, target, gather = f"__each__.{head}.in", f"__each__.{head}", f"__each__.{head}.out"

    def begin(state: _ChainState) -> dict[str, Any]:
        done = state.get("items", {}).get(head, {}).get("pass", 0)
        return {"items": {head: {"n": len(_items(chain, state["values"])), "pass": done + 1, "rows": {}}}}

    def fan(state: MASState) -> list[Send] | list[str]:
        items = _items(chain, state["values"])
        if not items:
            return [gather]
        values = {name: state["values"][name] for name in chain.inputs if name in state["values"]}
        return [
            Send(target, {"index": index, "item": item, "key": key, "values": values, "prompts": state["prompts"]})
            for index, (item, key) in enumerate(items)
        ]

    def run_item(task: dict[str, Any]) -> dict[str, Any]:
        run = _current_meter(mas)
        run.adopt(system, into=mas)
        index, key = task["index"], task["key"]
        where = {"item": index, "item_key": key}
        with run.gate(f"each:{id(system)}", each.at_once):  # per chain, whatever its head is called
            run.emit("item_start", head, **where)
            kept = len(run.records_of(system))
            state = _initial_state({each.item: task["item"], **task["values"]}, task["prompts"])
            try:
                result = system.compiled.invoke(state, config=_run_config(system, run, item=(index, key)))
            except GraphBubbleUp:
                raise
            except Exception as error:
                kind = _kind_of(error)
                mine = [
                    record
                    for record in run.records_of(system)[kept:]
                    if record.agent in chain.nodes and record.item == index
                ]
                run.emit("item_end", head, status="failed", cost_usd=_cost(mine), **where)
                if kind in RUN_ENDS or kind is ErrorKind.MONEY:
                    raise
                row: dict[str, Any] = {"error": f"{type(error).__name__}: {error}", "kind": kind.value}
                return {"items": {head: {"rows": {index: row}}}, "records": mine}
        run.emit("item_end", head, status="ok", cost_usd=_cost(result["records"]), **where)
        produced = result["values"]
        row = {"values": {name: produced[name] for name in chain.writes if name in produced}}
        return {"items": {head: {"rows": {index: row}}}, "records": result["records"]}

    def finish(state: _ChainState) -> dict[str, Any]:
        entry = state.get("items", {}).get(head, {})
        rows = [entry.get("rows", {}).get(index, {}).get("values", {}) for index in range(entry.get("n", 0))]
        return {"values": {name: [row.get(name) for row in rows] for name in chain.writes}}

    builder.add_node(start, begin)
    builder.add_node(target, run_item)  # type: ignore[arg-type]
    builder.add_node(gather, finish)
    builder.add_conditional_edges(start, fan, [target, gather])
    builder.add_edge(target, gather)


def compile_mas(mas: MAS) -> Any:
    """Compile a MAS topology into an executable graph.

    In version 1, agents identified by :func:`xmas.graph.join_nodes` are
    deferred, so a node fed by predecessors at unequal depth runs once per
    query rather than once per wave of arriving inputs. In version 2 a node
    with conditional or error edges gets a :class:`_Decision`, a join is a
    barrier that runs its node once per pass, and a chain is its three nodes in
    place of its members, on :class:`_ChainState`.
    """
    chains = _chains(mas)
    members = {name for chain in chains.values() for name in chain.nodes}
    schema: type[MASState] = _ChainState if chains else MASState
    builder: StateGraph[MASState, Any, MASState, MASState] = StateGraph(schema)
    types = mas.info.types if mas.info is not None else {}
    bodies: dict[str, _NodeBody] = {}
    for agent in mas.agents.values():
        if agent.name not in members:
            llm = mas.resolve_llm(agent)
            bodies[agent.name] = _agent_body(mas, agent, llm, get_token_counter(llm), types)
    for step in mas.steps.values():
        if step.name not in members:
            bodies[step.name] = _step_body(mas, step, types)
    for box in mas.boxes.values():
        if box.name not in members:
            bodies[box.name] = _box_body(box, types)

    out = _out_edges(mas) if mas.version == 2 else {}
    decisions = {
        source: _Decision.of(edges, mas._checked().predicates)
        for source, edges in out.items()
        if any(e.kind is not EdgeKind.PLAIN for e in edges)
    }
    deferred = join_nodes(mas.graph) if mas.version != 2 else set()
    writes_error = mas.version == 2 and _reads(mas, "error")
    for name, body in bodies.items():
        node = _wrap(mas, name, body, decisions.get(name), writes_error=writes_error)
        # mypy rejects a TypedDict schema here; langgraph documents it as the usage.
        builder.add_node(name, node, defer=name in deferred, input_schema=schema)  # type: ignore[arg-type]
    if mas.version == 2:
        _wire_v2(builder, mas, out, set(decisions))
    else:
        _wire_v1(builder, mas)
    return builder.compile()


def _reads(mas: MAS, value: str) -> bool:
    """Return whether any agent, step or predicate of a version 2 system reads ``value``."""
    nodes = [*(agent.reads.values() for agent in mas.agents.values())]
    nodes += [step_maps(step, mas.resolve_callable(step)[0])[0].values() for step in mas.steps.values()]
    paths = {path[0] for predicate in mas._checked().predicates.values() for path in predicate.paths()}
    return value in paths or any(value in names for names in nodes)


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
    """Number a run's records and summarize its health; the budget entry is the meter owner's alone.

    The meter owner's version 2 trace carries the run's id and the system's ``describe()`` id.
    """
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    values = mas.info.types if mas.info is not None else {"query", *(a.output for a in mas.agents.values())}
    chains = {head: (chain.nodes, dict(chain.each.item_budget or {})) for head, chain in _chains(mas).items()}
    health = summarize(
        numbered,
        meter=meter if mas is meter.owner else None,
        settings=mas.health,
        while_edges=mas.while_edges,
        values=values,
        chains=chains,
    )
    if mas.version == 2 and mas is meter.owner:
        ending.update(run_id=meter.run_id, graph_id=meter.graph_id)
    return Trace(query=query, records=numbered, health=health, **ending)


def trace_from_state(mas: MAS, query: str, result: Any, meter: RunMeter) -> Trace:
    """Assemble a trace from a finished run's state, or from the error that ended it.

    With declared ``outputs`` the trace holds each one's final value, ``None``
    when nothing wrote it, and its output renders the first.
    """
    if isinstance(result, BaseException):
        return _failed_trace(mas, query, result, meter)
    outputs = mas.outputs
    if not outputs:
        return _trace(mas, query, result["records"], meter, output=_output(mas, result["records"]))
    values = result["values"]
    first, vtype = next(iter(outputs.items()))
    output = render_value(values.get(first), vtype, present=first in values)
    return _trace(mas, query, result["records"], meter, output=output, outputs={n: values.get(n) for n in outputs})


def _failed_trace(
    mas: MAS, query: str, error: BaseException, meter: RunMeter, start: int = 0, item: int | None = None
) -> Trace:
    """Assemble a failed run's trace from the records its meter kept for ``mas``.

    ``start`` skips the records of earlier runs on the meter (an adopted box
    system's earlier visits); ``item`` keeps only one chain item's records.
    """
    records = meter.records_of(mas)[start:]
    return _trace(
        mas,
        query,
        records if item is None else [record for record in records if record.item == item],
        meter,
        output="",
        error=f"{type(error).__name__}: {error}",
        error_kind=_kind_of(error),
    )


def _initial_state(query: str | Mapping[str, Any], prompts: Mapping[str, str]) -> MASState:
    """Build the state a run starts from: the query, or a box's inner values."""
    values = {"query": query} if isinstance(query, str) else dict(query)
    return MASState(values=values, prompts=dict(prompts), records=[], routes={})


def _run_config(mas: MAS, meter: RunMeter, item: tuple[int, str | None] | None = None) -> dict[str, Any]:
    """Build one run's config: the step cap, the meter the nodes read, and the chain item it works on, if any."""
    configurable: dict[str, Any] = {_RUN_KEY: meter}
    if item is not None:
        configurable[_ITEM_KEY] = item
    return {"recursion_limit": mas.max_steps, "configurable": configurable}


def run_one(mas: MAS, query: str, prompts: dict[str, str]) -> Trace:
    """Run a MAS on one input with a given prompt set; a failure is recorded in the trace, not raised."""
    return run_many(mas, [query], [prompts])[0]


def run_many(mas: MAS, queries: list[str], prompts: list[dict[str, str]]) -> list[Trace]:
    """Run a MAS over parallel lists of inputs and prompt sets.

    Every item runs once, with its own meter and budget. An item that fails
    becomes a failed trace that keeps its finished records; the others are
    unaffected.
    """
    return _run_all(mas, list(queries), queries, prompts)


def run_state(
    mas: MAS, values: Mapping[str, Any], prompts: dict[str, str], on_event: Callable[[Event], None] | None = None
) -> Trace:
    """Run a MAS on its declared inputs with a given prompt set; a failure is recorded in the trace, not raised.

    A version 2 system takes exactly its inputs, each fitting its type; a
    version 1 system takes text values, ``query`` among them as ``run`` gives
    it. ``on_event`` is called with each event of the run, on the run's
    threads. The trace's query is the ``query`` value, else the values as one
    line of JSON.

    Raises:
        ValueError: Before any call, listing every input missing, extra or of
            the wrong type.
    """
    _check_inputs(mas, values)
    query = values["query"] if isinstance(values.get("query"), str) else render_value(dict(values), JSON)
    return _run_all(mas, [dict(values)], [query], [prompts], on_event)[0]


def _check_inputs(mas: MAS, values: Mapping[str, Any]) -> None:
    """Refuse values that are not the system's inputs, listing every problem."""
    if mas.version != 2:
        problems = [] if "query" in values else ["input 'query' is missing"]
        problems += [f"{name!r} is not text" for name, value in values.items() if not isinstance(value, str)]
    else:
        problems = [f"input {name!r} is missing" for name in mas.inputs if name not in values]
        problems += [f"{name!r} is no input of {sorted(mas.inputs)}" for name in values if name not in mas.inputs]
        problems += [
            f"input {name!r} is not {vtype}: {problem}"
            for name, vtype in mas.inputs.items()
            if name in values and (problem := check_value(values[name], vtype)) is not None
        ]
    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"The values do not fit the system's inputs:\n{lines}")


def _run_all(
    mas: MAS,
    starts: list[str | Mapping[str, Any]],
    queries: list[str],
    prompts: list[dict[str, str]],
    on_event: Callable[[Event], None] | None = None,
) -> list[Trace]:
    """Run each start (a query or the input values) with its prompt set, on a meter of its own."""
    if not starts:
        return []
    graph_id = mas.describe()["id"] if mas.version == 2 else None
    meters = [RunMeter(mas.budget, owner=mas, listener=on_event, graph_id=graph_id) for _ in starts]
    states = [_initial_state(start, prompt_set) for start, prompt_set in zip(starts, prompts, strict=True)]
    configs = [_run_config(mas, meter) for meter in meters]
    results: list[Any] = mas.compiled.batch(states, config=configs, return_exceptions=True)
    return [
        trace_from_state(mas, query, result, meter)
        for query, result, meter in zip(queries, results, meters, strict=True)
    ]
