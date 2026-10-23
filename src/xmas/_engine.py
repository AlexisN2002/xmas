"""Private execution engine.

The topology compiles to a LangGraph state graph here and nowhere else. No type
from langgraph crosses back out: callers receive a :class:`~xmas.trace.Trace`,
which is xmas's own data. Keeping the dependency behind this wall is what makes
replacing it later a contained change.

Each run gets its own :class:`~xmas.budget.RunMeter`, handed to the nodes in the
run config under :data:`_RUN_KEY`. Nodes read it back to check the budget before
each call and to keep every finished agent's record, so a failed run still
returns what it did.
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
from promptolution.utils.token_counter import get_token_counter

from xmas import tools as _tools
from xmas.agent import Agent
from xmas.budget import RunMeter
from xmas.errors import EmptyResponse, ErrorKind, classify
from xmas.graph import END, back_edges, join_nodes, terminal_agent
from xmas.llms.meta import SupportsMeta, supports_meta
from xmas.prompts import safe_format
from xmas.trace import AgentRecord, TokenUsage, Trace

if TYPE_CHECKING:  # pragma: no cover
    from promptolution.llms.base_llm import BaseLLM

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


def _merge_values(current: dict[str, str], incoming: dict[str, str]) -> dict[str, str]:
    """Merge one agent's outputs into the shared value map.

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
        values: Every state variable, including ``query``, keyed by name.
        prompts: The prompt set for this run, keyed by agent name. Prompts travel
            in the state rather than being baked into the compiled graph, so one
            compilation serves every candidate an optimizer proposes.
        records: Accumulated per-agent records, in completion order.
        visits: How often each agent has run, which is what bounds a retry loop.
    """

    values: Annotated[dict[str, str], _merge_values]
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
"""What an agent does in one visit: read the state, call its model, return its record."""


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


def _wrap(mas: MAS, agent: Agent, body: _Body) -> Callable[[MASState], dict[str, Any]]:
    """Turn an agent's body into the node LangGraph runs.

    The node times the body, sets ``latency_s`` on its record, keeps the record
    on the run's meter, and writes the agent's output into the state.

    Args:
        mas: The system, whose run meter the node reads.
        agent: The agent.
        body: What the agent does in one visit.

    Returns:
        A function mapping the incoming state to this agent's state updates.
    """

    def node(state: MASState) -> dict[str, Any]:
        meter = _current_meter(mas)
        if meter is not None:
            meter.start()  # the run's wall clock starts with its first agent, not with the batch
        started = time.perf_counter()
        record = body(state, meter)
        record = dataclasses.replace(record, latency_s=time.perf_counter() - started)
        if meter is not None:
            meter.keep(record)
        return {"values": {agent.output: record.raw_response}, "records": [record], "visits": {agent.name: 1}}

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
    return _wrap(mas, agent, body)


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

    Returns:
        The compiled graph.
    """
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


def _to_trace(mas: MAS, query: str, result: dict[str, Any]) -> Trace:
    """Assemble a trace from a finished run's state."""
    records: list[AgentRecord] = list(result.get("records") or [])
    numbered = tuple(dataclasses.replace(record, step=index) for index, record in enumerate(records))
    values: dict[str, str] = result.get("values") or {}
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
    """
    meter = RunMeter(mas.budget, owner=mas)
    try:
        result: dict[str, Any] = mas.compiled.invoke(_initial_state(query, prompts), config=_run_config(mas, meter))
    except Exception as error:  # a failed rollout is data, not a crash
        return _failed_trace(mas, query, error, meter)
    return _to_trace(mas, query, result)


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
        _failed_trace(mas, query, result, meter) if isinstance(result, BaseException) else _to_trace(mas, query, result)
        for query, result, meter in zip(queries, results, meters, strict=True)
    ]
