"""Topology: the graph a MAS executes, and the questions asked of it.

The topology is a ``networkx.DiGraph`` whose nodes are node names, agents and
(version 2) steps alike, plus two sentinels. The functions here speak of agents
because version 1 has no other node; they treat every node the same. Keeping it
a plain nx graph is deliberate: a topology optimizer is then an ordinary
graph-rewriting component and needs nothing from xmas to work.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from xmas.edges import Edge, EdgeKind

START = "__start__"
"""Source sentinel. Every agent with no predecessor is wired from it."""

END = "__end__"
"""Terminal sentinel. Every agent with no successor is wired to it."""

SENTINELS = frozenset({START, END})
"""The node names reserved by xmas; an agent may not use either."""


def _reachable(graph: nx.DiGraph[str], source: str, *, forward: bool) -> set[str]:
    """Return every node reachable from ``source``, following edges either way.

    Written out rather than calling ``nx.descendants``/``nx.ancestors`` because
    the networkx stubs type those as taking ``Graph[object]``, which a
    ``DiGraph[str]`` is not. A six-line traversal keeps the public signatures
    honest about nodes being agent names.

    Args:
        graph: The topology.
        source: Node to start from.
        forward: Follow successors when true, predecessors when false.

    Returns:
        The reachable nodes, excluding ``source`` itself.
    """
    step = graph.successors if forward else graph.predecessors
    seen: set[str] = set()
    stack = list(step(source))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(step(node))
    return seen


def build_graph(
    agents: list[str], edges: list[tuple[str, str]], back: Collection[tuple[str, str]] = ()
) -> nx.DiGraph[str]:
    """Build the topology for a set of agents and the edges between them.

    Agents with no incoming edge are connected from :data:`START`, and agents with
    no outgoing edge to :data:`END`, so the caller writes only the edges that
    carry meaning. An edge in ``back`` (a version 2 ``while`` edge) closes a
    loop and does not count as incoming: the loop's first node is still wired
    from :data:`START`.

    Args:
        agents: Node names, agents and steps alike, which become the
            non-sentinel nodes.
        edges: Directed ``(source, target)`` pairs between agent names. Sentinels
            may be named explicitly to override the implicit wiring.
        back: Those of ``edges`` that close a loop.

    Returns:
        The topology, including both sentinels.

    Raises:
        ValueError: If an edge names an unknown agent, or an agent name collides
            with a sentinel.
    """
    reserved = SENTINELS.intersection(agents)
    if reserved:
        raise ValueError(f"Agent names {sorted(reserved)} are reserved sentinels.")
    duplicates = {name for name in agents if agents.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate agent names: {sorted(duplicates)}.")

    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from([START, *agents, END])

    known = set(agents) | SENTINELS
    for source, target in edges:
        for node in (source, target):
            if node not in known:
                raise ValueError(f"Edge ({source!r}, {target!r}) names unknown agent {node!r}.")
        graph.add_edge(source, target)

    # Wire the loose ends. Doing it here rather than asking the caller for it is
    # what lets `edges=[("solver", "checker")]` describe a whole system.
    for agent in agents:
        if not any((source, agent) not in back for source in graph.predecessors(agent)):
            graph.add_edge(START, agent)
        if graph.out_degree(agent) == 0:
            graph.add_edge(agent, END)

    return graph


def validate_graph(graph: nx.DiGraph[str]) -> None:
    """Check that a topology is runnable.

    A runnable topology has exactly one source sentinel and one terminal
    sentinel, every agent reachable from :data:`START`, and every agent reaching
    :data:`END`. An unreachable agent is an error rather than a warning, because
    silently never running is the failure mode hardest to notice in a result.

    Args:
        graph: The topology to check.

    Raises:
        ValueError: If the topology is not runnable, naming the offending nodes.
    """
    for sentinel in (START, END):
        if sentinel not in graph:
            raise ValueError(f"Topology is missing the {sentinel!r} sentinel.")
    if graph.in_degree(START) != 0:
        raise ValueError(f"{START!r} must have no incoming edges.")
    if graph.out_degree(END) != 0:
        raise ValueError(f"{END!r} must have no outgoing edges.")

    agents = set(graph.nodes) - SENTINELS
    if not agents:
        raise ValueError("Topology has no agents.")

    unreachable = agents - _reachable(graph, START, forward=True)
    if unreachable:
        raise ValueError(f"Agents unreachable from {START!r}: {sorted(unreachable)}.")

    dead_ends = agents - _reachable(graph, END, forward=False)
    if dead_ends:
        raise ValueError(f"Agents that never reach {END!r}: {sorted(dead_ends)}.")


def agent_names(graph: nx.DiGraph[str]) -> list[str]:
    """Return the non-sentinel nodes, in deterministic order.

    Args:
        graph: The topology.

    Returns:
        Agent names, topologically sorted where the graph is acyclic and sorted
        by name otherwise, so serialization is stable.
    """
    agents = set(graph.nodes) - SENTINELS
    if nx.is_directed_acyclic_graph(graph):
        return [node for node in nx.topological_sort(graph) if node in agents]
    return sorted(agents)


def _pending(graph: nx.DiGraph[str], node: str) -> list[str]:
    """Return a node's successors ready to be consumed by ``pop``.

    Reversed, because ``pop`` takes from the tail and the traversal is specified
    to visit successors in ascending order.

    Args:
        graph: The topology.
        node: The node whose successors are wanted.

    Returns:
        The successors, in descending order.
    """
    return sorted(graph.successors(node), reverse=True)


def back_edges(graph: nx.DiGraph[str]) -> set[tuple[str, str]]:
    """Return the edges whose removal makes the topology acyclic.

    These are the cycle-closing edges of a depth-first traversal from
    :data:`START`: an edge to a node that is still open on the search stack. They
    are the only edges a run may decline to follow, so identifying them is what
    turns a cyclic topology from an infinite loop into a retry loop. See
    :mod:`xmas._engine` for how they compile.

    Which of a cycle's edges is the closing one is a real decision, not
    bookkeeping: the back edge is the conditional one, so it decides which agent
    owns the loop. Depth-first from :data:`START` with successors taken in sorted
    order picks the edge that runs against the flow, so in ``a -> b -> c`` with
    ``c -> b``, it is ``c -> b`` that becomes the retry and ``c`` that decides.

    Args:
        graph: The topology.

    Returns:
        The cycle-closing edges. Empty for a directed acyclic graph.
    """
    found: set[tuple[str, str]] = set()
    finished: set[str] = set()
    for root in [START, *sorted(graph.nodes)]:
        if root not in graph or root in finished:
            continue
        # Iterative DFS. `open_nodes` is the current stack as a set, so an edge
        # into it is an edge into an ancestor, which is the definition of a back
        # edge. Recursion is avoided because a mutated topology can be deep.
        stack: list[tuple[str, list[str]]] = [(root, _pending(graph, root))]
        open_nodes = {root}
        while stack:
            node, pending = stack[-1]
            if not pending:
                stack.pop()
                open_nodes.discard(node)
                finished.add(node)
                continue
            target = pending.pop()
            if target in open_nodes:
                found.add((node, target))
            elif target not in finished:
                stack.append((target, _pending(graph, target)))
                open_nodes.add(target)
    return found


LOOP_REFUSAL = "a loop needs a while edge"
"""Why a version 2 system with a cycle of other edges is refused."""


def forward_graph(graph: nx.DiGraph[str], back: Collection[tuple[str, str]]) -> nx.DiGraph[str]:
    """Return a copy of the topology without its loop-closing edges: one pass of a version 2 run."""
    forward = graph.copy()
    forward.remove_edges_from(back)
    return forward


def loop_problems(graph: nx.DiGraph[str], back: Collection[tuple[str, str]] = ()) -> list[str]:
    """Return one problem per cycle a version 2 topology may not have.

    Version 1 loops on the ``<continue>`` marker of its back edges; version 2
    loops only through a ``while`` edge. Removing the ``while`` edges must
    leave a DAG, and each of them must close a cycle.

    Args:
        graph: The topology.
        back: Its ``while`` edges as ``(source, target)``.

    Returns:
        One line per cycle closed by another edge and per ``while`` edge that
        closes none; empty for a well-formed topology.
    """
    forward = forward_graph(graph, back)
    problems = [
        f"{LOOP_REFUSAL}: the edge {source} -> {target} closes a cycle"
        for source, target in sorted(back_edges(forward))
    ]
    for source, target in back:
        if source != target and source not in _reachable(forward, target, forward=True):
            problems.append(f"the while edge {source} -> {target} closes no loop; make it a when edge")
    return problems


ROUTE_CAP = 256
"""Most route combinations kept per node; past it a merge or join cannot be proved."""

_Point = str | tuple[str, str, str]
"""A node of the route graph: a node name, or a route ``(node, decision, label)``."""

_FAIL, _CHOOSE, _OK = "fail", "choose", ""


@dataclass(frozen=True)
class Join:
    """How a version 2 node with several incoming edges runs.

    A ``"merge"`` has incoming edges that never arrive in the same pass: it runs
    once on whichever arrives. A ``"join"`` has several groups of edges that
    each arrive on every pass that reaches it: it waits for one edge of every
    group, once per pass (a barrier).

    Attributes:
        kind: ``"merge"`` or ``"join"``.
        groups: The sources of the incoming edges, grouped; the edges of one
            group are alternatives. A merge has one group.
        relays: For a join, one entry per group: ``None`` when the group is one
            plain edge from a node without error edges, so the barrier waits
            for that node, else the name of the no-op relay node
            (``__in__.<node>.<k>``) the group's edges enter and the barrier
            waits for. Empty for a merge.
    """

    kind: str
    groups: tuple[tuple[str, ...], ...]
    relays: tuple[str | None, ...] = ()


class RouteGraph:
    """One pass of a version 2 topology, every route of every decision a node of its own.

    ``while`` edges are left out (``max_visits`` forces their exit). A node with
    ``when``, ``while``, ``switch`` or ``else`` edges makes one decision whose
    routes are those edges; a node with ``on error`` edges makes another, its
    plain and deciding edges hanging under an ``ok`` route. A node's choice sets
    are the minimal sets of routes under which it runs in one pass; two nodes
    are alternatives when every pair of their sets takes different routes of
    some decision, which is exact because a decision runs at most once a pass.

    Args:
        nodes: The node names, in a deterministic order.
        edges: The declared edges.

    Raises:
        ValueError: If the edges other than ``while`` edges form a cycle.
    """

    def __init__(self, nodes: Sequence[str], edges: Sequence[Edge]) -> None:
        """Build the route graph and every node's choice sets."""
        self.nodes = list(nodes)
        graph: nx.DiGraph[Any] = nx.DiGraph()
        graph.add_nodes_from([START, *self.nodes, END])
        failing = {edge.source for edge in edges if edge.kind is EdgeKind.ERROR}
        self.loops: list[tuple[str, str]] = []
        for edge in edges:
            if edge.kind is EdgeKind.WHILE:
                self.loops.append((edge.source, edge.target))
                continue
            head: _Point = edge.source
            if edge.source in failing and edge.kind is not EdgeKind.ERROR:
                head = (edge.source, _FAIL, _OK)
                graph.add_edge(edge.source, head, kind="route")
            if edge.kind is EdgeKind.PLAIN:
                graph.add_edge(head, edge.target, kind="plain")
                continue
            route = (edge.source, _FAIL if edge.kind is EdgeKind.ERROR else _CHOOSE, edge.target)
            graph.add_edge(head, route, kind="route")
            graph.add_edge(route, edge.target, kind="plain")
        sources = {edge.source for edge in edges}
        for name in self.nodes:
            if graph.in_degree(name) == 0:
                graph.add_edge(START, name, kind="plain")
            if name not in sources:
                graph.add_edge(name, END, kind="plain")
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"{LOOP_REFUSAL}: the edges other than while edges form a cycle")
        self.graph = graph
        self.choices: dict[_Point, set[frozenset[_Point]] | None] = {START: {frozenset()}}
        for point in nx.topological_sort(graph):
            if point != START:
                self.choices[point] = self._choice_sets(point)
        self._reach: dict[tuple[_Point, tuple[_Point, ...]], bool] = {}

    def _choice_sets(self, point: _Point) -> set[frozenset[_Point]] | None:
        """Return the minimal route sets under which ``point`` runs, ``None`` past :data:`ROUTE_CAP`."""
        extra = {point} if isinstance(point, tuple) else set()
        sets: set[frozenset[_Point]] = set()
        for parent in self.graph.predecessors(point):
            known = self.choices.get(parent)
            if known is None:
                return None
            sets |= {choice | extra for choice in known}
        minimal = {choice for choice in sets if not any(other < choice for other in sets)}
        return minimal if len(minimal) <= ROUTE_CAP else None

    def exclusive(self, first: _Point, second: _Point) -> bool:
        """Return whether two points never both run in one pass; ``False`` when that cannot be proved."""
        left, right = self.choices.get(first), self.choices.get(second)
        if left is None or right is None:
            return False
        return all(_clash(one, other) for one in left for other in right)

    def failed_before(self, name: str) -> frozenset[str]:
        """Return the nodes through whose error routes every pass that runs ``name`` goes."""
        sets = self.choices.get(name)
        if not sets:
            return frozenset()
        failed = [{_name(route) for route in choice if not _every_pass(route) and route[1] == _FAIL} for choice in sets]
        return frozenset(set.intersection(*failed))

    def _reaches(self, point: _Point, group: tuple[_Point, ...]) -> bool:
        """Return whether every pass that runs ``point`` also runs a member of ``group``."""
        key = (point, group)
        if key not in self._reach:
            plain: list[_Point] = []
            routes: list[_Point] = []
            for child in self.graph.successors(point):
                (plain if self.graph.edges[point, child]["kind"] == "plain" else routes).append(child)
            self._reach[key] = (
                point in group
                or any(self._reaches(child, group) for child in plain)
                or (bool(routes) and all(self._reaches(child, group) for child in routes))
            )
        return self._reach[key]

    def _loop_bodies(self) -> list[tuple[str, str, set[str]]]:
        """Return each loop as ``(source, target, the nodes it runs again)``."""
        bodies = []
        for source, target in self.loops:
            inside = {target, *nx.descendants(self.graph, target)} & {source, *nx.ancestors(self.graph, source)}
            bodies.append((source, target, {point for point in inside if isinstance(point, str)}))
        return bodies

    def classify(self) -> tuple[dict[str, Join], dict[str, str]]:
        """Classify every node with several incoming edges.

        Returns:
            ``(joins, refusals)``: the merge or join of each such node, and why
            each node that is neither was refused.
        """
        joins: dict[str, Join] = {}
        refusals: dict[str, str] = {}
        idom: dict[Any, Any] | None = None
        for name in [*self.nodes, END]:
            points: list[_Point] = list(self.graph.predecessors(name))
            if len(points) < 2:
                continue
            if any(self.choices.get(point) is None for point in points):
                refusals[name] = (
                    f"node {name!r}: more than {ROUTE_CAP} route combinations reach it, so whether it merges "
                    "or joins cannot be proved; put part of the system in a box"
                )
                continue
            groups: list[list[_Point]] = []
            for point in points:
                fits = [group for group in groups if all(self.exclusive(point, other) for other in group)]
                if fits:
                    fits[0].append(point)
                else:
                    groups.append([point])
            named = tuple(tuple(_name(point) for point in group) for group in groups)
            if len(groups) == 1:
                joins[name] = Join("merge", named)
                continue
            idom = idom if idom is not None else nx.immediate_dominators(self.graph, START)
            problem = self._join_problem(name, groups, idom[name])
            if problem:
                refusals[name] = problem
                continue
            # One plain edge from a node is waited on at the node; a route (a
            # decision's, or the success route of a node with error edges), the
            # start or several alternatives get a relay.
            relays = tuple(
                None if len(group) == 1 and group[0] != START and isinstance(group[0], str) else f"__in__.{name}.{k}"
                for k, group in enumerate(groups)
            )
            joins[name] = Join("join", named, relays)
        return joins, refusals

    def _join_problem(self, name: str, groups: list[list[_Point]], dominator: _Point) -> str | None:
        """Say why ``name`` cannot wait for one edge of every group, or return ``None``."""

        def edges(group: list[_Point]) -> str:
            return " or ".join(f"{_name(point)} -> {name}" for point in group)

        for group in groups:
            if not self._reaches(dominator, tuple(group)):
                other = next(rest for rest in groups if rest is not group)
                return (
                    f"node {name!r} is neither a merge nor a join: {edges(group)} does not run on every pass "
                    f"that runs {edges(other)}; join the alternatives in one node first"
                )
        for source, target, inside in self._loop_bodies():
            for point in (point for group in groups for point in group):
                if name not in inside and _every_pass(point) and _name(point) in inside:
                    return (
                        f"node {name!r} cannot wait for {_name(point)} -> {name}, which leaves the loop "
                        f"{target} .. {source} on every pass; leave a loop through its else edge"
                    )
        return None


def _name(point: _Point) -> str:
    """Return the node a point of the route graph belongs to."""
    return point if isinstance(point, str) else point[0]


def _every_pass(point: _Point) -> bool:
    """Return whether a point is a node, or the route its node takes whenever it succeeds."""
    return isinstance(point, str) or point[1:] == (_FAIL, _OK)


def _clash(first: frozenset[_Point], second: frozenset[_Point]) -> bool:
    """Return whether two route sets take different routes of one decision."""
    return any(
        a[:2] == b[:2] and a[2] != b[2] for a in first for b in second if isinstance(a, tuple) and isinstance(b, tuple)
    )


def classify_joins(nodes: Sequence[str], edges: Sequence[Edge]) -> dict[str, Join]:
    """Classify every version 2 node with several incoming edges as a merge or a join.

    Args:
        nodes: The node names, in a deterministic order; the result follows it,
            :data:`END` last.
        edges: The declared edges; ``while`` edges are left out of the pass.

    Returns:
        ``{node: Join}`` for every node with several incoming edges,
        :data:`END` included.

    Raises:
        ValueError: If a node is neither a merge nor a join, or the edges other
            than ``while`` edges form a cycle; the message names every such node.
    """
    joins, refusals = RouteGraph(nodes, edges).classify()
    if refusals:
        raise ValueError("; ".join(refusals.values()))
    return joins


def ancestors(graph: nx.DiGraph[str], node: str) -> set[str]:
    """Return every node with a path to ``node``, sentinels included.

    Args:
        graph: The topology.
        node: The node to inspect.

    Returns:
        Its ancestors, excluding ``node`` unless it sits on a cycle.
    """
    return _reachable(graph, node, forward=False)


def forward_predecessors(graph: nx.DiGraph[str], node: str) -> set[str]:
    """Return a node's predecessors excluding those that reach it by a back edge.

    Args:
        graph: The topology.
        node: The node to inspect.

    Returns:
        Predecessors on acyclic paths only.
    """
    back = back_edges(graph)
    return {source for source in graph.predecessors(node) if (source, node) not in back}


def join_nodes(graph: nx.DiGraph[str]) -> set[str]:
    """Return the agents that must wait for every inbound path before running.

    A node fed by predecessors at unequal depth would otherwise execute once per
    wave of arriving inputs: in ``A -> C`` together with ``A -> B -> C``, C sees
    A's write in one step and B's in the next, and runs twice for one query. The
    seed engine has this defect latent, and every mutated topology triggers it,
    so the set computed here is what the executor defers on.

    Back edges are excluded from the count. A node at the top of a retry loop is
    fed both from upstream and from the agent that decided to loop, but those two
    never arrive in the same pass, and deferring on the second would hold the node
    until an input that only exists if it runs first.

    Args:
        graph: The topology.

    Returns:
        Names of agents with more than one distinct forward predecessor.
    """
    agents = set(graph.nodes) - SENTINELS
    back = back_edges(graph)
    counts = {agent: sum(1 for source in graph.predecessors(agent) if (source, agent) not in back) for agent in agents}
    return {agent for agent, count in counts.items() if count > 1}


def terminal_agent(graph: nx.DiGraph[str]) -> str:
    """Return the single agent feeding :data:`END`.

    Args:
        graph: The topology.

    Returns:
        The terminal agent's name, whose output is the MAS output.

    Raises:
        ValueError: If the topology has zero or several terminal agents.
    """
    terminals = sorted(graph.predecessors(END))
    if not terminals:
        raise ValueError(f"No agent feeds {END!r}, so the MAS has no output.")
    if len(terminals) > 1:
        raise ValueError(
            f"Several agents feed {END!r} ({terminals}), so the MAS output is ambiguous. "
            "Add a final agent that consumes them."
        )
    return terminals[0]
