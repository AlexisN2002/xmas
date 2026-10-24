"""Route analysis: whether a version 2 node with several incoming edges merges or joins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from xmas.edges import Edge, EdgeKind
from xmas.graph import END, START

ROUTE_CAP = 256
"""Most route combinations kept per node; past it a merge or join cannot be proved."""

_Point = str | tuple[str, str, str]
"""A node of the route graph: a node name, or a route ``(node, decision, label)``."""

_FAIL, _CHOOSE, _OK = "fail", "choose", ""


@dataclass(frozen=True)
class Join:
    """How a version 2 node with several incoming edges runs.

    ``groups`` holds the sources of the incoming edges; the edges of one group
    are alternatives. A ``"merge"`` has one group, whose edges never arrive in
    the same pass: it runs once on whichever arrives. A ``"join"`` has several
    groups that each arrive on every pass reaching it: it waits for one edge of
    every group, once per pass (a barrier). A join has one relay per group:
    ``None`` when the group is one plain edge from a node without error edges,
    so the barrier waits for that node, else the no-op node
    (``__in__.<node>.<k>``) the group's edges enter.
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
    The edges other than ``while`` edges must not form a cycle.
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
        sources = {edge.source for edge in edges if edge.kind is not EdgeKind.ERROR}
        for name in self.nodes:
            if graph.in_degree(name) == 0:
                graph.add_edge(START, name, kind="plain")
            if name in sources:
                continue
            if name in failing:  # only error edges: a visit that succeeds ends the run
                ok = (name, _FAIL, _OK)
                graph.add_edge(name, ok, kind="route")
                graph.add_edge(ok, END, kind="plain")
            else:
                graph.add_edge(name, END, kind="plain")
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
        """Return each multi-input node's merge or join, and the refusal of each node that is neither."""
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
