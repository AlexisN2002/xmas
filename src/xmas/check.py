"""The dataflow check of a version 2 system, run when the MAS is built.

:func:`check_system` asks every question a run would otherwise answer late and
expensively: does each value a node reads exist before the node runs, is each
value written once (or by alternatives), does each prompt placeholder name a
read, do the names and types of steps and boxes fit, do the edges decide on
values that exist, and is every node with several incoming edges a merge or a
join. It collects every problem and raises them together, so one edit cycle
fixes them all. A version 1 system never comes here; it keeps
``validate_graph`` and ``terminal_agent``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from xmas.agent import Agent
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import ErrorKind
from xmas.graph import (
    END,
    SENTINELS,
    START,
    Join,
    RouteGraph,
    agent_names,
    ancestors,
    back_edges,
    build_graph,
    forward_graph,
    loop_problems,
    validate_graph,
)
from xmas.nodes import Box
from xmas.predicates import Predicate, PredicateError, parse_predicate
from xmas.prompts import extract_placeholders
from xmas.schemas import compile_form
from xmas.values import AVAILABLE, JSON, RESERVED, ValueType, parse_type

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.nodes import Step
    from xmas.tools import Tool

MERGE_LAST = "last"
"""The one merge rule: the latest write wins."""

ROUTED_ERRORS = frozenset(kind.value for kind in ErrorKind) - {ErrorKind.BUDGET.value, ErrorKind.STOPPED.value}
"""The error kinds an ``on error`` edge may route; a run always stops on the others."""


@dataclass(frozen=True)
class SystemInfo:
    """What the check learned about a version 2 system, for the engine and ``describe``.

    Attributes:
        types: The type of every value, inputs and writes alike.
        writers: The nodes writing each written value, in node order.
        joins: Each node with several incoming edges, :data:`~xmas.graph.END`
            included, as a merge or a join (:func:`~xmas.graph.classify_joins`).
    """

    types: dict[str, ValueType]
    writers: dict[str, tuple[str, ...]]
    joins: dict[str, Join]


def callable_writes(item: Tool) -> dict[str, ValueType]:
    """Return what a callable writes, ``{write: type}``, as :class:`~xmas.tools.Tool` typed it when built."""
    return cast("dict[str, ValueType]", item.writes)


def step_maps(step: Step, item: Tool) -> tuple[dict[str, str], dict[str, str]]:
    """Return a step's reads and writes as ``{name inside: name in the graph}``.

    Every parameter of the callable is a read; a parameter not renamed reads the
    value of its own name, and a write not renamed keeps its own name.

    Args:
        step: The step.
        item: The callable it uses.

    Returns:
        ``(reads, writes)``.
    """
    reads = {name: step.reads.get(name, name) for name in item.reads}
    writes = {name: step.writes.get(name, name) for name in callable_writes(item)}
    return reads, writes


def _schema_type(item: Tool, parameter: str) -> str:
    """Name a parameter's JSON-schema type for a message."""
    schema = item.parameters.get("properties", {}).get(parameter, {})
    return str(schema.get("type", "json")) if isinstance(schema, dict) else "json"


def _name_problem(name: str) -> str | None:
    """Say why a value name is not allowed, or return ``None``."""
    if name in RESERVED:
        return "is a reserved name; pick another"
    if name.startswith("__"):
        return "starts with '__'; such names belong to xmas"
    return None


def _type_problem(vtype: ValueType) -> str | None:
    """Say why a type cannot run in this phase, or return ``None``."""
    return None if vtype in AVAILABLE else f"has type {str(vtype)!r}, which arrives in phase 3"


def _contains(system: MAS, target: MAS, seen: set[int]) -> bool:
    """Return whether ``target`` is ``system`` or sits in one of its boxes, at any depth."""
    if system is target:
        return True
    if id(system) in seen:
        return False
    seen.add(id(system))
    return any(_contains(box.system, target, seen) for box in system.boxes.values())


def _edge_problems(edges: tuple[Edge, ...], endpoints: set[str]) -> tuple[list[str], dict[Edge, Predicate], bool]:
    """Check each edge on its own and each node's set of deciding edges.

    Returns:
        The problems; the parsed predicate (or switch path) of every
        ``when``, ``while`` and ``switch`` edge that parsed; and whether the
        edges make a graph (known endpoints, one edge per pair).
    """
    problems: list[str] = []
    parsed: dict[Edge, Predicate] = {}
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        bad = [n for n in (edge.source, edge.target) if n not in endpoints]
        if bad or edge.source == END or edge.target == START:
            problems.append(f"edge {edge.label}: {bad[0]!r} is no node" if bad else f"edge {edge.label} runs backwards")
            continue
        if (edge.source, edge.target) in pairs:
            problems.append(f"edge {edge.label} is listed twice; one edge per pair of nodes")
        pairs.add((edge.source, edge.target))
        if edge.kind is EdgeKind.ERROR:
            for kind in edge.errors:
                if kind == ANY_ERROR and len(edge.errors) > 1:
                    problems.append(f"edge {edge.label}: on error 'any' stands alone")
                elif kind in (ErrorKind.BUDGET, ErrorKind.STOPPED):
                    problems.append(f"edge {edge.label}: on error {kind!r} is never routed; the run stops on it")
                elif kind != ANY_ERROR and kind not in ROUTED_ERRORS:
                    problems.append(
                        f"edge {edge.label}: on error {kind!r} is no error kind; known: {sorted(ROUTED_ERRORS)}"
                    )
        text = edge.switch if edge.kind is EdgeKind.SWITCH else edge.when
        if text is None:
            continue
        try:
            predicate = parse_predicate(text)
        except PredicateError as exc:
            problems.append(f"edge {edge.label}: {exc}")
            continue
        if edge.kind is EdgeKind.SWITCH and predicate.tree[0] != "path":
            problems.append(f"edge {edge.label}: switch takes a path such as verdict.action, not {text!r}")
            continue
        parsed[edge] = predicate

    deciding = Counter(e.source for e in edges if e.kind in (EdgeKind.WHEN, EdgeKind.WHILE, EdgeKind.SWITCH))
    elses = Counter(e.source for e in edges if e.kind is EdgeKind.ELSE)
    for source in dict.fromkeys(e.source for e in edges):
        if (deciding[source] or elses[source]) and (elses[source] != 1 or not deciding[source]):
            problems.append(
                f"node {source!r} has {deciding[source]} when, while or switch edges and {elses[source]} else edges; "
                "a deciding node needs at least one of the first and exactly one else"
            )
    cases = Counter((e.source, e.case) for e in edges if e.kind is EdgeKind.SWITCH)
    problems.extend(f"node {source!r} switches to case {case!r} twice" for (source, case), n in cases.items() if n > 1)
    return problems, parsed, len(pairs) == len(edges)


def check_system(mas: MAS) -> SystemInfo:
    """Check a version 2 system and return what the engine needs to know.

    Args:
        mas: The system, with its agents, steps, boxes, edges, inputs, merges,
            frozen nodes and tools set; its graph is not needed yet.

    Returns:
        The types, writers and joins.

    Raises:
        ValueError: Listing every problem found, one per line.
    """
    problems: list[str] = []
    nodes: dict[str, Agent | Step | Box] = {**mas.agents, **mas.steps, **mas.boxes}

    for name in nodes:
        if name.startswith("__"):
            problems.append(f"node {name!r}: names starting with '__' belong to xmas")
    for name in sorted(mas.frozen - set(nodes)):
        problems.append(f"frozen names {name!r}, which is no node")

    types: dict[str, ValueType] = {}
    for name, vtype in mas.inputs.items():
        for problem in (_name_problem(name), _type_problem(vtype)):
            if problem:
                problems.append(f"input {name!r} {problem}")
        types[name] = vtype

    # What each node reads (value -> (name inside, its type, a word for that
    # type) for a step or a box; None for an agent, which renders anything) and
    # writes.
    reads: dict[str, dict[str, tuple[str, ValueType, str] | None]] = {}
    writes: dict[str, dict[str, ValueType]] = {}
    for name, node in nodes.items():
        if isinstance(node, Agent):
            reads[name] = dict.fromkeys(node.inputs)
            writes[name] = {node.output: parse_type(node.output_type)}
            if node.tools:
                try:
                    mas.resolve_tools(node)
                except ValueError as exc:
                    problems.append(str(exc))
            missing = sorted(extract_placeholders(node.prompt) - set(node.inputs))
            if missing:
                problems.append(f"agent {name!r}: prompt placeholders {missing} are not among its reads {node.inputs}")
            continue
        if isinstance(node, Box):
            if _contains(node.system, mas, set()):
                problems.append(f"box {name!r} contains the system it sits in")
                reads[name], writes[name] = {}, {}
                continue
            outer_reads, writes[name] = node.types()
            reads[name] = {
                outer: (inner, outer_reads[outer], str(outer_reads[outer])) for inner, outer in node.reads.items()
            }
            continue
        try:
            item, _source = mas.resolve_callable(node)
            parameters, produced = item.reads, callable_writes(item)
        except ValueError as exc:
            problems.append(str(exc))
            reads[name], writes[name] = {}, {}
            continue
        for label, renames, known in (("reads", node.reads, parameters), ("writes", node.writes, produced)):
            for inside in sorted(set(renames) - set(known)):
                what = "a parameter" if label == "reads" else "a write"
                problems.append(
                    f"step {name!r}: {label} renames {inside!r}, which is not {what} of {node.uses!r} "
                    f"(it has {sorted(known)})"
                )
        inner_reads, inner_writes = step_maps(node, item)
        reads[name] = {
            outer: (inner, parameters[inner], _schema_type(item, inner)) for inner, outer in inner_reads.items()
        }
        writes[name] = {outer: produced[inner] for inner, outer in inner_writes.items()}

    writers: dict[str, list[str]] = {}
    for name, produced in writes.items():
        for value, vtype in produced.items():
            for problem in (_name_problem(value), _type_problem(vtype)):
                if problem:
                    problems.append(f"{name!r} writes {value!r}, which {problem}")
            if value in types and types[value] != vtype:
                problems.append(f"value {value!r} is {types[value]} elsewhere but {name!r} writes it as {vtype}")
            types.setdefault(value, vtype)
            writers.setdefault(value, []).append(name)
    for value in sorted(set(mas.merges) - set(types)):
        problems.append(f"state names {value!r}, which no node writes and no input declares")

    edge_problems, predicates, buildable = _edge_problems(mas.edges, set(nodes) | SENTINELS)
    problems.extend(edge_problems)

    routes: RouteGraph | None = None
    joins: dict[str, Join] = {}
    if buildable:
        loops = [(edge.source, edge.target) for edge in mas.edges if edge.kind is EdgeKind.WHILE]
        graph = build_graph(list(nodes), [(edge.source, edge.target) for edge in mas.edges], back=loops)
        try:
            validate_graph(graph)
        except ValueError as exc:
            problems.append(str(exc))
        problems.extend(loop_problems(graph, loops))
        forward = forward_graph(graph, loops)
        if not back_edges(forward):
            routes = RouteGraph(agent_names(forward), mas.edges)
            joins, refusals = routes.classify()
            problems.extend(refusals.values())
            terminals = sorted(graph.predecessors(END))
            end = joins.get(END)
            if len(terminals) > 1 and END not in refusals and (end is None or end.kind != "merge"):
                problems.append(
                    f"several nodes feed {END!r} ({terminals}) on one pass, so the output is ambiguous; "
                    "add one final node"
                )

            def readable(value: str, at: str) -> bool:
                return value in mas.inputs or bool({at, *ancestors(graph, at)}.intersection(writers.get(value, ())))

            for name, needed in reads.items():
                failed = routes.failed_before(name)
                for value, parameter in needed.items():
                    if value not in mas.inputs and value not in writers:
                        problems.append(f"{name!r} reads {value!r}, which is neither an input nor written by any node")
                        continue
                    if value not in mas.inputs and not ancestors(graph, name).intersection(writers[value]):
                        problems.append(
                            f"{name!r} reads {value!r}, but its writers {writers[value]} do not run before it"
                        )
                    elif value not in mas.inputs and set(writers[value]) <= failed:
                        problems.append(
                            f"{name!r} runs only when {writers[value][0]!r} fails, so it cannot read {value!r}"
                        )
                    if parameter is None or types[value] == parameter[1]:
                        continue
                    if isinstance(nodes[name], Box):
                        problems.append(
                            f"box {name!r} passes {types[value]} {value!r} to its {parameter[2]} input {parameter[0]!r}"
                        )
                    else:
                        problems.append(
                            f"step {name!r} passes {types[value]} {value!r} to parameter {parameter[0]!r} "
                            f"({parameter[2]}); text goes into a string parameter only, json into any other"
                        )

            registered = set(mas._callables())
            for edge, predicate in predicates.items():
                for path in sorted(predicate.paths()):
                    root, fields = path[0], ".".join(path[1:])
                    if not readable(root, edge.source):
                        problems.append(
                            f"edge {edge.label} decides on {root!r}, which is not readable at {edge.source!r}"
                        )
                        continue
                    if not fields:
                        continue
                    if types.get(root) != JSON:
                        problems.append(f"edge {edge.label} reads the field {fields!r} of {root!r}, which is not json")
                        continue
                    for writer in writers.get(root, ()):
                        node = nodes[writer]
                        if not isinstance(node, Agent) or node.schema is None:
                            continue
                        form = compile_form(node.schema)
                        if fields not in form.paths():
                            problems.append(
                                f"edge {edge.label} reads {root}.{fields}, but {writer!r}'s answer form has no "
                                f"field {fields!r} (it has {list(form.paths())})"
                            )
                        elif edge.kind is EdgeKind.SWITCH:
                            choices = form.choices(fields)
                            if choices is not None and edge.case not in choices:
                                problems.append(
                                    f"edge {edge.label}: case {edge.case!r} is not one of {list(choices)}, "
                                    f"the choices of {root}.{fields}"
                                )
                for function in sorted(predicate.functions() - registered):
                    problems.append(f"edge {edge.label} calls {function!r}, which is no registered callable")

    for value, names in writers.items():
        count = len(names) + (value in mas.inputs)
        alternatives = (
            routes is not None
            and value not in mas.inputs
            and all(routes.exclusive(a, b) for i, a in enumerate(names) for b in names[i + 1 :])
        )
        if count > 1 and mas.merges.get(value) != MERGE_LAST and not alternatives:
            sources = (["the input"] if value in mas.inputs else []) + [repr(n) for n in names]
            problems.append(
                f"value {value!r} has {count} writers ({', '.join(sources)}); give it one writer, make the writers "
                f"alternatives, or declare state: {{{value}: {{merge: last}}}}"
            )

    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"The system does not check ({len(problems)} problem{'s' * (len(problems) > 1)}):\n{lines}")
    return SystemInfo(types, {value: tuple(names) for value, names in writers.items()}, joins)
