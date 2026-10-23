"""The dataflow check of a version 2 system, run when the MAS is built.

:func:`check_system` asks every question a run would otherwise answer late and
expensively: does each value a node reads exist before the node runs, is each
value written once, does each prompt placeholder name a read, do the names and
types of steps fit their callables, and does the topology run in this phase. It
collects every problem and raises them together, so one edit cycle fixes them
all. A version 1 system never comes here; it keeps ``validate_graph`` and
``terminal_agent``.

Phase 2a runs plain edges only: a conditional edge or a cycle is refused,
naming phase 2b, and every node with several predecessors is a join that waits
for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from xmas.agent import Agent
from xmas.edges import EdgeKind
from xmas.graph import END, SENTINELS, START, ancestors, build_graph, loop_problems, validate_graph
from xmas.prompts import extract_placeholders
from xmas.values import AVAILABLE, RESERVED, ValueType, parse_type

if TYPE_CHECKING:  # pragma: no cover
    from xmas.mas import MAS
    from xmas.nodes import Step
    from xmas.tools import Tool

MERGE_LAST = "last"
"""The one merge rule of phase 2a: the latest write wins."""


@dataclass(frozen=True)
class SystemInfo:
    """What the check learned about a version 2 system, for the engine and ``describe``.

    Attributes:
        types: The type of every value, inputs and writes alike.
        writers: The nodes writing each written value, in node order.
        joins: Each node that waits for several predecessors, with those
            predecessors sorted by name. In phase 2a each is a group of one and
            the engine joins them with one barrier.
    """

    types: dict[str, ValueType]
    writers: dict[str, tuple[str, ...]]
    joins: dict[str, tuple[str, ...]]


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


def check_system(mas: MAS) -> SystemInfo:
    """Check a version 2 system and return what the engine needs to know.

    Args:
        mas: The system, with its agents, steps, edges, inputs, merges and
            tools set; its graph is not needed yet.

    Returns:
        The types, writers and joins.

    Raises:
        ValueError: Listing every problem found, one per line.
    """
    problems: list[str] = []
    nodes: dict[str, Agent | Step] = {**mas.agents, **mas.steps}

    for name in nodes:
        if name.startswith("__"):
            problems.append(f"node {name!r}: names starting with '__' belong to xmas")

    types: dict[str, ValueType] = {}
    for name, vtype in mas.inputs.items():
        for problem in (_name_problem(name), _type_problem(vtype)):
            if problem:
                problems.append(f"input {name!r} {problem}")
        types[name] = vtype

    # What each node reads (value -> the parameter of a step, with its type and
    # schema type; None for an agent, which renders anything) and writes.
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

    for value, names in writers.items():
        count = len(names) + (value in mas.inputs)
        if count > 1 and mas.merges.get(value) != MERGE_LAST:
            sources = (["the input"] if value in mas.inputs else []) + [repr(n) for n in names]
            problems.append(
                f"value {value!r} has {count} writers ({', '.join(sources)}); give it one writer or "
                f"declare state: {{{value}: {{merge: last}}}}"
            )
    for value in sorted(set(mas.merges) - set(types)):
        problems.append(f"state names {value!r}, which no node writes and no input declares")

    endpoints = set(nodes) | SENTINELS
    pairs: list[tuple[str, str]] = []
    for edge in mas.edges:
        bad = [n for n in (edge.source, edge.target) if n not in endpoints]
        if bad or edge.source == END or edge.target == START:
            problems.append(f"edge {edge.label}: {bad[0]!r} is no node" if bad else f"edge {edge.label} runs backwards")
            continue
        if edge.kind is not EdgeKind.PLAIN:
            problems.append(f"edge {edge.label}: a {edge.kind.value} edge arrives in phase 2b")
        if (edge.source, edge.target) in pairs:
            problems.append(f"edge {edge.label} is listed twice; one edge per pair of nodes")
        pairs.append((edge.source, edge.target))

    joins: dict[str, tuple[str, ...]] = {}
    if len(pairs) == len(mas.edges):
        graph = build_graph(list(nodes), pairs)
        try:
            validate_graph(graph)
        except ValueError as exc:
            problems.append(str(exc))
        cycles = loop_problems(graph)
        problems.extend(cycles)
        terminals = sorted(graph.predecessors(END))
        if len(terminals) > 1:
            problems.append(f"several nodes feed {END!r} ({terminals}), so the output is ambiguous; add one final node")
        if not cycles:
            for name, needed in reads.items():
                upstream = ancestors(graph, name)
                for value, parameter in needed.items():
                    if value not in mas.inputs and value not in writers:
                        problems.append(f"{name!r} reads {value!r}, which is neither an input nor written by any node")
                        continue
                    if value not in mas.inputs and not upstream.intersection(writers[value]):
                        problems.append(
                            f"{name!r} reads {value!r}, but its writers {writers[value]} do not run before it"
                        )
                    if parameter is not None and types[value] != parameter[1]:
                        problems.append(
                            f"step {name!r} passes {types[value]} {value!r} to parameter {parameter[0]!r} "
                            f"({parameter[2]}); text goes into a string parameter only, json into any other"
                        )
            for name in [*nodes, END]:
                predecessors = sorted(graph.predecessors(name))
                if len(predecessors) > 1:
                    joins[name] = tuple(predecessors)

    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"The system does not check ({len(problems)} problem{'s' * (len(problems) > 1)}):\n{lines}")
    return SystemInfo(types, {value: tuple(names) for value, names in writers.items()}, joins)
