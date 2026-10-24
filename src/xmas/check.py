"""The dataflow check of a version 2 system, run when the MAS is built.

:func:`check_system` asks every question a run would otherwise answer late and
expensively: does each value a node reads exist before the node runs, is each
value written once (or by alternatives), does each prompt placeholder name a
read, do the names and types of steps and boxes fit, do the edges decide on
values that exist, and is every node with several incoming edges a merge or a
join. It collects every problem and raises them together, so one edit cycle
fixes them all. A version 1 system never comes here; it keeps
``validate_graph`` and ``terminal_agent``.

A ``for each`` chain is checked twice: inside, as a system of its own that one
item runs, and outside, collapsed into its head, which reads what the chain
reads from outside and writes each value the chain writes as a list.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xmas.agent import Agent
from xmas.chains import Chain, ChainShape, collapse, find_chains
from xmas.edges import ANY_ERROR, Edge, EdgeKind
from xmas.errors import RUN_ENDS, ErrorKind
from xmas.graph import (
    END,
    SENTINELS,
    START,
    agent_names,
    ancestors,
    back_edges,
    build_graph,
    forward_graph,
    loop_problems,
    validate_graph,
)
from xmas.nodes import Box
from xmas.predicates import BUILTINS, KEYWORDS, Predicate, PredicateError, parse_predicate
from xmas.prompts import extract_placeholders
from xmas.routes import Join, RouteGraph
from xmas.schemas import compile_form
from xmas.values import JSON, KINDS, LIST_PREFIX, TEXT, ValueType, fits, parse_type

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

    from xmas.mas import MAS
    from xmas.nodes import Step
    from xmas.tools import Tool

MERGE_LAST = "last"
"""The one merge rule: the latest write wins."""

ROUTED_ERRORS = frozenset(kind.value for kind in ErrorKind if kind not in RUN_ENDS)
"""The error kinds an ``on error`` edge may route; a run always stops on the others."""

AVAILABLE = frozenset({TEXT, JSON})
"""The value kinds a system may use; a ``list of`` them, at any depth, as well."""

RESERVED_READS = {"error": JSON, "run": JSON}
"""Values xmas fills and no node writes. ``error``, ``{kind, message, node}`` of
the failure routed to a node, is readable by a node an ``on error`` edge reaches
and on that node's edges; ``run``, ``{cost_usd, elapsed_s, calls, records}`` so
far, is readable anywhere but in a box."""

RESERVED = frozenset({*KINDS, *LIST_PREFIX.split(), *BUILTINS, *KEYWORDS, *RESERVED_READS})
"""Names no value may take: the type words, the words of predicates and the
reserved reads. Names starting with ``__`` belong to xmas as well."""

RETRYABLE = frozenset({"outage", "timeout", "rate_limited"})
"""The error kinds a node's ``retry`` may run it again on."""


@dataclass(frozen=True)
class SystemInfo:
    """What the check learned about a version 2 system, for the engine and ``describe``.

    Attributes:
        types: The type of every value, inputs and writes alike.
        writers: The nodes writing each written value, in node order.
        joins: Each node with several incoming edges, :data:`~xmas.graph.END`
            included, as a merge or a join.
        graph: The topology, with the implicit wiring from
            :data:`~xmas.graph.START` and to :data:`~xmas.graph.END`.
        predicates: The parsed predicate of every ``when``, ``while`` and
            ``switch`` edge the engine wires (a switch's is its path).
        chains: Each ``for each`` chain, keyed by its head.
        edges: The edges the engine wires: the declared ones, each chain
            collapsed into its head.
        flow: The topology of ``edges``, what the engine wires; ``graph``
            when there is no chain.

    A value a chain writes has its gathered type here, a list; the reserved
    reads (:data:`RESERVED_READS`) are among ``types`` when a node reads them.
    """

    types: dict[str, ValueType]
    writers: dict[str, tuple[str, ...]]
    joins: dict[str, Join]
    graph: nx.DiGraph[str]
    predicates: dict[Edge, Predicate]
    chains: dict[str, Chain]
    edges: tuple[Edge, ...]
    flow: nx.DiGraph[str]


class _CheckError(ValueError):
    """A system that does not check; ``problems`` holds its lines."""

    def __init__(self, problems: list[str]) -> None:
        lines = "\n".join(f"- {problem}" for problem in problems)
        super().__init__(f"The system does not check ({len(problems)} problem{'s' * (len(problems) > 1)}):\n{lines}")
        self.problems = problems


def step_maps(step: Step, item: Tool) -> tuple[dict[str, str], dict[str, str]]:
    """Return a step's ``(reads, writes)``, each ``{name inside: name in the graph}``.

    Every parameter of the callable is a read; a parameter not renamed reads the
    value of its own name, and a write not renamed keeps its own name.
    """
    reads = {name: step.reads.get(name, name) for name in item.reads}
    writes = {name: step.writes.get(name, name) for name in item.writes}
    return reads, writes


def _schema_type(item: Tool, parameter: str) -> str:
    """Name a parameter's JSON-schema type for a message."""
    schema = item.parameters.get("properties", {}).get(parameter, {})
    return str(schema.get("type", "json")) if isinstance(schema, dict) else "json"


def _available(vtype: ValueType) -> bool:
    """Return whether a system may use a type: an available kind, or a list of an available type."""
    return vtype in AVAILABLE or (vtype.item is not None and _available(vtype.item))


def _value_problems(name: str, vtype: ValueType) -> Iterator[str]:
    """Say why a value's name or type is not allowed."""
    if name in RESERVED:
        yield "is a reserved name; pick another"
    elif name.startswith("__"):
        yield "starts with '__'; such names belong to xmas"
    if not _available(vtype):
        yield f"has type {str(vtype)!r}; {vtype} values are not available yet"


def _holds_chain(system: MAS) -> bool:
    """Return whether a system, or a box in it at any depth, has a ``for each`` chain."""
    return bool(system.each) or any(_holds_chain(box.system) for box in system.boxes.values())


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

    Returns the problems, the parsed predicate (or switch path) of every edge
    that has one, and whether the edges make a graph (known endpoints, one edge
    per pair).
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
                elif kind in RUN_ENDS:
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
    # a bool never equals a number, as predicates compare: case false and case 0 are two cases
    cases = Counter((e.source, isinstance(e.case, bool), e.case) for e in edges if e.kind is EdgeKind.SWITCH)
    problems.extend(
        f"node {source!r} switches to case {case!r} twice" for (source, _, case), n in cases.items() if n > 1
    )
    return problems, parsed, len(pairs) == len(edges)


_Reads = dict[str, dict[str, tuple[str, ValueType, str] | None]]
"""Per node, each value it reads: ``(name inside, type, type word)`` for a step or a box, ``None`` for an agent."""
_Writes = dict[str, dict[str, ValueType]]


def _node_io(mas: MAS, nodes: dict[str, Agent | Step | Box]) -> tuple[_Reads, _Writes, dict[str, list[str]]]:
    """Return what each node reads and writes, and per node the problems of its tools, prompt, callable or box."""
    problems: dict[str, list[str]] = {name: [] for name in nodes}
    reads: _Reads = {}
    writes: _Writes = {}
    for name, node in nodes.items():
        reads[name], writes[name] = {}, {}
        if isinstance(node, Agent):
            reads[name] = dict.fromkeys(node.inputs)
            writes[name] = {node.output: parse_type(node.output_type)}
            if node.tools:
                try:
                    mas.resolve_tools(node)
                except ValueError as exc:
                    problems[name].append(str(exc))
            missing = sorted(extract_placeholders(node.prompt) - set(node.reads))
            if missing:
                problems[name].append(
                    f"agent {name!r}: prompt placeholders {missing} are not among its reads {list(node.reads)}"
                )
        elif isinstance(node, Box):
            if _contains(node.system, mas, set()):
                problems[name].append(f"box {name!r} contains the system it sits in")
                continue
            outer_reads, writes[name] = node.types()
            reads[name] = {
                outer: (inner, outer_reads[outer], str(outer_reads[outer])) for inner, outer in node.reads.items()
            }
        else:
            try:
                item, _source = mas.resolve_callable(node)
                parameters, produced = item.reads, item.writes
            except ValueError as exc:
                problems[name].append(str(exc))
                continue
            for label, renames, known in (("reads", node.reads, parameters), ("writes", node.writes, produced)):
                for inside in sorted(set(renames) - set(known)):
                    what = "a parameter" if label == "reads" else "a write"
                    problems[name].append(
                        f"step {name!r}: {label} renames {inside!r}, which is not {what} of {node.uses!r} "
                        f"(it has {sorted(known)})"
                    )
            inner_reads, inner_writes = step_maps(node, item)
            reads[name] = {
                outer: (inner, parameters[inner], _schema_type(item, inner)) for inner, outer in inner_reads.items()
            }
            writes[name] = {outer: produced[inner] for inner, outer in inner_writes.items()}
    return reads, writes, problems


def _writer_types(
    mas: MAS, writes: _Writes, heads: Sequence[str] = ()
) -> tuple[dict[str, ValueType], dict[str, list[str]], list[str]]:
    """Return the type of every value, the writers of each written value, and the problems of their names and types.

    A chain's head writes the chain's values, whose names its chain's own check judges.
    """
    problems: list[str] = []
    types = dict(mas.inputs)
    writers: dict[str, list[str]] = {}
    for name, produced in writes.items():
        for value, vtype in produced.items():
            if name not in heads:
                problems += [f"{name!r} writes {value!r}, which {problem}" for problem in _value_problems(value, vtype)]
            if value in types and types[value] != vtype:
                problems.append(f"value {value!r} is {types[value]} elsewhere but {name!r} writes it as {vtype}")
            types.setdefault(value, vtype)
            writers.setdefault(value, []).append(name)
    problems += [
        f"state names {v!r}, which no node writes and no input declares" for v in sorted({*mas.merges} - {*types})
    ]
    return types, writers, problems


def _graph(nodes: list[str], edges: Sequence[Edge]) -> nx.DiGraph[str]:
    """Build the topology of nodes and edges, ``while`` edges closing loops and error edges taken on failure."""
    loops = [(edge.source, edge.target) for edge in edges if edge.kind is EdgeKind.WHILE]
    errors = [(edge.source, edge.target) for edge in edges if edge.kind is EdgeKind.ERROR]
    return build_graph(nodes, [(edge.source, edge.target) for edge in edges], back=loops, errors=errors)


def _topology(
    nodes: list[str], edges: tuple[Edge, ...], outputs: bool
) -> tuple[nx.DiGraph[str], RouteGraph | None, dict[str, Join], list[str]]:
    """Build the graph and, when its forward part has no cycle, its routes and joins; with the problems found.

    Several nodes may feed the end only of a system that declares its outputs.
    """
    loops = [(edge.source, edge.target) for edge in edges if edge.kind is EdgeKind.WHILE]
    graph = _graph(nodes, edges)
    problems: list[str] = []
    try:
        validate_graph(graph)
    except ValueError as exc:
        problems.append(str(exc))
    problems.extend(loop_problems(graph, loops))
    forward = forward_graph(graph, loops)
    if back_edges(forward):
        return graph, None, {}, problems
    routes = RouteGraph(agent_names(forward), edges)
    joins, refusals = routes.classify()
    problems.extend(refusals.values())
    terminals = sorted(graph.predecessors(END))
    end = joins.get(END)
    if len(terminals) > 1 and not outputs and END not in refusals and (end is None or end.kind != "merge"):
        problems.append(
            f"several nodes feed {END!r} ({terminals}) on one pass, so the output is ambiguous; add one final node"
        )
    return graph, routes, joins, problems


def _read_problems(
    mas: MAS,
    reads: _Reads,
    types: dict[str, ValueType],
    writers: dict[str, list[str]],
    routes: RouteGraph,
    before: dict[str, set[str]],
    erred: set[str],
) -> Iterator[str]:
    """Say which reads have no writer that runs first, or have the wrong type for a step or a box.

    ``erred`` holds the nodes an ``on error`` edge reaches, the ones that may read ``error``.
    """
    for name, needed in reads.items():
        failed = routes.failed_before(name)
        for value, parameter in needed.items():
            if value in RESERVED_READS:
                problem = _reserved_problem(name, value, name in mas.boxes, erred)
                if problem is not None:
                    yield problem
                    continue
            elif value not in mas.inputs and value not in writers:
                yield f"{name!r} reads {value!r}, which is neither an input nor written by any node"
                continue
            elif value not in mas.inputs and not before[name].intersection(writers[value]):
                yield f"{name!r} reads {value!r}, but its writers {writers[value]} do not run before it"
            elif value not in mas.inputs and set(writers[value]) <= failed:
                yield f"{name!r} runs only when {writers[value][0]!r} fails, so it cannot read {value!r}"
            if parameter is None or fits(types[value], parameter[1]):
                continue
            inside, _, word = parameter
            if name in mas.boxes:
                yield f"box {name!r} passes {types[value]} {value!r} to its {word} input {inside!r}"
            else:
                yield (
                    f"step {name!r} passes {types[value]} {value!r} to parameter {inside!r} ({word}); "
                    "text goes into a string parameter only, json into any other"
                )


def _reserved_problem(name: str, value: str, box: bool, erred: set[str]) -> str | None:
    """Say why a node may not read a reserved value, or return ``None``."""
    if box:
        return f"box {name!r} reads {value!r}; a box reads neither 'error' nor 'run', the system inside it does"
    if value == "error" and name not in erred:
        return f"{name!r} reads 'error', which only a node an on error edge reaches can read"
    return None


def _predicate_problems(
    mas: MAS,
    predicates: dict[Edge, Predicate],
    types: dict[str, ValueType],
    writers: dict[str, list[str]],
    before: dict[str, set[str]],
    erred: set[str],
) -> Iterator[str]:
    """Say which predicates read values, fields or cases that do not exist, or call callables wrongly."""
    registered = mas._callables()
    for edge, predicate in predicates.items():
        where = f"edge {edge.label}"
        for path in sorted(predicate.paths()):
            root, fields = path[0], ".".join(path[1:])
            if root in RESERVED_READS:
                if root == "error" and edge.source not in erred:
                    yield f"{where} decides on 'error', which only a node an on error edge reaches can read"
                continue
            if root not in mas.inputs and not {edge.source, *before[edge.source]}.intersection(writers.get(root, ())):
                yield f"{where} decides on {root!r}, which is not readable at {edge.source!r}"
                continue
            if not fields:
                continue
            if types.get(root) != JSON:
                yield f"{where} reads the field {fields!r} of {root!r}, which is not json"
                continue
            for writer in writers.get(root, ()):
                agent = mas.agents.get(writer)
                if agent is None or agent.schema is None:
                    continue
                form = compile_form(agent.schema)
                if fields not in form.paths():
                    yield (
                        f"{where} reads {root}.{fields}, but {writer!r}'s answer form has no field {fields!r} "
                        f"(it has {list(form.paths())})"
                    )
                    continue
                choices = form.choices(fields) if edge.kind is EdgeKind.SWITCH else None
                if choices is not None and edge.case not in choices:
                    yield f"{where}: case {edge.case!r} is not one of {list(choices)}, the choices of {root}.{fields}"
        for function in sorted({name for name, _ in predicate.calls()} - set(registered)):
            yield f"{where} calls {function!r}, which is no registered callable"
        for function, count in sorted(predicate.calls()):
            if function not in registered:
                continue
            takes = list(registered[function][0].reads)
            if count != len(takes):
                yield (
                    f"{where} calls {function!r} with {count} argument{'s' * (count != 1)}, "
                    f"but it takes {len(takes)} ({', '.join(takes) or 'none'})"
                )


def _writer_count_problems(mas: MAS, writers: dict[str, list[str]], routes: RouteGraph | None) -> Iterator[str]:
    """Say which values have several writers that are neither alternatives nor merged."""
    for value, names in writers.items():
        count = len(names) + (value in mas.inputs)
        if count < 2 or mas.merges.get(value) == MERGE_LAST:
            continue
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        if routes is not None and value not in mas.inputs and all(routes.exclusive(a, b) for a, b in pairs):
            continue
        sources = (["the input"] if value in mas.inputs else []) + [repr(n) for n in names]
        yield (
            f"value {value!r} has {count} writers ({', '.join(sources)}); give it one writer, make the writers "
            f"alternatives, or declare state: {{{value}: {{merge: last}}}}"
        )


def _setting_problems(mas: MAS, nodes: dict[str, Agent | Step | Box], members: set[str]) -> Iterator[str]:
    """Say what is wrong with the ``for each``, ``retry`` and shared prompt settings.

    A chain node's retry is its chain's own check's question.
    """
    yield from (f"for each names {name!r}, which is no node" for name in mas.each if name not in nodes)
    for name, rules in mas.retry.items():
        node = nodes.get(name)
        if node is None:
            yield f"retry names {name!r}, which is no node"
        elif isinstance(node, Box):
            yield f"box {name!r} takes no retry; retry inside its system"
        elif name not in members:
            for kind, attempts in rules.items():
                count: Any = attempts
                if kind not in RETRYABLE:
                    yield f"{name!r} retries on {kind!r}; a node retries on {sorted(RETRYABLE)} only"
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    yield f"{name!r} retries on {kind!r} {count!r} times; give a positive integer"
    sharers: dict[str, list[str]] = {}
    for name, key in mas.prompt_names.items():
        if name not in mas.agents:
            yield f"prompt names {name!r}, which is no agent"
        elif key in mas.agents and key != name:
            yield f"agent {name!r} takes the prompt {key!r}, another agent's name; name the shared prompt apart"
        else:
            sharers.setdefault(key, []).append(name)
    for key, names in sharers.items():
        if len({mas.agents[name].prompt for name in names}) > 1:
            yield f"agents {names} share the prompt {key!r} but hold different texts; give them one text"


def _outside_reads(shape: ChainShape, reads: _Reads, writes: _Writes) -> list[str]:
    """Return the values a chain's nodes read from outside it, in order; the item, its writes and reserved reads not."""
    written = {value for name in shape.nodes for value in writes[name]}
    skip = {*written, shape.each.item, *RESERVED_READS}
    return list(dict.fromkeys(value for name in shape.nodes for value in reads[name] if value not in skip))


def _chains(
    mas: MAS,
    shapes: list[ChainShape],
    nodes: dict[str, Agent | Step | Box],
    io: tuple[_Reads, _Writes],
    types: dict[str, ValueType],
) -> tuple[dict[str, Chain], list[str]]:
    """Check each chain at its edges and build the system one item runs, whose own check judges the inside."""
    reads, writes = io
    chains: dict[str, Chain] = {}
    problems: list[str] = []
    for shape in shapes:
        each, where = shape.each, f"chain {shape.head!r}"
        root = types.get(each.root)
        listed = root.item if root is not None and not each.fields else None
        item = JSON if listed is None else listed
        found = []
        if root is not None and listed is None and root != JSON:
            found.append(f"for each {each} runs over {root} {each.root!r}; a chain runs over json or a list")
        if each.key is not None and item != JSON:
            found.append(f"key {each.key!r} needs json items, not {item}")
        found += [f"the item {each.item!r} {problem}" for problem in _value_problems(each.item, item)]
        if each.item in types:
            found.append(f"the item {each.item!r} is an input or a written value too; give it a name of its own")
        found += [
            f"{other!r} writes {value!r}, which the chain writes; a value written in a chain has no writer outside it"
            for name in shape.nodes
            for value in writes[name]
            for other in nodes
            if other not in shape.nodes and value in writes[other]
        ]
        found += [
            f"box {name!r} holds a for each chain; chains do not nest"
            for name in shape.nodes
            if isinstance(box := nodes[name], Box) and _holds_chain(box.system)
        ]
        if not shape.exits and not mas.outputs:
            found.append("it feeds the end, which needs declared outputs")
        inputs = {each.item: item, **{v: types[v] for v in _outside_reads(shape, reads, writes) if v in types}}
        try:
            system = mas._chain_system(shape, inputs, {v: t for name in shape.nodes for v, t in writes[name].items()})
        except _CheckError as exc:
            found += exc.problems
        except ValueError as exc:
            found.append(str(exc))
        problems += [f"{where}: {problem}" for problem in found]
        if not found:
            info = system._checked()
            written = {value: info.types[value] for value in info.writers}
            chains[shape.head] = Chain(shape.head, shape.nodes, each, inputs, written, system)
    return chains, problems


def check_system(mas: MAS) -> SystemInfo:
    """Check a version 2 system and return what the engine needs to know.

    Raises:
        ValueError: Listing every problem found, one per line.
    """
    nodes: dict[str, Agent | Step | Box] = {**mas.agents, **mas.steps, **mas.boxes}
    problems = [f"node {name!r}: names starting with '__' belong to xmas" for name in nodes if name.startswith("__")]
    problems += [f"frozen names {name!r}, which is no node" for name in sorted(mas.frozen - set(nodes))]
    for name, vtype in mas.inputs.items():
        problems += [f"input {name!r} {problem}" for problem in _value_problems(name, vtype)]
    reads, writes, io_problems = _node_io(mas, nodes)
    shapes, shape_problems = find_chains(list(nodes), mas.edges, {n: f for n, f in mas.each.items() if n in nodes})
    members = {name for shape in shapes for name in shape.nodes}
    problems += [problem for name, found in io_problems.items() if name not in members for problem in found]
    problems += _setting_problems(mas, nodes, members)
    if shape_problems:
        raise _CheckError(problems + shape_problems)

    # The flow: every chain collapsed into its head, which writes the chain's values as lists.
    heads = {shape.head: shape for shape in shapes}
    flow_nodes = [name for name in nodes if name not in members or name in heads]
    flow_writes = {
        name: {v: ValueType("list", t) for m in heads[name].nodes for v, t in writes[m].items()}
        if name in heads
        else writes[name]
        for name in flow_nodes
    }
    types, writers, type_problems = _writer_types(mas, flow_writes, list(heads))
    chains, chain_problems = _chains(mas, shapes, nodes, (reads, writes), types)
    flow_reads = {name: reads[name] for name in flow_nodes}
    for head, shape in heads.items():
        outside = [v for v in _outside_reads(shape, reads, writes) if v in types]
        flow_reads[head] = dict.fromkeys([*outside, shape.each.root])
    edges = collapse(shapes, mas.edges)
    edge_problems, predicates, buildable = _edge_problems(edges, {*flow_nodes, *SENTINELS})
    problems += type_problems + chain_problems + edge_problems

    flow: nx.DiGraph[str] | None = None
    routes: RouteGraph | None = None
    joins: dict[str, Join] = {}
    erred = {edge.target for edge in edges if edge.kind is EdgeKind.ERROR}
    typed = {**RESERVED_READS, **types}
    if buildable:
        flow, routes, joins, topology_problems = _topology(flow_nodes, edges, bool(mas.outputs))
        problems += topology_problems
        if routes is not None:
            before = {node: ancestors(flow, node) for node in flow}
            problems += _read_problems(mas, flow_reads, typed, writers, routes, before, erred)
            problems += _predicate_problems(mas, predicates, typed, writers, before, erred)
    problems += _writer_count_problems(mas, writers, routes)
    for name, vtype in mas.outputs.items():
        if name not in types:
            problems.append(f"output {name!r} is neither an input nor written by any node")
        elif types[name] != vtype:
            problems.append(f"output {name!r} is declared {vtype} but is {types[name]}")

    if problems or flow is None:  # an edge list that makes no graph is always a problem of the list
        raise _CheckError(problems)
    read = {v for needed in flow_reads.values() for v in needed} | {
        path[0] for predicate in predicates.values() for path in predicate.paths()
    }
    types.update((value, vtype) for value, vtype in RESERVED_READS.items() if value in read)
    graph = _graph(list(nodes), mas.edges) if shapes else flow
    writers_of = {value: tuple(names) for value, names in writers.items()}
    return SystemInfo(types, writers_of, joins, graph, predicates, chains, edges, flow)
