"""The MAS: a multi-agent system that is also a promptolution predictor."""

from __future__ import annotations

import copy as _copy
import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import BasePredictor

from xmas import runtime
from xmas.agent import Agent
from xmas.budget import Budget
from xmas.check import MERGE_LAST, SystemInfo, check_system, step_maps
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.graph import (
    END,
    SENTINELS,
    START,
    agent_names,
    back_edges,
    build_graph,
    forward_graph,
    join_nodes,
    terminal_agent,
    validate_graph,
)
from xmas.health import HealthSettings
from xmas.nodes import Box, Step
from xmas.prompts import decode_prompts, encode_prompts
from xmas.spec import validate_spec, validate_spec_v2
from xmas.tools import Tool, ToolIndex, ToolSource, index_tools, select_tools
from xmas.trace import Trace
from xmas.values import TEXT, ValueType, parse_type

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

_QUERY_ONLY = {"query": TEXT}


class MAS(BasePredictor):
    """A graph of agents that maps an input to an output.

    ``MAS`` subclasses promptolution's ``BasePredictor``, so a one-agent MAS is
    not merely analogous to a promptolution predictor, it is one: hand it to a
    promptolution task and its optimizers work on it unchanged. Everything that
    makes it a multi-agent system, the extra nodes and the edges between them, is
    additional structure on top of that contract.

    The parameters an optimizer may change, per-agent prompts, the topology and
    the per-node model bindings, are all reachable as ordinary attributes, and
    :meth:`copy` is how candidates are derived from them.

    A system is version 1 or version 2. Version 2 adds steps (nodes that run a
    callable without a model), boxes (nodes that run a whole system), typed
    values, answer forms, declared inputs and conditional edges, and is checked
    by :func:`xmas.check.check_system` when it is built.

    Attributes:
        agents: Agents keyed by name. Mutable, so ``mas.agents["solver"].prompt``
            is a supported edit.
        steps: Steps keyed by name; empty in version 1.
        boxes: Boxes keyed by name; empty in version 1.
        frozen: Names of the nodes no optimizer may change.
        health: The thresholds of the run's health signals, or ``None`` for
            the defaults.
        edges: The declared edges, in order, without the implicit wiring from
            :data:`~xmas.graph.START` and to :data:`~xmas.graph.END`.
        inputs: The values a run starts with, with their types; ``{query: text}``
            unless a version 2 system declares others.
        merges: ``{value: "last"}`` for values several nodes may write.
        version: 1 or 2.
        info: What the version 2 check learned (types, writers, joins); ``None``
            in version 1.
        graph: The topology as a ``networkx.DiGraph`` over node names plus the
            :data:`~xmas.graph.START` and :data:`~xmas.graph.END` sentinels.
        llm: The default model, used by any agent without its own binding.
        llms: Additional models by name, which agents bind to via ``Agent.model``.
        default_model: Name recorded for the default model in a serialized spec.
        budget: Limits for each run, or ``None`` for none; never part of the spec.
        max_visits: How often one agent of a version 1 cycle may run in a single
            query before its back edges close.
        last_trace: Trace of the most recent :meth:`__call__`, or ``None`` before any.
        last_traces: Traces of the most recent :meth:`predict` or :meth:`run_batch`
            call, assigned once at the end of the batch; this is the seam an
            attributor reads after a task evaluation. One MAS instance serves one
            batch at a time; :meth:`run` shares no state at all.
    """

    agents: dict[str, Agent]
    steps: dict[str, Step]
    boxes: dict[str, Box]
    frozen: frozenset[str]
    health: HealthSettings | None
    edges: tuple[Edge, ...]
    inputs: dict[str, ValueType]
    merges: dict[str, str]
    version: int
    info: SystemInfo | None
    graph: nx.DiGraph[str]
    budget: Budget | None

    def __init__(
        self,
        agents: list[Agent],
        edges: Sequence[tuple[str, str] | Edge] | None = None,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        max_steps: int = 50,
        default_model: str | None = None,
        tools: Sequence[ToolSource | Tool] = (),
        max_tool_rounds: int = 5,
        max_visits: int = 3,
        budget: Budget | None = None,
        *,
        steps: Sequence[Step] = (),
        inputs: Mapping[str, str | ValueType] | None = None,
        state: Mapping[str, Mapping[str, str]] | None = None,
        version: int | None = None,
        boxes: Sequence[Box] = (),
        frozen: Iterable[str] = (),
        health: HealthSettings | None = None,
    ) -> None:
        """Initialize a MAS.

        Args:
            agents: The agents, in any order.
            edges: ``(source, target)`` pairs between node names, or
                :class:`~xmas.edges.Edge` objects. Nodes left unconnected are
                wired from the source sentinel and to the terminal sentinel.
            llm: Default model for agents without their own binding.
            llms: Models by name, for per-agent bindings via ``Agent.model``.
            max_steps: Cap on executed steps per call, so a cyclic topology
                cannot run forever.
            default_model: Name to record for the default model when serializing;
                defaults to the model id ``llm`` reports.
            tools: Tools and tool sources (e.g. ``MCPTools``), mixed: the
                callables registry ``Agent.tools`` and ``Step.uses`` resolve in.
                Runtime state like ``llm``; never part of the spec.
            max_tool_rounds: Model turns per agent visit that may request tools,
                before one last turn with the tools withheld.
            max_visits: How often one agent may run in a single query before its
                version 1 back edges close; a ``while`` edge carries its own.
            budget: Limits for each run, and for each item of a batch; a run
                that reaches one ends with status ``over_budget``.
            steps: Nodes that run a registered callable.
            inputs: The values a run starts with, ``{name: type}``; defaults to
                ``{"query": "text"}``.
            state: ``{value: {"merge": "last"}}`` for each value several nodes
                may write.
            version: 1 or 2. Defaults to 2 when steps, boxes, conditional
                edges, a non-text agent, ``inputs``, ``state`` or ``frozen``
                are used, else 1.
            boxes: Nodes that run a whole version 2 system.
            frozen: Names of nodes no optimizer may change.
            health: Thresholds of the health signals; ``None`` keeps the
                defaults.

        Raises:
            ValueError: If the system is not runnable. A version 2 system
                raises once, listing every problem
                :func:`~xmas.check.check_system` finds.
        """
        self.edges = tuple(edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges or ())
        self.frozen = frozenset(frozen)
        self.health = health
        self.version = self._infer_version(version, agents, steps, boxes, inputs, state)
        if not agents and (self.version == 1 or not (steps or boxes)):
            raise ValueError(
                "A MAS needs at least one node." if self.version == 2 else "A MAS needs at least one agent."
            )
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be >= 0.")
        super().__init__(llm=llm)  # type: ignore[arg-type]

        names = [agent.name for agent in agents] + [step.name for step in steps] + [box.name for box in boxes]
        if len(set(names)) != len(names):
            raise ValueError(f"Node names must be unique: {sorted({n for n in names if names.count(n) > 1})}.")
        self.agents = {agent.name: agent for agent in agents}
        self.steps = {step.name: step for step in steps}
        self.boxes = {box.name: box for box in boxes}
        self.llms = dict(llms or {})
        for agent in agents:
            if agent.model is not None and agent.model not in self.llms:
                raise ValueError(
                    f"Agent {agent.name!r} binds to model {agent.model!r}, which is not in llms={sorted(self.llms)}."
                )
        self.max_steps = max_steps
        self.max_visits = max_visits
        self.budget = budget
        self.default_model = default_model if default_model is not None else getattr(llm, "model_id", None)
        self.tools = tuple(tools)
        self.tool_index: ToolIndex | None = None  # built on first use, so tool-free systems never touch their sources
        self.max_tool_rounds = max_tool_rounds
        self.compilation: tuple[str, Any] | None = None
        self.last_trace: Trace | None = None
        self.last_traces: tuple[Trace, ...] = ()
        if self.version == 2:
            self._build_v2(inputs, state)
        else:
            self._build_v1()

    def _infer_version(
        self,
        version: int | None,
        agents: list[Agent],
        steps: Sequence[Step],
        boxes: Sequence[Box],
        inputs: Mapping[str, str | ValueType] | None,
        state: Mapping[str, Mapping[str, str]] | None,
    ) -> int:
        """Return the version asked for, else 2 when a version 2 feature is used, else 1."""
        uses_v2 = (
            bool(steps or boxes or self.frozen)
            or inputs is not None
            or state is not None
            or any(edge.kind is not EdgeKind.PLAIN for edge in self.edges)
            or any(agent.output_type != "text" for agent in agents)
        )
        inferred = version if version is not None else 2 if uses_v2 else 1
        if inferred not in (1, 2):
            raise ValueError(f"version must be 1 or 2, not {inferred!r}.")
        if inferred == 1 and uses_v2:
            raise ValueError("Steps, boxes, inputs, state, frozen, conditional edges and typed writes need version 2.")
        return inferred

    def _build_v1(self) -> None:
        """Build and check a version 1 topology, and resolve every agent's tools, so it fails now, not mid-run."""
        self.inputs, self.merges, self.info = dict(_QUERY_ONLY), {}, None
        self.graph = build_graph(list(self.agents), [(edge.source, edge.target) for edge in self.edges])
        validate_graph(self.graph)
        # Two terminal agents only break the run once it produces an output, after a batch of model calls.
        terminal_agent(self.graph)
        for agent in self.agents.values():
            if agent.tools:
                self.resolve_tools(agent)

    def _build_v2(
        self, inputs: Mapping[str, str | ValueType] | None, state: Mapping[str, Mapping[str, str]] | None
    ) -> None:
        """Read the inputs and merges, and run :func:`~xmas.check.check_system`, which builds the graph."""
        given = inputs if inputs is not None else _QUERY_ONLY
        self.inputs = {name: kind if isinstance(kind, ValueType) else parse_type(kind) for name, kind in given.items()}
        self.merges = {}
        for value, entry in (state or {}).items():
            if entry != {"merge": MERGE_LAST}:
                raise ValueError(f"state[{value!r}] must be {{'merge': 'last'}}, got {entry!r}.")
            self.merges[value] = MERGE_LAST
        self.info = check_system(self)  # it checks the topology itself, so a bad edge is one problem of the list
        self.graph = self.info.graph

    def _checked(self, refusal: str = "A version 1 system is not checked.") -> SystemInfo:
        """Return what the check learned, which only a version 2 system has; ``refusal`` is the ValueError's text."""
        if self.info is None:
            raise ValueError(refusal)
        return self.info

    def _callables(self) -> ToolIndex:
        """Return the callables registry: every tool and tool source, indexed on first use."""
        if self.tool_index is None:
            self.tool_index = index_tools(self.tools)
        return self.tool_index

    def resolve_tools(self, agent: Agent) -> ToolIndex:
        """Return the tools an agent may call, each with the source that runs it.

        Raises:
            ValueError: If a name is unknown, or the agent's model cannot call tools.
        """
        llm = self.llms.get(agent.model) if agent.model is not None else self.llm
        return select_tools(self._callables(), agent, llm)

    def resolve_callable(self, step: Step) -> tuple[Tool, ToolSource]:
        """Return the callable a step uses, with the source that runs it, from the registry ``Agent.tools`` use."""
        index = self._callables()
        if step.uses not in index:
            raise ValueError(f"Step {step.name!r} uses unknown callable {step.uses!r}; available: {sorted(index)}.")
        return index[step.uses]

    @property
    def nodes(self) -> dict[str, Agent | Step | Box]:
        """Every node, agents, steps and boxes, keyed by name in the topological order of one pass."""
        everything: dict[str, Agent | Step | Box] = {**self.agents, **self.steps, **self.boxes}
        return {name: everything[name] for name in agent_names(forward_graph(self.graph, self.while_edges))}

    @property
    def while_edges(self) -> tuple[tuple[str, str], ...]:
        """The ``(source, target)`` of every ``while`` edge, in declared order: the loops of a version 2 run."""
        return tuple((edge.source, edge.target) for edge in self.edges if edge.kind is EdgeKind.WHILE)

    def resolve_llm(self, agent: Agent) -> BaseLLM:
        """Return the agent's bound model, or the MAS default; ``ValueError`` when there is neither."""
        if agent.model is not None:
            return self.llms[agent.model]
        if self.llm is None:
            raise ValueError(f"Agent {agent.name!r} has no model binding and the MAS has no default llm.")
        return self.llm

    def __call__(self, query: str) -> str:
        """Run the MAS on one input and return its output; a failed run raises ``RuntimeError``."""
        trace = self.run(query)
        self.last_trace = trace
        if trace.error is not None:
            raise RuntimeError(f"MAS run failed: {trace.error}")
        return trace.output

    def _check_runs_on_query(self) -> None:
        """Refuse to run a system whose inputs are not the one ``query`` text."""
        if self.inputs != _QUERY_ONLY:
            declared = {name: str(vtype) for name, vtype in self.inputs.items()}
            raise ValueError(
                f"This system's inputs are {declared}, not {{'query': 'text'}}, so run(query) cannot fill them; "
                "it runs as a box, or with run_state in phase 3."
            )

    def run(self, query: str) -> Trace:
        """Run the MAS on one input and return the trace of the run.

        This is :meth:`__call__` without the shared state: the trace is returned
        rather than stashed, which is what concurrent callers want. A system
        whose inputs are not ``{query: text}`` raises ``ValueError``.
        """
        self._check_runs_on_query()
        return runtime.run_one(self, query, self.prompts())

    def run_batch(self, queries: list[str]) -> list[Trace]:
        """Run the MAS over several inputs, concurrently where the LLM allows; one trace per query, in order."""
        self._check_runs_on_query()
        prompts = self.prompts()
        traces = runtime.run_many(self, queries, [prompts] * len(queries))
        self.last_traces = tuple(traces)
        return traces

    def prompts(self) -> dict[str, str]:
        """Return the current prompt of every agent, keyed by name."""
        return {name: agent.prompt for name, agent in self.agents.items()}

    def predict(
        self,
        prompts: str | list[str],
        xs: list[str],
        system_prompts: str | list[str] | None = None,  # noqa: ARG002
    ) -> tuple[list[str], list[str]]:
        """Evaluate prompt sets against inputs, as promptolution's tasks expect.

        ``prompts`` and ``xs`` are parallel: element ``i`` is the prompt set to
        run against input ``i``. Each prompt set is an encoding produced by
        :func:`xmas.prompts.encode_prompts`, which for a single-agent MAS is the
        bare prompt string, so promptolution's optimizers need no adapter.

        The MAS itself is not mutated; each element runs against its own prompts.

        ``system_prompts`` is accepted for the interface and not applied: agent
        prompts carry their instructions. Returns ``(predictions, sequences)``,
        matching ``BasePredictor``; a failed rollout's sequence is empty.
        """
        if not xs:
            return [], []
        self._check_runs_on_query()

        encoded = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        if len(encoded) != len(xs):
            raise ValueError(f"Got {len(encoded)} prompt sets for {len(xs)} inputs; they must be parallel.")

        names = list(self.agents)
        decoded = [decode_prompts(item, names) for item in encoded]
        traces = runtime.run_many(self, list(xs), decoded)
        self.last_traces = tuple(traces)

        preds = [trace.output for trace in traces]
        # A failed rollout keeps its finished records, but promptolution gets an empty sequence for it.
        seqs = [
            "" if trace.failed else "\n".join(f"{r.rendered_prompt}\n{r.raw_response}" for r in trace.records)
            for trace in traces
        ]
        return preds, seqs

    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; promptolution declares this abstract and forces the name."""
        return list(preds)

    def encode(self) -> str:
        """Return this MAS's prompts as a single promptolution prompt string."""
        return encode_prompts(self.prompts())

    def copy(
        self,
        prompts: dict[str, str] | None = None,
        edges: Sequence[tuple[str, str] | Edge] | None = None,
        models: dict[str, str | None] | None = None,
    ) -> MAS:
        """Derive a new MAS with some components replaced.

        This is how an optimizer proposes a candidate. Models are shared with the
        original by reference, and the budget is carried over; everything else is
        copied, so the original is never touched.

        ``prompts`` and ``models`` are keyed by agent name; a ``None`` model
        unbinds an agent back to the MAS default. ``edges`` replaces the edge
        set entirely. A key naming an unknown agent, or a result that is not
        runnable, raises ``ValueError``.
        """
        prompts = prompts or {}
        models = models or {}
        unknown = (set(prompts) | set(models)) - set(self.agents)
        if unknown:
            raise ValueError(f"copy() names unknown agents: {sorted(unknown)}.")
        new_agents = [
            agent.copy(prompt=prompts.get(name, agent.prompt), model=models.get(name, agent.model))
            for name, agent in self.agents.items()
        ]

        if edges is None:
            edges = list(self.graph.edges) if self.version == 1 else list(self.edges)
        return MAS(
            agents=new_agents,
            edges=edges,
            llm=self.llm,
            llms=self.llms,
            max_steps=self.max_steps,
            default_model=self.default_model,
            tools=self.tools,
            max_tool_rounds=self.max_tool_rounds,
            max_visits=self.max_visits,
            budget=self.budget,
            **self._v2_kwargs(),
        )

    def _v2_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments that rebuild this system's version 2 fields.

        :meth:`copy` and every optimizer that constructs a ``MAS(...)`` from an
        existing one pass ``**mas._v2_kwargs()``, so a field added here reaches
        all of them.
        """
        v2 = self.version == 2
        return {
            "steps": tuple(self.steps.values()),
            "inputs": dict(self.inputs) if v2 else None,
            "state": {value: {"merge": rule} for value, rule in self.merges.items()} if v2 else None,
            "version": self.version,
            "boxes": tuple(self.boxes.values()),
            "frozen": self.frozen,
            "health": self.health,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data spec for this MAS: prompts, topology and model bindings by name.

        The LLM objects, tools and budget are runtime state, not spec. A
        version 2 spec is canonical: sections in :data:`~xmas.spec.V2_KEYS`
        order, empty or default sections left out, edges as
        :func:`~xmas.edges.dump_edges` writes them, prompts in a top-level
        ``prompts`` section, and a ``callables`` entry for every callable a node
        uses.
        """
        if self.version == 1:
            return {
                "version": 1,
                "agents": [self.agents[name].to_dict() for name in agent_names(self.graph)],
                "edges": [[source, target] for source, target in self.graph.edges],
                "default_model": self.default_model,
            }
        sections: dict[str, Any] = {
            "agents": [_agent_spec(agent) for agent in self.agents.values()],
            "steps": [step.to_dict() for step in self.steps.values()],
            "boxes": [box.to_dict() for box in self.boxes.values()],
            "edges": dump_edges(self.edges),
            "state": {value: {"merge": rule} for value, rule in self.merges.items()},
            "frozen": sorted(self.frozen),
            "health": _health_spec(self.health),
            "callables": self._callable_specs(),
            "default_model": self.default_model,
            "prompts": self.prompts(),
        }
        inputs = {} if self.inputs == _QUERY_ONLY else {"inputs": _type_words(self.inputs)}
        return {
            "version": 2,
            **inputs,
            **{key: value for key, value in sections.items() if value not in (None, [], {})},
        }

    def _callable_specs(self) -> dict[str, dict[str, Any]]:
        """Return the ``callables`` section: every callable a node uses, steps' first."""
        index = self._callables()
        used = [step.uses for step in self.steps.values()] + [t for agent in self.agents.values() for t in agent.tools]
        specs: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(used):
            item, source = index[name]
            if item.fn is not None:
                module = getattr(item.fn, "__module__", "?")
                entry: dict[str, Any] = {
                    "kind": "python",
                    "import": f"{module}:{getattr(item.fn, '__qualname__', name)}",
                }
            else:
                entry = {"kind": "mcp"}
                url = getattr(source, "url", None)
                if isinstance(url, str):
                    entry["server"] = url
            reads = _type_words(item.reads)
            if reads:
                entry["reads"] = reads
            entry["writes"] = _type_words(item.writes)
            specs[name] = entry
        return specs

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
    ) -> MAS:
        """Rebuild a MAS from a spec.

        A spec with ``version: 2`` is read as version 2; any other goes through
        :func:`~xmas.spec.validate_spec` as version 1.

        ``llm``, ``llms`` and ``tools`` resolve the spec's model bindings, the
        agents' tool names and the steps' callables.

        Raises:
            ValueError: If the spec is malformed, binds to an unsupplied model,
                or (version 2) its ``callables`` do not match those supplied.
        """
        if data.get("version") == 2:
            return cls._from_v2(data, llm=llm, llms=llms, tools=tools)
        spec = validate_spec(data)
        return cls(
            agents=[Agent.from_dict(dict(item)) for item in spec["agents"]],
            edges=[(source, target) for source, target in spec.get("edges", [])],
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
        )

    @classmethod
    def _from_v2(
        cls,
        data: dict[str, Any],
        llm: BaseLLM | None,
        llms: dict[str, BaseLLM] | None,
        tools: Sequence[ToolSource | Tool],
    ) -> MAS:
        """Rebuild a version 2 MAS and check its ``callables`` against those supplied."""
        spec: dict[str, Any] = dict(validate_spec_v2(data))
        prompts = spec.get("prompts", {})
        agents = []
        for item in spec.get("agents", []):
            ((output, output_type),) = item["writes"].items()
            agents.append(
                Agent(
                    item["name"],
                    prompts[item["name"]],
                    output,
                    item.get("reads"),
                    item.get("tools"),
                    item.get("model"),
                    output_type,
                    item.get("schema"),
                )
            )
        mas = cls(
            agents=agents,
            edges=parse_edges(spec.get("edges", [])),
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
            steps=[Step(**item) for item in spec.get("steps", [])],
            inputs=spec.get("inputs"),
            state=spec.get("state"),
            version=2,
            boxes=[
                Box(item["name"], cls._from_v2(item["system"], llm, llms, tools), item["reads"], item["writes"])
                for item in spec.get("boxes", [])
            ],
            frozen=spec.get("frozen", ()),
            health=_health_settings(spec["health"]) if "health" in spec else None,
        )
        mas._check_callables(spec.get("callables", {}))
        return mas

    def _check_callables(self, declared: Mapping[str, Mapping[str, Any]]) -> None:
        """Check a spec's ``callables`` section against the callables this system resolved.

        Raises:
            ValueError: Listing each callable the spec names that no node uses,
                or whose reads or writes differ from the supplied callable's.
        """
        actual = self._callable_specs()
        problems = []
        for name, entry in declared.items():
            if name not in actual:
                problems.append(f"callables lists {name!r}, which no node uses")
                continue
            for key in ("reads", "writes"):
                want = {value: str(parse_type(word)) for value, word in entry.get(key, {}).items()}
                got = actual[name].get(key, {})
                if want != got:
                    problems.append(f"callable {name!r} {key} {want} in the spec, but the one supplied {key} {got}")
        if problems:
            lines = "\n".join(f"- {problem}" for problem in problems)
            raise ValueError(f"The spec's callables do not match the callables supplied:\n{lines}")

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file.

        Raises:
            ValueError: For a version 2 system: its file format comes with the
                ``.mas`` bundle; :meth:`to_dict` gives its spec.
        """
        if self.version == 2:
            raise ValueError(
                "A version 2 system is not saved as one YAML file: its format comes with the .mas bundle "
                "(name.mas/); to_dict() gives its spec."
            )
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        llm: BaseLLM | None = None,
        llms: dict[str, BaseLLM] | None = None,
        tools: Sequence[ToolSource | Tool] = (),
    ) -> MAS:
        """Read a MAS spec from a YAML file, as :meth:`from_dict` reads it."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, llm=llm, llms=llms, tools=tools)

    def describe(self) -> dict[str, Any]:
        """Return the system as plain data, for drawing it and naming it.

        Every node with its kind, model, callable or inner system (a box's own
        description), typed reads and writes, and whether it is frozen; every
        edge with its condition, the implicit wiring marked ``"implicit":
        True``; ``"merge"`` or ``"join"`` for each node with several incoming
        edges; and an ``id``, the SHA-256 of the canonical JSON of the rest.
        Prompts are not part of it, so editing one keeps the id. A version 1
        system's ``<continue>`` back edges are ``"loop"`` edges.
        """
        declared = {(edge.source, edge.target): edge for edge in self.edges}
        loops = back_edges(self.graph) if self.info is None else set()
        wiring = [pair for pair in self.graph.edges if pair not in declared]
        # Wiring from START first, then the declared edges in their order, then wiring to END.
        ordered = sorted([*declared, *wiring], key=lambda pair: (pair[0] != START) + (pair[1] == END))
        if self.info is not None:
            joins = {name: join.kind for name, join in self.info.joins.items()}
        else:
            joins = dict.fromkeys(sorted(join_nodes(self.graph)), "join")
        description: dict[str, Any] = {
            "version": self.version,
            "inputs": _type_words(self.inputs),
            "nodes": [self._describe_node(name, node) for name, node in self.nodes.items()],
            "edges": [self._describe_edge(pair, declared.get(pair), pair in loops) for pair in ordered],
            "joins": joins,
        }
        canonical = json.dumps(description, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        description["id"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return description

    def _describe_node(self, name: str, node: Agent | Step | Box) -> dict[str, Any]:
        """Return one node of :meth:`describe`."""
        types = self.info.types if self.info is not None else {}

        def typed(names: Iterable[str]) -> dict[str, str]:
            return {name: str(types.get(name, TEXT)) for name in names}

        if isinstance(node, Agent):
            entry: dict[str, Any] = {
                "name": name,
                "kind": "agent",
                "model": node.model if node.model is not None else self.default_model,
                "reads": typed(node.inputs),
                "writes": {node.output: str(parse_type(node.output_type))},
                "tools": list(node.tools),
                "frozen": name in self.frozen,
            }
            if node.schema is not None:
                entry["schema"] = _copy.deepcopy(node.schema)
            return entry
        if isinstance(node, Box):
            entry = {"name": name, "kind": "box", "reads": typed(node.reads.values())}
            entry.update(writes=typed(node.writes.values()), system=node.system.describe())
        else:
            reads, writes = step_maps(node, self.resolve_callable(node)[0])
            entry = {"name": name, "kind": "step", "uses": node.uses, "reads": typed(reads.values())}
            entry["writes"] = typed(writes.values())
        if name in self.frozen:
            entry["frozen"] = True
        return entry

    def _describe_edge(self, pair: tuple[str, str], edge: Edge | None, loop: bool) -> dict[str, Any]:
        """Return one edge of :meth:`describe`: ``edge`` when declared, ``loop`` for a version 1 back edge."""
        source, target = pair
        item: dict[str, Any] = {"from": source, "to": target, "kind": EdgeKind.PLAIN.value}
        if loop:
            item.update(kind="loop", when=f"{runtime.CONTINUE_MARKER} in output", max_visits=self.max_visits)
        elif edge is not None:
            item["kind"] = edge.kind.value
            fields = {"when": edge.when, "max_visits": edge.max_visits, "switch": edge.switch, "case": edge.case}
            item.update({key: value for key, value in fields.items() if value is not None})
            if edge.errors:
                item["errors"] = list(edge.errors)
        if self.info is not None:
            implicit = edge is None
        else:
            implicit = (source == START and self.graph.in_degree(target) == 1) or (
                target == END and self.graph.out_degree(source) == 1
            )
        if implicit:
            item["implicit"] = True
        return item

    @property
    def structure(self) -> str:
        """What the compiled graph depends on, and what makes two MASs swappable.

        Two systems with the same structure differ only in prompts, so one can
        evaluate the other's prompt sets; an optimizer batches a population of
        prompt candidates through one predictor this way. Prompts travel in the
        run state, so they are absent. A version 2 system adds lines for its
        steps, edges, answer types and forms, inputs, merges and boxes (each with
        its inner structure, indented); frozen nodes and health settings change
        no compiled graph and stay out.
        """
        lines = [
            f"{agent.name} {','.join(agent.inputs)} -> {agent.output} model={agent.model} tools={','.join(agent.tools)}"
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        ]
        lines += [f"{source} -> {target}" for source, target in sorted(self.graph.edges)]
        if self.info is None:
            return "\n".join(lines)
        lines += [
            f"step {step.name} uses={step.uses} reads={_canonical(step.reads)} writes={_canonical(step.writes)}"
            for step in sorted(self.steps.values(), key=lambda s: s.name)
        ]
        lines += [f"edge {_canonical(dataclasses.astuple(edge))}" for edge in self.edges]
        lines += [
            f"form {agent.name} {parse_type(agent.output_type)} {_canonical(agent.schema)}"
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        ]
        lines += [f"input {name} {vtype}" for name, vtype in sorted(self.inputs.items())]
        lines += [f"merge {value} {rule}" for value, rule in sorted(self.merges.items())]
        for box in sorted(self.boxes.values(), key=lambda b: b.name):
            lines.append(f"box {box.name} reads={_canonical(box.reads)} writes={_canonical(box.writes)}")
            lines += [f"  {line}" for line in box.system.structure.splitlines()]
        return "\n".join(lines)

    @property
    def compiled(self) -> Any:
        """The compiled executable graph, rebuilt whenever the structure changes; its type is not API."""
        if self.compilation is None or self.compilation[0] != self.structure:
            self.compilation = (self.structure, runtime.compile_mas(self))
        return self.compilation[1]

    @property
    def terminal(self) -> str:
        """Name of the node whose output is the MAS output."""
        return terminal_agent(self.graph)

    def __repr__(self) -> str:
        """Return an unambiguous representation naming agents and edge count."""
        edges = sum(1 for source, target in self.graph.edges if source not in SENTINELS and target not in SENTINELS)
        terminals = sorted(self.graph.predecessors(END))
        ending = f"terminal={terminals[0]!r}" if len(terminals) == 1 else f"terminals={terminals!r}"
        return f"MAS(agents={agent_names(self.graph)!r}, edges={edges}, {ending})"


def _agent_spec(agent: Agent) -> dict[str, Any]:
    """Return one agent's version 2 entry, defaults left out; its prompt goes to ``prompts``."""
    data: dict[str, Any] = {"name": agent.name}
    if agent.model is not None:
        data["model"] = agent.model
    if agent.inputs != ["query"]:
        data["reads"] = list(agent.inputs)
    data["writes"] = {agent.output: str(parse_type(agent.output_type))}
    if agent.tools:
        data["tools"] = list(agent.tools)
    if agent.schema is not None:
        data["schema"] = _copy.deepcopy(agent.schema)
    return data


def _health_spec(settings: HealthSettings | None) -> dict[str, Any]:
    """Return the health settings that differ from the defaults, as a spec writes them."""
    if settings is None:
        return {}
    default = HealthSettings()
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name in ("length_factor", "min_samples", "leak")
        if (value := getattr(settings, name)) != getattr(default, name)
    }


def _health_settings(spec: Mapping[str, Any]) -> HealthSettings:
    """Read a spec's ``health`` section."""
    data: dict[str, Any] = {name: tuple(value) if name == "leak" else value for name, value in spec.items()}
    return HealthSettings(**data)


def _type_words(types: Mapping[str, ValueType]) -> dict[str, str]:
    """Write ``{name: type}`` as type words."""
    return {name: str(vtype) for name, vtype in types.items()}


def _canonical(value: Any) -> str:
    """Write a structure part as canonical JSON, so equal parts give equal lines."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
