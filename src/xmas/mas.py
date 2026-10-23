"""The MAS: a multi-agent system that is also a promptolution predictor."""

from __future__ import annotations

import copy as _copy
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import yaml
from promptolution.llms.base_llm import BaseLLM
from promptolution.predictors.base_predictor import BasePredictor

from xmas import _engine
from xmas.agent import DEFAULT_INPUT, Agent
from xmas.budget import Budget
from xmas.check import MERGE_LAST, SystemInfo, callable_writes, check_system, step_maps
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.graph import (
    END,
    SENTINELS,
    START,
    agent_names,
    back_edges,
    build_graph,
    join_nodes,
    terminal_agent,
    validate_graph,
)
from xmas.nodes import Step
from xmas.prompts import decode_prompts, encode_prompts
from xmas.spec import (
    SPEC_VERSION,
    SPEC_VERSION_2,
    AgentSpec,
    AgentSpecV2,
    CallableSpec,
    MASSpec,
    MASSpecV2,
    StepSpec,
    validate_spec,
    validate_spec_v2,
)
from xmas.tools import Tool, ToolIndex, ToolSource, index_tools, select_tools
from xmas.trace import Trace
from xmas.values import TEXT, ValueType, parse_type

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from xmas._engine import CompiledGraph

_Structure = tuple[object, ...]

_QUERY_ONLY = {DEFAULT_INPUT: TEXT}


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
    callable without a model), typed values, answer forms and declared inputs,
    and is checked by :func:`xmas.check.check_system` when it is built. A
    version 1 system behaves, serializes and compiles exactly as before.

    Attributes:
        agents: Agents keyed by name. Mutable, so ``mas.agents["solver"].prompt``
            is a supported edit.
        steps: Steps keyed by name (version 2); empty in version 1.
        edges: The declared edges, in order, without the implicit wiring from
            :data:`~xmas.graph.START` and to :data:`~xmas.graph.END`.
        inputs: The values a run starts with, with their types; ``{query: text}``
            unless a version 2 system declares others.
        merges: ``{value: rule}`` for values several nodes may write (version 2;
            the rule is ``"last"``).
        version: 1 or 2.
        info: What the version 2 check learned (types, writers, joins); ``None``
            in version 1.
        graph: The topology as a ``networkx.DiGraph`` over node names plus the
            :data:`~xmas.graph.START` and :data:`~xmas.graph.END` sentinels.
        llm: The default model, used by any agent without its own binding.
        llms: Additional models by name, which agents bind to via ``Agent.model``.
        default_model: Name recorded for the default model in a serialized spec.
        budget: Limits for each run, or ``None`` for none. Runtime state, never
            part of the spec.
    """

    agents: dict[str, Agent]
    steps: dict[str, Step]
    edges: tuple[Edge, ...]
    inputs: dict[str, ValueType]
    merges: dict[str, str]
    version: int
    info: SystemInfo | None
    graph: nx.DiGraph[str]
    llms: dict[str, BaseLLM]
    max_steps: int
    default_model: str | None
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
        budget: Budget | None = None,
        *,
        steps: Sequence[Step] = (),
        inputs: Mapping[str, str | ValueType] | None = None,
        state: Mapping[str, Mapping[str, str]] | None = None,
        version: int | None = None,
    ) -> None:
        """Initialize a MAS.

        Args:
            agents: The agents, in any order. Names must be unique.
            edges: Directed ``(source, target)`` pairs between node names, or
                :class:`~xmas.edges.Edge` objects. Nodes left unconnected are
                wired from the source sentinel and to the terminal sentinel
                automatically.
            llm: Default model for agents without their own binding.
            llms: Models by name, for per-agent bindings via ``Agent.model``.
            max_steps: Cap on executed agent steps per call, so a cyclic topology
                cannot run forever.
            default_model: Name to record for the default model when serializing.
                Defaults to the model id ``llm`` reports.
            tools: Tool objects and tool sources (e.g. ``MCPTools``), mixed, that
                ``Agent.tools`` names and ``Step.uses`` resolve against: the
                callables registry. Runtime state like ``llm``; never part of
                the spec.
            max_tool_rounds: Model turns per agent visit that may request tools,
                before one last turn with the tools withheld.
            budget: Limits for each run: every run, and every item of a batch,
                gets its own meter with these limits. A run that reaches one
                ends with status ``over_budget``. Runtime state like ``llm``;
                never part of the spec.
            steps: Nodes that run a registered callable (version 2).
            inputs: The values a run starts with, ``{name: type}`` (version 2).
                Defaults to ``{"query": "text"}``.
            state: ``{value: {"merge": "last"}}`` for each value several nodes
                may write (version 2).
            version: 1 or 2. Defaults to 2 when steps, conditional edges, a
                non-text agent, ``inputs`` or ``state`` are used, else 1.

        Raises:
            ValueError: If names collide, an edge names an unknown node, the
                topology is not runnable, an agent binds to an unknown model or
                tool, two sources offer the same tool name, or a tool-using
                agent's model cannot call tools. A version 2 system raises once,
                listing every problem :func:`~xmas.check.check_system` finds.
        """
        self.edges = tuple(edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges or ())
        uses_v2 = (
            bool(steps)
            or inputs is not None
            or state is not None
            or any(edge.kind is not EdgeKind.PLAIN for edge in self.edges)
            or any(agent.output_type != "text" for agent in agents)
        )
        self.version = version if version is not None else (SPEC_VERSION_2 if uses_v2 else SPEC_VERSION)
        if self.version not in (SPEC_VERSION, SPEC_VERSION_2):
            raise ValueError(f"version must be 1 or 2, not {self.version!r}.")
        if self.version == SPEC_VERSION and uses_v2:
            raise ValueError("Steps, inputs, state, conditional edges and typed writes need version 2.")

        if not agents and (self.version == SPEC_VERSION or not steps):
            raise ValueError("A MAS needs at least one agent." if not steps else "A MAS needs at least one node.")

        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agent names must be unique.")
        self.steps = {step.name: step for step in steps}
        names = [*self.agents, *(step.name for step in steps)]
        if len(set(names)) != len(names):
            raise ValueError(f"Node names must be unique: {sorted({n for n in names if names.count(n) > 1})}.")
        self.inputs = (
            {name: kind if isinstance(kind, ValueType) else parse_type(kind) for name, kind in inputs.items()}
            if inputs is not None
            else dict(_QUERY_ONLY)
        )
        self.merges = {}
        for value, entry in (state or {}).items():
            if dict(entry) != {"merge": MERGE_LAST}:
                raise ValueError(f"state[{value!r}] must be {{'merge': 'last'}}, got {entry!r}.")
            self.merges[value] = MERGE_LAST
        self.info = None

        pairs = [(edge.source, edge.target) for edge in self.edges]
        if self.version == SPEC_VERSION:
            self.graph = build_graph([agent.name for agent in agents], pairs)
            validate_graph(self.graph)
            # Force the terminal check now. A MAS with two terminal agents is only
            # broken at the moment it produces an output, and discovering that after
            # a batch of model calls is an expensive way to learn it.
            terminal_agent(self.graph)

        self.llms = dict(llms or {})
        self.max_steps = max_steps
        self.budget = budget
        self.default_model = default_model if default_model is not None else getattr(llm, "model_id", None)

        for agent in agents:
            if agent.model is not None and agent.model not in self.llms:
                raise ValueError(
                    f"Agent {agent.name!r} binds to model {agent.model!r}, which is not in llms={sorted(self.llms)}."
                )

        self._compiled: CompiledGraph | None = None
        self._structure: _Structure | None = None
        self._last_trace: Trace | None = None
        self._last_traces: tuple[Trace, ...] = ()

        super().__init__(llm=llm)  # type: ignore[arg-type]

        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must be >= 0.")
        self.tools = tuple(tools)
        self.max_tool_rounds = max_tool_rounds
        self._tool_index: tuple[tuple[ToolSource | Tool, ...], ToolIndex] | None = None
        if self.version == SPEC_VERSION_2:
            # The check builds and checks the topology itself, so that a bad edge
            # is one problem in the list rather than the first and only error.
            self.info = check_system(self)
            self.graph = build_graph(names, pairs)
            return
        for agent in agents:  # fail at construction, not mid-run; tool-free systems never touch their sources
            if agent.tools:
                self.resolve_tools(agent)

    def _callables(self) -> ToolIndex:
        """Return the callables registry, indexed once per ``tools`` tuple."""
        if self._tool_index is None or self._tool_index[0] is not self.tools:
            self._tool_index = (self.tools, index_tools(self.tools))
        return self._tool_index[1]

    def resolve_tools(self, agent: Agent) -> ToolIndex:
        """Return the tools an agent may call, keyed by name.

        Args:
            agent: The agent whose tools are wanted.

        Returns:
            Each of its tools with the source that runs it.

        Raises:
            ValueError: If a name is unknown, or the agent's model cannot call tools.
        """
        llm = self.llms.get(agent.model) if agent.model is not None else self.llm
        return select_tools(self._callables(), agent, llm)

    def resolve_callable(self, step: Step) -> tuple[Tool, ToolSource]:
        """Return the callable a step uses, with the source that runs it.

        Args:
            step: The step.

        Returns:
            ``(tool, source)`` from the same registry ``Agent.tools`` resolve in.

        Raises:
            ValueError: If no tool or tool source offers ``step.uses``.
        """
        index = self._callables()
        if step.uses not in index:
            raise ValueError(f"Step {step.name!r} uses unknown callable {step.uses!r}; available: {sorted(index)}.")
        return index[step.uses]

    @property
    def nodes(self) -> dict[str, Agent | Step]:
        """Every node, agents and steps, keyed by name in topological order."""
        return {name: self.agents.get(name) or self.steps[name] for name in agent_names(self.graph)}

    # ----------------------------------------------------------------- models

    def resolve_llm(self, agent: Agent) -> BaseLLM:
        """Return the model that serves an agent.

        Args:
            agent: The agent whose model is wanted.

        Returns:
            The agent's bound model, or the MAS default when it has none.

        Raises:
            ValueError: If neither a binding nor a default is available.
        """
        if agent.model is not None:
            return self.llms[agent.model]
        if self.llm is None:
            raise ValueError(f"Agent {agent.name!r} has no model binding and the MAS has no default llm.")
        return self.llm

    # ---------------------------------------------------------------- running

    def __call__(self, query: str) -> str:
        """Run the MAS on one input and return the terminal agent's output.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The terminal agent's output.

        Raises:
            RuntimeError: If the run failed.
        """
        trace = self.run(query)
        self._last_trace = trace
        if trace.error is not None:
            raise RuntimeError(f"MAS run failed: {trace.error}")
        return trace.output

    def _check_runs_on_query(self) -> None:
        """Refuse to run a system whose inputs are not the one ``query`` text.

        Raises:
            ValueError: If the system declares other inputs.
        """
        if self.inputs != _QUERY_ONLY:
            declared = {name: str(vtype) for name, vtype in self.inputs.items()}
            raise ValueError(
                f"This system's inputs are {declared}, not {{'query': 'text'}}, so run(query) cannot fill them; "
                "it runs as a box, or with run_state in phase 3."
            )

    def run(self, query: str) -> Trace:
        """Run the MAS on one input and return the full record of the run.

        This is :meth:`__call__` without the shared-state convenience: the trace
        is returned rather than stashed, which is what concurrent callers want.

        Args:
            query: The input, bound to the ``query`` state variable.

        Returns:
            The trace, whose ``output`` field is what :meth:`__call__` returns.

        Raises:
            ValueError: If the system's inputs are not ``{query: text}``.
        """
        self._check_runs_on_query()
        return _engine.run_one(self, query, self.prompts())

    def run_batch(self, queries: list[str]) -> list[Trace]:
        """Run the MAS over several inputs, concurrently where the LLM allows.

        Args:
            queries: The inputs.

        Returns:
            One trace per query, in the order given.

        Raises:
            ValueError: If the system's inputs are not ``{query: text}``.
        """
        self._check_runs_on_query()
        prompts = self.prompts()
        traces = _engine.run_many(self, queries, [prompts] * len(queries))
        self._last_traces = tuple(traces)
        return traces

    def prompts(self) -> dict[str, str]:
        """Return the current prompt of every agent, keyed by name."""
        return {name: agent.prompt for name, agent in self.agents.items()}

    @property
    def last_trace(self) -> Trace | None:
        """Trace of the most recent :meth:`__call__`, or ``None`` before any.

        This is the single-call ergonomic. Batched execution records into
        :attr:`last_traces` instead; see there for the concurrency contract.
        """
        return self._last_trace

    @property
    def last_traces(self) -> tuple[Trace, ...]:
        """Traces of the most recent :meth:`predict` or :meth:`run_batch` call.

        Assigned once, at the end of the batch, rather than appended to as the
        batch progresses, so a reader either sees the previous batch or the whole
        new one. This is the seam an attributor reads after a task evaluation.
        One MAS instance still serves one batch at a time; run concurrent batches
        on separate instances, or use :meth:`run`, which shares no state at all.
        """
        return self._last_traces

    # ------------------------------------------------- promptolution predictor

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

        Args:
            prompts: Encoded prompt sets, one per input.
            xs: The inputs.
            system_prompts: Accepted for interface compatibility. Agent prompts
                already carry their instructions, so this is not applied.

        Returns:
            A ``(predictions, sequences)`` pair, matching ``BasePredictor``. A
            failed rollout's sequence is empty.
        """
        if not xs:
            return [], []
        self._check_runs_on_query()

        encoded = [prompts] * len(xs) if isinstance(prompts, str) else list(prompts)
        if len(encoded) != len(xs):
            raise ValueError(f"Got {len(encoded)} prompt sets for {len(xs)} inputs; they must be parallel.")

        names = list(self.agents)
        decoded = [decode_prompts(item, names) for item in encoded]
        traces = _engine.run_many(self, list(xs), decoded)
        self._last_traces = tuple(traces)

        preds = self._extract_preds([trace.output for trace in traces])
        # A failed rollout keeps its finished agents' records, but promptolution
        # gets an empty sequence for it, as it did before records were kept.
        seqs = [
            "" if trace.failed else "\n".join(f"{r.rendered_prompt}\n{r.raw_response}" for r in trace.records)
            for trace in traces
        ]
        return preds, seqs

    def _extract_preds(self, preds: list[str]) -> list[str]:
        """Return predictions unchanged; a MAS's terminal agent already extracts.

        Args:
            preds: Raw terminal-agent outputs.

        Returns:
            The same list.
        """
        return list(preds)

    def encode(self) -> str:
        """Return this MAS's prompts as a single promptolution prompt string."""
        return encode_prompts(self.prompts())

    # ------------------------------------------------------------- components

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

        Args:
            prompts: Replacement prompts, keyed by agent name.
            edges: A replacement topology. Replaces the edge set entirely.
            models: Replacement model bindings, keyed by agent name. A ``None``
                value unbinds an agent back to the MAS default.

        Returns:
            The derived MAS.

        Raises:
            ValueError: If a key names an unknown agent, or the result is not
                runnable.
        """
        for label, mapping in (("prompts", prompts), ("models", models)):
            unknown = set(mapping or {}) - set(self.agents)
            if unknown:
                raise ValueError(f"copy({label}=...) names unknown agents: {sorted(unknown)}.")

        new_agents = []
        for name, agent in self.agents.items():
            overrides: dict[str, Any] = {}
            if prompts is not None and name in prompts:
                overrides["prompt"] = prompts[name]
            if models is not None and name in models:
                overrides["model"] = models[name]
            new_agents.append(agent.copy(**overrides) if overrides else agent.copy())

        if edges is None:
            edges = list(self.graph.edges) if self.version == SPEC_VERSION else list(self.edges)
        return MAS(
            agents=new_agents,
            edges=edges,
            llm=self.llm,
            llms=self.llms,
            max_steps=self.max_steps,
            default_model=self.default_model,
            tools=self.tools,
            max_tool_rounds=self.max_tool_rounds,
            budget=self.budget,
            **self._v2_kwargs(),
        )

    def _v2_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments that rebuild this system's version 2 fields.

        The one place they are forwarded from: :meth:`copy`, :meth:`from_dict`
        and every optimizer that constructs a ``MAS(...)`` from an existing one
        pass ``**mas._v2_kwargs()``, so a field added here reaches all of them.

        Returns:
            ``steps``, ``inputs``, ``state`` and ``version``.
        """
        v2 = self.version == SPEC_VERSION_2
        return {
            "steps": tuple(self.steps.values()),
            "inputs": dict(self.inputs) if v2 else None,
            "state": {value: {"merge": rule} for value, rule in self.merges.items()} if v2 else None,
            "version": self.version,
        }

    # ---------------------------------------------------------- serialization

    def to_dict(self) -> MASSpec | MASSpecV2:
        """Return the plain-data spec for this MAS.

        A version 2 spec is canonical: sections in :data:`~xmas.spec.V2_KEYS`
        order, empty or default sections left out, edges as
        :func:`~xmas.edges.dump_edges` writes them, prompts in a top-level
        ``prompts`` section, and a ``callables`` entry for every callable a node
        uses.

        Returns:
            Prompts, topology and model bindings, and nothing else. Models are
            recorded by name; the LLM objects, tools and budget are runtime
            state, not spec.
        """
        if self.version == SPEC_VERSION_2:
            return self._to_dict_v2()
        ordered = agent_names(self.graph)
        agents: list[AgentSpec] = [cast("AgentSpec", self.agents[name].to_dict()) for name in ordered]
        return MASSpec(
            version=SPEC_VERSION,
            agents=agents,
            edges=[[source, target] for source, target in self.graph.edges],
            default_model=self.default_model,
        )

    def _to_dict_v2(self) -> MASSpecV2:
        """Return the canonical version 2 spec."""
        spec = MASSpecV2(version=SPEC_VERSION_2)
        if self.inputs != _QUERY_ONLY:
            spec["inputs"] = {name: str(vtype) for name, vtype in self.inputs.items()}
        if self.agents:
            spec["agents"] = [_agent_spec(agent) for agent in self.agents.values()]
        if self.steps:
            spec["steps"] = [cast("StepSpec", step.to_dict()) for step in self.steps.values()]
        if self.edges:
            spec["edges"] = dump_edges(self.edges)
        if self.merges:
            spec["state"] = {value: {"merge": rule} for value, rule in self.merges.items()}
        callables = self._callable_specs()
        if callables:
            spec["callables"] = callables
        if self.default_model is not None:
            spec["default_model"] = self.default_model
        if self.agents:
            spec["prompts"] = {name: agent.prompt for name, agent in self.agents.items()}
        return spec

    def _callable_specs(self) -> dict[str, CallableSpec]:
        """Return the ``callables`` section: every callable a node uses, steps' first."""
        index = self._callables()
        used = [step.uses for step in self.steps.values()] + [t for agent in self.agents.values() for t in agent.tools]
        specs: dict[str, CallableSpec] = {}
        for name in dict.fromkeys(used):
            item, source = index[name]
            if item.fn is not None:
                module = getattr(item.fn, "__module__", "?")
                entry: CallableSpec = {"kind": "python", "import": f"{module}:{getattr(item.fn, '__qualname__', name)}"}
            else:
                entry = {"kind": "mcp"}
                url = getattr(source, "url", None)
                if isinstance(url, str):
                    entry["server"] = url
            reads, writes = _type_words(item.reads), _type_words(callable_writes(item))
            if reads:
                entry["reads"] = reads
            entry["writes"] = writes
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

        Args:
            data: A spec produced by :meth:`to_dict`.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.
            tools: Tools and tool sources, resolving the agents' tool names and
                the steps' callables.

        Returns:
            The reconstructed MAS.

        Raises:
            ValueError: If the spec is malformed, binds to an unsupplied model,
                or (version 2) its ``callables`` do not match those supplied.
        """
        if data.get("version", SPEC_VERSION) == SPEC_VERSION_2:
            return cls._from_v2(data, llm=llm, llms=llms, tools=tools)
        spec = validate_spec(data)
        agents = [Agent.from_dict(dict(item)) for item in spec["agents"]]
        edges = [(source, target) for source, target in spec.get("edges", [])]
        return cls(
            agents=agents,
            edges=edges,
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
        spec = validate_spec_v2(data)
        prompts = spec.get("prompts", {})
        agents = []
        for item in spec.get("agents", []):
            ((output, output_type),) = item["writes"].items()
            agents.append(
                Agent(
                    name=item["name"],
                    prompt=prompts[item["name"]],
                    output=output,
                    inputs=item.get("reads"),
                    tools=item.get("tools"),
                    model=item.get("model"),
                    output_type=output_type,
                    schema=item.get("schema"),
                )
            )
        mas = cls(
            agents=agents,
            edges=parse_edges(spec.get("edges", [])),
            llm=llm,
            llms=llms,
            default_model=spec.get("default_model"),
            tools=tools,
            steps=[Step.from_dict(item) for item in spec.get("steps", [])],
            inputs=spec.get("inputs"),
            state=cast("Mapping[str, Mapping[str, str]] | None", spec.get("state")),
            version=SPEC_VERSION_2,
        )
        mas._check_callables(spec.get("callables", {}))
        return mas

    def _check_callables(self, declared: Mapping[str, CallableSpec]) -> None:
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
                want = _type_words(_parsed(cast("Mapping[str, str]", entry.get(key, {}))))
                got = actual[name].get(key, {})
                if want != got:
                    problems.append(f"callable {name!r} {key} {want} in the spec, but the one supplied {key} {got}")
        if problems:
            lines = "\n".join(f"- {problem}" for problem in problems)
            raise ValueError(f"The spec's callables do not match the callables supplied:\n{lines}")

    def to_yaml(self, path: str | Path) -> None:
        """Write the spec to a YAML file.

        Args:
            path: Destination file.

        Raises:
            ValueError: For a version 2 system, which is saved as a ``.mas``
                bundle (phase 5); until then, use :meth:`to_dict`.
        """
        if self.version == SPEC_VERSION_2:
            raise ValueError(
                "A version 2 system is saved as a .mas bundle (name.mas/), which arrives in phase 5; "
                "until then keep its to_dict()."
            )
        Path(path).write_text(
            yaml.safe_dump(dict(self.to_dict()), sort_keys=False, allow_unicode=True),
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
        """Read a MAS spec from a YAML file.

        Args:
            path: Source file.
            llm: Default model to attach.
            llms: Models by name, resolving the spec's bindings.
            tools: Tools and tool sources, resolving the agents' tool names.

        Returns:
            The reconstructed MAS.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, llm=llm, llms=llms, tools=tools)

    # --------------------------------------------------------------- describe

    def describe(self) -> dict[str, Any]:
        """Return the system as plain data, for drawing it and naming it.

        Every node with its kind, model or callable, typed reads and writes;
        every edge, the implicit wiring marked ``"implicit": True``; the join
        kind of each node that waits for several predecessors; and an ``id``,
        the SHA-256 of the canonical JSON of the rest. Prompts are not part of
        it, so editing one keeps the id. A version 1 system describes too: its
        ``<continue>`` back edges are ``"loop"`` edges.

        Returns:
            ``{"version", "inputs", "nodes", "edges", "joins", "groups", "id"}``,
            JSON-dumpable.
        """
        types = self.info.types if self.info is not None else {}

        def typed(names: Iterable[str]) -> dict[str, str]:
            return {name: str(types.get(name, TEXT)) for name in names}

        nodes: list[dict[str, Any]] = []
        for name, node in self.nodes.items():
            if isinstance(node, Agent):
                entry: dict[str, Any] = {
                    "name": name,
                    "kind": "agent",
                    "model": node.model if node.model is not None else self.default_model,
                    "reads": typed(node.inputs),
                    "writes": {node.output: str(parse_type(node.output_type))},
                    "tools": list(node.tools),
                    "frozen": False,
                }
                if node.schema is not None:
                    entry["schema"] = _copy.deepcopy(node.schema)
            else:
                reads, writes = step_maps(node, self.resolve_callable(node)[0])
                entry = {
                    "name": name,
                    "kind": "step",
                    "uses": node.uses,
                    "reads": typed(reads.values()),
                    "writes": typed(writes.values()),
                }
            nodes.append(entry)

        declared = {(edge.source, edge.target): edge for edge in self.edges}
        loops = back_edges(self.graph) if self.version == SPEC_VERSION else set()
        # Wiring from START first, then the declared edges in their order, then wiring to END.
        wiring = [pair for pair in self.graph.edges if pair not in declared]
        ordered = sorted([*declared, *wiring], key=lambda pair: (pair[0] != START) + (pair[1] == END))
        edges: list[dict[str, Any]] = []
        for source, target in ordered:
            item: dict[str, Any] = {"from": source, "to": target, "kind": EdgeKind.PLAIN.value}
            if (source, target) in loops:
                when = f"{_engine.CONTINUE_MARKER} in output"
                item.update(kind="loop", when=when, max_visits=_engine.MAX_NODE_VISITS)
            elif (source, target) in declared:
                spec = declared[(source, target)]
                item["kind"] = spec.kind.value
                fields = {"when": spec.when, "max_visits": spec.max_visits, "switch": spec.switch, "case": spec.case}
                item.update({key: value for key, value in fields.items() if value is not None})
                if spec.errors:
                    item["errors"] = list(spec.errors)
            if (source == START and self.graph.in_degree(target) == 1) or (
                target == END and self.graph.out_degree(source) == 1
            ):
                item["implicit"] = True
            edges.append(item)

        joined = self.info.joins if self.info is not None else sorted(join_nodes(self.graph))
        description: dict[str, Any] = {
            "version": self.version,
            "inputs": {name: str(vtype) for name, vtype in self.inputs.items()},
            "nodes": nodes,
            "edges": edges,
            "joins": dict.fromkeys(joined, "join"),
            "groups": [],
        }
        canonical = json.dumps(description, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        description["id"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return description

    # -------------------------------------------------------------- internals

    @property
    def structure(self) -> _Structure:
        """What the compiled graph depends on, and what makes two MASs swappable.

        Two systems with the same structure differ only in prompts, so one can
        evaluate the other's prompt sets. An optimizer uses this to batch a whole
        population of prompt candidates through a single predictor.

        Prompts are deliberately absent: they travel in the run state, so editing
        one must not throw away a compilation. Anything else about an agent, its
        wiring or its model, changes the graph itself. A version 2 system adds
        its steps, edges, answer types and forms, inputs and merges.
        """
        agents = tuple(
            (agent.name, ",".join(agent.inputs), agent.output, agent.model, ",".join(agent.tools))
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        )
        base = (agents, tuple(sorted(self.graph.edges)))
        if self.version == SPEC_VERSION:
            return base
        steps = tuple(
            (step.name, step.uses, tuple(sorted(step.reads.items())), tuple(sorted(step.writes.items())))
            for step in sorted(self.steps.values(), key=lambda s: s.name)
        )
        forms = tuple(
            (agent.name, str(parse_type(agent.output_type)), json.dumps(agent.schema, sort_keys=True))
            for agent in sorted(self.agents.values(), key=lambda a: a.name)
        )
        return (
            *base,
            steps,
            tuple(edge.as_tuple() for edge in self.edges),
            forms,
            tuple(sorted((name, str(vtype)) for name, vtype in self.inputs.items())),
            tuple(sorted(self.merges.items())),
        )

    @property
    def compiled(self) -> CompiledGraph:
        """The compiled executable graph, rebuilt whenever the structure changes.

        The compiled object's type is an implementation detail and never appears
        in a public signature.
        """
        current = self.structure
        if self._compiled is None or self._structure != current:
            self._compiled = _engine.compile_mas(self)
            self._structure = current
        return self._compiled

    @property
    def terminal(self) -> str:
        """Name of the node whose output is the MAS output."""
        return terminal_agent(self.graph)

    def __repr__(self) -> str:
        """Return an unambiguous representation naming agents and edge count."""
        edges = sum(1 for source, target in self.graph.edges if source not in SENTINELS and target not in SENTINELS)
        return f"MAS(agents={agent_names(self.graph)!r}, edges={edges}, terminal={self.terminal!r})"


def _agent_spec(agent: Agent) -> AgentSpecV2:
    """Return one agent's version 2 entry, defaults left out; its prompt goes to ``prompts``."""
    data = AgentSpecV2(name=agent.name)
    if agent.model is not None:
        data["model"] = agent.model
    if agent.inputs != [DEFAULT_INPUT]:
        data["reads"] = list(agent.inputs)
    data["writes"] = {agent.output: str(parse_type(agent.output_type))}
    if agent.tools:
        data["tools"] = list(agent.tools)
    if agent.schema is not None:
        data["schema"] = _copy.deepcopy(agent.schema)
    return data


def _parsed(words: Mapping[str, str]) -> dict[str, ValueType]:
    """Read ``{name: type word}``."""
    return {name: parse_type(word) for name, word in words.items()}


def _type_words(types: Mapping[str, ValueType]) -> dict[str, str]:
    """Write ``{name: type}`` as type words."""
    return {name: str(vtype) for name, vtype in types.items()}
