"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.

Two versions exist. A spec without ``version: 2`` is version 1 and goes through
:func:`validate_spec` exactly as before. A ``version: 2`` spec adds steps, boxes,
typed values, answer forms, the edge forms of :mod:`xmas.edges`, ``frozen``
nodes and ``health`` settings, keeps its prompts
in a top-level ``prompts:`` section, and has its shape checked by
:func:`validate_spec_v2`.
"""

from __future__ import annotations

from typing import Any, TypedDict, cast

SPEC_VERSION = 1
"""Schema version written into every version 1 spec, so old files stay readable."""

SPEC_VERSION_2 = 2
"""The version of a spec with steps, typed values and answer forms."""

V2_KEYS = (
    "version",
    "inputs",
    "agents",
    "steps",
    "boxes",
    "edges",
    "state",
    "frozen",
    "health",
    "callables",
    "default_model",
    "prompts",
)
"""Every top-level key of a version 2 spec, in the order a dump writes them."""

_BOX_KEYS = frozenset({"name", "reads", "writes", "system"})
_HEALTH_KEYS = {"length_factor": (int, float), "min_samples": (int,), "leak": (list,)}
_AGENT_V2_KEYS = frozenset({"name", "model", "reads", "writes", "tools", "schema"})
_STEP_KEYS = frozenset({"name", "uses", "reads", "writes"})
_CALLABLE_KEYS = frozenset({"kind", "import", "server", "reads", "writes"})
_CALLABLE_KINDS = ("python", "mcp")
_MERGES = ("last",)


class AgentSpec(TypedDict, total=False):
    """Serialized form of one agent."""

    name: str
    prompt: str
    inputs: list[str]
    output: str
    tools: list[str]
    model: str | None


class MASSpec(TypedDict, total=False):
    """Serialized form of a whole MAS."""

    version: int
    agents: list[AgentSpec]
    edges: list[list[str]]
    default_model: str | None


class AgentSpecV2(TypedDict, total=False):
    """Serialized form of one agent in a version 2 spec; its prompt sits in ``prompts``."""

    name: str
    model: str
    reads: list[str]
    writes: dict[str, str]
    tools: list[str]
    schema: dict[str, Any]


class StepSpec(TypedDict, total=False):
    """Serialized form of one step, its maps renaming ``{inside: in the graph}``."""

    name: str
    uses: str
    reads: dict[str, str]
    writes: dict[str, str]


CallableSpec = TypedDict(
    "CallableSpec",
    {"kind": str, "import": str, "server": str, "reads": dict[str, str], "writes": dict[str, str]},
    total=False,
)
"""Serialized form of one callable: where it lives (information until the ``.mas``
bundle resolves it) and its typed reads and writes, checked against the callable
supplied on load."""


class MergeSpec(TypedDict):
    """How several writes of one value combine."""

    merge: str


class BoxSpec(TypedDict):
    """Serialized form of one box: its maps ``{inside: in the graph}`` and the inner system's spec."""

    name: str
    reads: dict[str, str]
    writes: dict[str, str]
    system: dict[str, Any]


class HealthSpec(TypedDict, total=False):
    """The health settings a spec changes from their defaults."""

    length_factor: float
    min_samples: int
    leak: list[str]


class MASSpecV2(TypedDict, total=False):
    """Serialized form of a version 2 MAS; keys in :data:`V2_KEYS` order, empty ones left out."""

    version: int
    inputs: dict[str, str]
    agents: list[AgentSpecV2]
    steps: list[StepSpec]
    boxes: list[BoxSpec]
    edges: list[str | dict[str, Any]]
    state: dict[str, MergeSpec]
    frozen: list[str]
    health: HealthSpec
    callables: dict[str, CallableSpec]
    default_model: str
    prompts: dict[str, str]


def validate_spec(data: dict[str, Any]) -> MASSpec:
    """Check that a mapping is a well-formed spec and return it typed.

    Args:
        data: Candidate spec, typically parsed from YAML.

    Returns:
        The same mapping, narrowed to :class:`MASSpec`.

    Raises:
        ValueError: If a required key is missing, a key is unknown, the version
            is unsupported, or a value has the wrong type.
    """
    unknown = set(data) - {"version", "agents", "edges", "default_model"}
    if unknown:
        raise ValueError(f"Unknown spec keys: {sorted(unknown)}.")

    version = data.get("version", SPEC_VERSION)
    if version != SPEC_VERSION:
        raise ValueError(f"Unsupported spec version {version!r}; this build reads version {SPEC_VERSION}.")

    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Spec needs a non-empty 'agents' list.")

    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {index} is not a mapping.")
        unknown_agent = set(agent) - {"name", "prompt", "inputs", "output", "tools", "model"}
        if unknown_agent:
            raise ValueError(f"Agent {index} has unknown keys: {sorted(unknown_agent)}.")
        for key in ("name", "prompt", "output"):
            if not isinstance(agent.get(key), str):
                raise ValueError(f"Agent {index} needs a string {key!r}.")
        for key in ("inputs", "tools"):
            value = agent.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Agent {agent['name']!r} needs {key!r} to be a list of strings.")
        model = agent.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"Agent {agent['name']!r} needs 'model' to be a string or null.")

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Spec needs 'edges' to be a list.")
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or not all(isinstance(node, str) for node in edge):
            raise ValueError(f"Edge {edge!r} must be a two-element list of strings.")

    default_model = data.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("Spec needs 'default_model' to be a string or null.")

    return cast(MASSpec, data)


def _names_map(value: Any, where: str) -> dict[str, str]:
    """Return a ``{name: name}`` mapping, refusing anything else."""
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"{where} must map names to names, got {value!r}.")
    return value


def _entries(data: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    """Return a section that is a list of mappings, each with a string ``name``."""
    entries = data.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Spec needs {key!r} to be a list.")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"{label} {index} must be a mapping with a string 'name'.")
    return entries


def _check_agent(agent: dict[str, Any], prompts: dict[str, str]) -> None:
    """Check one version 2 agent entry."""
    where = f"Agent {agent['name']!r}"
    if "prompt" in agent:
        raise ValueError(f"{where}: a version 2 prompt sits in the top-level 'prompts' section.")
    unknown = set(agent) - _AGENT_V2_KEYS
    if unknown:
        raise ValueError(f"{where} has unknown keys: {sorted(unknown)}.")
    writes = _names_map(agent.get("writes"), f"{where} 'writes'")
    if len(writes) != 1:
        raise ValueError(f"{where} needs exactly one write such as {{draft: text}}, got {writes!r}.")
    for key in ("reads", "tools"):
        value = agent.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{where} needs {key!r} to be a list of strings.")
    if not isinstance(agent.get("model", ""), str):
        raise ValueError(f"{where} needs 'model' to be a string.")
    if not isinstance(agent.get("schema", {}), dict):
        raise ValueError(f"{where} needs 'schema' to be a mapping.")
    if agent["name"] not in prompts:
        raise ValueError(f"{where} has no prompt in the 'prompts' section.")


def validate_spec_v2(data: dict[str, Any]) -> MASSpecV2:
    """Check the shape of a version 2 spec and return it typed.

    Only the shape is checked here: the keys, and the type of every value.
    Whether the system makes sense (names, types, dataflow, edges) is
    :func:`xmas.check.check_system`'s question, asked when the MAS is built.

    Args:
        data: Candidate spec with ``version: 2``, typically parsed from YAML.

    Returns:
        The same mapping, narrowed to :class:`MASSpecV2`; a box's inner system
        is checked as a spec of its own when the box is built.

    Raises:
        ValueError: If a key is unknown, a section has the wrong shape, an
            agent has no prompt or not exactly one write, or ``prompts`` names
            an unknown agent.
    """
    unknown = set(data) - set(V2_KEYS)
    if unknown:
        raise ValueError(f"Unknown spec keys: {sorted(unknown)}.")
    if data.get("version") != SPEC_VERSION_2:
        raise ValueError(f"A version 2 spec needs 'version: 2', not {data.get('version')!r}.")

    _names_map(data.get("inputs", {}), "Spec 'inputs'")
    prompts = _names_map(data.get("prompts", {}), "Spec 'prompts'")
    agents = _entries(data, "agents", "Agent")
    for agent in agents:
        _check_agent(agent, prompts)
    stray = sorted(set(prompts) - {agent["name"] for agent in agents})
    if stray:
        raise ValueError(f"Spec 'prompts' names unknown agents: {stray}.")

    steps = _entries(data, "steps", "Step")
    for step in steps:
        where = f"Step {step['name']!r}"
        unknown_step = set(step) - _STEP_KEYS
        if unknown_step:
            raise ValueError(f"{where} has unknown keys: {sorted(unknown_step)}.")
        if not isinstance(step.get("uses"), str):
            raise ValueError(f"{where} needs a string 'uses'.")
        for key in ("reads", "writes"):
            _names_map(step.get(key, {}), f"{where} {key!r}")
    boxes = _entries(data, "boxes", "Box")
    for box in boxes:
        where = f"Box {box['name']!r}"
        if set(box) != _BOX_KEYS:
            raise ValueError(f"{where} needs exactly the keys {sorted(_BOX_KEYS)}, got {sorted(box)}.")
        for key in ("reads", "writes"):
            _names_map(box[key], f"{where} {key!r}")
        if not isinstance(box["system"], dict):
            raise ValueError(f"{where} needs 'system' to be a version 2 spec.")
    if not agents and not steps and not boxes:
        raise ValueError("Spec needs at least one agent, step or box.")

    frozen = data.get("frozen", [])
    if not isinstance(frozen, list) or not all(isinstance(name, str) for name in frozen):
        raise ValueError("Spec needs 'frozen' to be a list of node names.")
    health = data.get("health", {})
    if not isinstance(health, dict):
        raise ValueError("Spec needs 'health' to be a mapping.")
    for key, value in health.items():
        kinds = _HEALTH_KEYS.get(key)
        if kinds is None:
            raise ValueError(f"Spec 'health' has unknown key {key!r}; known: {sorted(_HEALTH_KEYS)}.")
        wrong = isinstance(value, bool) or not isinstance(value, kinds)
        if wrong or (key == "leak" and not all(isinstance(pattern, str) for pattern in value)):
            raise ValueError(f"Spec 'health' {key!r} has the wrong type: {value!r}.")

    if not isinstance(data.get("edges", []), list):
        raise ValueError("Spec needs 'edges' to be a list.")

    state = data.get("state", {})
    if not isinstance(state, dict):
        raise ValueError("Spec needs 'state' to be a mapping.")
    for name, entry in state.items():
        if not isinstance(entry, dict) or set(entry) != {"merge"} or entry["merge"] not in _MERGES:
            raise ValueError(f"State {name!r} must be {{merge: last}}, got {entry!r}.")

    callables = data.get("callables", {})
    if not isinstance(callables, dict):
        raise ValueError("Spec needs 'callables' to be a mapping.")
    for name, entry in callables.items():
        where = f"Callable {name!r}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping.")
        unknown_callable = set(entry) - _CALLABLE_KEYS
        if unknown_callable:
            raise ValueError(f"{where} has unknown keys: {sorted(unknown_callable)}.")
        if entry.get("kind") not in _CALLABLE_KINDS:
            raise ValueError(f"{where} needs 'kind' to be one of {list(_CALLABLE_KINDS)}.")
        for key in ("reads", "writes"):
            _names_map(entry.get(key, {}), f"{where} {key!r}")

    if not isinstance(data.get("default_model", ""), str):
        raise ValueError("Spec needs 'default_model' to be a string.")

    return cast(MASSpecV2, data)
