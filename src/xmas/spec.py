"""The serialized form of a MAS.

A spec is plain data: dicts, lists, strings and ints, and nothing else. It holds
prompts, topology and model bindings, which per the design principles is the
framework-agnostic artefact an optimization run produces. No LLM object, no
compiled graph and no callable is ever part of it.

A ``version: 2`` spec adds steps, boxes, typed values, answer forms, the edge
forms of :mod:`xmas.edges`, ``frozen`` nodes and ``health`` settings, and keeps
its prompts in a top-level ``prompts`` section.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Names mas.py, runtime.py and graph_updater.py still import; a spec is a plain dict.
SPEC_VERSION = 1
SPEC_VERSION_2 = 2
MASSpec = dict[str, Any]
MASSpecV2 = dict[str, Any]
AgentSpecV2 = dict[str, Any]
HealthSpec = dict[str, Any]
CallableSpec = dict[str, Any]
AgentSpec = Mapping[str, Any]
StepSpec = Mapping[str, Any]
BoxSpec = Mapping[str, Any]

V2_KEYS: dict[str, Any] = {
    "version": int,
    "inputs": dict,
    "agents": list,
    "steps": list,
    "boxes": list,
    "edges": list,
    "state": dict,
    "frozen": list,
    "health": dict,
    "callables": dict,
    "default_model": str,
    "prompts": dict,
}
"""Every top-level key of a version 2 spec with the type of its value, in the order a dump writes them."""

_ENTRIES: dict[str, tuple[str, frozenset[str], dict[str, Any]]] = {
    "agents": (
        "Agent",
        frozenset({"writes"}),
        {"name": str, "model": str, "reads": list, "writes": dict, "tools": list, "schema": dict},
    ),
    "steps": ("Step", frozenset({"uses"}), {"name": str, "uses": str, "reads": dict, "writes": dict}),
    "boxes": (
        "Box",
        frozenset({"reads", "writes", "system"}),
        {"name": str, "reads": dict, "writes": dict, "system": dict},
    ),
}
_CALLABLE = {"kind": str, "import": str, "server": str, "reads": dict, "writes": dict}
_HEALTH = {"length_factor": (int, float), "min_samples": int, "leak": list}
_WORDS = {int: "an integer", (int, float): "a number", str: "a string", list: "a list", dict: "a mapping"}


def validate_spec(data: dict[str, Any]) -> dict[str, Any]:
    """Check the top-level shape of a spec and return it.

    Each agent mapping is checked by :meth:`xmas.Agent.from_dict`.

    Args:
        data: Candidate spec, typically parsed from YAML.

    Returns:
        The same mapping.

    Raises:
        ValueError: If a key is unknown, the version is unsupported, the agents
            are not a non-empty list of mappings, or an edge or the default model
            has the wrong type.
    """
    unknown = set(data) - {"version", "agents", "edges", "default_model"}
    if unknown:
        raise ValueError(f"Unknown spec keys: {sorted(unknown)}.")

    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported spec version {version!r}; this build reads version 1.")

    agents = data.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Spec needs a non-empty 'agents' list.")
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {index} is not a mapping.")

    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("Spec needs 'edges' to be a list.")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(node, str) for node in edge):
            raise ValueError(f"Edge {edge!r} must be a two-element list of strings.")

    default_model = data.get("default_model")
    if default_model is not None and not isinstance(default_model, str):
        raise ValueError("Spec needs 'default_model' to be a string or null.")

    return data


def _check_fields(entry: Any, where: str, fields: dict[str, Any], required: frozenset[str] = frozenset()) -> None:
    """Refuse a mapping with an unknown or missing key or a value of the wrong type."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be a mapping.")
    unknown, missing = sorted(set(entry) - set(fields)), sorted(required - set(entry))
    if unknown or missing:
        problem = f"has unknown keys {unknown}" if unknown else f"is missing the keys {missing}"
        raise ValueError(f"{where} {problem}; it takes {sorted(fields)}.")
    for key, value in entry.items():
        if isinstance(value, bool) or not isinstance(value, fields[key]):
            raise ValueError(f"{where} needs {key!r} to be {_WORDS[fields[key]]}, got {value!r}.")


def validate_spec_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Check the shape of a version 2 spec and return it.

    The shape is the keys and the type of each value. The names inside the maps
    are checked by the constructors, and whether the system makes sense by
    :func:`xmas.check.check_system`, both when the MAS is built.
    """
    _check_fields(data, "Spec", V2_KEYS)
    if data.get("version") != 2:
        raise ValueError(f"A version 2 spec needs 'version: 2', not {data.get('version')!r}.")
    for section, (label, required, fields) in _ENTRIES.items():
        for index, entry in enumerate(data.get(section, [])):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError(f"{label} {index} must be a mapping with a string 'name'.")
            where = f"{label} {entry['name']!r}"
            if section == "agents" and "prompt" in entry:
                raise ValueError(f"{where}: a version 2 prompt sits in the top-level 'prompts' section.")
            _check_fields(entry, where, fields, required)
    for name, entry in data.get("callables", {}).items():
        _check_fields(entry, f"Callable {name!r}", _CALLABLE, frozenset({"kind"}))
        if entry["kind"] not in ("python", "mcp"):
            raise ValueError(f"Callable {name!r} needs 'kind' to be 'python' or 'mcp', not {entry['kind']!r}.")
    _check_fields(data.get("health", {}), "Spec 'health'", _HEALTH)

    prompts = data.get("prompts", {})
    if not all(isinstance(text, str) for text in prompts.values()):
        raise ValueError("Spec 'prompts' must map agent names to prompt text.")
    for agent in data.get("agents", []):
        if len(agent["writes"]) != 1:
            raise ValueError(
                f"Agent {agent['name']!r} needs exactly one write such as {{draft: text}}, got {agent['writes']!r}."
            )
        if agent["name"] not in prompts:
            raise ValueError(f"Agent {agent['name']!r} has no prompt in the 'prompts' section.")
    stray = sorted(set(prompts) - {agent["name"] for agent in data.get("agents", [])})
    if stray:
        raise ValueError(f"Spec 'prompts' names unknown agents: {stray}.")
    return data
