"""Nodes that are not agents: a :class:`Step` runs a callable when the graph reaches it.

A step lives in ``MAS.steps``, never in ``MAS.agents``, so prompts, encodings and
prompt optimizers see agents only. Its maps rename between the callable and the
graph: every map is ``{name inside: name in the graph}``. Boxes, which run a
whole system as one node, join this module in phase 2b.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_STEP_KEYS = ("name", "uses", "reads", "writes")


@dataclass(frozen=True)
class Step:
    """A node that runs a registered callable (a tool or a step function) without a model.

    Attributes:
        name: Unique node name in the MAS.
        uses: Name of the callable in the MAS's callables registry (its ``tools=``).
        reads: ``{parameter: value in the graph}`` for parameters whose value has
            another name; a parameter not listed reads the value of its own name.
        writes: ``{callable write: value in the graph}`` for writes stored under
            another name; a write not listed keeps its own name.

    Raises:
        ValueError: If ``name`` or ``uses`` is empty, or a map holds a name that
            is not a non-empty string.
    """

    name: str
    uses: str
    reads: Mapping[str, str] = field(default_factory=dict)
    writes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the names and keep plain copies of the maps."""
        if not self.name:
            raise ValueError("A step needs a name.")
        if not self.uses:
            raise ValueError(f"Step {self.name!r} needs the name of the callable it uses.")
        for label in ("reads", "writes"):
            mapping = getattr(self, label)
            if not isinstance(mapping, Mapping) or not all(
                isinstance(k, str) and k and isinstance(v, str) and v for k, v in mapping.items()
            ):
                raise ValueError(f"Step {self.name!r}: {label} must map names to names, got {mapping!r}.")
            object.__setattr__(self, label, dict(mapping))

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form; ``reads`` and ``writes`` only when not empty."""
        data: dict[str, Any] = {"name": self.name, "uses": self.uses}
        if self.reads:
            data["reads"] = dict(self.reads)
        if self.writes:
            data["writes"] = dict(self.writes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Rebuild a step from :meth:`to_dict` output or a spec entry.

        Raises:
            ValueError: If a required key is missing or an unknown key is present.
        """
        unknown = set(data) - set(_STEP_KEYS)
        if unknown:
            raise ValueError(f"Unknown step keys: {sorted(unknown)}.")
        missing = {"name", "uses"} - set(data)
        if missing:
            raise ValueError(f"Step is missing required keys: {sorted(missing)}.")
        return cls(data["name"], data["uses"], data.get("reads") or {}, data.get("writes") or {})
