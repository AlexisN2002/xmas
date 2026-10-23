"""Health: what looked wrong in a finished run, read from its records.

:func:`summarize` runs at the end of every run and never changes it: a flag is
information for the person reading the trace, not a rule. It reads three things:

- per record, a :class:`Flag` for each signal in :data:`SIGNALS`;
- per ``while`` edge, how often the loop was taken and whether its cap was hit;
- per run, each limit the budget sets, with what the run spent of it.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xmas.errors import ErrorKind
from xmas.prompts import extract_placeholders

if TYPE_CHECKING:
    from xmas.budget import Budget, RunMeter
    from xmas.trace import AgentRecord

SIGNALS = ("retried", "bad_format", "error_routed", "empty", "stuck", "unfilled", "leak", "long", "short")
"""The per-record signals, in the order a record's flags are listed."""

DEFAULT_LEAKS = (r"<\|[A-Za-z_]{2,30}\|>", r"</?think>", r"<<<xmas:agent ")
"""Patterns that should never reach an output: chat-template tokens, reasoning tags, xmas's encoding header."""


@dataclass(frozen=True)
class HealthSettings:
    """How sensitive the signals are; a v2 spec sets them under ``health:``.

    Attributes:
        length_factor: An output is ``long`` beyond this many times its node's
            median length, and ``short`` under the median divided by it.
        min_samples: The records a node needs before its median is used.
        leak: Regular expressions flagged as ``leak``, on top of
            :data:`DEFAULT_LEAKS`.

    Raises:
        ValueError: If ``length_factor`` is under 1, ``min_samples`` under 1, or
            a leak pattern does not compile.
    """

    length_factor: float = 3.0
    min_samples: int = 3
    leak: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot be applied."""
        if not self.length_factor >= 1:
            raise ValueError(f"health length_factor must be >= 1, got {self.length_factor!r}")
        if self.min_samples < 1:
            raise ValueError(f"health min_samples must be >= 1, got {self.min_samples!r}")
        for pattern in self.leak:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"health leak pattern {pattern!r} does not compile: {exc}") from exc


@dataclass(frozen=True)
class Flag:
    """One signal on one record.

    Attributes:
        signal: Which signal, one of :data:`SIGNALS`.
        node: The node the record belongs to.
        step: The record's step in the run.
        detail: What was seen, in a short plain sentence.
    """

    signal: str
    node: str
    step: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return {"signal": self.signal, "node": self.node, "step": self.step, "detail": self.detail}


@dataclass(frozen=True)
class Health:
    """What :func:`summarize` found in one run.

    Attributes:
        flags: The per-record flags, in record order.
        loops: Per ``while`` edge taken or capped, keyed ``"source -> target"``:
            ``{"taken": n, "cap_hit": bool}``.
        budget: Per limit the budget sets (``"calls"``, ``"cost_usd"``,
            ``"wall_s"``): ``{"spent": x, "limit": y}``.
    """

    flags: tuple[Flag, ...] = ()
    loops: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form with only the parts that report something; empty for a healthy run."""
        data: dict[str, Any] = {}
        if self.flags:
            data["flags"] = [flag.to_dict() for flag in self.flags]
        if self.loops:
            data["loops"] = {edge: dict(entry) for edge, entry in self.loops.items()}
        if self.budget:
            data["budget"] = {limit: dict(entry) for limit, entry in self.budget.items()}
        return data


def summarize(
    records: Sequence[AgentRecord],
    *,
    budget: Budget | None = None,
    meter: RunMeter | None = None,
    settings: HealthSettings | None = None,
    while_edges: Iterable[tuple[str, str]] | None = None,
) -> Health:
    """Read a run's records for the health signals.

    Args:
        records: The run's records, in run order.
        budget: The run's budget; ``meter.budget`` when ``None``.
        meter: The run's meter, which says what was spent. Without one there is
            no budget entry.
        settings: The thresholds and extra leak patterns; the defaults when
            ``None``.
        while_edges: The system's ``while`` edges as ``(source, target)``. When
            ``None`` they are read from the records: a route to a node that first
            ran before the deciding node first ran goes back, so it is a loop.

    Returns:
        The flags, loops and budget use.
    """
    settings = settings or HealthSettings()
    return Health(
        flags=_flags(records, settings),
        loops=_loops(records, while_edges),
        budget=_budget_use(budget, meter),
    )


def _text(record: AgentRecord) -> str:
    """Return a record's outputs as one text."""
    return "\n".join(record.outputs.values())


def _failed(record: AgentRecord) -> bool:
    """Whether the record is of a visit that failed and was routed."""
    return record.error is not None


def _flags(records: Sequence[AgentRecord], settings: HealthSettings) -> tuple[Flag, ...]:
    """Return every record's flags, in record order and :data:`SIGNALS` order within a record."""
    leaks = [re.compile(pattern) for pattern in (*DEFAULT_LEAKS, *settings.leak)]
    lengths: dict[str, list[int]] = {}
    for record in records:
        if not _failed(record):
            lengths.setdefault(record.agent, []).append(len(_text(record)))
    medians = {node: statistics.median(sizes) for node, sizes in lengths.items() if len(sizes) >= settings.min_samples}
    previous: dict[str, AgentRecord] = {}
    flags: list[Flag] = []
    for record in records:
        found = _record_signals(record, previous.get(record.agent), leaks, medians.get(record.agent), settings)
        flags.extend(Flag(signal, record.agent, record.step, detail) for signal, detail in found)
        previous[record.agent] = record
    return tuple(flags)


def _record_signals(
    record: AgentRecord,
    before: AgentRecord | None,
    leaks: list[re.Pattern[str]],
    median: float | None,
    settings: HealthSettings,
) -> list[tuple[str, str]]:
    """Return ``(signal, detail)`` for each signal one record shows."""
    found: list[tuple[str, str]] = []
    if record.rejected:
        found.append(("retried", f"{len(record.rejected)} answer(s) rejected: {record.rejected[0].error}"))
    if record.error_kind == ErrorKind.BAD_FORMAT:
        found.append(("bad_format", "the answer did not match its form after the retry"))
    if _failed(record):
        kind = "unknown" if record.error_kind is None else record.error_kind.value
        found.append(("error_routed", f"{kind}: {record.error}"))
        return found
    text = _text(record)
    if not text.strip():
        found.append(("empty", "the output is empty"))
    elif before is not None and not _failed(before) and _text(before) == text:
        found.append(("stuck", f"the same output as step {before.step}"))
    missing = sorted(extract_placeholders(record.prompt) - set(record.inputs))
    if missing:
        found.append(("unfilled", f"placeholders without a value: {', '.join(missing)}"))
    for pattern in leaks:
        match = pattern.search(text)
        if match:
            found.append(("leak", f"the output holds {match.group()!r}"))
            break
    if median:
        size = len(text)
        if size > settings.length_factor * median:
            found.append(("long", f"{size} characters, the node's median is {median:g}"))
        elif size < median / settings.length_factor:
            found.append(("short", f"{size} characters, the node's median is {median:g}"))
    return found


def _route(record: AgentRecord) -> tuple[str, ...]:
    """Return the targets a record's decision routed to; empty when it made none."""
    return record.route or ()


def _loops(records: Sequence[AgentRecord], while_edges: Iterable[tuple[str, str]] | None) -> dict[str, dict[str, Any]]:
    """Return ``{"source -> target": {"taken", "cap_hit"}}`` for each while edge taken or capped."""
    if while_edges is None:
        first: dict[str, int] = {}
        for index, record in enumerate(records):
            first.setdefault(record.agent, index)
        edges = {
            (record.agent, target): None
            for record in records
            for target in _route(record)
            if target in first and first[target] < first[record.agent]
        }
    else:
        edges = dict.fromkeys(while_edges)
    loops: dict[str, dict[str, Any]] = {}
    for source, target in edges:
        mine = [record for record in records if record.agent == source]
        taken = sum(target in _route(record) for record in mine)
        cap_hit = any(record.capped and target not in _route(record) for record in mine)
        if taken or cap_hit:
            loops[f"{source} -> {target}"] = {"taken": taken, "cap_hit": cap_hit}
    return loops


def _budget_use(budget: Budget | None, meter: RunMeter | None) -> dict[str, dict[str, float]]:
    """Return ``{limit: {"spent", "limit"}}`` for each limit set, read from the meter."""
    if meter is None:
        return {}
    budget = budget if budget is not None else meter.budget
    if budget is None:
        return {}
    use: dict[str, dict[str, float]] = {}
    for name, limit, spent in (
        ("calls", budget.max_calls, meter.calls),
        ("cost_usd", budget.max_cost_usd, meter.cost_usd),
        ("wall_s", budget.max_wall_s, meter.elapsed_s),
    ):
        if limit is not None:
            use[name] = {"spent": spent, "limit": limit}
    return use
