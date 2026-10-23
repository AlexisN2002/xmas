"""Budgets: limits on what one run may spend, and the meter that enforces them.

A :class:`Budget` is runtime state, like a MAS's tools, and never part of the
spec. Each run gets its own :class:`RunMeter`, which counts the run's calls, cost
and wall time and refuses a call before it starts once a limit is reached.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xmas.errors import BudgetExceeded
from xmas.trace import AgentRecord

if TYPE_CHECKING:
    from xmas.llms.meta import CallMeta


@dataclass(frozen=True)
class Budget:
    """Limits for one run; ``None`` means no limit.

    A call may start only while the run is below every limit, so a limit of ``0``
    starts no call. A cost limit that cannot be checked, because a call reported
    no cost and the model has no price, refuses the next call instead of silently
    not binding.

    Attributes:
        max_cost_usd: The most the run may spend, in USD.
        max_wall_s: The most wall seconds the run may take.
        max_calls: The most model calls the run may start.

    Raises:
        ValueError: If a limit is negative or NaN.
    """

    max_cost_usd: float | None = None
    max_wall_s: float | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        """Refuse negative limits."""
        for name in ("max_cost_usd", "max_wall_s", "max_calls"):
            value = getattr(self, name)
            if value is not None and not value >= 0:
                raise ValueError(f"Budget.{name} must be >= 0, got {value!r}")


class RunMeter:
    """What one run has used so far, checked against its budget before each call.

    The counters and the record list are shared by the run's parallel branches,
    so every update holds one lock.

    Attributes:
        budget: The run's limits, or ``None`` to count without limits.
        owner: The MAS the meter belongs to, so a nested run of another MAS that
            inherits this meter can tell it is not its own.
        started_s: ``time.monotonic()`` when the meter was made, then when the
            run's first agent started (:meth:`start`).
        calls: Calls started so far.
        cost_usd: Cost reported or priced so far, in USD.
        uncosted_calls: Finished calls whose cost is unknown.
        records: Records of the agents that finished, in wall-clock order.
    """

    def __init__(self, budget: Budget | None = None, owner: object | None = None) -> None:
        """Start the clock with every counter at zero."""
        self.budget = budget
        self.owner = owner
        self.started_s = time.monotonic()
        self.calls = 0
        self.cost_usd = 0.0
        self.uncosted_calls = 0
        self.records: list[AgentRecord] = []
        self._lock = threading.Lock()
        self._started = False

    @property
    def elapsed_s(self) -> float:
        """Wall seconds since the meter was made, or since :meth:`start`."""
        return time.monotonic() - self.started_s

    def start(self) -> None:
        """Restart the clock when the run's first agent starts; later calls do nothing.

        A batch makes every item's meter before it runs any, and an item can wait
        for a free worker; that wait is not the run's wall time.
        """
        with self._lock:
            if not self._started:
                self._started = True
                self.started_s = time.monotonic()

    def before_call(self) -> None:
        """Count a call that is about to start, or refuse it.

        Raises:
            BudgetExceeded: If the run has made ``max_calls`` calls, has run for
                ``max_wall_s``, has a cost limit and a call of unknown cost
                (limit ``"cost_unknown"``), or has spent ``max_cost_usd``; checked
                in that order.
        """
        with self._lock:
            self._check(count_calls=True)
            self.calls += 1

    def check_boundary(self) -> None:
        """Check the wall and cost limits before work that makes no model call, such as a step.

        Nothing is counted: ``max_calls`` limits model calls only.

        Raises:
            BudgetExceeded: If the run has run for ``max_wall_s``, has a cost
                limit and a call of unknown cost, or has spent ``max_cost_usd``.
        """
        with self._lock:
            self._check(count_calls=False)

    def _check(self, *, count_calls: bool) -> None:
        """Raise for the first limit reached, the call limit only when ``count_calls``; the caller holds the lock."""
        budget = self.budget
        if budget is None:
            return
        if count_calls and budget.max_calls is not None and self.calls >= budget.max_calls:
            raise BudgetExceeded("calls", self.calls, budget.max_calls, self.calls)
        elapsed = self.elapsed_s
        if budget.max_wall_s is not None and elapsed >= budget.max_wall_s:
            raise BudgetExceeded("wall_s", elapsed, budget.max_wall_s, self.calls)
        if budget.max_cost_usd is not None:
            if self.uncosted_calls > 0:
                raise BudgetExceeded("cost_unknown", self.uncosted_calls, budget.max_cost_usd, self.calls)
            if self.cost_usd >= budget.max_cost_usd:
                raise BudgetExceeded("cost_usd", self.cost_usd, budget.max_cost_usd, self.calls)

    def after_call(self, meta: CallMeta | None) -> None:
        """Add a finished call's cost, or count it as uncosted when it is unknown.

        Args:
            meta: The call's metadata, or ``None`` when the model gives none.
        """
        cost = None if meta is None else meta.cost_usd
        with self._lock:
            if cost is None:
                self.uncosted_calls += 1
            else:
                self.cost_usd += cost

    def keep(self, record: AgentRecord) -> None:
        """Keep the record of an agent that finished, so a failed run still has it.

        Args:
            record: The finished agent's record.
        """
        with self._lock:
            self.records.append(record)
