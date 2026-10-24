"""Budgets: limits on what one run may spend, and the meter that enforces them.

A :class:`Budget` is runtime state, like a MAS's tools, and never part of the
spec. Each run gets its own :class:`RunMeter`, which counts the run's calls, cost
and wall time and refuses a call before it starts once a limit is reached. The
meter also holds the run's caps on calls and items in flight, and passes the
run's events to a listener.
"""

from __future__ import annotations

import threading
import time
import uuid
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from xmas.errors import BudgetExceeded, EmptyResponse
from xmas.trace import AgentRecord, Event, Trace

if TYPE_CHECKING:
    from xmas.llms.meta import CallMeta


@dataclass(frozen=True)
class Budget:
    """Limits for one run, in USD, wall seconds and model calls; ``None`` means no limit.

    A call may start only while the run is below every limit, so a limit of ``0``
    starts no call. A cost limit that cannot be checked, because a call reported
    no cost and the model has no price, refuses the next call instead of silently
    not binding. A negative or NaN limit is refused. ``max_parallel_calls`` caps
    the model calls in flight at once, across every branch, box and item of the
    run; a call waits for a free slot. It is at least ``1``.
    """

    max_cost_usd: float | None = None
    max_wall_s: float | None = None
    max_calls: int | None = None
    max_parallel_calls: int | None = None

    def __post_init__(self) -> None:
        """Refuse negative and NaN limits, and a parallel cap below 1."""
        for name in ("max_cost_usd", "max_wall_s", "max_calls"):
            value = getattr(self, name)
            if value is not None and not value >= 0:
                raise ValueError(f"Budget.{name} must be >= 0, got {value!r}")
        if self.max_parallel_calls is not None and not self.max_parallel_calls >= 1:
            raise ValueError(f"Budget.max_parallel_calls must be >= 1, got {self.max_parallel_calls!r}")


class _Counts(Protocol):
    def slot(self) -> AbstractContextManager[Any]: ...
    def before_call(self) -> None: ...
    def after_call(self, meta: CallMeta | None) -> None: ...


class _Metering:
    def metered[T](self: _Counts, call: Callable[[], tuple[T, CallMeta | None]]) -> tuple[T, CallMeta | None]:
        """Make one model call on this meter: in a free slot, refused at a limit, counted when it ends.

        ``call`` returns its result and the call's metadata. An empty answer
        raises, but its paid attempts are counted first.
        """
        with self.slot():
            self.before_call()
            try:
                result, meta = call()
            except EmptyResponse as error:
                self.after_call(error.meta)
                raise
            self.after_call(meta)
            return result, meta


class RunMeter(_Metering):
    """What one run has used so far, checked against its budget before each call.

    The counters and the record list are shared by the run's parallel branches,
    so every update holds one lock.

    Attributes:
        budget: The run's limits, or ``None`` to count without limits.
        owner: The MAS the meter belongs to, so a nested run of another MAS that
            inherits this meter can tell it is not its own, unless the meter
            adopted that MAS (:meth:`adopt`).
        started_s: ``time.monotonic()`` when the meter was made, then when the
            run's first node started (:meth:`start`).
        calls: Calls started so far.
        cost_usd: Cost reported or priced so far, in USD.
        uncosted_calls: Finished calls whose cost is unknown.
        records: Records of the owner's finished visits, in wall-clock order;
            an adopted system's are kept apart (:meth:`records_of`).
        run_id: A new random id for the run.
        graph_id: The ``describe()`` id of the owner, when the run was given it.
        listener: Called with each of the run's events (:meth:`emit`), or ``None``.
    """

    def __init__(
        self,
        budget: Budget | None = None,
        owner: object | None = None,
        *,
        listener: Callable[[Event], None] | None = None,
        graph_id: str | None = None,
    ) -> None:
        """Start the clock with every counter at zero."""
        self.budget = budget
        self.owner = owner
        self.run_id = uuid.uuid4().hex
        self.graph_id = graph_id
        self.listener = listener
        self.started_s = time.monotonic()
        self.calls = 0
        self.cost_usd = 0.0
        self.uncosted_calls = 0
        self.records: list[AgentRecord] = []
        self._adopted: dict[int, tuple[object, list[AgentRecord]]] = {}
        self._lock = threading.Lock()
        self._started = False
        cap = None if budget is None else budget.max_parallel_calls
        self._slots = None if cap is None else threading.BoundedSemaphore(cap)
        self._gates: dict[str, threading.BoundedSemaphore] = {}

    def adopt(self, system: object, into: object | None = None) -> None:
        """Let a nested system, such as a box's, run on this meter: one budget for both.

        Its records are kept apart, or with those of ``into``, a system this
        meter owns or adopted, as a chain's items keep theirs with the system
        the chain sits in.
        """
        with self._lock:
            if id(system) not in self._adopted:
                records = [] if into is None else self.records_of(into)
                self._adopted[id(system)] = (system, records)

    def owns(self, system: object) -> bool:
        """Whether ``system`` is the owner or an adopted system."""
        return system is self.owner or id(system) in self._adopted

    def records_of(self, system: object) -> list[AgentRecord]:
        """Return the records kept for ``system``: the owner's, or an adopted system's."""
        if system is self.owner:
            return self.records
        return self._adopted[id(system)][1] if id(system) in self._adopted else []

    @property
    def elapsed_s(self) -> float:
        """Wall seconds since the meter was made, or since :meth:`start`."""
        return time.monotonic() - self.started_s

    def start(self) -> None:
        """Restart the clock when the run's first node starts; later calls do nothing.

        A batch makes every item's meter before it runs any, and an item can wait
        for a free worker; that wait is not the run's wall time.
        """
        with self._lock:
            if not self._started:
                self._started = True
                self.started_s = time.monotonic()

    def before_call(self) -> None:
        """Count a call that is about to start, or refuse it with :class:`~xmas.errors.BudgetExceeded`.

        The limits are checked in order: calls, wall time, a cost limit with a
        call of unknown cost (``"cost_unknown"``), cost.
        """
        with self._lock:
            self._check(count_calls=True)
            self.calls += 1

    def check_boundary(self) -> None:
        """Check the wall and cost limits before work that makes no model call, such as a step; nothing is counted."""
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
        """Add a finished call's cost, or count it as uncosted when it is unknown."""
        cost = None if meta is None else meta.cost_usd
        with self._lock:
            if cost is None:
                self.uncosted_calls += 1
            else:
                self.cost_usd += cost

    def keep(self, record: AgentRecord, system: object | None = None) -> None:
        """Keep a finished visit's record, so a failed run still has it; ``system`` names an adopted system."""
        with self._lock:
            self.records_of(self.owner if system is None else system).append(record)

    def slot(self) -> AbstractContextManager[Any]:
        """Hold one of the run's ``max_parallel_calls`` slots for a model call; no cap holds nothing."""
        return nullcontext() if self._slots is None else self._slots

    def gate(self, key: str, limit: int | None) -> AbstractContextManager[Any]:
        """Hold one of ``limit`` places under ``key``, such as a chain's items at once; ``None`` holds nothing."""
        if limit is None:
            return nullcontext()
        with self._lock:
            return self._gates.setdefault(key, threading.BoundedSemaphore(limit))

    def emit(self, kind: str, node: str, **fields: Any) -> None:
        """Pass an event to the listener, which is called on the run's threads.

        A listener that raises is warned about once and gets no more events
        this run: a viewer's bug does not end a paid run.
        """
        listener = self.listener
        if listener is None:
            return
        try:
            listener(Event(kind, node, self.run_id, self.elapsed_s, **fields))
        except Exception as error:
            with self._lock:
                first = self.listener is listener
                self.listener = None
            if first:
                warnings.warn(
                    f"The event listener raised {type(error).__name__}: {error}; it gets no more events this run.",
                    RuntimeWarning,
                    stacklevel=2,
                )


class VisitMeter(_Metering):
    """One node visit's share of a run's meter.

    Every call counts on the run's meter; the visit also keeps the metadata of
    its own finished calls (``metas``, ``None`` for a model that gives none),
    so a visit that fails after paid attempts still records what they cost. A
    box's visit keeps the trace of its inner system when that fails (``inner``).
    """

    def __init__(self, run: RunMeter) -> None:
        """Start a visit with no call made."""
        self.run = run
        self.metas: list[CallMeta | None] = []
        self.inner: Trace | None = None

    def slot(self) -> AbstractContextManager[Any]:
        """Hold a slot of the run's parallel call cap."""
        return self.run.slot()

    def before_call(self) -> None:
        """Count a call that is about to start on the run's meter, or refuse it."""
        self.run.before_call()

    def check_boundary(self) -> None:
        """Check the run's wall and cost limits."""
        self.run.check_boundary()

    def after_call(self, meta: CallMeta | None) -> None:
        """Count a finished call on the run's meter and keep its metadata for the visit."""
        self.run.after_call(meta)
        self.metas.append(meta)

    @property
    def meta(self) -> CallMeta | None:
        """The visit's calls combined, or ``None`` when it made none or a model gave none."""
        if not self.metas or any(meta is None for meta in self.metas):
            return None
        first, *rest = cast("list[CallMeta]", self.metas)
        for meta in rest:
            first = first.combine(meta)
        return first
