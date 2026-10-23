"""Typed errors: what kind of failure ended a call or a run.

xmas classifies exceptions; it never re-wraps them. :func:`classify` reads an
exception, and the exceptions it was raised from, and returns an
:class:`ErrorKind`, so callers branch on a closed set instead of matching words
in ``Trace.error``. :func:`status_for` turns a kind into the run's
:class:`RunStatus`.

This is a leaf module: at runtime it imports no other xmas module. ``httpx2``,
which openai 3.x and the MCP client raise from, is read when it is installed.
"""

from __future__ import annotations

import enum
import json
import re
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import httpx
import openai
import pydantic

if TYPE_CHECKING:  # pragma: no cover
    from xmas.llms.meta import CallMeta

try:  # openai 3.x and the MCP client raise httpx2's errors, which httpx's types do not match
    import httpx2
except ImportError:  # pragma: no cover (httpx2 is optional here, not a dependency)
    _HTTPX2_TIMEOUTS: tuple[type[BaseException], ...] = ()
    _HTTPX2_OUTAGES: tuple[type[BaseException], ...] = ()
else:
    _HTTPX2_TIMEOUTS = (httpx2.TimeoutException,)
    _HTTPX2_OUTAGES = (httpx2.TransportError,)


class ErrorKind(enum.StrEnum):
    """Why a call or a run failed.

    ``CHECK_NOT_RUN`` is reserved for the verify phase and has no producer yet.
    """

    MONEY = "money"
    LENGTH_CUT = "length_cut"
    OUTAGE = "outage"
    BAD_FORMAT = "bad_format"
    REFUSAL = "refusal"
    STOPPED = "stopped"
    CHECK_NOT_RUN = "check_not_run"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    BUDGET = "budget"
    UNKNOWN = "unknown"


class RunStatus(enum.StrEnum):
    """How a run ended."""

    OK = "ok"
    FAILED = "failed"
    STOPPED = "stopped"
    OVER_BUDGET = "over_budget"


class XmasError(RuntimeError):
    """The error xmas raises itself; the message reads '<source>: <what> (<code>)', e.g. 'llm: rate limited (429)'.

    A plain ``XmasError`` is a failed call; its subclasses below name a run's
    typed failures.

    Attributes:
        status: The HTTP-style status code, or ``None``.
        retry_after: Seconds the endpoint asked the caller to wait, or ``None``.
        kind: What :func:`classify` returns for this error when it carries no
            ``status``.
    """

    kind: ErrorKind = ErrorKind.UNKNOWN

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the status and the wait, which the message alone would lose."""
        return type(self), (str(self), self.status, self.retry_after)


class BudgetExceeded(XmasError):  # noqa: N818 (public name fixed by the phase-1 contract)
    """A run reached a limit of its budget, so the next call was not started.

    Attributes:
        limit: The limit that bound: ``"cost_usd"``, ``"wall_s"``, ``"calls"``, or
            ``"cost_unknown"`` when a cost limit is set but a call reported no cost.
        spent: What the run had used of that limit. For ``"cost_unknown"``, the
            number of calls that reported no cost.
        allowed: The limit's value. For ``"cost_unknown"``, ``max_cost_usd``.
        calls: The calls the run had started.
    """

    kind = ErrorKind.BUDGET

    def __init__(
        self, limit: Literal["cost_usd", "wall_s", "calls", "cost_unknown"], spent: float, allowed: float, calls: int
    ) -> None:
        """Build the message that names the limit and the refused call."""
        if limit == "cost_unknown":
            uncosted = "1 call reported no cost and has" if spent == 1 else f"{spent:g} calls reported no cost and have"
            message = (
                f"cost_unknown: {uncosted} no price, "
                f"so max_cost_usd {allowed:g} cannot be checked before call {calls + 1}"
            )
        else:
            message = f"{limit} {spent:g} >= max {allowed:g} before call {calls + 1}"
        super().__init__(message)
        self.limit = limit
        self.spent = spent
        self.allowed = allowed
        self.calls = calls

    def __reduce__(self) -> tuple[type[BudgetExceeded], tuple[str, float, float, int]]:
        """Pickle by the constructor's arguments, not by the message."""
        return type(self), (self.limit, self.spent, self.allowed, self.calls)


class Stopped(XmasError):  # noqa: N818 (public name fixed by the phase-1 contract)
    """A run was stopped on request.

    A caller's model wrapper (and later the Stop button) raises it to end the run
    with status ``stopped``.
    """

    kind = ErrorKind.STOPPED


class EmptyResponse(XmasError):  # noqa: N818 (public name fixed by the phase-1 contract)
    """A model answered with no content.

    The message starts with promptolution's words, "Empty content from model", so
    checks written against promptolution's error keep matching.

    Attributes:
        kind: Why the answer was empty, such as ``LENGTH_CUT`` or ``REFUSAL``.
        meta: The combined metadata of the paid attempts behind the error, so a
            run's meter can count what they cost, or ``None`` when not known.
    """

    def __init__(self, message: str, kind: ErrorKind = ErrorKind.UNKNOWN, meta: CallMeta | None = None) -> None:
        """Keep the message, the kind and the paid attempts' metadata."""
        super().__init__(message)
        self.kind = kind
        self.meta = meta

    def __reduce__(self) -> tuple[type[EmptyResponse], tuple[str, ErrorKind, CallMeta | None]]:
        """Pickle with the kind and the meta, which the message alone would lose."""
        return type(self), (str(self), self.kind, self.meta)


class BadFormat(XmasError):  # noqa: N818 (public name fixed by the phase-2 contract)
    """A node's output did not have its declared form.

    An agent with an answer form raises it when its answer and its one retry
    both fail the form; a step raises it when a write has the wrong type.

    Attributes:
        node: The node whose output was refused.
        problems: One line per refused attempt, first attempt first.
        meta: The combined metadata of the paid attempts behind the error, so a
            run's meter can count what they cost, or ``None`` when not known.
    """

    kind = ErrorKind.BAD_FORMAT

    def __init__(self, node: str, problems: tuple[str, ...], meta: CallMeta | None = None) -> None:
        """Build the message that names the node and every problem."""
        super().__init__(f"Bad format from {node!r}: " + "; after the retry: ".join(problems))
        self.node = node
        self.problems = tuple(problems)
        self.meta = meta

    def __reduce__(self) -> tuple[type[BadFormat], tuple[str, tuple[str, ...], CallMeta | None]]:
        """Pickle by the constructor's arguments, not by the message."""
        return type(self), (self.node, self.problems, self.meta)


MONEY_WORDS = re.compile(
    r"key limit exceeded|requires more credits|insufficient credits|insufficient_quota"
    r"|exceeded your current quota|payment required|in_flight_budget_exhausted|insufficient_budget",
    re.I,
)
"""Provider texts that mean the key or account is out of money."""

MODERATION_WORDS = re.compile(r"flagged|moderation", re.I)
"""Provider texts that mean a moderation filter refused the request."""

_MAX_DEPTH = 8


def classify(error: BaseException) -> ErrorKind:
    """Return the kind of an exception, reading the exceptions it came from too.

    The walk visits ``error``, then its ``__cause__`` and ``__context__``, nearest
    first, at most 8 deep, and visits each exception once, so a cycle ends it.
    The first rule that matches any visited exception wins:

    0. an :class:`XmasError` with a ``status``: :func:`kind_of_status` of it and
       the message, except that 401 and 403 (the source refused the caller) are
       ``REFUSAL`` when no money words appear;
    1. a ``kind`` attribute that is an :class:`ErrorKind`;
    2. ``openai.APIStatusError``: :func:`kind_of_status` of its status code and its
       message plus body;
    3. ``openai.APITimeoutError``, ``TimeoutError`` or ``TimeoutException`` of
       ``httpx`` or ``httpx2``: ``TIMEOUT`` (before rule 4, whose types they
       subclass);
    4. ``openai.APIConnectionError``, ``ConnectionError`` or ``TransportError``
       of ``httpx`` or ``httpx2``: ``OUTAGE``;
    5. ``pydantic.ValidationError`` or ``json.JSONDecodeError``: ``BAD_FORMAT``.

    Otherwise the kind is ``UNKNOWN``. Walking the chain is what classifies
    promptolution's ``RuntimeError("LLM batch failed: ...")`` by its cause.

    Args:
        error: The exception to classify.

    Returns:
        The kind.
    """
    seen: set[int] = set()
    queue: deque[tuple[BaseException, int]] = deque([(error, 0)])
    while queue:
        exc, depth = queue.popleft()
        if id(exc) in seen or depth >= _MAX_DEPTH:
            continue
        seen.add(id(exc))
        kind = _own_kind(exc)
        if kind is not None:
            return kind
        queue.extend((linked, depth + 1) for linked in (exc.__cause__, exc.__context__) if linked is not None)
    return ErrorKind.UNKNOWN


def _own_kind(exc: BaseException) -> ErrorKind | None:
    """Return the kind of one exception by rules 0 to 5, or ``None`` if none matches."""
    if isinstance(exc, XmasError) and exc.status is not None:
        status_kind = kind_of_status(exc.status, str(exc))
        if status_kind == ErrorKind.UNKNOWN and exc.status in (401, 403):
            return ErrorKind.REFUSAL
        return status_kind
    kind = getattr(exc, "kind", None)
    if isinstance(kind, ErrorKind):
        return kind
    if isinstance(exc, openai.APIStatusError):
        text = str(exc) if exc.body is None else f"{exc} {exc.body}"
        return kind_of_status(exc.status_code, text)
    if isinstance(exc, (openai.APITimeoutError, TimeoutError, httpx.TimeoutException, *_HTTPX2_TIMEOUTS)):
        return ErrorKind.TIMEOUT
    if isinstance(exc, (openai.APIConnectionError, ConnectionError, httpx.TransportError, *_HTTPX2_OUTAGES)):
        return ErrorKind.OUTAGE
    if isinstance(exc, pydantic.ValidationError | json.JSONDecodeError):
        return ErrorKind.BAD_FORMAT
    return None


def kind_of_status(status: int, text: str = "") -> ErrorKind:
    """Return the kind of an HTTP error status.

    402, or any 4xx whose text names money (403 "Key limit exceeded", 429
    "insufficient_quota"), is ``MONEY``; 429 is ``RATE_LIMITED``; 408 is
    ``TIMEOUT``; 500 and above are ``OUTAGE``; 400 or 403 naming moderation is
    ``REFUSAL``; anything else (401, 404, 422) is ``UNKNOWN``.

    Args:
        status: The HTTP status code.
        text: The error's message and body, searched for provider words.

    Returns:
        The kind.
    """
    if status == 402 or (400 <= status < 500 and MONEY_WORDS.search(text)):
        return ErrorKind.MONEY
    if status == 429:
        return ErrorKind.RATE_LIMITED
    if status == 408:
        return ErrorKind.TIMEOUT
    if status >= 500:
        return ErrorKind.OUTAGE
    if status in (400, 403) and MODERATION_WORDS.search(text):
        return ErrorKind.REFUSAL
    return ErrorKind.UNKNOWN


def kind_of_finish(finish_reason: str | None, refused: bool = False) -> ErrorKind | None:
    """Return the kind of an answer that ended badly but still arrived.

    Args:
        finish_reason: The provider's finish reason.
        refused: Whether the provider flagged the answer as a refusal.

    Returns:
        ``LENGTH_CUT`` for ``"length"``, ``REFUSAL`` for ``"content_filter"`` or a
        refusal, otherwise ``None``.
    """
    if finish_reason == "length":
        return ErrorKind.LENGTH_CUT
    if finish_reason == "content_filter" or refused:
        return ErrorKind.REFUSAL
    return None


def status_for(kind: ErrorKind | None) -> RunStatus:
    """Return the status of a failed run with this error kind.

    Args:
        kind: The run's error kind, or ``None`` when it is not known.

    Returns:
        ``OVER_BUDGET`` for ``BUDGET``, ``STOPPED`` for ``STOPPED``, otherwise
        ``FAILED``.
    """
    if kind == ErrorKind.BUDGET:
        return RunStatus.OVER_BUDGET
    if kind == ErrorKind.STOPPED:
        return RunStatus.STOPPED
    return RunStatus.FAILED
