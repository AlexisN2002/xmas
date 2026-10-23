"""Execution records produced by a MAS run.

A trace is the input an :class:`~xmas.optimization.Attributor` computes on, so it
is modelled as data rather than as a log: every field is a plain value, the whole
structure round-trips through :meth:`Trace.to_dict`, and nothing in it is
formatted for human reading.

The plain-data form stays readable by v1 readers: every v1 key is always emitted,
in v1 order, and a newer key appears only when its value is set. Enum values are
emitted as plain strings, so both ``json.dumps`` and ``yaml.safe_dump`` accept the
dicts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xmas.errors import ErrorKind, RunStatus, status_for

if TYPE_CHECKING:  # pragma: no cover
    from xmas.health import Health


@dataclass(frozen=True)
class TokenUsage:
    """Input and output token counts for a single model call.

    Attributes:
        input_tokens: Tokens sent to the model.
        output_tokens: Tokens the model produced.
        cached_tokens: The part of ``input_tokens`` read from a prompt cache, or
            ``None`` when the provider did not say.
        reasoning_tokens: The part of ``output_tokens`` spent on reasoning, or
            ``None`` when the provider did not say.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two usages component-wise, so usage aggregates over a trace.

        A cached or reasoning count is ``None`` only when it is ``None`` on both
        sides.
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=_add_counts(self.cached_tokens, other.cached_tokens),
            reasoning_tokens=_add_counts(self.reasoning_tokens, other.reasoning_tokens),
        )

    def to_dict(self) -> dict[str, int]:
        """Return the plain-data form, with the cached and reasoning counts only when set."""
        data = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens is not None:
            data["cached_tokens"] = self.cached_tokens
        if self.reasoning_tokens is not None:
            data["reasoning_tokens"] = self.reasoning_tokens
        return data


def _add_counts(left: int | None, right: int | None) -> int | None:
    """Sum two optional counts, giving ``None`` only when both are ``None``."""
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation made by an agent."""

    tool: str
    arguments: dict[str, Any]
    result: str
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form."""
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "result": self.result,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class Rejected:
    """An answer refused by the node's answer form before its retry.

    Attributes:
        raw_response: The refused answer as the model gave it.
        error: Why it was refused, the problem the retry note carried.
    """

    raw_response: str
    error: str

    def to_dict(self) -> dict[str, str]:
        """Return the plain-data form."""
        return {"raw_response": self.raw_response, "error": self.error}


@dataclass(frozen=True)
class AgentRecord:
    """Everything one agent did during one run of the graph.

    Attributes:
        agent: Name of the agent, matching a node in the MAS graph.
        step: Zero-based order in which this record was produced within the run.
        prompt: The prompt template the agent held for this run, before rendering.
        rendered_prompt: The prompt actually sent, after placeholder substitution.
        inputs: The state values the agent read, keyed by variable name.
        outputs: The state values the agent wrote, keyed by variable name.
        tool_calls: Tool invocations made while producing the outputs, in order.
        usage: Token usage attributable to this agent for this run, as xmas
            estimates it. The provider's own counts are in ``tokens``.
        model: The agent's model binding name, or ``None`` for the default. The
            model the provider says served the call is ``served_model``.
        raw_response: The unparsed model response, kept because output extraction
            is lossy and an attributor may need what was discarded.
        latency_s: Wall seconds of the agent's model work, tool turns and tool
            calls included. Left out of ``==``, so equal runs compare equal.
        cost_usd: What the agent's calls cost in USD, or ``None`` when unknown;
            never ``0`` for an unknown cost.
        tokens: The provider's token counts, or ``None`` when it reported none.
        finish_reason: Why the model stopped, as the provider said.
        error_kind: The kind of a cut or refused answer that still produced text.
        served_model: The model the provider says served the call.
        kind: What kind of node ran: ``"agent"``, ``"step"`` for a step, or
            ``"box"`` for a box.
        rejected: Answers refused by the agent's answer form before the one it
            kept in ``raw_response``, in order.
        error: Why the visit failed, when it failed and the run went on.
        route: The nodes the visit's decision sent the run to, for a node with
            conditional or error edges; ``None`` for a node without.
        capped: Whether a ``while`` edge that held was skipped because its
            target had run ``max_visits`` times.
        inner: The inner system's trace, for a box.
    """

    agent: str
    step: int
    prompt: str
    rendered_prompt: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = TokenUsage()
    model: str | None = None
    raw_response: str = ""
    latency_s: float | None = field(default=None, compare=False)
    cost_usd: float | None = None
    tokens: TokenUsage | None = None
    finish_reason: str | None = None
    error_kind: ErrorKind | None = None
    served_model: str | None = None
    kind: str = "agent"
    rejected: tuple[Rejected, ...] = ()
    error: str | None = None
    route: tuple[str, ...] | None = None
    capped: bool = False
    inner: Trace | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form: the ten v1 keys, then each newer field that is set.

        ``kind`` is set when it is not ``"agent"``, ``rejected`` when it is not
        empty and ``capped`` when it is true.
        """
        data: dict[str, Any] = {
            "agent": self.agent,
            "step": self.step,
            "prompt": self.prompt,
            "rendered_prompt": self.rendered_prompt,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "raw_response": self.raw_response,
        }
        newer = {
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "tokens": None if self.tokens is None else self.tokens.to_dict(),
            "finish_reason": self.finish_reason,
            "error_kind": None if self.error_kind is None else self.error_kind.value,
            "served_model": self.served_model,
            "kind": None if self.kind == "agent" else self.kind,
            "rejected": [rejected.to_dict() for rejected in self.rejected] or None,
            "error": self.error,
            "route": None if self.route is None else list(self.route),
            "capped": self.capped or None,
            "inner": None if self.inner is None else self.inner.to_dict(),
        }
        data.update((key, value) for key, value in newer.items() if value is not None)
        return data


@dataclass(frozen=True)
class Trace:
    """The record of a single MAS call.

    Attributes:
        query: The input the MAS was called with.
        output: The terminal agent's output, i.e. what ``__call__`` returned.
        records: One :class:`AgentRecord` per agent execution, in execution order.
            An agent that runs twice (a cycle) contributes two records.
        error: Populated when the run failed, so a failed rollout is still data an
            attributor can reason about rather than a raised exception.
        error_kind: What kind of failure ``error`` is, when it is known.
        health: What looked wrong in the run (:func:`xmas.health.summarize`),
            or ``None`` for a trace the engine did not finish. Left out of
            ``==``, since its budget entry holds the wall time.
    """

    query: str
    output: str
    records: tuple[AgentRecord, ...] = ()
    error: str | None = None
    error_kind: ErrorKind | None = None
    health: Health | None = field(default=None, compare=False)

    def __getitem__(self, agent: str) -> tuple[AgentRecord, ...]:
        """Return every record for one agent, in execution order.

        Args:
            agent: The agent name.

        Returns:
            Its records. Empty if the agent did not run, which is a legitimate
            outcome for a conditional branch rather than an error.
        """
        return tuple(record for record in self.records if record.agent == agent)

    def walk(self) -> Iterator[tuple[tuple[str, ...], AgentRecord]]:
        """Yield every record depth first, a box's inner records right after the box's own.

        Yields:
            ``(path, record)``, where ``path`` names the boxes enclosing the
            record, outermost first; ``()`` for this trace's own records.
        """
        for record in self.records:
            yield (), record
            if record.inner is not None:
                for path, inner in record.inner.walk():
                    yield (record.agent, *path), inner

    @property
    def agents(self) -> tuple[str, ...]:
        """Names of the agents that ran, in first-execution order."""
        names: list[str] = []
        for record in self.records:
            if record.agent not in names:
                names.append(record.agent)
        return tuple(names)

    @property
    def usage(self) -> TokenUsage:
        """Token usage summed over every record."""
        total = TokenUsage()
        for record in self.records:
            total = total + record.usage
        return total

    @property
    def failed(self) -> bool:
        """Whether the run ended in an error."""
        return self.error is not None

    @property
    def status(self) -> RunStatus:
        """How the run ended: ``ok`` without an error, else derived from ``error_kind``."""
        return RunStatus.OK if self.error is None else status_for(self.error_kind)

    @property
    def cost_usd(self) -> float | None:
        """The records' costs summed, or ``None`` when there is no record or any record's cost is unknown."""
        if not self.records:  # nothing ran, or nothing finished: unknown, not 0
            return None
        total = 0.0
        for record in self.records:
            if record.cost_usd is None:
                return None
            total += record.cost_usd
        return total

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-data form, suitable for JSON or YAML.

        The four v1 keys come first. ``"status"`` follows only for a failed run,
        so it is never ``"ok"``, ``"error_kind"`` only when it is known, and
        ``"health"`` only when it reports something. A reader that finds no
        ``"status"`` reads ``"ok"`` when ``error`` is null and ``"failed"``
        otherwise.
        """
        data: dict[str, Any] = {
            "query": self.query,
            "output": self.output,
            "records": [record.to_dict() for record in self.records],
            "error": self.error,
        }
        if self.error is not None:
            data["status"] = self.status.value
        if self.error_kind is not None:
            data["error_kind"] = self.error_kind.value
        health = None if self.health is None else self.health.to_dict()
        if health:
            data["health"] = health
        return data
