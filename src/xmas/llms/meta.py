"""What one model call cost, how it ended, and which model served it.

promptolution's generation keeps only the message text and drops everything else
the provider sent back. This module holds what xmas keeps instead: provider token
counts, a reported (or priced) cost, the finish reason, a refusal flag and the
served model name, gathered into a :class:`CallMeta` that the engine copies onto
the agent's :class:`~xmas.trace.AgentRecord`.

A cost that is not known is ``None``, never ``0``: a budget can only bind on a
number it can trust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import ErrorKind, kind_of_finish
from xmas.trace import TokenUsage

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionUsage

    from xmas.tools import Tool, ToolCallRequest


@dataclass(frozen=True)
class TokenPrice:
    """A model's price in USD per million tokens, for providers that report no cost.

    Reasoning tokens are already part of the completion tokens, so they are not
    priced a second time. Tokens written to a prompt cache are priced as plain
    input.

    Attributes:
        input_per_m: USD per million uncached input tokens.
        output_per_m: USD per million output tokens, reasoning included.
        cached_input_per_m: USD per million input tokens read from a cache, or
            ``None`` to price them as plain input.
    """

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None = None

    def cost_usd(self, tokens: TokenUsage) -> float:
        """Return the price of one call's tokens.

        Args:
            tokens: The provider's counts for the call.

        Returns:
            The cost in USD.
        """
        cached = tokens.cached_tokens or 0
        # cache reads are part of the input (openai's schema); more of them than input
        # means the provider counted them apart, and the whole input is uncached
        uncached = tokens.input_tokens - cached if cached <= tokens.input_tokens else tokens.input_tokens
        cached_rate = self.input_per_m if self.cached_input_per_m is None else self.cached_input_per_m
        return (uncached * self.input_per_m + cached * cached_rate + tokens.output_tokens * self.output_per_m) / 1e6


@dataclass(frozen=True)
class CallMeta:
    """What the provider reported about one call, or about several combined.

    Attributes:
        latency_s: Wall seconds the call took.
        tokens: The provider's token counts, or ``None`` when it sent none.
        cost_usd: Reported or priced cost in USD, or ``None`` when unknown.
        finish_reason: Why the model stopped (``"stop"``, ``"length"``,
            ``"tool_calls"``, ``"content_filter"``, ...).
        refused: Whether the model refused to answer.
        served_model: The model the provider says served the call, which can
            differ from the name that was asked for.
    """

    latency_s: float
    tokens: TokenUsage | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    refused: bool = False
    served_model: str | None = None

    @property
    def error_kind(self) -> ErrorKind | None:
        """The kind of a cut or refused answer, or ``None`` for a clean finish."""
        return kind_of_finish(self.finish_reason, self.refused)

    def combine(self, other: CallMeta) -> CallMeta:
        """Return the meta of this call followed by ``other``.

        Latency, tokens and cost are summed. An unknown count or cost on either
        side makes the sum unknown, because a partial sum would understate what
        was paid. How the pair ended (finish reason, refusal, served model) is
        ``other``'s, the later call.

        Args:
            other: The later call.

        Returns:
            The combined meta.
        """
        tokens = None if self.tokens is None or other.tokens is None else self.tokens + other.tokens
        cost = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return CallMeta(
            latency_s=self.latency_s + other.latency_s,
            tokens=tokens,
            cost_usd=cost,
            finish_reason=other.finish_reason,
            refused=other.refused,
            served_model=other.served_model,
        )

    def record_fields(self) -> dict[str, Any]:
        """Return the :class:`~xmas.trace.AgentRecord` keywords this meta fills.

        ``latency_s`` is not among them: the engine times the whole agent,
        tool turns and tool calls included, and sets it itself.

        Returns:
            ``cost_usd``, ``tokens``, ``finish_reason``, ``error_kind`` and
            ``served_model``.
        """
        return {
            "cost_usd": self.cost_usd,
            "tokens": self.tokens,
            "finish_reason": self.finish_reason,
            "error_kind": self.error_kind,
            "served_model": self.served_model,
        }


@runtime_checkable
class SupportsMeta(Protocol):
    """A model that returns call metadata next to its answers."""

    def generate_with_meta(self, prompt: str, system_prompt: str = DEFAULT_SYS_PROMPT) -> tuple[str, CallMeta]:
        """Answer one prompt.

        Args:
            prompt: The user message.
            system_prompt: The system message.

        Returns:
            The answer text and the call's metadata.
        """
        ...

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Run one chat turn that may request tools.

        Args:
            messages: OpenAI-shaped history.
            tools: Tools on offer. Empty forces a text answer.

        Returns:
            The message text, the requested calls, and the turn's metadata.
        """
        ...


def supports_meta(llm: object) -> bool:
    """Whether the engine should take ``llm``'s metadata path.

    Having the methods is not enough: the model must also have opted in with
    ``full_response`` set to ``True``. A subclass that hooks the old call path
    keeps it, so its own accounting is never bypassed.

    Args:
        llm: Any model.

    Returns:
        ``True`` when the model has both meta methods and ``full_response is True``.
    """
    return isinstance(llm, SupportsMeta) and getattr(llm, "full_response", False) is True


def meta_from_completion(completion: ChatCompletion, latency_s: float, price: TokenPrice | None) -> CallMeta:
    """Read a chat completion's metadata.

    The cost is the first that is known of: ``usage.cost`` as the provider sent
    it, a top-level ``cost`` field some proxies add, the tokens priced at
    ``price``. Otherwise it is ``None``. Only non-negative numbers count as a
    reported cost.

    Args:
        completion: The provider's response.
        latency_s: Wall seconds the call took.
        price: The model's price, or ``None``.

    Returns:
        The call's metadata.
    """
    usage = completion.usage
    tokens = _tokens(usage) if usage is not None else None
    cost = _reported_cost(usage.model_extra) if usage is not None else None
    if cost is None:
        cost = _reported_cost(completion.model_extra)
    if cost is None and price is not None and tokens is not None:
        cost = price.cost_usd(tokens)

    choice = completion.choices[0] if completion.choices else None
    finish = None if choice is None or choice.finish_reason is None else str(choice.finish_reason)
    refused = bool(choice is not None and choice.message.refusal) or finish == "content_filter"
    return CallMeta(
        latency_s=latency_s,
        tokens=tokens,
        cost_usd=cost,
        finish_reason=finish,
        refused=refused,
        served_model=completion.model or None,
    )


def _tokens(usage: CompletionUsage) -> TokenUsage | None:
    """Return the provider's counts, with cache reads and reasoning when it sent them.

    ``None`` when a count is missing: the client does not validate responses, so
    ``"usage": {}`` arrives as ``None`` counts whatever the types say.
    """
    input_tokens: object = usage.prompt_tokens
    output_tokens: object = usage.completion_tokens
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=None if prompt_details is None else prompt_details.cached_tokens,
        reasoning_tokens=None if completion_details is None else completion_details.reasoning_tokens,
    )


def _reported_cost(extra: dict[str, Any] | None) -> float | None:
    """Return a ``cost`` field when it is a non-negative number, else ``None``.

    A bool is an int to Python but never a cost, and a string is not trusted.
    """
    value = (extra or {}).get("cost")
    if isinstance(value, bool) or not isinstance(value, int | float) or not value >= 0:
        return None
    return float(value)
