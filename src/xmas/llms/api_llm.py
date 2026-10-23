"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import TYPE_CHECKING, Any

import httpx
from promptolution.llms import api_llm as _promptolution
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import EmptyResponse, ErrorKind
from xmas.llms.meta import CallMeta, TokenPrice, meta_from_completion
from xmas.llms.scoring import ScoredSequence
from xmas.llms.telemetry import GENERATE, SCORE, CallLog, CallRecord, approximate_tokens
from xmas.tools import Tool, ToolCallRequest, _create_with_retries

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

ENV_BASE_URL = "XMAS_BASE_URL"
"""Environment variable holding the endpoint base URL."""

ENV_API_KEY = "XMAS_API_KEY"
"""Environment variable holding the API key."""

SCORE_PATH = "score"
"""Scoring endpoint, resolved relative to ``base_url``."""


class ScoringUnsupportedError(RuntimeError):
    """Raised when a backend cannot expose prompt log-probabilities.

    Carries the endpoint's own explanation, which names the models that would
    work, so the caller is not left guessing what to switch to.
    """


class RateLimitedError(RuntimeError):
    """Raised when the endpoint refuses a call because the key is over its limit.

    Typed, and carrying ``retry_after``, because a limiter wall is the system
    working and the only correct response is to wait exactly as long as the
    endpoint asked. A caller that could not tell this apart from a malformed
    request would have to either give up or guess, and guessing is what turns
    one 429 into a retry storm.

    Attributes:
        kind: Always :attr:`~xmas.errors.ErrorKind.RATE_LIMITED`, so
            :func:`~xmas.errors.classify` reads it without matching the message.
        retry_after: Seconds the endpoint asked the caller to wait, or ``None``
            when it did not say.
    """

    kind: ErrorKind = ErrorKind.RATE_LIMITED

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Initialize the error.

        Args:
            message: The endpoint's own explanation.
            retry_after: Value of the ``Retry-After`` header, in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after


class APILLM(_promptolution.APILLM):
    """An OpenAI-compatible model that can also score a fixed continuation.

    :meth:`get_response` is inherited from promptolution: batching, concurrency,
    retries and token metering live there. What xmas adds is :meth:`score`, the
    tool turn :meth:`get_tool_response`, and a second plain path,
    :meth:`generate_with_meta`, that sends the very same request but keeps what
    the provider reported (tokens, cost, finish reason, served model).

    The engine takes the second path when ``full_response`` is ``True``. By
    default it is, unless a subclass overrides a hook of the old path
    (``get_response``, ``_get_response``, ``_aget_batch``,
    ``_ainvoke_with_retries``, ``_ainvoke_once`` or ``get_tool_response``): such
    a subclass keeps the path it hooked, and its own accounting with it.

    The constructor accepts ``model`` and ``base_url``, which is what the xmas
    surface reads like, and forwards them to promptolution's ``model_id`` and
    ``api_url``. Both spellings are accepted so that code written against either
    library keeps working.

    Attributes:
        full_response: Whether the engine reads this model through
            :meth:`generate_with_meta` and :meth:`get_tool_response_meta`.
        price: The price used for a call whose response reports no cost.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        telemetry: CallLog | None = None,
        backend_tag: str = "gateway",
        full_response: bool | None = None,
        price: TokenPrice | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``. Alias of the
                parent's ``model_id``.
            base_url: Endpoint base URL. Alias of the parent's ``api_url``.
                Defaults to the ``XMAS_BASE_URL`` environment variable.
            api_key: API key. Defaults to the ``XMAS_API_KEY`` environment
                variable. Never read from a file in the repository.
            telemetry: Collector for per-call statistics. ``None`` records
                nothing, which is the default so that ordinary use pays no
                attention to instrumentation.
            backend_tag: What this backend calls itself in a record. Calls from
                two serving paths must never be pooled into one statistic, and
                this is what keeps them separable afterwards.
            full_response: ``True`` sends plain agents through
                :meth:`generate_with_meta`, ``False`` keeps promptolution's
                path, and ``None`` decides by class: on, unless the class
                overrides a hook of the old path. The class is read once, at
                construction, so a hook patched onto an instance afterwards is
                not seen; pass ``full_response=False`` to keep promptolution's
                path.
            price: USD per million tokens, used only when a response reports
                no cost. ``None`` leaves such a cost unknown.
            **kwargs: Forwarded to promptolution's ``APILLM``.

        Raises:
            ValueError: If no model is given, or no base URL can be resolved.
        """
        model = model or kwargs.pop("model_id", None)
        base_url = base_url or kwargs.pop("api_url", None) or os.environ.get(ENV_BASE_URL)
        api_key = api_key or os.environ.get(ENV_API_KEY)

        if not model:
            raise ValueError("APILLM needs a model, e.g. APILLM(model='llama-3.1-8b').")
        if not base_url:
            raise ValueError(f"APILLM needs a base_url, or {ENV_BASE_URL} set in the environment.")

        self.base_url = base_url.rstrip("/")
        self.telemetry = telemetry
        self.backend_tag = backend_tag
        self.full_response: bool = not _overrides_call_path(type(self)) if full_response is None else full_response
        self.price = price
        super().__init__(model_id=model, api_url=self.base_url, api_key=api_key, **kwargs)

    def _record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.telemetry is not None:
            self.telemetry.record(record)

    def get_response(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None = None,
    ) -> list[str]:
        """Generate, timing the call and recording what it cost.

        The generation itself is promptolution's; this adds the measurement.
        Token counts are read from the parent's own counters, which split on
        whitespace, so they are an approximation and are reported as one.

        Args:
            prompts: Prompt or prompts to answer.
            system_prompts: System prompt or prompts.

        Returns:
            One response per prompt.
        """
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        before = self.get_token_count()
        started = time.perf_counter()
        try:
            responses = super().get_response(prompts, system_prompts)
        except Exception as error:
            self._record(
                CallRecord(
                    operation=GENERATE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    n_prompts=n_prompts,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        latency = time.perf_counter() - started
        after = self.get_token_count()
        self._record(
            CallRecord(
                operation=GENERATE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=latency,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
                n_prompts=n_prompts,
            )
        )
        return responses

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """Run one chat turn that may request tools.

        Scheduled on promptolution's background loop and semaphore, with its
        timeout and backoff, and recorded like :meth:`get_response`. Token counts
        are the provider's when it reports usage.

        Args:
            messages: OpenAI-shaped history (system, user, assistant with
                ``tool_calls``, tool).
            tools: Tools on offer. Empty omits the ``tools`` key, which forces a
                text answer.

        Returns:
            The message text (``""`` when the model only called tools) and the
            requested calls, in order.
        """
        text, requests, _ = self.get_tool_response_meta(messages, tools)
        return text, requests

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Run one chat turn that may request tools, and keep its metadata.

        The request carries ``model``, ``messages``, ``max_tokens`` and the tools;
        sampling parameters (``temperature``, ``top_p``, ``stop``, ...) are sent
        only when set in ``call_kwargs``, never as defaults, so a route that
        refuses them still serves the call. One ``GENERATE`` call is recorded,
        with the provider's exact token counts when it reports them.

        A response without choices is asked again, with the attempt count and
        backoff of an empty plain answer, and every paid attempt is summed into
        the meta.

        Args:
            messages: OpenAI-shaped history (system, user, assistant with
                ``tool_calls``, tool).
            tools: Tools on offer. Empty omits the ``tools`` key, which forces a
                text answer.

        Returns:
            The message text (``""`` when the model only called tools), the
            requested calls in order, and the turn's metadata.

        Raises:
            EmptyResponse: If no attempt's response had a choice.
        """
        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        kwargs.update(self._call_kwargs)
        if tools:
            kwargs["tools"] = [spec.to_openai() for spec in tools]
        started = time.perf_counter()
        try:
            completion, meta = self._tool_completion(kwargs)
        except EmptyResponse as error:  # a failed _chat records itself; a turn without choices is recorded here
            self._record(
                CallRecord(
                    operation=GENERATE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        message = completion.choices[0].message
        text = message.content or ""
        requests = [
            ToolCallRequest(call.id, call.function.name, call.function.arguments or "{}")
            for call in message.tool_calls or []
            if call.type == "function"
        ]
        if meta.tokens is not None:  # the provider's counts, when it sent them
            input_tokens, output_tokens = meta.tokens.input_tokens, meta.tokens.output_tokens
        else:
            input_tokens = sum(approximate_tokens(str(item.get("content") or "")) for item in messages)
            output_tokens = approximate_tokens(text) + sum(approximate_tokens(r.arguments) for r in requests)
        self.input_token_count += input_tokens
        self.output_token_count += output_tokens
        self._record(
            CallRecord(
                operation=GENERATE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        return text, requests, meta

    def generate_with_meta(self, prompt: str, system_prompt: str = DEFAULT_SYS_PROMPT) -> tuple[str, CallMeta]:
        """Answer one prompt with promptolution's exact request, and keep its metadata.

        The request is built as promptolution's ``_ainvoke_once`` builds it, key
        for key and in the same order, so the bytes on the wire are the same.
        Token counters and the call log read as after :meth:`get_response`.

        Retries are xmas's: a 408, 409, 429, 5xx, connection error or timeout is
        retried with promptolution's attempt count and backoff, and any other
        4xx, or a 4xx whose text names money, is raised at once. An empty answer is asked again the same way,
        unless the model was cut by ``max_tokens`` or refused, which the same
        request would only repeat. Every paid attempt is summed into the meta.

        Args:
            prompt: The user message.
            system_prompt: The system message.

        Returns:
            The answer text and the metadata of the attempts it took.

        Raises:
            EmptyResponse: If the answer has no content: at once, with kind
                ``length_cut`` or ``refusal``, when the model was cut or
                refused; otherwise after the last attempt.
            TimeoutError: If the call outlives ``gather_timeout_s + 5``.
        """
        messages = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(prompt)},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        kwargs.update(self._call_kwargs)

        before = self.get_token_count()
        started = time.perf_counter()
        try:
            text, meta = self._answer(kwargs)
        except EmptyResponse as error:  # a failed _chat records itself; an empty answer is recorded here
            self._record(
                CallRecord(
                    operation=GENERATE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        self.update_token_count([prompt, system_prompt], [text])
        after = self.get_token_count()
        self._record(
            CallRecord(
                operation=GENERATE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
            )
        )
        return text, meta

    def _answer(self, kwargs: dict[str, Any]) -> tuple[str, CallMeta]:
        """Send one plain request until it has content, summing the paid attempts.

        Args:
            kwargs: The request.

        Returns:
            The content and the combined meta.

        Raises:
            EmptyResponse: As described in :meth:`generate_with_meta`, with the
                paid attempts' meta.
        """
        paid: CallMeta | None = None
        for attempt in range(self.max_retries + 1):
            completion, latency = self._chat(kwargs, "model call")
            meta = meta_from_completion(completion, latency, self.price)
            paid = meta if paid is None else paid.combine(meta)
            # no choices at all is an empty answer too, asked again as promptolution does
            content = completion.choices[0].message.content if completion.choices else None
            if content is not None:
                return content, paid
            if meta.error_kind is not None:
                raise EmptyResponse(
                    f"Empty content from model (finish_reason={meta.finish_reason}, refused={meta.refused})",
                    kind=meta.error_kind,
                    meta=paid,
                )
            if attempt < self.max_retries:
                time.sleep(self.retry_base_delay_s * (2**attempt))
        raise EmptyResponse(f"Empty content from model after {self.max_retries + 1} attempts", meta=paid)

    def _tool_completion(self, kwargs: dict[str, Any]) -> tuple[ChatCompletion, CallMeta]:
        """Send one tool turn until its response has a choice, summing the paid attempts.

        Args:
            kwargs: The request.

        Returns:
            The completion and the combined meta.

        Raises:
            EmptyResponse: After the last attempt, with the kind of the last
                finish reason and the paid attempts' meta.
        """
        paid: CallMeta | None = None
        for attempt in range(self.max_retries + 1):
            completion, latency = self._chat(kwargs, "tool turn")
            meta = meta_from_completion(completion, latency, self.price)
            paid = meta if paid is None else paid.combine(meta)
            if completion.choices:
                return completion, paid
            if attempt < self.max_retries:
                time.sleep(self.retry_base_delay_s * (2**attempt))
        kind = ErrorKind.UNKNOWN if paid is None or paid.error_kind is None else paid.error_kind
        raise EmptyResponse(
            f"Empty content from model after {self.max_retries + 1} attempts (no choices in a tool turn)",
            kind=kind,
            meta=paid,
        )

    def _chat(self, kwargs: dict[str, Any], what: str) -> tuple[ChatCompletion, float]:
        """Send one chat-completions request on the model's loop, with retries.

        Runs on promptolution's background loop and semaphore with its per-call
        timeout and backoff; a permanent 4xx is raised at once. A failure is
        recorded as one failed ``GENERATE`` call and re-raised.

        Args:
            kwargs: The request, passed to ``client.chat.completions.create``.
            what: How a timeout names the call, e.g. ``"tool turn"``.

        Returns:
            The completion and the wall seconds it took.

        Raises:
            TimeoutError: If the call outlives ``gather_timeout_s + 5``.
        """
        started = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(_create_with_retries(self, kwargs), self._loop)
        try:
            try:
                completion = future.result(timeout=self.gather_timeout_s + 5.0)
            except FuturesTimeout:
                future.cancel()
                raise TimeoutError(f"{what} timed out after {self.gather_timeout_s + 5.0}s") from None
        except Exception as error:
            self._record(
                CallRecord(
                    operation=GENERATE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        return completion, time.perf_counter() - started

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def _score_endpoint(self) -> str:
        """Return the absolute URL of the scoring endpoint."""
        return f"{self.base_url}/{SCORE_PATH}"

    def _headers(self) -> dict[str, str]:
        """Return the auth headers for a direct call."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        This is teacher forcing: the continuation is fixed and scored, not
        generated. It calls the gateway's scoring endpoint, which sits next to
        ``base_url`` and needs a backend that echoes prompt log-probabilities.

        Args:
            context: The text the continuation is conditioned on.
            continuation: The exact text to score.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            ScoringUnsupportedError: If the endpoint reports that the configured
                backend cannot echo prompt log-probabilities.
            ValueError: If the continuation does not align with token boundaries,
                or the request is otherwise rejected.
        """
        if not continuation:
            raise ValueError("Cannot score an empty continuation.")

        started = time.perf_counter()
        try:
            response = httpx.post(
                self._score_endpoint(),
                json={"model": self.model, "prompt": context, "continuation": continuation},
                headers=self._headers(),
                timeout=self.call_timeout_s,
            )
            scored = self._parse_score(response)
        except Exception as error:
            self._record(
                CallRecord(
                    operation=SCORE,
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        self._record(
            CallRecord(
                operation=SCORE,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                # A probe is billed for everything it sends, which is the prompt and
                # the continuation both, once per candidate-reference pair. It decodes
                # nothing, so it has no output tokens at all, and what it measures is
                # the continuation alone.
                input_tokens=approximate_tokens(context) + approximate_tokens(continuation),
                output_tokens=0,
                scored_tokens=scored.token_count,
            )
        )
        return scored

    def score_batch(self, contexts: list[str], continuations: list[str]) -> list[ScoredSequence]:
        """Score several continuations.

        Sequential for now: the endpoint has no batch form, and issuing
        concurrent requests against a metered gateway is a decision to take
        deliberately rather than inherit.

        Args:
            contexts: The conditioning texts.
            continuations: The texts to score, aligned to ``contexts``.

        Returns:
            One result per pair, in the order given.
        """
        pairs = zip(contexts, continuations, strict=True)
        return [self.score(context, continuation) for context, continuation in pairs]

    @staticmethod
    def _parse_score(response: httpx.Response) -> ScoredSequence:
        """Turn a scoring response into a :class:`ScoredSequence`.

        Args:
            response: The HTTP response.

        Returns:
            The parsed result.

        Raises:
            ScoringUnsupportedError: On a 501 capability refusal.
            RateLimitedError: On a 429, carrying ``Retry-After`` when the endpoint
                sent one.
            ValueError: On any other error status or a malformed body.
        """
        if response.status_code == httpx.codes.NOT_IMPLEMENTED:
            raise ScoringUnsupportedError(_error_message(response))
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise RateLimitedError(_error_message(response), retry_after=_retry_after(response))
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ValueError(f"Scoring failed ({response.status_code}): {_error_message(response)}")

        body = response.json()
        tokens = body.get("tokens")
        logprobs = body.get("token_logprobs")
        if not isinstance(tokens, list) or not isinstance(logprobs, list):
            raise ValueError(f"Scoring response is missing tokens/token_logprobs: {body!r}")

        return ScoredSequence(
            tokens=tuple(str(token) for token in tokens),
            token_logprobs=tuple(float(value) for value in logprobs),
        )


_CALL_PATH = (
    "get_response",
    "_get_response",
    "_aget_batch",
    "_ainvoke_with_retries",
    "_ainvoke_once",
    "get_tool_response",
)
"""The old path's hooks: a class that overrides one keeps that path by default."""

_OWN_HOOKS = {name: getattr(APILLM, name) for name in _CALL_PATH}
"""The hooks as this module made them, so a later patch counts too, even on promptolution's class."""


def _overrides_call_path(cls: type) -> bool:
    """Whether ``cls`` replaces any of xmas :class:`APILLM`'s call hooks.

    Compared by identity, so an override that merely calls ``super()`` still
    counts: the class asked to see the calls. So does a hook patched onto
    ``APILLM`` or promptolution's ``APILLM`` after import, the way a test fakes
    the gateway.

    Args:
        cls: ``APILLM`` or a subclass.

    Returns:
        ``True`` when at least one hook is not ``APILLM``'s own.
    """
    return any(getattr(cls, name, None) is not _OWN_HOOKS[name] for name in _CALL_PATH)


def _retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` header in seconds, if the endpoint sent a usable one.

    Only the delta-seconds form is read. The HTTP-date form is legal and no
    endpoint in this stack sends it; parsing it speculatively would add a clock
    skew problem to a wait that is already conservative.

    Args:
        response: The HTTP response.

    Returns:
        The number of seconds to wait, or ``None``.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _error_message(response: httpx.Response) -> str:
    """Extract the endpoint's own explanation, falling back to the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and "message" in error:
        return str(error["message"])
    return str(body)
