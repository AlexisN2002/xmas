"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import ContextVar
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion
from promptolution.llms import api_llm as promptolution
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import MONEY_WORDS, EmptyResponse, ErrorKind, XmasError
from xmas.llms.call_log import CallLog, CallRecord
from xmas.llms.meta import CallMeta, TokenPrice, meta_from_completion
from xmas.llms.scoring import ScoredSequence
from xmas.tools import Tool, ToolCallRequest
from xmas.utils import approximate_tokens, error_message, retry_after_seconds, timestamp

# Completion-id plumbing: _completion_id, _Completion and the _ainvoke_once /
# _ainvoke_with_retries overrides below exist only because promptolution's
# _ainvoke_with_retries returns the response text and drops the ChatCompletion,
# and with it the provider's completion id that CallRecord.response_id(s) carry.
# Temporary: delete all of it once promptolution returns the completion (or its id).
_completion_id: ContextVar[list[str | None] | None] = ContextVar("xmas_completion_id", default=None)
"""Per-request slot for the provider's completion id.

promptolution keeps only a completion's text, and each prompt of a batch runs as its
own asyncio task, so a context variable is the one place the id can travel from the
request that saw it to the response it belongs to without reimplementing the batch.
"""


class _Completion(str):
    """A response string that remembers which provider completion it came from."""

    completion_id: str | None

    def __new__(cls, text: str, completion_id: str | None) -> _Completion:
        """Wrap ``text``, attaching ``completion_id``."""
        instance = super().__new__(cls, text)
        instance.completion_id = completion_id
        return instance


class APILLM(promptolution.APILLM):
    """promptolution's APILLM plus :meth:`score`, tool turns, call metadata and a per-call log.

    ``model`` and ``base_url`` are forwarded to promptolution's ``model_id`` and ``api_url``.

    :meth:`get_response` is promptolution's: batching, concurrency, retries and
    token metering live there. :meth:`generate_with_meta` is a second plain path
    that sends the very same request but keeps what the provider reported
    (tokens, cost, finish reason, served model) as a :class:`~xmas.llms.meta.CallMeta`;
    :meth:`get_tool_response_meta` does the same for a tool turn.

    The engine takes the metadata path when ``full_response`` is ``True``. By
    default it is, unless a subclass overrides a hook of the old path
    (``get_response``, ``_get_response``, ``_aget_batch``,
    ``_ainvoke_with_retries``, ``_ainvoke_once`` or ``get_tool_response``): such
    a subclass keeps the path it hooked, and its own accounting with it.

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
        call_log: CallLog | None = None,
        backend_tag: str = "gateway",
        full_response: bool | None = None,
        price: TokenPrice | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``.
            base_url: Endpoint base URL. Defaults to ``XMAS_BASE_URL``.
            api_key: API key. Defaults to ``XMAS_API_KEY``.
            call_log: Collector for per-call statistics; ``None`` records nothing.
            backend_tag: What this backend calls itself in a record.
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
        base_url = base_url or os.environ.get("XMAS_BASE_URL")
        if not model:
            raise ValueError("APILLM needs a model, e.g. APILLM(model='llama-3.1-8b').")
        if not base_url:
            raise ValueError("APILLM needs a base_url, or XMAS_BASE_URL set in the environment.")
        self.call_log = call_log
        self.backend_tag = backend_tag
        self.full_response: bool = not _overrides_call_path(type(self)) if full_response is None else full_response
        self.price = price
        api_key = api_key or os.environ.get("XMAS_API_KEY")
        super().__init__(model_id=model, api_url=base_url.rstrip("/"), api_key=api_key, **kwargs)

    def record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.call_log is not None:
            self.call_log.record(record)

    def call_params(self) -> dict[str, Any]:
        """Return what a generation is sent besides the messages, merged as promptolution merges it."""
        params: dict[str, Any] = {"max_tokens": self.max_tokens}
        params.update(self._call_kwargs)
        # Headers can carry credentials, so they never reach a record.
        params.pop("extra_headers", None)
        return params

    async def _ainvoke_once(self, prompt: str, system_prompt: str) -> ChatCompletion:
        """Make one request, noting the provider's completion id for the caller."""
        completion = await super()._ainvoke_once(prompt, system_prompt)
        slot = _completion_id.get()
        if slot is not None:
            slot[0] = completion.id
        return completion

    async def _ainvoke_with_retries(self, prompt: str, system_prompt: str) -> str:
        """Run promptolution's retry loop and tag the answer with its completion id.

        The slot is overwritten by every attempt that reaches the provider, and the
        attempt that returns is the last one, so the id that survives belongs to the
        text that is returned.
        """
        slot: list[str | None] = [None]
        token = _completion_id.set(slot)
        try:
            content = await super()._ainvoke_with_retries(prompt, system_prompt)
        finally:
            _completion_id.reset(token)
        return _Completion(content, slot[0])

    def get_response(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None = None,
    ) -> list[str]:
        """Generate, timing the call and recording what it cost.

        The generation itself is promptolution's; this adds the measurement.
        Token counts are read from the parent's own counters, which split on
        whitespace, so they are an approximation and are reported as one.

        A batch is one call and stays one record. Its provenance (start time,
        sampling parameters, completion ids) is always recorded; its text only
        when the collector was built with ``capture_text=True``.

        Args:
            prompts: Prompt or prompts to answer.
            system_prompts: System prompt or prompts.

        Returns:
            One response per prompt.
        """
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        params = self.call_params()
        before = self.get_token_count()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            tagged = super().get_response(prompts, system_prompts)
        except Exception as error:
            self.record(
                CallRecord(
                    operation="generate",
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    n_prompts=n_prompts,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    ts=started_at,
                    params=params,
                    **self._provenance(prompts, system_prompts, None, None),
                )
            )
            raise
        latency = time.perf_counter() - started
        after = self.get_token_count()
        response_ids = [getattr(response, "completion_id", None) for response in tagged]
        # Hand the caller ordinary strings, not the tagged subclass.
        responses = [str(response) for response in tagged]
        self.record(
            CallRecord(
                operation="generate",
                model=self.model,
                backend=self.backend_tag,
                latency_s=latency,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
                n_prompts=n_prompts,
                ts=started_at,
                params=params,
                **self._provenance(prompts, system_prompts, responses, response_ids),
            )
        )
        return responses

    def _provenance(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None,
        responses: list[str] | None,
        response_ids: list[str | None] | None,
    ) -> dict[str, Any]:
        """Return the text and id fields of a generation record.

        Args:
            prompts: The prompts as the caller passed them.
            system_prompts: The system prompts as the caller passed them, resolved
                here exactly as promptolution resolves them before sending.
            responses: The responses, or ``None`` when the call failed.
            response_ids: The provider's completion ids, aligned to ``responses``.

        Returns:
            Keyword arguments for :class:`CallRecord`: singular fields for a
            one-prompt call, list fields for a batch. The text is present only when
            the collector asked for it.
        """
        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        known_ids = response_ids if response_ids and any(i is not None for i in response_ids) else None
        capture = self.call_log is not None and self.call_log.capture_text
        if system_prompts is None:
            system_prompts = DEFAULT_SYS_PROMPT
        system_list = [system_prompts] * len(prompt_list) if isinstance(system_prompts, str) else list(system_prompts)

        if len(prompt_list) == 1:
            fields: dict[str, Any] = {"response_id": known_ids[0] if known_ids else None}
            if capture:
                fields["prompt"] = prompt_list[0]
                fields["system_prompt"] = system_list[0] if system_list else None
                fields["response"] = responses[0] if responses else None
            return fields
        fields = {"response_ids": known_ids}
        if capture:
            fields["prompts"] = prompt_list
            fields["system_prompts"] = system_list
            fields["responses"] = responses
        return fields

    def get_tool_response(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest]]:
        """Run one chat turn that may request tools.

        :meth:`get_tool_response_meta` without the metadata.

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

        Scheduled on promptolution's background loop and semaphore, with its
        timeout and backoff (see :meth:`tool_turn_once`), and recorded like
        :meth:`get_response`. The request carries ``model``, ``messages``,
        ``max_tokens`` and the tools; sampling parameters (``temperature``,
        ``top_p``, ``stop``, ...) are sent only when set in ``call_kwargs``, never
        as defaults, so a route that refuses them still serves the call. Token
        counts are the provider's when it reports usage.

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
            EmptyResponse: If no attempt's response had a choice, or the model
                was cut or refused before it wrote any text or call.
        """
        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        kwargs.update(self._call_kwargs)
        if tools:
            kwargs["tools"] = [spec.to_openai() for spec in tools]
        params = self.call_params()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            completion, meta = self._tool_completion(kwargs)
        except Exception as error:
            self.record(self._failed(error, started, started_at, params))
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
        self.record(
            CallRecord(
                operation="generate",
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ts=started_at,
                params=params,
                response_id=completion.id,
            )
        )
        return text, requests, meta

    def generate_with_meta(self, prompt: str, system_prompt: str = DEFAULT_SYS_PROMPT) -> tuple[str, CallMeta]:
        """Answer one prompt with promptolution's exact request, and keep its metadata.

        The request is built as promptolution's ``_ainvoke_once`` builds it, key
        for key and in the same order, so the bytes on the wire are the same.
        Token counters and the call log read as after :meth:`get_response`: one
        ``generate`` record per call, failed or not, whatever its attempts.

        Retries are xmas's (see :meth:`tool_turn_once`): a 408, 409, 429, 5xx,
        connection error or timeout is retried with promptolution's attempt count
        and backoff, and any other 4xx, or a 4xx whose text names money, is sent
        once. An empty answer is asked again the same way, unless the model was
        cut by ``max_tokens`` or refused, which the same request would only
        repeat. Every paid attempt is summed into the meta.

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

        params = self.call_params()
        before = self.get_token_count()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            text, meta, completion_id = self._answer(kwargs)
        except Exception as error:
            self.record(
                self._failed(error, started, started_at, params, **self._provenance(prompt, system_prompt, None, None))
            )
            raise
        self.update_token_count([prompt, system_prompt], [text])
        after = self.get_token_count()
        self.record(
            CallRecord(
                operation="generate",
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                input_tokens=after["input_tokens"] - before["input_tokens"],
                output_tokens=after["output_tokens"] - before["output_tokens"],
                ts=started_at,
                params=params,
                **self._provenance(prompt, system_prompt, [text], [completion_id]),
            )
        )
        return text, meta

    def _answer(self, kwargs: dict[str, Any]) -> tuple[str, CallMeta, str]:
        """Send one plain request until it has content, summing the paid attempts.

        Args:
            kwargs: The request.

        Returns:
            The content, the combined meta, and the completion id of the attempt
            that answered.

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
                return content, paid, completion.id
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
                finish reason and the paid attempts' meta; at once, with kind
                ``length_cut`` or ``refusal``, when the model was cut or refused
                before it wrote any text or call, as a plain answer is.
        """
        paid: CallMeta | None = None
        for attempt in range(self.max_retries + 1):
            completion, latency = self._chat(kwargs, "tool turn")
            meta = meta_from_completion(completion, latency, self.price)
            paid = meta if paid is None else paid.combine(meta)
            if completion.choices:
                message = completion.choices[0].message
                if message.content is None and not message.tool_calls and meta.error_kind is not None:
                    raise EmptyResponse(
                        f"Empty content from model (finish_reason={meta.finish_reason}, refused={meta.refused})",
                        kind=meta.error_kind,
                        meta=paid,
                    )
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
        """Send one chat-completions request on the model's loop, with :meth:`tool_turn_once`'s retries.

        Args:
            kwargs: The request, passed to ``client.chat.completions.create``.
            what: How a timeout names the call, e.g. ``"tool turn"``.

        Returns:
            The completion and the wall seconds it took.

        Raises:
            TimeoutError: If the call outlives ``gather_timeout_s + 5``.
        """
        started = time.perf_counter()
        # Five seconds of slack over the batch timeout, so promptolution's own timeout fires first.
        deadline = self.gather_timeout_s + 5.0
        future = asyncio.run_coroutine_threadsafe(self.tool_turn_once(kwargs), self._loop)
        try:
            completion = future.result(timeout=deadline)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(f"{what} timed out after {deadline}s") from None
        return completion, time.perf_counter() - started

    def _failed(
        self, error: BaseException, started: float, started_at: str, params: dict[str, Any], **fields: Any
    ) -> CallRecord:
        """Return the record of a failed ``generate`` call."""
        return CallRecord(
            operation="generate",
            model=self.model,
            backend=self.backend_tag,
            latency_s=time.perf_counter() - started,
            ok=False,
            error=f"{type(error).__name__}: {error}",
            ts=started_at,
            params=params,
            **fields,
        )

    async def tool_turn_once(self, kwargs: dict[str, Any]) -> ChatCompletion:
        """One chat-completions call with promptolution's timeout and backoff.

        A permanent client error is raised at once instead of being retried: any
        4xx but 408 request timeout, 409 conflict and 429 rate limit, and any 4xx
        whose message or body names money (:data:`~xmas.errors.MONEY_WORDS`: 402,
        403 "Key limit exceeded", 429 "insufficient_quota"), which asking again
        only repeats. Connection errors, timeouts, 5xx and a plain 429 are retried.
        """
        attempt = 0
        while True:
            try:
                async with self._sem:
                    completion: ChatCompletion = await asyncio.wait_for(
                        self.client.chat.completions.create(**kwargs), timeout=self.call_timeout_s
                    )
                return completion
            except Exception as exc:
                if _is_permanent(exc) or attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_base_delay_s * (2**attempt))
                attempt += 1

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Teacher forcing: the continuation is fixed and scored, not generated, through
        the gateway's ``/score`` endpoint next to ``base_url``.

        Args:
            context: The text the continuation is conditioned on.
            continuation: The exact text to score.

        Returns:
            The continuation's per-token and aggregate log-probabilities.

        Raises:
            ValueError: If the continuation is empty.
            XmasError: If the endpoint refuses (501 unsupported, 429 rate limited,
                any other 4xx/5xx) or the body is malformed.
        """
        if not continuation:
            raise ValueError("Cannot score an empty continuation.")

        started_at = timestamp()
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.api_url}/score",
                json={"model": self.model, "prompt": context, "continuation": continuation},
                headers=headers,
                timeout=self.call_timeout_s,
            )
            scored = parse_score(response)
        except Exception as error:
            self.record(
                CallRecord(
                    operation="score",
                    model=self.model,
                    backend=self.backend_tag,
                    latency_s=time.perf_counter() - started,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    ts=started_at,
                )
            )
            raise
        self.record(
            CallRecord(
                operation="score",
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
                ts=started_at,
            )
        )
        return scored


def _is_permanent(exc: BaseException) -> bool:
    """Whether retrying cannot help: a 4xx other than 408, 409 and 429, or a 4xx that names money."""
    if not isinstance(exc, openai.APIStatusError) or not 400 <= exc.status_code < 500:
        return False
    text = str(exc) if exc.body is None else f"{exc} {exc.body}"
    return exc.status_code not in (408, 409, 429) or MONEY_WORDS.search(text) is not None


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
"""The hooks as this module made them, so a later patch counts too."""

_BASE_HOOKS = {name: getattr(promptolution.APILLM, name, None) for name in _CALL_PATH}
"""promptolution's hooks at import, so a patch of its class counts even where xmas's hook calls ``super()``."""


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
    return any(
        getattr(cls, name, None) is not _OWN_HOOKS[name]
        or getattr(promptolution.APILLM, name, None) is not _BASE_HOOKS[name]
        for name in _CALL_PATH
    )


def parse_score(response: httpx.Response) -> ScoredSequence:
    """Turn a scoring response into a :class:`ScoredSequence`.

    Raises:
        XmasError: On an error status or a body without tokens/token_logprobs.
    """
    status = response.status_code
    if status == 501:
        raise XmasError(f"llm: scoring unsupported (501): {error_message(response)}", status=501)
    if status == 429:
        retry_after = retry_after_seconds(response.headers)
        message = f"llm: rate limited (429), retry after {retry_after}s: {error_message(response)}"
        raise XmasError(message, status=429, retry_after=retry_after)
    if status >= 400:
        raise XmasError(f"llm: scoring failed ({status}): {error_message(response)}", status=status)
    body = response.json()
    tokens = body.get("tokens")
    logprobs = body.get("token_logprobs")
    if not isinstance(tokens, list) or not isinstance(logprobs, list):
        raise XmasError(f"llm: scoring response missing tokens/token_logprobs: {body!r}")
    return ScoredSequence(
        tokens=tuple(str(token) for token in tokens),
        token_logprobs=tuple(float(value) for value in logprobs),
    )
