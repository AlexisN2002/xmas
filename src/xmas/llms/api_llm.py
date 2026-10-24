"""API-backed model with teacher-forced scoring."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import time
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import ContextVar
from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from promptolution.llms import api_llm as promptolution
from promptolution.utils.templates import DEFAULT_SYS_PROMPT

from xmas.errors import MONEY_WORDS, EmptyResponse, XmasError
from xmas.llms.call_log import CallLog, CallRecord
from xmas.llms.meta import CallMeta, TokenPrice, meta_from_completion
from xmas.llms.parts import Part, has_media, record_text, to_content
from xmas.llms.scoring import ScoredSequence
from xmas.tools import Tool, ToolCallRequest, _turn_usage
from xmas.trace import TokenUsage
from xmas.utils import approximate_tokens, error_message, retry_after_seconds, timestamp

# promptolution's _ainvoke_with_retries returns only the text, so the provider's
# completion id travels to CallRecord.response_id(s) in this per-request slot (each
# prompt of a batch runs as its own task). Delete it, _Completion and the two hook
# overrides once promptolution returns the completion.
_completion_id: ContextVar[list[str | None] | None] = ContextVar("xmas_completion_id", default=None)

_IMAGE_MODALITIES = ("image", "text")
"""What an image request asks for: OpenRouter's ``modalities``, images first."""

_DATA_URI = re.compile(r"data:([^;,]+);base64,(.*)", re.DOTALL)


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

    :meth:`get_response` is promptolution's batching, concurrency and retries,
    timed and logged. :meth:`generate_with_meta` sends the same request through
    ``_ainvoke_once`` and keeps what the provider reported (tokens, cost, finish
    reason, served model) as a :class:`~xmas.llms.meta.CallMeta`;
    :meth:`get_tool_response_meta` does the same for a tool turn,
    :meth:`generate_parts_with_meta` for a prompt with pictures or sound in it,
    and :meth:`generate_image_with_meta` for an answer with images. Each sends a
    permanent client error once (see :meth:`tool_turn_once`) and asks an empty
    answer again with promptolution's attempt count and backoff, unless the model
    was cut or refused; every paid attempt is summed into the meta.

    Attributes:
        full_response: Whether the engine reads this model through
            :meth:`generate_with_meta` and :meth:`get_tool_response_meta`.
        price: The price used for a call whose response reports no cost.
        accepts: The kinds the model reads, as stated: ``"text"``, and
            ``"image"`` or ``"audio"`` for a model that sees or hears.
        produces: The kinds the model answers with, as stated: ``"text"``, and
            ``"image"`` for a model that draws.
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
        accepts: Sequence[str] = ("text",),
        produces: Sequence[str] = ("text",),
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model identifier, e.g. ``"llama-3.1-8b"``; promptolution's ``model_id``.
            base_url: Endpoint base URL, promptolution's ``api_url``. Defaults to ``XMAS_BASE_URL``.
            api_key: API key. Defaults to ``XMAS_API_KEY``.
            call_log: Collector for per-call statistics; ``None`` records nothing.
            backend_tag: What this backend calls itself in a record.
            full_response: ``True`` sends plain agents through
                :meth:`generate_with_meta`, ``False`` keeps promptolution's path,
                and ``None`` decides by class, once, at construction: on, unless
                the class overrides one of promptolution's call hooks or
                ``get_tool_response``, so a subclass keeps the path it hooked.
            price: USD per million tokens, used only when a response reports
                no cost. ``None`` leaves such a cost unknown.
            accepts: The kinds the model reads; a MAS sends media only to a
                model that accepts their kind.
            produces: The kinds the model answers with.
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
        self.accepts = tuple(accepts)
        self.produces = tuple(produces)
        api_key = api_key or os.environ.get("XMAS_API_KEY")
        super().__init__(model_id=model, api_url=base_url.rstrip("/"), api_key=api_key, **kwargs)

    def record(self, record: CallRecord) -> None:
        """Hand one call to the collector, if there is one."""
        if self.call_log is not None:
            self.call_log.record(record)

    def _log(
        self, operation: str, started: float, started_at: str, *, error: BaseException | None = None, **fields: Any
    ) -> None:
        """Record one call that started at ``started``; ``error`` marks it failed."""
        if error is not None:
            fields.update(ok=False, error=f"{type(error).__name__}: {error}")
        self.record(
            CallRecord(
                operation=operation,
                model=self.model,
                backend=self.backend_tag,
                latency_s=time.perf_counter() - started,
                ts=started_at,
                **fields,
            )
        )

    def _log_answer(
        self, started: float, started_at: str, meta: CallMeta, estimate: TokenUsage, **fields: Any
    ) -> TokenUsage:
        """Record an answered ``generate`` call with the provider's token counts, else ``estimate``; return them."""
        tokens = estimate if meta.tokens is None else meta.tokens
        fields.setdefault("params", self.call_params())
        self._log(
            "generate",
            started,
            started_at,
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            **fields,
        )
        return tokens

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
        """Run promptolution's retry loop and tag the answer with the id of the attempt that returned it."""
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
        """Generate through promptolution, timing the call and recording what it cost.

        Token counts are promptolution's whitespace-word approximation. A batch
        is one call and stays one record.
        """
        n_prompts = 1 if isinstance(prompts, str) else len(prompts)
        params = self.call_params()
        before = self.get_token_count()
        started_at = timestamp()
        started = time.perf_counter()
        try:
            tagged = super().get_response(prompts, system_prompts)
        except Exception as error:
            self._log(
                "generate",
                started,
                started_at,
                error=error,
                n_prompts=n_prompts,
                params=params,
                **self._provenance(prompts, system_prompts, None, None),
            )
            raise
        after = self.get_token_count()
        response_ids = [getattr(response, "completion_id", None) for response in tagged]
        responses = [str(response) for response in tagged]  # plain strings, not the tagged subclass
        self._log(
            "generate",
            started,
            started_at,
            input_tokens=after["input_tokens"] - before["input_tokens"],
            output_tokens=after["output_tokens"] - before["output_tokens"],
            n_prompts=n_prompts,
            params=params,
            **self._provenance(prompts, system_prompts, responses, response_ids),
        )
        return responses

    def _provenance(
        self,
        prompts: str | list[str],
        system_prompts: str | list[str] | None,
        responses: list[str] | None,
        response_ids: list[str | None] | None,
    ) -> dict[str, Any]:
        """Return a generation record's id fields, and its text when the collector asked for it.

        Singular fields for a one-prompt call, list fields for a batch; the
        system prompts are resolved as promptolution resolves them.
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
        """:meth:`get_tool_response_meta` without the metadata."""
        text, requests, _ = self.get_tool_response_meta(messages, tools)
        return text, requests

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Run one chat turn that may request tools, and keep its metadata.

        The request carries ``model``, ``messages``, ``max_tokens`` and the
        tools; sampling parameters are sent only when set in ``call_kwargs``,
        so a route that refuses them still serves the call. Empty ``tools``
        omits the key, which forces a text answer.

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
        started_at = timestamp()
        started = time.perf_counter()
        try:
            completion, meta = self._ask(lambda: self.tool_turn_once(kwargs), "tool turn", tool_turn=True)
        except Exception as error:
            self._log("generate", started, started_at, error=error, params=self.call_params())
            raise
        message = completion.choices[0].message
        text = message.content or ""
        requests = [
            ToolCallRequest(call.id, call.function.name, call.function.arguments or "{}")
            for call in message.tool_calls or []
            if call.type == "function"
        ]
        estimate = _turn_usage(approximate_tokens, messages, text, requests)
        tokens = self._log_answer(started, started_at, meta, estimate, response_id=completion.id)
        self.input_token_count += tokens.input_tokens
        self.output_token_count += tokens.output_tokens
        return text, requests, meta

    def generate_with_meta(self, prompt: str, system_prompt: str = DEFAULT_SYS_PROMPT) -> tuple[str, CallMeta]:
        """Answer one prompt with promptolution's own request, and keep its metadata.

        One ``generate`` record per call, failed or not, whatever its attempts;
        the token counters move as after :meth:`get_response`.

        Raises:
            EmptyResponse: If the answer has no content: at once, with kind
                ``length_cut`` or ``refusal``, when the model was cut or
                refused; otherwise after the last attempt.
            TimeoutError: If the call outlives ``gather_timeout_s + 5``.
        """
        _, text, meta = self._generate(
            lambda: self._retrying(lambda: self._ainvoke_once(prompt, system_prompt)), prompt, system_prompt
        )
        return text, meta

    def generate_parts_with_meta(
        self, parts: Sequence[Part], system_prompt: str = DEFAULT_SYS_PROMPT
    ) -> tuple[str, CallMeta]:
        """Answer a prompt of text and media parts, as :meth:`generate_with_meta` answers text.

        The request has the same keys (``model``, ``messages``, ``max_tokens``,
        then ``call_kwargs``); the user content is the parts'
        :func:`~xmas.llms.parts.to_content`, or the plain string when no part is
        media. The record's prompt shows a marker per ref, never its bytes.

        Raises:
            ValueError: If a part cannot be sent (see :func:`~xmas.llms.parts.to_content`).
            EmptyResponse: As :meth:`generate_with_meta`.
            TimeoutError: As :meth:`generate_with_meta`.
        """
        request = self._request(parts, system_prompt)
        _, text, meta = self._generate(lambda: self.tool_turn_once(request), parts, system_prompt)
        return text, meta

    def generate_image_with_meta(
        self, prompt: str | Sequence[Part], system_prompt: str = DEFAULT_SYS_PROMPT
    ) -> tuple[str, list[tuple[bytes, str]], CallMeta]:
        """Ask for an answer with images, and keep its metadata.

        The request adds ``"modalities": ["image", "text"]`` after
        ``max_tokens`` (``call_kwargs`` still override it). The images are read
        from the message's ``images`` list of ``data:`` URIs, as OpenRouter
        sends them; images without text are an answer, not an empty one.

        Returns:
            The answer's text (``""`` when it has none), each image as
            ``(bytes, media type)`` in order, and the call's metadata.

        Raises:
            XmasError: If an image is not a base64 ``data:`` URI.
            ValueError: If a part cannot be sent (see :func:`~xmas.llms.parts.to_content`).
            EmptyResponse: If the answer has neither text nor images, as :meth:`generate_with_meta`.
            TimeoutError: As :meth:`generate_with_meta`.
        """
        request = self._request(prompt, system_prompt, modalities=list(_IMAGE_MODALITIES))
        params = self.call_params()
        params.setdefault("modalities", list(_IMAGE_MODALITIES))
        completion, text, meta = self._generate(
            lambda: self.tool_turn_once(request), prompt, system_prompt, images=True, params=params
        )
        return text, [_decode_image(url) for url in _image_urls(completion.choices[0].message)], meta

    def _request(self, prompt: str | Sequence[Part], system_prompt: str, **extra: Any) -> dict[str, Any]:
        """Return promptolution's request for ``prompt``, with ``extra`` keys before ``call_kwargs``."""
        content: str | list[dict[str, Any]]
        if isinstance(prompt, str) or not has_media(prompt):
            content = prompt if isinstance(prompt, str) else record_text(prompt)
        else:
            content = to_content(prompt)
        messages = [{"role": "system", "content": str(system_prompt)}, {"role": "user", "content": content}]
        request: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": self.max_tokens}
        request.update(extra)
        request.update(self._call_kwargs)
        return request

    def _generate(
        self,
        send: Callable[[], Coroutine[Any, Any, ChatCompletion]],
        prompt: str | Sequence[Part],
        system_prompt: str,
        *,
        images: bool = False,
        params: dict[str, Any] | None = None,
    ) -> tuple[ChatCompletion, str, CallMeta]:
        """Send one generation, record it, and move the token counters by its text parts."""
        shown = prompt if isinstance(prompt, str) else record_text(prompt)
        counted = prompt if isinstance(prompt, str) else "".join(part for part in prompt if isinstance(part, str))
        logged = {"params": self.call_params() if params is None else params}
        started_at = timestamp()
        started = time.perf_counter()
        try:
            completion, meta = self._ask(send, "model call", images=images)
        except Exception as error:
            fields = self._provenance(shown, system_prompt, None, None)
            self._log("generate", started, started_at, error=error, **logged, **fields)
            raise
        text = completion.choices[0].message.content or ""
        self.update_token_count([counted, system_prompt], [text])
        estimate = TokenUsage(approximate_tokens(counted) + approximate_tokens(system_prompt), approximate_tokens(text))
        fields = self._provenance(shown, system_prompt, [text], [completion.id])
        self._log_answer(started, started_at, meta, estimate, **logged, **fields)
        return completion, text, meta

    def _ask(
        self,
        send: Callable[[], Coroutine[Any, Any, ChatCompletion]],
        what: str,
        *,
        tool_turn: bool = False,
        images: bool = False,
    ) -> tuple[ChatCompletion, CallMeta]:
        """Send until the response has an answer, summing the paid attempts into one meta.

        A plain answer needs content, or with ``images`` an image; a tool turn
        needs a choice, and its text may be empty when it calls tools or
        finished cleanly. An empty answer
        that was cut or refused raises at once, since the same request would
        only repeat it.
        """
        paid: CallMeta | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            deadline = self.gather_timeout_s + 5.0  # as long as promptolution's _get_response waits for a batch
            future = asyncio.run_coroutine_threadsafe(send(), self._loop)
            try:
                completion: ChatCompletion = future.result(timeout=deadline)
            except FuturesTimeout:
                future.cancel()
                raise TimeoutError(f"{what} timed out after {deadline}s") from None
            meta = meta_from_completion(completion, time.perf_counter() - started, self.price)
            paid = meta if paid is None else paid.combine(meta)
            if completion.choices:
                message = completion.choices[0].message
                empty = (
                    message.content is None
                    and not (tool_turn and message.tool_calls)
                    and not (images and _image_urls(message))
                )
                if empty and meta.error_kind is not None:
                    raise EmptyResponse(
                        f"Empty content from model (finish_reason={meta.finish_reason}, refused={meta.refused})",
                        kind=meta.error_kind,
                        meta=paid,
                    )
                if tool_turn or not empty:
                    return completion, paid
            if attempt < self.max_retries:
                time.sleep(self.retry_base_delay_s * (2**attempt))
        raise EmptyResponse(f"Empty content from model after {self.max_retries + 1} attempts", meta=paid)

    async def _retrying(self, send: Callable[[], Coroutine[Any, Any, ChatCompletion]]) -> ChatCompletion:
        """Await ``send()``, retrying with promptolution's backoff unless the error is permanent."""
        attempt = 0
        while True:
            try:
                return await send()
            except Exception as exc:
                if _is_permanent(exc) or attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_base_delay_s * (2**attempt))
                attempt += 1

    async def tool_turn_once(self, kwargs: dict[str, Any]) -> ChatCompletion:
        """One chat-completions call with promptolution's timeout and backoff.

        A permanent client error is raised at once instead of being retried: any
        4xx but 408 request timeout, 409 conflict and 429 rate limit, and any 4xx
        whose message or body names money (:data:`~xmas.errors.MONEY_WORDS`: 402,
        403 "Key limit exceeded", 429 "insufficient_quota"), which asking again
        only repeats. Connection errors, timeouts, 5xx and a plain 429 are retried.
        """

        async def send() -> ChatCompletion:
            async with self._sem:
                completion: ChatCompletion = await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs), timeout=self.call_timeout_s
                )
            return completion

        return await self._retrying(send)

    @property
    def model(self) -> str:
        """The model identifier, under the name the xmas surface uses."""
        return str(self.model_id)

    def score(self, context: str, continuation: str) -> ScoredSequence:
        """Return the log-probability the model assigns to ``continuation``.

        Teacher forcing: the continuation is fixed and scored, not generated, through
        the gateway's ``/score`` endpoint next to ``base_url``.

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
            self._log("score", started, started_at, error=error)
            raise
        # A probe is billed for the prompt and the continuation both, decodes
        # nothing, and measures the continuation alone.
        self._log(
            "score",
            started,
            started_at,
            input_tokens=approximate_tokens(context) + approximate_tokens(continuation),
            output_tokens=0,
            scored_tokens=scored.token_count,
        )
        return scored


def _image_urls(message: ChatCompletionMessage) -> list[str]:
    """Return the image URLs of an answer's ``images`` list, which the client keeps as an extra field."""
    entries = (message.model_extra or {}).get("images") or []
    urls = []
    for entry in entries:
        image_url = entry.get("image_url") if isinstance(entry, dict) else None
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(url, str):
            urls.append(url)
    return urls


def _decode_image(url: str) -> tuple[bytes, str]:
    """Return a ``data:`` URI's bytes and media type.

    Raises:
        XmasError: For any other URL, which xmas would have to fetch, or invalid base64.
    """
    found = _DATA_URI.fullmatch(url)
    if found is None:
        raise XmasError(f"llm: an image answer is not a base64 data URI ({url[:40]}...); xmas reads no network")
    try:
        return base64.b64decode(found.group(2), validate=True), found.group(1)
    except binascii.Error as exc:
        raise XmasError(f"llm: an image answer's base64 is invalid: {exc}") from None


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
"""The hooks whose override keeps a class on promptolution's path by default."""


def _overrides_call_path(cls: type) -> bool:
    """Whether ``cls`` replaces a hook of :data:`_CALL_PATH`."""
    return any(getattr(cls, name) is not getattr(APILLM, name) for name in _CALL_PATH)


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
