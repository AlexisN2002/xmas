"""Shared test doubles: the scripted model, a recording tool source, a patched APILLM and the worked example.

The mock model is the reason CI can run the whole framework end to end without a
key: it satisfies promptolution's LLM contract exactly, so nothing under test has
to know it is not talking to a real endpoint.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion
from promptolution.llms.base_llm import BaseLLM

from xmas import MAS, Agent, TokenUsage
from xmas.edges import Edge, EdgeKind
from xmas.llms.api_llm import APILLM
from xmas.llms.meta import CallMeta
from xmas.nodes import Step
from xmas.tools import Tool, ToolCallRequest, ToolSource, tool

Turn = tuple[str, list[ToolCallRequest]]
"""One tool turn: the model's text and the calls it requests."""

META = CallMeta(latency_s=0.5, tokens=TokenUsage(10, 2), cost_usd=0.01)
"""What a ``MockLLM(meta=True)`` reports for every call."""


class MockLLM(BaseLLM):
    """A deterministic model that records every prompt it was given.

    It answers each prompt with ``responder`` (by default it echoes the prompt
    back, which makes substitution bugs visible in output), or with the next of
    ``answers``, raising an exception found there. Built with ``turns`` it also takes tool turns from that script,
    and ``default`` once the script runs out; built without, it has no
    ``get_tool_response``, as a plain promptolution model has none. With
    ``meta=True`` it reports :data:`META` for every call, so a meter sums cost.
    """

    get_tool_response: Callable[[list[dict[str, Any]], Sequence[Tool]], Turn]

    def __init__(
        self,
        responder: Callable[[str], str] | None = None,
        *,
        answers: list[str | Exception] | None = None,
        turns: list[Turn] | None = None,
        default: Turn = ("done", []),
        meta: bool = False,
    ) -> None:
        """Script the answers and, optionally, the tool turns and the call metadata."""
        super().__init__()
        self.answers = list(answers or [])
        self.responder = responder or (self._next_answer if answers is not None else lambda prompt: f"echo:{prompt}")
        self.calls: list[str] = []
        self.turns = list(turns or [])
        self.default = default
        self.seen: list[tuple[list[dict[str, Any]], list[str]]] = []
        self.full_response = meta
        if turns is not None:
            self.get_tool_response = self._tool_turn

    def _next_answer(self, prompt: str) -> str:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        """Return one scripted response per prompt."""
        self.calls.extend(prompts)
        return [self.responder(prompt) for prompt in prompts]

    def _tool_turn(self, messages: list[dict[str, Any]], tools: Sequence[Tool]) -> Turn:
        """Record a snapshot of the history and the tools offered, and answer from the script."""
        self.seen.append((json.loads(json.dumps(messages)), [t.name for t in tools]))
        return self.turns.pop(0) if self.turns else self.default

    def generate_with_meta(self, prompt: str, system_prompt: str = "") -> tuple[str, CallMeta]:
        """Record the prompt and answer it, with metadata."""
        self.calls.append(prompt)
        return self.responder(prompt), META

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, list[ToolCallRequest], CallMeta]:
        """Answer from the tool script, with metadata."""
        text, requests = self.get_tool_response(messages, tools)
        return text, requests, META


@pytest.fixture
def mock_llm() -> MockLLM:
    """A model that echoes its prompt."""
    return MockLLM()


def req(name: str, arguments: str = "{}", call_id: str | None = None) -> ToolCallRequest:
    """A requested tool call; its id defaults to ``call_<name>``."""
    return ToolCallRequest(call_id or f"call_{name}", name, arguments)


class FakeToolSource(ToolSource):
    """A ToolSource with scripted answers; records every call."""

    def __init__(
        self,
        names: Sequence[str] = ("run_python",),
        answers: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Offer ``names``; ``answers`` maps a name to its result, or to an exception to raise.

        ``properties`` is the JSON Schema ``properties`` every tool declares.
        """
        schema = {"type": "object", "properties": dict(properties or {})}
        self.specs = [Tool(name, f"{name} tool", schema) for name in names]
        self.answers = answers or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.listed = 0

    def tools(self) -> list[Tool]:
        """Return the tools, counting how often they were listed."""
        self.listed += 1
        return list(self.specs)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Record the call and answer it."""
        self.calls.append((name, arguments))
        answer = self.answers.get(name, f"{name} ok")
        if isinstance(answer, Exception):
            raise answer
        return str(answer)


@tool
def add(a: int, b: int = 2) -> int:
    """Add two integers.

    Longer prose that is not part of the description.

    Args:
        a: The first summand.
        b: The second summand,
            continued on a second line.

    Returns:
        The sum.
    """
    return a + b


# ------------------------------------------------------------ APILLM doubles

BASE_URL = "https://gateway.example/v1"
URL = f"{BASE_URL}/chat/completions"


def status_error(
    cls: type[openai.APIStatusError], status: int, message: str = "", body: object | None = None
) -> openai.APIStatusError:
    """An openai error as the client raises it for an HTTP error status."""
    response = httpx2.Response(status, request=httpx2.Request("POST", URL))
    return cls(message or f"Error code: {status}", response=response, body=body)


def answer(
    content: str | None,
    *,
    finish: str | None = None,
    calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    model: str = "m",
    **extra: Any,
) -> ChatCompletion:
    """A chat completion as a provider would send it; ``extra`` lands at the top level.

    ``finish`` defaults to ``tool_calls`` when there are calls, else ``stop``.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    body: dict[str, Any] = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish or ("tool_calls" if calls else "stop"), "message": message}],
        **extra,
    }
    if usage is not None:
        body["usage"] = usage
    return ChatCompletion.model_validate(body)


def recording_llm(
    monkeypatch: pytest.MonkeyPatch, answers: list[Any], cls: type[APILLM] = APILLM, **kwargs: Any
) -> tuple[APILLM, list[dict[str, Any]]]:
    """An APILLM whose client records each request, as sent, and replays ``answers``; an exception is raised."""
    llm = cls(model="m", base_url=BASE_URL, api_key="k", retry_base_delay_s=0, **kwargs)
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> ChatCompletion:
        sent.append(copy.deepcopy(request))  # the tool loop appends to the same messages list later
        reply = answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        assert isinstance(reply, ChatCompletion)
        return reply

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    return llm, sent


# ------------------------------------------------------- the worked example


@tool(writes={"counts": "json"})
def count_words(draft: str) -> dict[str, int]:
    """Count the words of a draft."""
    return {"words": len(draft.split())}


@tool(writes={"published": "text"})
def publish_text(text: str) -> str:
    """Publish a text."""
    return f"Published: {text}"


@tool(writes={"published": "text"})
def apology_text(query: str) -> str:
    """Apologize for a query."""
    return f"Sorry, no short answer about {query}."


VERDICT = {
    "fits": {"type": "bool", "means": "the draft makes one claim in under 300 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"], "means": "what should happen next"},
}
"""The judge's answer form."""


def worked_example(verdicts: Iterator[str], **kwargs: Any) -> MAS:
    """The worked example without its box and web_fetch, the judge answering from ``verdicts``.

    The writer drafts, ``count`` counts its words and the judge decides: back to
    the writer while the draft does not fit (at most 3 visits), else publish or
    apologize by the verdict's action.
    """
    llm = MockLLM(lambda p: "A short draft about cats." if p.startswith("Answer") else next(verdicts))
    writer = Agent(
        "writer",
        "Answer {query} in a short draft. The last verdict, empty at first: {verdict}",
        output="draft",
        inputs=["query", "verdict"],
    )
    judge = Agent(
        "judge",
        "Judge this draft: {draft}\nIts word counts: {counts}",
        output="verdict",
        inputs=["draft", "counts"],
        output_type="json",
        schema=VERDICT,
    )
    return MAS(
        [writer, judge],
        [
            ("writer", "count"),
            ("count", "judge"),
            Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
            Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action"),
            Edge("judge", "apology", EdgeKind.ELSE),
        ],
        steps=[
            Step("count", "count_words"),
            Step("publish", "publish_text", {"text": "draft"}),
            Step("apology", "apology_text"),
        ],
        state={"published": {"merge": "last"}},
        llm=llm,
        tools=[count_words, publish_text, apology_text],
        **kwargs,
    )


# ------------------------------------------------------------- live gateway


def gateway_configured() -> bool:
    """Whether the environment has what a live gateway call needs."""
    return bool(os.environ.get("XMAS_BASE_URL")) and bool(os.environ.get("XMAS_API_KEY"))


requires_gateway = pytest.mark.skipif(
    not gateway_configured(),
    reason="set XMAS_BASE_URL and XMAS_API_KEY to run live gateway tests",
)
"""Skip marker for tests that need Tom's gateway. CI has no key, so these skip."""
