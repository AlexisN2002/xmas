"""Shared test doubles: the scripted model, a recording tool source, a patched APILLM, the worked example and the deck.

The mock model is the reason CI can run the whole framework end to end without a
key: it satisfies promptolution's LLM contract exactly, so nothing under test has
to know it is not talking to a real endpoint.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import httpx2
import openai
import pytest
import yaml
from openai.types.chat import ChatCompletion
from promptolution.llms.base_llm import BaseLLM

import xmas.trace
from xmas import MAS, Agent, TokenUsage, runtime
from xmas.check import SystemInfo, step_maps
from xmas.edges import Edge, EdgeKind
from xmas.graph import END, agent_names, build_graph, forward_graph
from xmas.llms.api_llm import APILLM
from xmas.llms.meta import CallMeta
from xmas.nodes import Box, Step
from xmas.predicates import parse_predicate
from xmas.routes import RouteGraph
from xmas.tools import Tool, ToolCallRequest, ToolSource, index_tools, tool
from xmas.trace import AgentRecord
from xmas.values import JSON, TEXT, ValueType, parse_type

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


# ------------------------------------------------- the deck: for each chains

DECK = """\
version: 2
inputs: {plan: json, house: text}
outputs: {deck: json, notes: text}
agents:
- {name: drafter, model: writer-model, for each: {item: page, in: plan.slides, at once: 4, key: id, item budget: {cost_usd: 0.05}},
   reads: {slide: page, rules: house, verdict: verdict}, writes: {draft: text}}
- {name: checker, model: judge-model, prompt: judge, for each: page in plan.slides, reads: {planned: page, shown: measure},
   writes: {verdict: json}, schema: {fits: {type: bool, means: the page makes its claim at a glance}, fix: {type: text}}}
- {name: captioner, model: writer-model, for each: {item: figure, in: plan.figures, key: id},
   reads: {figure: figure, rules: house}, writes: {caption: text}}
- {name: figure_checker, model: judge-model, prompt: judge, for each: figure in plan.figures,
   reads: {planned: figure, shown: caption}, writes: {figure_verdict: json}, schema: {fits: {type: bool}, fix: {type: text}}}
- {name: noter, model: writer-model, reads: [plan], writes: {notes: text}}
steps:
- {name: render, uses: render_page, for each: page in plan.slides, retry: {outage: 1}}
- {name: assemble, uses: assemble_deck}
edges:
- drafter -> render -> checker
- 'checker -> drafter': {when: not verdict.fits, max_visits: 2}
- 'checker -> assemble': else
- captioner -> figure_checker -> assemble
callables:
  render_page: {kind: python, import: 'tests.conftest:render_page', reads: {draft: text}, writes: {measure: json}}
  assemble_deck: {kind: python, import: 'tests.conftest:assemble_deck', reads: {draft: json, verdict: json, caption: json},
                  writes: {deck: json}}
prompts:
  drafter: "Draft slide {slide} in this house style: {rules}\\nThe last verdict, empty at first: {verdict}"
  judge: "Does the page make the claim of its plan?\\nPlan: {planned}\\nAs drawn: {shown}"
  captioner: "Caption figure {figure} in this house style: {rules}"
  noter: "Write speaker notes for this plan: {plan}"
"""  # noqa: E501
"""The deck of two chains: a page is drafted, rendered and checked, redrawn once while it does not fit; figures are
captioned and checked beside it; ``assemble`` joins both, and END joins it with ``noter``."""

PLAN = {
    "slides": [
        {"id": name, "claim": f"the {name} slide"} for name in ("intro", "method", "result", "limits", "outlook")
    ],
    "figures": [{"id": "f1", "shows": "the setup"}, {"id": "f2", "shows": "the curve"}],
}
"""The deck's plan: five slides and two figures. ``limits`` never fits; ``intro`` is drawn slowly."""

LONG_PAGE = "A long page with far too many words on it for one slide."
"""A first draft: more than eight words, so the rendered page overflows."""


@tool(writes={"measure": "json"})
def render_page(draft: str) -> dict[str, Any]:
    """Render a page and measure it: its words, and whether they overflow it."""
    words = len(draft.split())
    return {"words": words, "overflow": words > 8}


@tool(writes={"deck": "json"})
def assemble_deck(draft: list[str], verdict: list[Any], caption: list[str]) -> dict[str, Any]:
    """Put the pages and the captions together, with whether each page fits."""
    return {"pages": draft, "captions": caption, "fits": [item["fits"] for item in verdict]}


def _draft(prompt: str) -> str:
    """The writer: a long first draft, a short one once a verdict came back; ``limits`` stays long."""
    if prompt.startswith("Draft slide"):
        if '"intro"' in prompt:
            time.sleep(0.05)  # the first item finishes last
        time.sleep(0.01)
        redrawn = prompt.rstrip().endswith("}") and '"limits"' not in prompt
        return "Short page." if redrawn else LONG_PAGE
    if prompt.startswith("Caption figure"):
        return "A caption."
    return "Notes."


def _judge(prompt: str) -> str:
    """The judge: a page fits unless its render overflows."""
    if '"overflow": true' in prompt:
        return '{"fits": false, "fix": "cut it"}'
    return '{"fits": true, "fix": ""}'


def deck_llms() -> dict[str, MockLLM]:
    """The deck's scripted models, by the names its agents bind; each call costs 0.01."""
    return {"writer-model": MockLLM(_draft, meta=True), "judge-model": MockLLM(_judge, meta=True)}


# ------------------------------------- chains until MAS declares them itself

HAS_EACH = "each" in inspect.signature(MAS.__init__).parameters
"""Whether ``MAS`` takes ``each=``, ``outputs=`` and ``retry=``; until it does, :func:`build` stands in."""

HAS_FIELDS = {"item", "item_key", "visit", "retries"} <= {f.name for f in dataclasses.fields(AgentRecord)} and hasattr(
    xmas.trace, "Event"
)
"""Whether records carry their item, visit and retries, and runs emit events."""

needs_fields = pytest.mark.skipif(not HAS_FIELDS, reason="needs the record fields item, visit, retries and Event")


@dataclasses.dataclass(frozen=True)
class StandInForEach:
    """P2: ``xmas.chains.ForEach``."""

    item: str
    over: str
    at_once: int | None = None
    key: str | None = None
    item_budget: Mapping[str, float] | None = None

    @property
    def root(self) -> str:
        """The value the path starts at."""
        return self.over.split(".")[0]

    @property
    def fields(self) -> tuple[str, ...]:
        """The path's field steps."""
        return tuple(self.over.split(".")[1:])

    def __str__(self) -> str:
        """Return the short form."""
        return f"{self.item} in {self.over}"

    @classmethod
    def parse(cls, form: str | Mapping[str, Any]) -> StandInForEach:
        """Read ``item in path`` or ``{item, in, at once, key, item budget}``."""
        if isinstance(form, str):
            item, over = form.split(" in ")
            return cls(item.strip(), over.strip())
        return cls(form["item"], form["in"], form.get("at once"), form.get("key"), form.get("item budget"))


@dataclasses.dataclass(frozen=True)
class StandInChain:
    """P2: ``xmas.chains.Chain``."""

    head: str
    nodes: tuple[str, ...]
    each: StandInForEach
    inputs: dict[str, ValueType]
    writes: dict[str, ValueType]
    system: MAS


@dataclasses.dataclass(frozen=True)
class StandInInfo(SystemInfo):
    """P2: ``SystemInfo`` with ``chains``, ``edges`` and ``flow``."""

    chains: dict[str, StandInChain] = dataclasses.field(default_factory=dict)
    edges: tuple[Edge, ...] = ()
    flow: Any = None


class StandIn(MAS):
    """P2: real ``MAS(..., each=, outputs=, retry=)``: the outer view of a system, its info built by hand.

    It skips the version 2 check, which refuses what the real one admits (END
    joining parallel branches, the reserved reads ``error`` and ``run``).
    """

    def __init__(
        self,
        *args: Any,
        chains: dict[str, StandInChain],
        outputs: Mapping[str, str] | None,
        retry: Mapping[str, Mapping[str, int]],
        **kwargs: Any,
    ) -> None:
        """Build the system, then set what the engine reads."""
        self.planned = chains
        super().__init__(*args, version=2, **kwargs)
        self.outputs = {name: parse_type(word) for name, word in (outputs or {}).items()}
        self.retry = dict(retry)

    def _build_v2(
        self, inputs: Mapping[str, str | ValueType] | None, state: Mapping[str, Mapping[str, str]] | None
    ) -> None:
        """Build the graph, joins, predicates and types without checking them."""
        given = inputs if inputs is not None else {"query": "text"}
        self.inputs = {name: kind if isinstance(kind, ValueType) else parse_type(kind) for name, kind in given.items()}
        self.merges = dict.fromkeys(state or {}, "last")
        loops = self.while_edges
        errors = [(edge.source, edge.target) for edge in self.edges if edge.kind is EdgeKind.ERROR]
        pairs = [(edge.source, edge.target) for edge in self.edges]
        graph = build_graph([*self.agents, *self.steps, *self.boxes], pairs, back=loops, errors=errors)
        joins, _ = RouteGraph(agent_names(forward_graph(graph, loops)), self.edges).classify()
        predicates = {
            edge: parse_predicate(text)
            for edge in self.edges
            if (text := edge.switch if edge.kind is EdgeKind.SWITCH else edge.when) is not None
        }
        types = dict(self.inputs)
        writers: dict[str, tuple[str, ...]] = {}
        nodes: dict[str, Agent | Step | Box] = {**self.agents, **self.steps, **self.boxes}
        for name, node in nodes.items():
            if isinstance(node, Agent):
                writes = {node.output: parse_type(node.output_type)}
            elif isinstance(node, Step):
                spec = self.resolve_callable(node)[0]
                writes = {outer: spec.writes[inner] for inner, outer in step_maps(node, spec)[1].items()}
            else:
                writes = node.types()[1]
            types.update(writes)
            writers.update({value: (*writers.get(value, ()), name) for value in writes})
        info = StandInInfo(types, writers, joins, graph, predicates, chains=self.planned, edges=self.edges, flow=graph)
        self.info = info
        self.graph = graph


def _stands_for_a_chain(**values: Any) -> None:
    """Never runs: the engine runs a chain's head as the chain."""
    raise AssertionError("a chain's head runs as its chain")


def build(
    agents: list[Agent],
    edges: Sequence[tuple[str, str] | Edge] = (),
    *,
    each: Mapping[str, str | Mapping[str, Any]] | None = None,
    outputs: Mapping[str, str] | None = None,
    retry: Mapping[str, Mapping[str, int]] | None = None,
    **kwargs: Any,
) -> MAS:
    """``MAS(agents, edges, each=, outputs=, retry=, **kwargs)``; P2: real ``MAS(...)`` once it takes them.

    Until then a chain is built as the check would build it: one system of its
    members, its exits ending it, on the item and the outside values they read;
    outside, the head stands for it, writing the chain's writes.
    """
    declared: dict[str, Any] = {"each": each, "outputs": outputs, "retry": retry}
    if HAS_EACH:
        return MAS(agents, edges, **{key: value for key, value in declared.items() if value is not None}, **kwargs)
    left = dict(retry or {})
    links = [edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges]
    steps: list[Step] = list(kwargs.pop("steps", ()))
    boxes: list[Box] = list(kwargs.pop("boxes", ()))
    tools: list[Any] = list(kwargs.pop("tools", ()))
    everything: list[Agent | Step | Box] = [*agents, *steps, *boxes]
    nodes = {node.name: node for node in everything}
    index = index_tools(tools)
    types = {name: parse_type(word) for name, word in (kwargs.get("inputs") or {"query": "text"}).items()}
    reads: dict[str, list[str]] = {}
    writes: dict[str, dict[str, ValueType]] = {}
    for name, node in nodes.items():
        if isinstance(node, Agent):
            reads[name], writes[name] = list(node.inputs), {node.output: parse_type(node.output_type)}
        elif isinstance(node, Step):
            spec = index[node.uses][0]
            read_map, write_map = step_maps(node, spec)
            reads[name], writes[name] = list(read_map.values()), {o: spec.writes[i] for i, o in write_map.items()}
        else:
            reads[name], writes[name] = list(node.reads.values()), node.types()[1]
        types.update(writes[name])

    forms = {name: StandInForEach.parse(form) for name, form in (each or {}).items()}
    group = {name: name for name in forms}

    def top(name: str) -> str:
        while group[name] != name:
            name = group[name]
        return name

    for edge in links:
        source, target = forms.get(edge.source), forms.get(edge.target)
        if source and target and (source.item, source.over) == (target.item, target.over):
            group[top(edge.source)] = top(edge.target)
    chains: dict[str, StandInChain] = {}
    head_of: dict[str, str] = {}
    for root in dict.fromkeys(top(name) for name in forms):
        members = [name for name in nodes if name in forms and top(name) == root]
        inside = set(members)
        entered = [n for n in members if any(e.target == n and e.source not in inside for e in links)]
        head = (
            entered[0]
            if entered
            else next(
                n
                for n in members
                if not any(e.target == n and e.source in inside and e.kind is not EdgeKind.WHILE for e in links)
            )
        )
        form = forms[head]
        written = {value for name in members for value in writes[name]}
        over = types[form.root]
        inner_inputs = {form.item: over.item if over.item is not None and not form.fields else JSON}
        inner_inputs.update({v: types[v] for n in members for v in reads[n] if v != form.item and v not in written})
        inner_edges = [e for e in links if e.source in inside and e.target in inside]
        inner_edges += [
            dataclasses.replace(e, target=END) for e in links if e.source in inside and e.target not in inside
        ]
        system = MAS(
            [nodes[n] for n in members if isinstance(nodes[n], Agent)],  # type: ignore[misc]
            inner_edges,
            steps=[nodes[n] for n in members if isinstance(nodes[n], Step)],  # type: ignore[misc]
            boxes=[nodes[n] for n in members if isinstance(nodes[n], Box)],  # type: ignore[misc]
            inputs=inner_inputs,
            version=2,
            tools=tools,
            **{k: v for k, v in kwargs.items() if k in ("llm", "llms", "max_steps", "max_visits")},
        )
        system.retry = {name: left.pop(name) for name in members if name in left}  # type: ignore[attr-defined]
        chain_writes = {value: vtype for name in members for value, vtype in writes[name].items()}
        chains[head] = StandInChain(head, tuple(members), form, inner_inputs, chain_writes, system)
        head_of.update(dict.fromkeys(members, head))
        reads_outside = {form.root: types[form.root], **{v: t for v, t in inner_inputs.items() if v != form.item}}
        properties = {v: {"type": "string"} if t == TEXT else {} for v, t in reads_outside.items()}
        parameters = {"type": "object", "properties": properties}
        tools.append(
            Tool(
                f"each_{head}",
                "Stands in for a chain.",
                parameters,
                _stands_for_a_chain,
                None,
                dict.fromkeys(chain_writes, JSON),
            )
        )
    outer_edges: list[Edge] = []
    for edge in links:
        if edge.source in head_of and edge.target in head_of:
            continue
        lifted = Edge(head_of[edge.source], edge.target) if edge.source in head_of else edge
        if lifted not in outer_edges:
            outer_edges.append(lifted)
    return StandIn(
        [agent for agent in agents if agent.name not in head_of],
        outer_edges,
        chains=chains,
        outputs=outputs,
        retry=left,
        steps=[step for step in steps if step.name not in head_of] + [Step(head, f"each_{head}") for head in chains],
        boxes=[box for box in boxes if box.name not in head_of],
        tools=tools,
        **kwargs,
    )


def run_values(mas: MAS, values: Mapping[str, Any], on_event: Callable[[Any], None] | None = None) -> Any:
    """``mas.run_state(values, on_event=on_event)``, a :class:`~xmas.trace.Trace`; P2: ``MAS.run_state``."""
    run_state = getattr(mas, "run_state", None)
    if run_state is not None:
        return run_state(values, on_event=on_event)
    prompts = mas.prompts()
    for chain in (getattr(mas.info, "chains", None) or {}).values():
        prompts.update(chain.system.prompts())
    return runtime.run_state(mas, values, prompts, on_event=on_event)


def deck() -> MAS:
    """The deck of :data:`DECK` on :func:`deck_llms`; P2: ``MAS.from_dict`` once it reads chains.

    Until then it is built in Python, each read under its own value's name.
    """
    llms: dict[str, BaseLLM] = dict(deck_llms())
    tools = [render_page, assemble_deck]
    if HAS_EACH:
        return MAS.from_dict(yaml.safe_load(DECK), llms=llms, tools=tools)
    judged = {"fits": {"type": "bool", "means": "the page makes its claim at a glance"}, "fix": {"type": "text"}}
    agents = [
        Agent(
            "drafter",
            "Draft slide {page} in this house style: {house}\nThe last verdict, empty at first: {verdict}",
            "draft",
            ["page", "house", "verdict"],
            model="writer-model",
        ),
        Agent(
            "checker",
            "Does the page make the claim of its plan?\nPlan: {page}\nAs drawn: {measure}",
            "verdict",
            ["page", "measure"],
            model="judge-model",
            output_type="json",
            schema=judged,
        ),
        Agent(
            "captioner",
            "Caption figure {figure} in this house style: {house}",
            "caption",
            ["figure", "house"],
            model="writer-model",
        ),
        Agent(
            "figure_checker",
            "Does the page make the claim of its plan?\nPlan: {figure}\nAs drawn: {caption}",
            "figure_verdict",
            ["figure", "caption"],
            model="judge-model",
            output_type="json",
            schema={"fits": {"type": "bool"}, "fix": {"type": "text"}},
        ),
        Agent("noter", "Write speaker notes for this plan: {plan}", "notes", ["plan"], model="writer-model"),
    ]
    return build(
        agents,
        [
            ("drafter", "render"),
            ("render", "checker"),
            Edge("checker", "drafter", EdgeKind.WHILE, when="not verdict.fits", max_visits=2),
            Edge("checker", "assemble", EdgeKind.ELSE),
            ("captioner", "figure_checker"),
            ("figure_checker", "assemble"),
        ],
        each={
            "drafter": {
                "item": "page",
                "in": "plan.slides",
                "at once": 4,
                "key": "id",
                "item budget": {"cost_usd": 0.05},
            },
            "render": "page in plan.slides",
            "checker": "page in plan.slides",
            "captioner": {"item": "figure", "in": "plan.figures", "key": "id"},
            "figure_checker": "figure in plan.figures",
        },
        outputs={"deck": "json", "notes": "text"},
        retry={"render": {"outage": 1}},
        steps=[Step("render", "render_page"), Step("assemble", "assemble_deck")],
        inputs={"plan": "json", "house": "text"},
        llms=llms,
        tools=tools,
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
