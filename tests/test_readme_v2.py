"""The README's version 2 sections, executed as written.

Each test mirrors one code block of the README line for line, with a scripted
``MockLLM`` in place of ``APILLM``, and checks the prints the README shows. If a
test needs editing to keep passing, the README changed with it. The last tests
pin the README's claims that have no block of their own, and that nothing the
README calls planned exists yet.
"""

from __future__ import annotations

import inspect
import itertools
import re
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import xmas
from xmas import MAS, Agent, Box, Budget, Edge, EdgeKind, ErrorKind, GSM8KTask, HealthSettings, Step, TokenUsage
from xmas.predicates import Context, parse_predicate
from xmas.tools import Tool, ToolSource, tool

from .conftest import MockLLM, req

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
QUERY = "Why is the sky blue?"
DRAFT = "Sunlight scatters off air."
FITS = '{"fits": true, "action": "publish"}'
SHORTEN = '{"fits": false, "action": "shorten"}'


def scripted(judge: Callable[[], str] | None = None, review: str = "A fine answer.") -> MockLLM:
    """A model that drafts DRAFT for the writer and answers the judge from ``judge``."""

    def respond(prompt: str) -> str:
        if prompt.startswith("Write"):
            return DRAFT
        if prompt.startswith("Cut"):
            return "Air scatters blue light."
        if "Is it a good answer?" in prompt:
            return review
        return judge() if judge is not None else FITS

    return MockLLM(respond)


def answers(*replies: str) -> Callable[[], str]:
    """Answer the judge with ``replies`` in order, repeating the last one."""
    stream: Iterator[str] = itertools.chain(replies, itertools.repeat(replies[-1]))
    return lambda: next(stream)


# --------------------------------------------------- the README's definitions


@tool
def count_words(draft: str) -> int:
    """Count the words of a draft."""
    return len(draft.split())


@tool
def publish(draft: str) -> str:
    """Publish a draft."""
    return f"Published: {draft}"


@tool
def apologize(query: str) -> str:
    """Say sorry when no draft fits."""
    return f"Sorry, no short answer to: {query}"


verdict = {
    "fits": {"type": "bool", "means": "the draft answers the question in under 50 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"]},
}


def json_judge() -> Agent:
    """The judge of 'Answers with a shape'."""
    return Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema=verdict,
    )


def looping_system(llm: MockLLM, counter: Tool = count_words) -> MAS:
    """The system of 'Conditions and loops'."""
    judge = json_judge()
    writer = Agent(
        "writer",
        prompt="Write a short answer to: {query}\nThe judge's last verdict, empty at first: {verdict}",
        inputs=["query", "verdict"],
        output="draft",
    )
    mas = MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words"), Step("publish", uses="publish"), Step("sorry", uses="apologize")],
        edges=[
            ("writer", "count"),
            ("count", "judge"),
            Edge("judge", "writer", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
            Edge("judge", "publish", kind=EdgeKind.SWITCH, switch="verdict.action"),
            Edge("judge", "sorry", kind=EdgeKind.ELSE),
            Edge("count", "sorry", kind=EdgeKind.ERROR, errors=("timeout", "outage")),
        ],
        tools=[counter, publish, apologize],
        llm=llm,
    )
    return mas


def quickstart_agents() -> tuple[Agent, Agent]:
    """The quickstart's intern and boss."""
    intern = Agent("intern", prompt="Answer this question, showing your working: {query}", output="draft")
    boss = Agent(
        "boss",
        prompt="Question: {query}\nYour intern's draft: {draft}\nGive the final answer in one sentence, as your own.",
        inputs=["query", "draft"],
        output="prediction",
    )
    return intern, boss


def quickstart_llm() -> MockLLM:
    """The quickstart test's model."""
    return MockLLM(lambda prompt: "Rayleigh scattering." if "as your own" in prompt else "Short waves scatter more.")


# ------------------------------------------------- Code between agents: steps


def test_steps_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(review="A fine answer.")

    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nIs it a good answer? Say why in one sentence.",
        inputs=["query", "draft", "count_words"],
        output="review",
    )
    mas = MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    print(mas("Why is the sky blue?"))  # the judge's review
    assert mas.last_trace is not None
    print([record.agent for record in mas.last_trace.records])  # ['writer', 'count', 'judge']

    assert capsys.readouterr().out.splitlines() == ["A fine answer.", "['writer', 'count', 'judge']"]
    count = mas.last_trace["count"][0]
    assert (count.kind, count.outputs, count.cost_usd) == ("step", {"count_words": "4"}, 0.0)
    assert mas.last_trace.cost_usd is None, "the writer's cost is unknown, so the sum is too"
    assert f"Draft, 4 words: {DRAFT}" in llm.calls[-1]
    assert len(llm.calls) == 2, "a step makes no model call"


def test_a_step_writes_text_for_str_json_otherwise_and_renames_both_sides() -> None:
    @tool
    def shout(answer: str) -> str:
        """Shout an answer."""
        return answer.upper()

    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="answer")
    judge = Agent("judge", prompt="{words} words: {shout}", inputs=["words", "shout"], output="review")
    mas = MAS(
        [writer, judge],
        steps=[
            Step("count", uses="count_words", reads={"draft": "answer"}, writes={"count_words": "words"}),
            Step("loud", uses="shout"),
        ],
        edges=[("writer", "count"), ("writer", "loud"), ("count", "judge"), ("loud", "judge")],
        tools=[count_words, shout],
        llm=llm,
    )

    trace = mas.run(QUERY)
    assert trace.error is None
    assert trace["count"][0].outputs == {"words": "4"}
    described = {node["name"]: node for node in mas.describe()["nodes"]}
    assert described["count"]["writes"] == {"words": "json"}
    assert described["loud"]["writes"] == {"shout": "text"}


def test_a_tool_needs_a_docstring_and_a_hint_on_every_parameter() -> None:
    def no_doc(draft: str) -> int:
        return len(draft)

    def no_hint(draft) -> int:  # type: ignore[no-untyped-def]
        """Count."""
        return len(draft)

    with pytest.raises(ValueError, match="docstring"):
        tool(no_doc)
    with pytest.raises(ValueError):
        tool(no_hint)


def test_the_same_function_is_a_step_and_an_agents_tool() -> None:
    model = MockLLM(turns=[("", [req("count_words", '{"draft": "one two three"}')])], default=(DRAFT, []))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft", tools=["count_words"])
    judge = Agent("judge", prompt="{count_words}: {draft}", inputs=["draft", "count_words"], output="review")
    mas = MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=model,
    )

    trace = mas.run(QUERY)
    assert trace.error is None
    assert [(call.tool, call.result) for call in trace["writer"][0].tool_calls] == [("count_words", "3")]
    assert trace["count"][0].outputs == {"count_words": "4"}


class RemotePages(ToolSource):
    """A tool source standing in for an MCP server."""

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        return [Tool("web_search", "Search the web.", schema)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer every call with the same page."""
        return f"A page about {arguments['query']}"


def test_an_mcp_style_tool_runs_as_a_step() -> None:
    llm = scripted(review="ok")
    judge = Agent("judge", prompt="Is it a good answer? {web_search}", inputs=["web_search"], output="review")
    mas = MAS(
        [judge], steps=[Step("search", uses="web_search")], edges=[("search", "judge")], tools=[RemotePages()], llm=llm
    )

    trace = mas.run(QUERY)
    assert trace["search"][0].outputs == {"web_search": f"A page about {QUERY}"}


# ------------------------------------------------------- Answers with a shape


def test_answer_form_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(answers("Looks good to me.", FITS))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")

    judge = Agent(
        "judge",
        prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
        inputs=["query", "draft", "count_words"],
        output="verdict",
        output_type="json",
        schema=verdict,
    )
    mas = MAS(
        [writer, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run("Why is the sky blue?")
    print(trace["judge"][0].rendered_prompt)

    shown = (
        "Question: Why is the sky blue?\n"
        "Draft, 4 words: Sunlight scatters off air.\n"
        "Judge the draft.\n"
        "\n"
        "Answer with one JSON object and nothing else. Its fields:\n"
        "- fits (true or false): the draft answers the question in under 50 words\n"
        '- action (one of "publish", "shorten")\n'
    )
    assert capsys.readouterr().out == shown
    assert f"```text\n{shown}```" in README

    print(trace.output)  # {"fits": true, "action": "publish"}
    print(trace["judge"][0].rejected)  # the answer refused before the retry, if there was one
    out = capsys.readouterr().out.splitlines()
    assert out[0] == '{"fits": true, "action": "publish"}'
    assert "Looks good to me." in out[1]
    assert [rejected.raw_response for rejected in trace["judge"][0].rejected] == ["Looks good to me."]
    judge_calls = [call for call in llm.calls if call.startswith("Question:")]
    assert len(judge_calls) == 2, "exactly one retry"
    assert "Your previous answer did not match the required format" in judge_calls[1]


def test_two_misses_end_the_run_with_bad_format() -> None:
    llm = scripted(answers("Looks good.", "Still looks good."))
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    mas = MAS(
        [writer, json_judge()],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("count", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run(QUERY)
    assert (trace.status, trace.error_kind) == ("failed", "bad_format")
    assert len([call for call in llm.calls if call.startswith("Question:")]) == 2
    with pytest.raises(RuntimeError, match="BadFormat"):
        mas(QUERY)


def test_the_form_dialect_the_readme_names() -> None:
    schema = {
        "fits": {"type": "bool", "means": "whether it fits"},
        "words": {"type": "int", "optional": True},
        "score": {"type": "number"},
        "extra": {"type": "json"},
        "where": {"type": "object", "fields": {"line": {"type": "int"}}},
        "problems": {"type": "list of object", "fields": {"text": {"type": "text"}}},
        "tags": {"type": "list of text"},
        "action": {"type": "text", "one of": ["publish", "shorten"]},
    }
    Agent("judge", prompt="{query}", output="verdict", output_type="json", schema=schema)
    with pytest.raises(ValueError):
        Agent("judge", prompt="{query}", output="verdict", output_type="json", schema={"fits": {"type": "boolean"}})


# ------------------------------------------------ Mistakes found before any call


def test_mistakes_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")

    judge_with_typos = Agent(
        "judge",
        prompt="Draft: {draft}\nIt has {count_words} words. The style guide: {style}",
        inputs=["draft", "notes"],
        output="verdict",
    )
    try:
        MAS(
            [writer, judge_with_typos],
            steps=[Step("count", uses="count_word")],
            edges=[("writer", "count"), ("count", "judge")],
            tools=[count_words],
            llm=llm,
        )
    except ValueError as problems:
        print(problems)

    shown = (
        "The system does not check (3 problems):\n"
        "- agent 'judge': prompt placeholders ['count_words', 'style'] are not among its reads ['draft', 'notes']\n"
        "- Step 'count' uses unknown callable 'count_word'; available: ['count_words'].\n"
        "- 'judge' reads 'notes', which is neither an input nor written by any node\n"
    )
    assert capsys.readouterr().out == shown
    assert f"```text\n{shown}```" in README
    assert llm.calls == [], "no model call"


def test_the_check_refuses_two_writers_a_type_clash_and_a_bad_condition() -> None:
    @tool
    def shout(draft: str) -> str:
        """Shout a draft."""
        return draft.upper()

    llm = scripted()
    two_writers = [Agent("a", "{query}", output="draft"), Agent("b", "{query}", output="draft")]
    with pytest.raises(ValueError, match="draft"):
        MAS(two_writers, steps=[Step("s", uses="shout")], tools=[shout], llm=llm)

    json_writer = Agent("judge", "{query}", output="draft", output_type="json")
    with pytest.raises(ValueError, match="json"):
        MAS([json_writer], steps=[Step("s", uses="shout")], edges=[("judge", "s")], tools=[shout], llm=llm)

    with pytest.raises(ValueError, match="position"):
        looping = [Agent("writer", "{query}", output="draft"), json_judge()]
        MAS(
            looping,
            steps=[Step("count", uses="count_words"), Step("publish", uses="publish")],
            edges=[
                ("writer", "count"),
                ("count", "judge"),
                Edge("judge", "publish", kind=EdgeKind.WHEN, when="verdict.fits = true"),
            ],
            tools=[count_words, publish],
            llm=llm,
        )


# ---------------------------------------------------------- Conditions and loops


def test_conditions_block_publishes_a_draft_that_fits(capsys: pytest.CaptureFixture[str]) -> None:
    mas = looping_system(scripted(answers(SHORTEN, FITS)))

    print(mas("Why is the sky blue?"))  # "Published: ..." or "Sorry, no short answer to: ..."

    assert capsys.readouterr().out == f"Published: {DRAFT}\n"
    assert mas.last_trace is not None
    assert [record.agent for record in mas.last_trace.records] == ["writer", "count", "judge"] * 2 + ["publish"]
    assert mas.last_trace["judge"][0].route == ("writer",)


def test_at_the_cap_the_else_edge_runs_even_when_the_switch_would_match() -> None:
    trace = looping_system(scripted(answers('{"fits": false, "action": "publish"}'))).run(QUERY)

    assert [record.agent for record in trace.records] == ["writer", "count", "judge"] * 3 + ["sorry"]
    assert [record.capped for record in trace["judge"]] == [False, False, True]
    assert trace.output == f"Sorry, no short answer to: {QUERY}"


def test_an_action_outside_the_cases_falls_through_to_else() -> None:
    trace = looping_system(scripted(answers('{"fits": true, "action": "shorten"}'))).run(QUERY)

    assert [record.agent for record in trace.records] == ["writer", "count", "judge", "sorry"]


def test_a_timeout_in_the_count_is_routed_and_keeps_the_failed_record() -> None:
    @tool(name="count_words")
    def slow_count(draft: str) -> int:
        """Count the words of a draft, too slowly."""
        raise TimeoutError("the counter took too long")

    trace = looping_system(scripted(), counter=slow_count).run(QUERY)

    assert trace.error is None
    assert [record.agent for record in trace.records] == ["writer", "count", "sorry"]
    assert trace["count"][0].error is not None
    assert trace.output == f"Sorry, no short answer to: {QUERY}"


def test_an_unrouted_error_fails_the_run_and_budget_and_stop_are_never_routable() -> None:
    @tool(name="count_words")
    def broken_count(draft: str) -> int:
        """Count the words of a draft, badly."""
        raise RuntimeError("no idea")

    trace = looping_system(scripted(), counter=broken_count).run(QUERY)
    assert trace.status == "failed"
    assert [record.agent for record in trace.records] == ["writer", "count"]

    for kind in ("budget", "stopped"):
        with pytest.raises(ValueError, match=kind):
            MAS(
                [Agent("writer", "{query}", output="draft")],
                steps=[Step("sorry", uses="apologize")],
                edges=[Edge("writer", "sorry", kind=EdgeKind.ERROR, errors=(kind,))],
                tools=[apologize],
                llm=scripted(),
            )


def test_the_operator_table_matches_the_edge_kinds_and_the_spec_forms() -> None:
    assert {kind.value for kind in EdgeKind} == {"plain", "when", "else", "while", "switch", "error"}
    assert Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action").case == "b"
    forms = {
        "a -> b: {when: verdict.fits}": Edge("a", "b", kind=EdgeKind.WHEN, when="verdict.fits"),
        "a -> b: else": Edge("a", "b", kind=EdgeKind.ELSE),
        "a -> b: {when: not verdict.fits, max_visits: 3}": Edge(
            "a", "b", kind=EdgeKind.WHILE, when="not verdict.fits", max_visits=3
        ),
        "a -> b: {switch: verdict.action}": Edge("a", "b", kind=EdgeKind.SWITCH, switch="verdict.action"),
        "a -> b: {on error: [timeout]}": Edge("a", "b", kind=EdgeKind.ERROR, errors=("timeout",)),
    }
    from xmas.edges import parse_edges

    for text, edge in forms.items():
        assert parse_edges(yaml.safe_load(f"- {text}")) == [edge], text


def test_every_condition_the_readme_names_parses_and_behaves() -> None:
    values = {"verdict": {"fits": False, "action": "publish"}, "draft": "four words right here"}
    ctx = Context(values=values, visits={"writer": 2, "judge": 1}, node="judge", cost_usd=0.2, elapsed_s=12.0)
    holds = {
        "not verdict.fits": True,
        "verdict.action in ['publish', 'shorten']": True,
        "exists(verdict.action)": True,
        "exists(verdict.notes)": False,
        "len(draft) > 200": False,
        "visits == 1 and visits(writer) >= 2": True,
        "cost < 0.5 and elapsed < 60": True,
        "verdict.fits == false or null == true": True,
        '"publish" == verdict.action': True,
    }
    for text, expected in holds.items():
        assert parse_predicate(text).evaluate(ctx) is expected, text
    with pytest.raises(ValueError, match="position 13"):
        parse_predicate("verdict.fits = true")


def test_a_join_runs_once_after_both_edges() -> None:
    llm = scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    critic = Agent("critic", prompt="Critic of {draft}", inputs=["draft"], output="notes")
    judge = Agent("judge", prompt="{draft} {notes} {count_words}", inputs=["draft", "notes", "count_words"], output="v")
    mas = MAS(
        [writer, critic, judge],
        steps=[Step("count", uses="count_words")],
        edges=[("writer", "count"), ("writer", "critic"), ("count", "judge"), ("critic", "judge")],
        tools=[count_words],
        llm=llm,
    )

    trace = mas.run(QUERY)
    order = [record.agent for record in trace.records]
    assert order.count("judge") == 1
    assert order.index("judge") > max(order.index("count"), order.index("critic"))
    assert mas.describe()["joins"]["judge"] == "join"
    assert looping_system(scripted()).describe()["joins"]["sorry"] == "merge"


# ----------------------------------------------------------------- Reading a run


def never_accepting() -> MAS:
    """The looping system with a judge that never accepts a draft."""
    return looping_system(scripted(answers(SHORTEN)))


def test_reading_a_run_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    mas = never_accepting()

    trace = mas.run("Why is the sky blue?")
    print(trace.status)  # ok
    print([record.agent for record in trace.records])
    # ['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']
    print(trace["judge"][-1].capped)  # True
    assert trace.health is not None
    print(trace.health.loops)  # {'judge -> writer': {'taken': 2, 'cap_hit': True}}

    assert capsys.readouterr().out.splitlines() == [
        "ok",
        "['writer', 'count', 'judge', 'writer', 'count', 'judge', 'writer', 'count', 'judge', 'sorry']",
        "True",
        "{'judge -> writer': {'taken': 2, 'cap_hit': True}}",
    ]
    record = trace["judge"][0]
    for name in ("inputs", "outputs", "rendered_prompt", "latency_s", "cost_usd", "tokens", "finish_reason"):
        assert hasattr(record, name), name
    for name in ("error_kind", "served_model", "route", "capped", "rejected", "kind"):
        assert hasattr(record, name), name
    assert record.cost_usd is None, "a model that reports no cost leaves it unknown, never 0"
    assert trace["count"][0].kind == "step"
    assert {status.value for status in xmas.RunStatus} >= {"ok", "failed", "over_budget"}


def test_health_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    trace = never_accepting().run(QUERY)
    assert trace.health is not None

    print(sorted({(flag.signal, flag.node) for flag in trace.health.flags}))
    # [('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]

    assert capsys.readouterr().out == "[('stuck', 'count'), ('stuck', 'judge'), ('stuck', 'writer')]\n"


def test_health_signal_names_and_a_healthy_run() -> None:
    named = ["retried", "bad_format", "error_routed", "empty", "stuck", "unfilled", "leak", "long", "short"]
    sentence = README.split("The signals are ", 1)[1].split("`trace.health.loops`", 1)[0]
    assert re.findall(r"`(\w+)`", sentence) == named

    healthy = looping_system(scripted(), counter=count_words)
    trace = healthy.run(QUERY)
    assert "health" not in trace.to_dict()

    empty = MAS(
        [Agent("writer", "{query}", output="draft")],
        llm=MockLLM(lambda prompt: ""),
        health=HealthSettings(length_factor=4),
    )
    flagged = empty.run(QUERY)
    assert flagged.health is not None
    assert [flag.signal for flag in flagged.health.flags] == ["empty"]


def test_budget_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = quickstart_llm()
    intern, boss = quickstart_agents()

    thrifty = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_calls=1))
    spent = thrifty.run("Why is the sky blue?")
    print(spent.status)  # over_budget
    print(spent.error)  # BudgetExceeded: calls 1 >= max 1 before call 2
    print(spent["intern"][0].outputs)  # the intern's draft is kept

    assert capsys.readouterr().out.splitlines() == [
        "over_budget",
        "BudgetExceeded: calls 1 >= max 1 before call 2",
        "{'draft': 'Short waves scatter more.'}",
    ]
    assert spent.error_kind == "budget"


def test_a_zero_budget_spends_nothing_and_an_unknown_cost_stops_a_cost_limit() -> None:
    intern, boss = quickstart_agents()
    llm = quickstart_llm()
    nothing = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_cost_usd=0))
    assert nothing.run(QUERY).status == "over_budget"
    assert llm.calls == []

    unknown = MAS([intern, boss], edges=[("intern", "boss")], llm=llm, budget=Budget(max_cost_usd=0.5))
    trace = unknown.run(QUERY)
    assert (trace.status, len(llm.calls)) == ("over_budget", 1)
    assert trace["intern"][0].cost_usd is None
    Budget(max_cost_usd=0.5, max_wall_s=600, max_calls=40)


# ------------------------------------------------------------- Drawing a system


def test_describe_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    mas = looping_system(scripted())

    picture = mas.describe()
    print([(node["name"], node["kind"]) for node in picture["nodes"]])
    # [('writer', 'agent'), ('count', 'step'), ('judge', 'agent'), ('publish', 'step'), ('sorry', 'step')]
    print([edge["kind"] for edge in picture["edges"] if not edge.get("implicit")])
    # ['plain', 'plain', 'while', 'switch', 'else', 'error']
    mas.agents["judge"].prompt += "\nBe strict."
    print(mas.describe()["id"] == picture["id"])  # True

    assert capsys.readouterr().out.splitlines() == [
        "[('writer', 'agent'), ('count', 'step'), ('judge', 'agent'), ('publish', 'step'), ('sorry', 'step')]",
        "['plain', 'plain', 'while', 'switch', 'else', 'error']",
        "True",
    ]
    rewired = looping_system(scripted()).copy(edges=[edge for edge in mas.edges if edge.kind is not EdgeKind.ERROR])
    assert rewired.describe()["id"] != picture["id"]
    node = picture["nodes"][2]
    assert {"model", "reads", "writes", "frozen"} <= set(node)


# ----------------------------------------------------- A system inside a system


def test_box_block_runs_as_written(capsys: pytest.CaptureFixture[str]) -> None:
    llm = MockLLM(
        lambda prompt: (
            "Air scatters blue light." if prompt.startswith("Cut") else "Sunlight hits air, which scatters blue."
        )
    )

    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = Box(
        "shorten",
        MAS([cutter], inputs={"page": "text"}, llm=llm),
        reads={"page": "draft"},
        writes={"short": "summary"},
    )
    first_draft = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    outer = MAS([first_draft], boxes=[shorten], edges=[("writer", "shorten")], llm=llm)

    trace = outer.run("Why is the sky blue?")
    print(trace.output)  # the cutter's shorter draft
    print([(path, record.agent) for path, record in trace.walk()])
    # [((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]

    assert capsys.readouterr().out.splitlines() == [
        "Air scatters blue light.",
        "[((), 'writer'), ((), 'shorten'), (('shorten',), 'cutter')]",
    ]
    box = trace["shorten"][0]
    assert box.kind == "box"
    assert box.inner is not None
    assert box.cost_usd is None and trace.cost_usd is None, "a model that reports no cost leaves every sum unknown"
    assert (
        box.inner["cutter"][0].rendered_prompt == "Cut this to half its length: Sunlight hits air, which scatters blue."
    )


def test_a_box_costs_its_inner_records_sum() -> None:
    llm = MockLLM(lambda prompt: "Air scatters blue light.", meta=True)
    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = Box("shorten", MAS([cutter], inputs={"page": "text"}, llm=llm), {"page": "draft"}, {"short": "summary"})
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    trace = MAS([writer], boxes=[shorten], edges=[("writer", "shorten")], llm=llm).run(QUERY)

    (box,) = trace["shorten"]
    assert box.cost_usd == pytest.approx(0.01)
    assert box.tokens == TokenUsage(10, 2)
    assert trace.cost_usd == pytest.approx(0.02)


def test_a_box_spends_from_the_outer_budget() -> None:
    llm = scripted()
    cutter = Agent("cutter", prompt="Cut this to half its length: {page}", inputs=["page"], output="short")
    shorten = Box("shorten", MAS([cutter], inputs={"page": "text"}, llm=llm), {"page": "draft"}, {"short": "summary"})
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft")
    outer = MAS([writer], boxes=[shorten], edges=[("writer", "shorten")], llm=llm, budget=Budget(max_calls=1))

    assert outer.run(QUERY).status == "over_budget"


# ---------------------------------------------------------------- Save and share


def test_version_1_save_block_runs_as_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    llm = quickstart_llm()
    intern, boss = quickstart_agents()

    team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)  # the quickstart's system
    team.to_yaml("team.yaml")
    team = MAS.from_yaml("team.yaml", llm=llm)

    assert team("Why is the sky blue?") == "Rayleigh scattering."
    text = (tmp_path / "team.yaml").read_text(encoding="utf-8")
    assert "http" not in text and "key" not in text.lower()


def test_version_2_save_block_runs_as_written(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    llm = scripted()
    mas = looping_system(llm)

    spec = mas.to_dict()  # version, agents, steps, edges, callables, prompts
    rebuilt = MAS.from_dict(spec, llm=llm, tools=[count_words, publish, apologize])
    print(rebuilt.describe()["id"] == mas.describe()["id"])  # True

    assert capsys.readouterr().out == "True\n"
    assert list(spec) == ["version", "agents", "steps", "edges", "callables", "prompts"]
    assert spec["version"] == 2
    shown = (
        "- writer -> count -> judge\n"
        "- judge -> writer: {when: not verdict.fits, max_visits: 3}\n"
        "- judge -> publish: {switch: verdict.action}\n"
        "- judge -> sorry: else\n"
        "- count -> sorry: {on error: [timeout, outage]}\n"
    )
    assert f"```yaml\n{shown}```" in README
    assert yaml.safe_load(shown) == spec["edges"]
    from_yaml_edges = MAS.from_dict(
        {**spec, "edges": yaml.safe_load(shown)}, llm=llm, tools=[count_words, publish, apologize]
    )
    assert from_yaml_edges.describe()["id"] == mas.describe()["id"]
    assert "http" not in yaml.safe_dump(spec) and "api_key" not in yaml.safe_dump(spec)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    from_file = MAS.from_yaml(tmp_path / "spec.yaml", llm=llm, tools=[count_words, publish, apologize])
    assert (from_file.version, from_file.describe()["id"]) == (2, mas.describe()["id"])
    with pytest.raises(ValueError, match=r"\.mas bundle .*to_dict\(\) gives its spec"):
        mas.to_yaml(tmp_path / "team.yaml")


# ------------------------------------------------------------ Optimization built in


def test_cantante_block_runs_as_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"question": ["What is 2 + 2?", "What is 1 + 3?"], "answer": ["#### 4", "#### 4"]})
    monkeypatch.setattr(GSM8KTask, "from_huggingface", classmethod(lambda cls, split, n: cls(frame.head(n))))
    llm = MockLLM(lambda prompt: "<prompt>Solve: {query}</prompt>" if "<prompt>" in prompt else "The answer is 4.")
    intern, boss = quickstart_agents()
    team = MAS([intern, boss], edges=[("intern", "boss")], llm=llm)

    from promptolution.optimizers import OPRO

    from xmas.optimization import CantanteOptimizer, ContrastiveAttributor

    task = GSM8KTask.from_huggingface(split="train", n=50)  # needs: uv run --with datasets
    optimizer = CantanteOptimizer(team, task, ContrastiveAttributor(llm), OPRO, optimizer_kwargs={"meta_llm": llm})
    best = optimizer.optimize(n_steps=5)
    best.to_yaml("team-optimized.yaml")

    assert MAS.from_yaml("team-optimized.yaml", llm=llm).agents.keys() == {"intern", "boss"}


def test_a_frozen_node_is_never_an_optimizer_target_and_makes_the_system_version_2() -> None:
    from xmas.optimization.updaters import movable_agents

    intern, boss = quickstart_agents()
    frozen = MAS([intern, boss], edges=[("intern", "boss")], llm=quickstart_llm(), frozen=("boss",))

    assert movable_agents(frozen) == ["intern"]
    assert frozen.version == 2
    assert frozen.frozen == frozenset({"boss"})


# ------------------------------------------------------------ xmas and promptolution


def test_a_one_agent_system_encodes_to_its_bare_prompt_with_or_without_steps() -> None:
    solver = Agent(
        "solver", prompt="Solve, using {count_words} words: {query}", inputs=["query", "count_words"], output="answer"
    )
    with_step = MAS(
        [solver],
        steps=[Step("count", uses="count_words", reads={"draft": "query"})],
        edges=[("count", "solver")],
        tools=[count_words],
        llm=scripted(),
    )
    alone = MAS([Agent("solver", prompt="Solve: {query}", output="answer")], llm=scripted())

    assert with_step.encode() == solver.prompt
    assert alone.encode() == "Solve: {query}"


# ------------------------------------------------- claims without a block of their own


def test_error_kinds_the_readme_names_are_the_real_ones() -> None:
    assert {kind.value for kind in ErrorKind} == {
        "money",
        "rate_limited",
        "outage",
        "timeout",
        "length_cut",
        "bad_format",
        "refusal",
        "stopped",
        "budget",
        "unknown",
    }
    for kind in ("timeout", "outage", "rate_limited", "money", "length_cut", "bad_format", "refusal"):
        assert f"`{kind}`" in README


def test_per_agent_models_bind_by_name_and_an_unknown_name_is_refused() -> None:
    fast, slow = scripted(), scripted()
    writer = Agent("writer", prompt="Write a short answer to: {query}", output="draft", model="fast")
    judge = Agent("judge", prompt="{draft}", inputs=["draft"], output="review", model="slow")
    mas = MAS([writer, judge], edges=[("writer", "judge")], llms={"fast": fast, "slow": slow})
    mas.run(QUERY)
    assert (len(fast.calls), len(slow.calls)) == (1, 1)

    with pytest.raises(ValueError, match="not in llms"):
        MAS([writer, judge], edges=[("writer", "judge")], llms={"fast": fast})


def test_nothing_the_readme_calls_planned_exists_yet() -> None:
    for name in ("save", "load", "verify"):
        assert not hasattr(MAS, name), name
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"].get("scripts", {})
    assert "xmas" not in scripts
    assert not {"for each", "first of", "for_each", "first_of"} & {kind.value for kind in EdgeKind}
    with pytest.raises(ValueError, match="image"):
        MAS([Agent("painter", prompt="Draw {query}", output="picture", output_type="image")], llm=scripted())
    assert "langgraph" not in " ".join(inspect.signature(MAS).parameters).lower()


def test_the_readme_keeps_the_rulings() -> None:
    assert ".xmas" not in README
    assert "name.mas/" in README
    assert "langgraph" not in README.lower()
    planned = README.split("## Where this is going", 1)[1].split("## 📚", 1)[0]
    assert "> **Planned, not in the code yet.**" in planned
    for block in planned.split("```yaml")[1:]:
        assert "http" not in block.split("```", 1)[0]
        assert "key" not in block.split("```", 1)[0]


def test_every_component_the_readme_lists_exists() -> None:
    import xmas.optimization

    components = README.split("## 🏗 Components", 1)[1].split("\n## ", 1)[0]
    names = [line.split("`")[1] for line in components.splitlines() if line.startswith("- **`")]
    assert len(names) == 11
    for name in names:
        assert hasattr(xmas, name) or hasattr(xmas.optimization, name), name
