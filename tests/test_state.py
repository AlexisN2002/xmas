"""State in, several outputs out: ``run_state``, declared outputs, retries, the reserved reads, events and caps.

The models are scripted doubles, so no test spends money.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import pytest

from xmas import MAS, Agent, Box, Budget, ErrorKind, TokenUsage
from xmas.edges import Edge, EdgeKind
from xmas.nodes import Step
from xmas.tools import tool

from .conftest import MockLLM


def deck_notes(llm: MockLLM) -> MAS:
    """``summarizer`` and ``noter`` run in parallel on the plan; END joins them; both are outputs."""
    return MAS(
        [
            Agent("summarizer", "Summarize {plan} in the style {house}", "summary", ["plan", "house"]),
            Agent("noter", "Write notes for {plan}", "notes", ["plan"]),
        ],
        outputs={"summary": "text", "notes": "text"},
        inputs={"plan": "json", "house": "text"},
        llm=llm,
    )


def test_run_state_checks_inputs_and_returns_outputs() -> None:
    llm = MockLLM(lambda p: "A summary." if p.startswith("Summarize") else "Some notes.", meta=True)
    mas = deck_notes(llm)

    # every problem at once, before any call
    with pytest.raises(ValueError, match="do not fit the system's inputs") as caught:
        mas.run_state({"plan": {1, 2}, "query": "x"})
    lines = str(caught.value).splitlines()[1:]
    assert len(lines) == 3 and llm.calls == []
    assert lines[0] == "- input 'house' is missing"
    assert lines[1].startswith("- 'query' is no input") and lines[2].startswith("- input 'plan' is not json")

    values = {"plan": {"slides": 2}, "house": "plain"}
    trace = mas.run_state(values)
    assert trace.error is None
    assert trace.outputs == {"summary": "A summary.", "notes": "Some notes."}
    assert trace.output == "A summary."  # the first output, which a task scores
    assert sorted(trace.agents) == ["noter", "summarizer"]  # END joined both branches
    assert trace.query == json.dumps(values)
    assert re.fullmatch("[0-9a-f]{32}", trace.run_id or "") and trace.graph_id == mas.describe()["id"]
    again = mas.run_state(values)
    assert again.run_id != trace.run_id and again.graph_id == trace.graph_id
    assert again == trace  # the run id is left out of ==

    # a failure is a failed trace, with no outputs
    broken = deck_notes(MockLLM(answers=[RuntimeError("boom"), RuntimeError("boom")]))
    failed = broken.run_state(values)
    assert failed.error is not None and failed.outputs == {}

    # version 1: text values, a preset beside the query; equal to run(query) record for record
    v1 = MAS([Agent("a", "Answer {query} {style}", "answer", ["query", "style"])], llm=MockLLM())
    plain = v1.run_state({"query": "cats"})
    assert plain.records == v1.run("cats").records and plain.query == "cats"
    assert plain.run_id is None and plain.graph_id is None and plain.records[0].visit is None
    assert v1.run_state({"query": "cats", "style": "tersely"}).output == "echo:Answer cats tersely"
    assert v1("cats") == "echo:Answer cats "
    with pytest.raises(ValueError, match="'query' is not text"):
        v1.run_state({"query": 3})

    # a box's inner trace is no run of its own
    inner = MAS([Agent("cut", "Cut {page}", "short", ["page"])], inputs={"page": "text"}, llm=MockLLM())
    boxed = MAS(
        [],
        [("summarize", "fin")],
        boxes=[Box("summarize", inner, reads={"page": "query"}, writes={"short": "short"})],
        steps=[Step("fin", "shout")],
        tools=[shout],
        llm=inner.llm,
    )
    outer: Any = boxed.run("cats")
    assert outer.run_id is not None and outer.records[0].inner.run_id is None


@tool(writes={"loud": "text"})
def shout(short: str) -> str:
    """Shout a text."""
    return short.upper()


def down_at_first(fails: int) -> tuple[Any, list[str]]:
    """A fetch step's callable that meets an outage on its first ``fails`` calls, and the queries it was called on."""
    calls: list[str] = []

    @tool(writes={"page": "text"})
    def fetch(query: str) -> str:
        """Fetch a page; the server is down at first."""
        calls.append(query)
        if len(calls) <= fails:
            raise ConnectionError("the server is down")
        return f"page on {query}"

    return fetch, calls


def test_retry_and_reserved_reads() -> None:
    # a step whose callable fails once with an outage is run again: one record, one retry
    for fails, ok in ((1, True), (2, False)):
        fetch, calls = down_at_first(fails)
        mas = MAS([], steps=[Step("fetch", "fetch")], retry={"fetch": {"outage": 1}}, tools=[fetch])
        trace: Any = mas.run("cats")
        (record,) = trace["fetch"]
        assert (trace.error is None, record.retries, len(calls)) == (ok, 1, 2)
        assert record.error_kind is (None if ok else ErrorKind.OUTAGE)

    # a retried agent costs what every attempt paid: a refused answer, a lost call, then a good answer
    judge = Agent("judge", "Judge {query}", "verdict", output_type="json", schema={"fits": {"type": "bool"}})
    llm = MockLLM(answers=["not json", ConnectionError("down"), '{"fits": true}'], meta=True)
    retried: Any = MAS([judge], retry={"judge": {"outage": 1}}, llm=llm).run("cats")
    (judged,) = retried["judge"]
    assert judged.retries == 1 and judged.outputs == {"verdict": '{"fits": true}'}
    assert judged.cost_usd == pytest.approx(0.02) and judged.tokens == TokenUsage(20, 4)

    # error: the failure routed to a fallback, as {kind, message, node}
    @tool(writes={"page": "text"})
    def slow(query: str) -> str:
        """Time out."""
        raise TimeoutError("the server took too long")

    sorry = Agent("sorry", "Sorry: {error}", "answer", ["error"])
    routed = MAS(
        [sorry],
        [Edge("fetch", "sorry", EdgeKind.ERROR, errors=("timeout",))],
        steps=[Step("fetch", "slow")],
        tools=[slow],
        llm=MockLLM(),
    ).run("cats")
    error = {"kind": "timeout", "message": "TimeoutError: the server took too long", "node": "fetch"}
    assert routed["sorry"][0].rendered_prompt == f"Sorry: {json.dumps(error)}"

    # run: the spend so far, read by an agent, a step and a predicate
    @tool(writes={"spent": "json"})
    def spend(run: dict[str, Any]) -> dict[str, Any]:
        """Keep what the run has spent."""
        return run

    looked = MAS(
        [
            Agent("a", "Answer {query}", "answer"),
            Agent("b", "So far: {run}", "note", ["run"]),
            Agent("more", "More on {query}", "extra"),
        ],
        [
            ("a", "b"),
            ("b", "spend"),
            Edge("spend", "more", EdgeKind.WHEN, when="run.records >= 3 and run.calls == 2"),
            Edge("spend", "fin", EdgeKind.ELSE),
        ],
        steps=[Step("spend", "spend"), Step("fin", "spend", writes={"spent": "final"})],
        tools=[spend],
        llm=MockLLM(meta=True),
    ).run("cats")
    note = json.loads(looked["b"][0].rendered_prompt.removeprefix("So far: "))
    assert (note["calls"], note["records"], note["cost_usd"]) == (1, 1, pytest.approx(0.01))
    spent = json.loads(looked["spend"][0].outputs["spent"])
    assert (spent["calls"], spent["records"], spent["cost_usd"]) == (2, 2, pytest.approx(0.02))
    assert spent["elapsed_s"] >= 0 and looked["spend"][0].route == ("fin",)  # three records only after spend

    reader = Agent("b", "So far: {run}", "note", ["run"])
    uncosted = MAS([Agent("a", "Answer {query}", "answer"), reader], [("a", "b")], llm=MockLLM(), version=2).run("cats")
    assert json.loads(uncosted["b"][0].rendered_prompt.removeprefix("So far: "))["cost_usd"] is None


def test_events_and_the_parallel_call_cap() -> None:
    from .test_chains import pages, plan_of

    def writer(prompt: str) -> str:
        return "a broken page" if '"b"' in prompt else "Page in four words"

    mas, _ = pages(writer)
    events: list[Any] = []
    trace = mas.run_state(plan_of("a", "b"), on_event=events.append)
    assert {event.run_id for event in events} == {trace.run_id}
    per_item = [[(e.kind, e.node, e.visit, e.status) for e in events if e.item == item] for item in (0, 1)]
    assert per_item[0] == [
        ("item_start", "writer", None, None),
        ("start", "writer", 1, None),
        ("end", "writer", 1, "ok"),
        ("start", "count", 1, None),
        ("end", "count", 1, "ok"),
        ("item_end", "writer", None, "ok"),
    ]
    assert [step[3] for step in per_item[1]] == [None, None, "ok", None, "failed", "failed"]
    assert {e.item_key for e in events if e.item == 1} == {"b"}
    fin = [e for e in events if e.node == "fin"]
    assert [e.kind for e in fin] == ["start", "end"] and fin[1].item is None
    assert list(fin[1].to_dict()) == ["kind", "node", "run_id", "at_s", "visit", "status", "cost_usd"]
    item_end = next(e for e in events if e.kind == "item_end" and e.item == 0)
    assert item_end.cost_usd == pytest.approx(0.01)  # the writer's one call; the count is free

    # a listener that raises is warned about once and dropped; the run goes on
    heard: list[Any] = []

    def broken(event: Any) -> None:
        heard.append(event)
        raise RuntimeError("the viewer broke")

    two = MAS(
        [Agent("a", "Answer {query}", "answer"), Agent("b", "Check {answer}", "check", ["answer"])],
        [("a", "b")],
        llm=MockLLM(),
    )
    with pytest.warns(RuntimeWarning, match="the viewer broke") as warned:
        heard_trace = two.run_state({"query": "cats"}, on_event=broken)
    assert heard_trace.error is None and len(heard) == 1 and len(warned) == 1

    # eight agents in parallel never hold more than two calls at once
    lock = threading.Lock()
    now = {"in": 0, "most": 0}

    def slow(prompt: str) -> str:
        with lock:
            now["in"] += 1
            now["most"] = max(now["most"], now["in"])
        time.sleep(0.03)
        with lock:
            now["in"] -= 1
        return "done"

    names = [f"a{i}" for i in range(8)]
    agents = [Agent(name, "Answer {query}", name) for name in names] + [Agent("sum", "Sum", "total", names)]
    for cap in (None, 2):
        now["most"] = 0
        wide = MAS(agents, [(name, "sum") for name in names], llm=MockLLM(slow), budget=Budget(max_parallel_calls=cap))
        assert wide.run("cats").error is None
        assert now["most"] > 2 if cap is None else now["most"] == cap
    with pytest.raises(ValueError, match="max_parallel_calls must be >= 1"):
        Budget(max_parallel_calls=0)
