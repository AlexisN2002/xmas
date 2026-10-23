"""Version 2 in the engine: typed values, steps, one callable registry, answer forms with one retry.

No network but the in-process MCP server of one case: the models are scripted
doubles.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from xmas import MAS, Agent, Budget, ErrorKind, TokenUsage, ToolCall
from xmas import runtime as engine
from xmas.budget import RunMeter
from xmas.edges import Edge, EdgeKind
from xmas.errors import BadFormat, EmptyResponse, XmasError
from xmas.nodes import Step
from xmas.schemas import RETRY_NOTE
from xmas.tools import Tool, ToolSource, index_tools, tool
from xmas.trace import Rejected
from xmas.values import JSON, TEXT

from .conftest import META, MockLLM, req


def invoke(mas: MAS, query: str) -> tuple[RunMeter, dict[str, Any] | BaseException]:
    """Run once on a meter of our own, returning the meter and the final state or the error."""
    meter = RunMeter(mas.budget, owner=mas)
    try:
        return meter, mas.compiled.invoke(engine._initial_state(query, mas.prompts()), engine._run_config(mas, meter))
    except Exception as error:
        return meter, error


# ---------------------------------------------------------------- callables


@tool(writes={"counts": "json"})
def count_points(plan: dict[str, Any]) -> dict[str, int]:
    """Count a plan's points and their words."""
    return {"points": len(plan["points"]), "words": sum(len(point.split()) for point in plan["points"])}


@tool
def shout(text: str) -> str:
    """Shout a text."""
    return text.upper() + "!"


FORM = {"topic": {"type": "text"}, "points": {"type": "list of text"}}
FORMAT = "Answer with one JSON object and nothing else. Its fields:\n- topic (text)\n- points (list of text)"
PLAN = {"topic": "cats", "points": ["sleep a lot", "purr"]}


# -------------------------------------------------------------------- tests


def test_a_step_runs_between_agents_with_typed_values() -> None:
    def answer(prompt: str) -> str:
        return "```json\n" + json.dumps(PLAN) + "\n```" if prompt.startswith("Plan") else "A note."

    def build(**kwargs: Any) -> tuple[MAS, MockLLM]:
        llm = MockLLM(answer)
        planner = Agent("planner", "Plan a note on {query}.", output="plan", output_type="json", schema=FORM)
        writer = Agent("writer", "Write about {plan} using {counts}.", output="note", inputs=["plan", "counts"])
        mas = MAS(
            [planner, writer],
            # writer is fed by planner and by count, at unequal depth: a barrier, so it runs once
            [("planner", "count"), ("count", "writer"), ("planner", "writer")],
            steps=[Step("count", "count_points")],
            llm=llm,
            tools=[count_points],
            **kwargs,
        )
        return mas, llm

    mas, llm = build()
    assert count_points.reads == {"plan": JSON}
    assert count_points.writes == {"counts": JSON}
    trace = mas.run("cats")

    assert trace.error is None
    assert [(r.agent, r.kind) for r in trace.records] == [("planner", "agent"), ("count", "step"), ("writer", "agent")]
    plan_text = '{"topic": "cats", "points": ["sleep a lot", "purr"]}'
    assert llm.calls == [
        f"Plan a note on cats.\n\n{FORMAT}",
        f'Write about {plan_text} using {{"points": 2, "words": 4}}.',
    ]
    planner, step, writer = trace.records
    assert planner.outputs == {"plan": plan_text}
    assert planner.raw_response.startswith("```json")
    assert step.prompt == step.rendered_prompt == ""
    assert step.inputs == {"plan": plan_text}
    assert step.outputs == {"counts": '{"points": 2, "words": 4}'}
    assert step.raw_response == '{"points": 2, "words": 4}'
    assert step.model is None and step.cost_usd == 0.0 and step.latency_s is not None  # a step makes no call
    assert writer.inputs == {"plan": plan_text, "counts": '{"points": 2, "words": 4}'}
    assert trace.output == "A note."
    assert trace.to_dict()["records"][1]["kind"] == "step"

    # a step makes no model call: with one call allowed, it still runs, and the writer is refused
    limited, _ = build(budget=Budget(max_calls=1))
    failed = limited.run("cats")
    assert failed.error_kind == ErrorKind.BUDGET
    assert [r.agent for r in failed.records] == ["planner", "count"]

    # a system whose inputs are not exactly {query: text} does not run on a query
    boxed, _ = build(inputs={"query": TEXT, "style": TEXT})
    with pytest.raises(ValueError, match="runs as a box, or with run_state in phase 3"):
        boxed.run("cats")


@contextmanager
def mcp_server() -> Iterator[str]:
    """Serve ``shout`` and ``tally`` from an in-process MCP server; yield its URL."""
    pytest.importorskip("mcp")
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("v2-steps")

    @server.tool(name="shout")
    def remote_shout(text: str) -> str:
        """Shout a text."""
        return text.upper() + "!"

    @server.tool()
    def tally(text: str) -> str:
        """Count a text's characters, as JSON."""
        return json.dumps({"chars": len(text)})

    config = uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started:
        assert time.monotonic() < deadline, "the MCP server did not start"
        time.sleep(0.01)
    yield f"http://127.0.0.1:{uv.servers[0].sockets[0].getsockname()[1]}/mcp"
    uv.should_exit = True
    thread.join(timeout=10)


@pytest.mark.parametrize("where", ["local", "mcp"])
def test_one_callable_is_a_tool_and_a_step(where: str) -> None:
    def run(tools: list[Any]) -> Any:
        llm = MockLLM(turns=[("", [req("shout", '{"text": "hi"}')]), ("said hi", [])])
        writer = Agent("writer", "Greet {query}.", output="draft", tools=["shout"])
        steps = [Step("loud", "shout", reads={"text": "draft"}, writes={"shout": "loud"})]
        edges = [("writer", "loud")]
        if where == "mcp":
            steps.append(Step("size", "tally", reads={"text": "loud"}))
            edges.append(("loud", "size"))
        mas = MAS([writer], edges, steps=steps, llm=llm, tools=tools)
        assert list(index_tools(mas.tools)) == (["shout"] if where == "local" else ["shout", "tally"])
        return mas.run("Tom")

    if where == "local":
        trace = run([shout])
        assert shout.reads == {"text": TEXT} and shout.writes == {"shout": TEXT}
    else:
        from xmas.tools.mcp import MCPTools

        with mcp_server() as url, MCPTools(url, "sk-any", writes={"tally": {"tally": "json"}}) as remote:
            listed = {spec.name: spec for spec in remote.tools()}
            assert listed["shout"].writes == {"shout": TEXT}
            assert listed["tally"].writes == {"tally": JSON}
            assert listed["shout"].reads == {"text": TEXT}
            trace = run([remote])

    assert trace.error is None
    writer, loud = trace.records[:2]
    assert writer.tool_calls == (ToolCall("shout", {"text": "hi"}, "HI!"),)
    assert (loud.kind, loud.inputs, loud.outputs) == ("step", {"draft": "said hi"}, {"loud": "SAID HI!"})
    if where == "local":
        assert trace.output == "SAID HI!"
    else:
        size = trace.records[2]
        assert (size.inputs, size.outputs) == ({"loud": "SAID HI!"}, {"tally": '{"chars": 8}'})
        assert trace.output == '{"chars": 8}'


VERDICT = {"fits": {"type": "bool", "means": "the draft fits"}}
JUDGE_PROMPT = (
    "Judge {query}.\n\nAnswer with one JSON object and nothing else. Its fields:\n"
    "- fits (true or false): the draft fits"
)
PROSE = "<root>: Invalid JSON: expected value at line 1 column 1"


def judge_mas(llm: MockLLM, tools: list[Any] | None = None) -> MAS:
    judge = Agent(
        "judge",
        "Judge {query}.",
        output="verdict",
        output_type="json",
        schema=VERDICT,
        tools=["shout"] if tools else None,
    )
    return MAS([judge], [], llm=llm, tools=tools or [])


@pytest.mark.parametrize("case", ["plain ok after retry", "plain bad twice", "tool agent's extra turn"])
def test_answer_form_retries_once_then_bad_format(case: str) -> None:
    rendered = JUDGE_PROMPT.replace("{query}", "this")
    retry = RETRY_NOTE.format(problem=PROSE)

    if case == "plain ok after retry":
        llm = MockLLM(answers=['Sure: {"fits": true}', '{"fits": true}'], meta=True)
        meter, state = invoke(judge_mas(llm), "this")
        assert not isinstance(state, BaseException)
        assert llm.calls == [rendered, f"{rendered}\n\n{retry}"]
        (record,) = state["records"]
        assert record.raw_response == '{"fits": true}'
        assert record.rejected == (Rejected('Sure: {"fits": true}', PROSE),)
        assert record.rendered_prompt == rendered
        assert state["values"]["verdict"] == {"fits": True}
        assert record.cost_usd == pytest.approx(0.02)
        assert record.tokens == TokenUsage(20, 4)
        assert meter.calls == 2 and meter.cost_usd == pytest.approx(0.02)
        trace = judge_mas(MockLLM(answers=['Sure: {"fits": true}', '{"fits": true}'], meta=True)).run("this")
        assert trace.output == '{"fits": true}'
        assert trace.to_dict()["records"][0]["rejected"] == [{"raw_response": 'Sure: {"fits": true}', "error": PROSE}]

    elif case == "plain bad twice":
        llm = MockLLM(answers=["Sure.", '{"fits": "perhaps"}'], meta=True)
        meter, error = invoke(judge_mas(llm), "this")
        assert isinstance(error, BadFormat)
        assert error.node == "judge"
        assert error.problems == (PROSE, "fits: Input should be a valid boolean, unable to interpret input")
        assert error.meta is not None and error.meta.cost_usd == pytest.approx(0.02)
        assert meter.calls == 2 and meter.cost_usd == pytest.approx(0.02)
        assert [(r.agent, r.error_kind) for r in meter.records] == [("judge", ErrorKind.BAD_FORMAT)]
        assert meter.records[0].cost_usd == pytest.approx(0.02)
        trace = judge_mas(MockLLM(answers=["Sure.", "Sure."], meta=True)).run("this")
        assert trace.error_kind == ErrorKind.BAD_FORMAT
        assert trace.error is not None and "after the retry" in trace.error
        # the retry passes the meter too: with one call allowed it is refused
        limited = judge_mas(MockLLM(answers=["Sure.", '{"fits": true}'], meta=True))
        limited.budget = Budget(max_calls=1)
        assert limited.run("this").error_kind == ErrorKind.BUDGET

    else:
        llm = MockLLM(meta=True, turns=[("", [req("shout", '{"text": "hi"}')]), ("Sure.", []), ('{"fits": true}', [])])
        meter, state = invoke(judge_mas(llm, tools=[shout]), "this")
        assert not isinstance(state, BaseException)
        offered = [names for _, names in llm.seen]
        assert offered == [["shout"], ["shout"], []]
        last = llm.seen[2][0]
        assert last[1] == {"role": "user", "content": rendered}
        assert last[-2:] == [
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": RETRY_NOTE.format(problem=PROSE)},
        ]
        (record,) = state["records"]
        assert record.tool_calls == (ToolCall("shout", {"text": "hi"}, "HI!"),)
        assert record.raw_response == '{"fits": true}'
        assert record.rejected == (Rejected("Sure.", PROSE),)
        assert meter.calls == 3 and meter.cost_usd == record.cost_usd == pytest.approx(0.03)
        # a second miss fails the run with both problems
        again = MockLLM(meta=True, turns=[("Sure.", []), ("Sure!", [])])
        _, error = invoke(judge_mas(again, tools=[shout]), "this")
        assert isinstance(error, BadFormat) and error.problems == (PROSE, PROSE)


def test_a_trace_costs_what_the_meter_paid() -> None:
    """A step costs 0; a failed visit's record holds every paid call of the visit, retries and tool turns included."""
    loud = Agent("writer", "Write {query}", output="draft")
    mas = MAS(
        [loud],
        [("writer", "shout")],
        steps=[Step("shout", "shout", {"text": "draft"})],
        tools=[shout],
        llm=MockLLM(answers=["a draft"], meta=True),
    )
    trace = mas.run("cats")
    assert [(r.agent, r.cost_usd) for r in trace.records] == [("writer", 0.01), ("shout", 0.0)]
    assert trace.cost_usd == pytest.approx(0.01) and trace.output == "A DRAFT!"

    cut = EmptyResponse("Empty content from model", kind=ErrorKind.LENGTH_CUT, meta=META)  # on the retry

    class Refusing(ToolSource):
        """Offers shout, and refuses every call of it."""

        def tools(self) -> list[Tool]:
            return [Tool("shout", "Shout.", {"type": "object", "properties": {"text": {"type": "string"}}})]

        def call(self, name: str, arguments: dict[str, Any]) -> str:
            raise XmasError("forbidden (403)", status=403)

    sorry = Agent("sorry", "Sorry about {query}", output="said")
    judge = Agent("judge", "Judge {query}", output="verdict", output_type="json", schema={"fits": {"type": "bool"}})
    caller = Agent("judge", "Judge {query}", output="verdict", tools=["shout"])
    cases = [
        (judge, MockLLM(answers=["Sure.", cut, "We are sorry."], meta=True), (), ErrorKind.LENGTH_CUT, 0.02),
        (
            caller,
            MockLLM(answers=["We are sorry."], meta=True, turns=[("", [req("shout", '{"text": "hi"}')])]),
            [Refusing()],
            ErrorKind.REFUSAL,
            0.01,
        ),
    ]
    for agent, llm, tools, kind, paid in cases:
        meter, _ = invoke(
            MAS([agent, sorry], [Edge("judge", "sorry", EdgeKind.ERROR, errors=("any",))], llm=llm, tools=tools), "cats"
        )
        failed, apology = meter.records
        assert (failed.agent, failed.error_kind, failed.route) == ("judge", kind, ("sorry",))
        calls = round(paid / 0.01)
        assert failed.cost_usd == pytest.approx(paid) and failed.tokens == TokenUsage(10 * calls, 2 * calls)
        assert apology.cost_usd == pytest.approx(0.01) and meter.cost_usd == pytest.approx(paid + 0.01)


def test_optional_text_parameter_and_json_null() -> None:
    @tool
    def tidy(draft: str | None = None) -> str:
        """Tidy a draft."""
        return (draft or "").strip().upper()

    assert tidy.reads == {"draft": TEXT}  # str | None reads text, so a text value may feed it
    writer = Agent("writer", "Write {query}", output="draft")
    mas = MAS(
        [writer],
        [("writer", "tidy")],
        steps=[Step("tidy", "tidy")],
        tools=[tidy],
        llm=MockLLM(answers=[" cats "], meta=True),
    )
    assert mas.run("q").output == "CATS"

    # a json null is a value and renders as null (a value not there yet renders empty: test_operators)
    judge = Agent("judge", "Judge {query}", output="verdict", output_type="json")
    reader = Agent("reader", "Read {verdict}", output="note", inputs=["verdict"])
    llm = MockLLM(answers=["null", "noted"], meta=True)
    trace = MAS([judge, reader], [("judge", "reader")], llm=llm, inputs={"query": "text"}).run("q")
    assert trace["judge"][0].outputs == {"verdict": "null"}
    assert trace["reader"][0].inputs == {"verdict": "null"} and llm.calls[1] == "Read null"
    alone = MAS([judge], llm=MockLLM(answers=["null"], meta=True), inputs={"query": "text"}).run("q")
    assert alone.output == "null" and alone.health is not None and alone.health.flags == ()
