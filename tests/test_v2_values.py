"""Version 2 in the engine: typed values, steps, one callable registry, answer forms with one retry.

No network but the in-process MCP server of one case: the models are scripted
doubles.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from xmas import MAS, Agent, Budget, ErrorKind, TokenUsage, ToolCall
from xmas import _engine as engine
from xmas.budget import RunMeter
from xmas.errors import BadFormat
from xmas.llms.meta import CallMeta
from xmas.nodes import Step
from xmas.schemas import RETRY_NOTE
from xmas.tools import Tool, index_tools, tool
from xmas.trace import Rejected
from xmas.values import JSON, TEXT

from .test_tools import ToolMockLLM, Turn, req

# --------------------------------------------------------------------- doubles


META = CallMeta(latency_s=0.5, tokens=TokenUsage(10, 2), cost_usd=0.01)


class MetaLLM(ToolMockLLM):
    """A scripted model that reports call metadata, so the meter can sum cost."""

    full_response = True

    def __init__(self, answers: list[str] | None = None, turns: list[Turn] | None = None) -> None:
        """Script the plain answers (in order) and the tool turns."""
        super().__init__(turns)
        self.answers = list(answers or [])

    def generate_with_meta(self, prompt: str, system_prompt: str = "") -> tuple[str, CallMeta]:
        """Record the prompt and give the next scripted answer."""
        self.calls.append(prompt)
        return self.answers.pop(0), META

    def get_tool_response_meta(
        self, messages: list[dict[str, Any]], tools: Sequence[Tool]
    ) -> tuple[str, Any, CallMeta]:
        """Answer from the tool script, with metadata."""
        text, requests = self.get_tool_response(messages, tools)
        return text, requests, META


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

    def build(**kwargs: Any) -> tuple[MAS, ToolMockLLM]:
        llm = ToolMockLLM()
        llm.responder = answer
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
    assert step.model is None and step.cost_usd is None and step.latency_s is not None
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
        llm = ToolMockLLM([("", [req("shout", '{"text": "hi"}')]), ("said hi", [])])
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


def judge_mas(llm: MetaLLM, tools: list[Any] | None = None) -> MAS:
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
        llm = MetaLLM(['Sure: {"fits": true}', '{"fits": true}'])
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
        trace = judge_mas(MetaLLM(['Sure: {"fits": true}', '{"fits": true}'])).run("this")
        assert trace.output == '{"fits": true}'
        assert trace.to_dict()["records"][0]["rejected"] == [{"raw_response": 'Sure: {"fits": true}', "error": PROSE}]

    elif case == "plain bad twice":
        llm = MetaLLM(["Sure.", '{"fits": "perhaps"}'])
        meter, error = invoke(judge_mas(llm), "this")
        assert isinstance(error, BadFormat)
        assert error.node == "judge"
        assert error.problems == (PROSE, "fits: Input should be a valid boolean, unable to interpret input")
        assert error.meta is not None and error.meta.cost_usd == pytest.approx(0.02)
        assert meter.calls == 2 and meter.cost_usd == pytest.approx(0.02) and meter.records == []
        trace = judge_mas(MetaLLM(["Sure.", "Sure."])).run("this")
        assert trace.error_kind == ErrorKind.BAD_FORMAT
        assert trace.error is not None and "after the retry" in trace.error
        # the retry passes the meter too: with one call allowed it is refused
        limited = judge_mas(MetaLLM(["Sure.", '{"fits": true}']))
        limited.budget = Budget(max_calls=1)
        assert limited.run("this").error_kind == ErrorKind.BUDGET

    else:
        llm = MetaLLM(turns=[("", [req("shout", '{"text": "hi"}')]), ("Sure.", []), ('{"fits": true}', [])])
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
        again = MetaLLM(turns=[("Sure.", []), ("Sure!", [])])
        _, error = invoke(judge_mas(again, tools=[shout]), "this")
        assert isinstance(error, BadFormat) and error.problems == (PROSE, PROSE)
