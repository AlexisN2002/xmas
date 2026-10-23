"""Version 2 operators in the engine: when, else, while, switch, merges, joins and ``on error``.

The models are scripted doubles. Until the phase 2b check (package B2) accepts
conditional edges and loops, :func:`system` builds the systems around the
base's ``check_system`` and supplies the joins the engine compiles; every use is
marked "B2: real MAS(...)" and switches to the plain constructor at integration.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple
from unittest import mock

from xmas import MAS, Agent, Budget, ErrorKind
from xmas import mas as mas_module
from xmas.check import SystemInfo, callable_writes, step_maps
from xmas.edges import Edge, EdgeKind
from xmas.errors import BudgetExceeded
from xmas.graph import END, build_graph
from xmas.nodes import Step
from xmas.tools import tool
from xmas.values import ValueType, parse_type

from .conftest import MockLLM
from .test_v2_values import MetaLLM

# ------------------------------------------------------------------- building


class Join(NamedTuple):
    """B2: graph.Join, as ``classify_joins`` gives it."""

    kind: str
    groups: tuple[tuple[str, ...], ...]


def system(
    agents: list[Agent],
    edges: Sequence[tuple[str, str] | Edge],
    *,
    joins: dict[str, Join] | None = None,
    boxes: Sequence[Any] = (),
    **kwargs: Any,
) -> MAS:
    """B2: real MAS(...). Build a phase 2b system the base check still refuses.

    The base check is replaced by one that only types the values, and the joins
    are the ones given; the topology wires start and end on the forward edges
    (``while`` edges left out), as phase 2b's check does.
    """
    declared = [edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges]

    def check(mas: MAS) -> SystemInfo:
        types: dict[str, ValueType] = dict(mas.inputs)
        for agent in mas.agents.values():
            types[agent.output] = parse_type(agent.output_type)
        for step in mas.steps.values():
            item, _ = mas.resolve_callable(step)
            for inner, outer in step_maps(step, item)[1].items():
                types[outer] = callable_writes(item)[inner]
        for box in boxes:
            inner_types = box.system.info.types
            types.update({outer: inner_types[inner] for inner, outer in box.writes.items()})
        return SystemInfo(types, {}, dict(joins or {}))  # type: ignore[arg-type]

    names = [agent.name for agent in agents]
    names += [step.name for step in kwargs.get("steps", ())] + [box.name for box in boxes]
    graph = build_graph(names, [(e.source, e.target) for e in declared if e.kind is not EdgeKind.WHILE])
    graph.add_edges_from((e.source, e.target) for e in declared if e.kind is EdgeKind.WHILE)
    with (
        mock.patch.object(mas_module, "check_system", check),
        mock.patch.object(mas_module, "build_graph", lambda *_: graph),
    ):
        built = MAS(agents, declared, **kwargs)
    setattr(built, "boxes", {box.name: box for box in boxes})  # noqa: B010 (B2 adds the attribute)
    return built


# ----------------------------------------------------------------- callables


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


def worked_example(verdicts: Iterator[str], **kwargs: Any) -> MAS:
    """§3's system without the box and web_fetch, the judge answering from ``verdicts``."""
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
    return system(  # B2: real MAS(...)
        [writer, judge],
        [
            ("writer", "count"),
            ("count", "judge"),
            Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3),
            Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action"),
            Edge("judge", "apology", EdgeKind.ELSE),
        ],
        joins={END: Join("merge", (("apology", "publish"),))},
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


# --------------------------------------------------------------------- tests


def test_when_else_while_switch() -> None:
    no, yes = '{"fits": false, "action": "publish"}', '{"fits": true, "action": "publish"}'

    # the back edge twice, then the switch case
    trace = worked_example(iter([no, no, yes])).run("cats")
    assert trace.error is None
    assert [r.agent for r in trace.records] == ["writer", "count", "judge"] * 3 + ["publish"]
    assert [r.route for r in trace["judge"]] == [("writer",), ("writer",), ("publish",)]
    assert [r.route for r in trace.records if r.agent != "judge"] == [None] * 7
    assert not any(r.capped for r in trace.records)
    assert trace["writer"][1].inputs["verdict"] == '{"fits": false, "action": "publish"}'
    assert trace["writer"][0].inputs["verdict"] == ""  # a first-visit gap renders empty, so nothing is unfilled
    assert trace.output == "Published: A short draft about cats."
    assert trace.health is not None and trace.health.loops == {"judge -> writer": {"taken": 2, "cap_hit": False}}
    assert trace.to_dict()["records"][2]["route"] == ["writer"]
    assert "capped" not in trace.to_dict()["records"][2]

    # the loop at its cap: capped, and the else route (§7's 2b snippet)
    trace = worked_example(itertools.repeat(no)).run("cats")
    assert [r.agent for r in trace.records] == ["writer", "count", "judge"] * 3 + ["apology"]
    assert [(r.route, r.capped) for r in trace["judge"]] == [(("writer",), False)] * 2 + [(("apology",), True)]
    assert trace.output == "Sorry, no short answer about cats."
    assert trace.health is not None
    assert trace.health.loops == {"judge -> writer": {"taken": 2, "cap_hit": True}}
    assert sorted({(f.signal, f.node) for f in trace.health.flags}) == [
        ("stuck", "count"),
        ("stuck", "judge"),
        ("stuck", "writer"),
    ]
    assert trace.to_dict()["records"][8]["capped"] is True
    assert trace.to_dict()["health"]["loops"] == {"judge -> writer": {"taken": 2, "cap_hit": True}}

    # a switch value outside the cases takes else
    trace = worked_example(iter(['{"fits": true, "action": "shorten"}'])).run("cats")
    assert [r.agent for r in trace.records] == ["writer", "count", "judge", "apology"]
    assert trace["judge"][0].route == ("apology",)

    # a when edge: the first that holds wins, in declared order
    llm = MockLLM(lambda p: "12" if p.startswith("Score") else f"done: {p}")
    scorer = Agent("scorer", "Score {query}", output="score", output_type="json")
    high, low = Agent("high", "High {score}", output="note", inputs=["score"]), Agent("low", "Low", output="note")
    mas = system(  # B2: real MAS(...)
        [scorer, high, low],
        [
            Edge("scorer", "high", EdgeKind.WHEN, when="score > 10"),
            Edge("scorer", "low", EdgeKind.WHEN, when="score > 5"),
        ],
        joins={END: Join("merge", (("high", "low"),))},
        state={"note": {"merge": "last"}},
        llm=llm,
    )
    trace = mas.run("x")
    assert [r.agent for r in trace.records] == ["scorer", "high"] and trace.output == "done: High 12"


def test_merges_and_joins() -> None:
    """S2: a loop with a when/else merge ``m`` inside joins a parallel chain at ``fin``, which runs once."""
    drafts = iter(["A rather long first draft.", "Short.", "Short."])

    def answer(prompt: str) -> str:
        return next(drafts) if prompt.startswith("Draft") else prompt.split(":")[0]

    @tool(writes={"verdict": "json"})
    def check(draft: str) -> dict[str, bool]:
        """Say whether a draft is long."""
        return {"long": len(draft) > 10}

    def agent(name: str, prompt: str, output: str, inputs: list[str]) -> Agent:
        return Agent(name, prompt, output=output, inputs=inputs)

    agents = [
        agent("s1", "S1: {query}", "one", ["query"]),
        agent("s2", "S2: {one}", "two", ["one"]),
        agent("s3", "S3: {two}", "three", ["two"]),
        agent("w", "Draft {query}", "draft", ["query"]),
        agent("a", "A: {draft}", "cut", ["draft"]),
        agent("m", "M: {draft}", "note", ["draft"]),
        agent("fin", "Fin: {three} {note}", "final", ["three", "note"]),
    ]
    mas = system(  # B2: real MAS(...)
        agents,
        [
            ("s1", "s2"),
            ("s2", "s3"),
            ("s3", "fin"),
            ("w", "c"),
            Edge("c", "a", EdgeKind.WHEN, when="verdict.long"),
            Edge("c", "m", EdgeKind.ELSE),
            ("a", "m"),
            Edge("m", "w", EdgeKind.WHILE, when="true", max_visits=3),
            Edge("m", "fin", EdgeKind.ELSE),
        ],
        # B2: classify_joins gives these
        joins={"m": Join("merge", (("a", "c"),)), "fin": Join("join", (("s3",), ("m",)))},
        steps=[Step("c", "check")],
        llm=MockLLM(answer),
        tools=[check],
    )
    trace = mas.run("cats")
    assert trace.error is None
    names = [r.agent for r in trace.records]
    assert names.count("fin") == 1 and names[-1] == "fin"
    assert names.count("m") == 3 and names.count("w") == 3 and names.count("a") == 1
    assert names.index("s3") < names.index("fin")
    assert [r.route for r in trace["c"]] == [("a",), ("m",), ("m",)]
    assert [(r.route, r.capped) for r in trace["m"]] == [(("w",), False)] * 2 + [(("fin",), True)]
    assert trace["fin"][0].inputs == {"three": "S3", "note": "M"}
    assert trace.output == "Fin"
    assert "__in__.fin.1" not in names  # a relay runs, but leaves no record


def test_on_error_routes_by_kind() -> None:
    @tool(writes={"page": "text"})
    def fetch(query: str) -> str:
        """Fetch a page, or fail as the query says."""
        if query == "slow":
            raise TimeoutError("the server took too long")
        if query == "down":
            raise ConnectionError("the server is down")
        return f"page on {query}"

    @tool(writes={"page": "text"})
    def fallback(query: str) -> str:
        """Give a cached page."""
        return f"cached page on {query}"

    @tool(writes={"extra": "text"})
    def side(query: str) -> str:
        """Give a side note."""
        return f"note on {query}"

    joiner = Agent("j", "Join {page} and {extra}", output="answer", inputs=["page", "extra"])
    mas = system(  # B2: real MAS(...)
        [joiner],
        [
            ("fetch", "j"),
            Edge("fetch", "fb", EdgeKind.ERROR, errors=("timeout",)),
            ("fb", "j"),
            ("side", "j"),
        ],
        # B2: classify_joins gives this; fetch's ok route and fb are alternatives
        joins={"j": Join("join", (("fb", "fetch"), ("side",)))},
        steps=[Step("fetch", "fetch"), Step("fb", "fallback"), Step("side", "side")],
        state={"page": {"merge": "last"}},
        llm=MockLLM(lambda p: p.upper()),
        tools=[fetch, fallback, side],
    )

    ok = mas.run("cats")
    assert ok.error is None and ok.output == "JOIN PAGE ON CATS AND NOTE ON CATS"
    assert sorted(r.agent for r in ok.records) == ["fetch", "j", "side"]
    assert ok["fetch"][0].route == ("j",)

    # a timeout takes its route: a failed record, no writes, and the join still runs once
    routed = mas.run("slow")
    assert routed.error is None and routed.error_kind is None
    assert routed.output == "JOIN CACHED PAGE ON SLOW AND NOTE ON SLOW"
    assert [r.agent for r in routed.records].count("j") == 1
    (failed,) = routed["fetch"]
    assert failed.error == "TimeoutError: the server took too long"
    assert failed.error_kind == ErrorKind.TIMEOUT and failed.outputs == {} and failed.route == ("fb",)
    assert failed.kind == "step" and failed.latency_s is not None
    assert failed.to_dict()["error"] == "TimeoutError: the server took too long"
    assert routed.health is not None
    assert [(f.signal, f.node) for f in routed.health.flags] == [("error_routed", "fetch")]

    # an outage has no route: the run fails as in phase 1, and keeps the failed record
    unrouted = mas.run("down")
    assert unrouted.error_kind == ErrorKind.OUTAGE and unrouted.output == ""
    assert [(r.agent, r.error_kind) for r in unrouted.records if r.error] == [("fetch", ErrorKind.OUTAGE)]
    assert unrouted["fetch"][0].route is None

    # a bad format after the retry is routed with the cost of both paid attempts
    judge = Agent("judge", "Judge {query}", output="verdict", output_type="json", schema=VERDICT)
    sorry = Agent("sorry", "Sorry about {query}", output="verdict_text")
    llm = MetaLLM(["not json", "still not json", "We are sorry."])
    mas = system(  # B2: real MAS(...)
        [judge, sorry], [Edge("judge", "sorry", EdgeKind.ERROR, errors=("bad_format",))], llm=llm
    )
    trace = mas.run("cats")
    assert trace.error is None and trace.output == "We are sorry."
    (failed,) = trace["judge"]
    assert failed.error_kind == ErrorKind.BAD_FORMAT and failed.cost_usd is not None
    assert abs(failed.cost_usd - 0.02) < 1e-9 and failed.route == ("sorry",)
    assert trace.health is not None
    assert [f.signal for f in trace.health.flags] == ["bad_format", "error_routed"]

    # a budget refusal is never routed, not even by `any`; B2: MAS(...) refuses `on error: [budget]`
    llm = MetaLLM(["unused"])
    mas = system(  # B2: real MAS(...)
        [judge, sorry],
        [Edge("judge", "sorry", EdgeKind.ERROR, errors=("any",))],
        llm=llm,
        budget=Budget(max_calls=0),
    )
    trace = mas.run("cats")
    assert trace.error_kind == ErrorKind.BUDGET and trace.records == ()
    assert trace.error is not None and trace.error.startswith(BudgetExceeded.__name__)
