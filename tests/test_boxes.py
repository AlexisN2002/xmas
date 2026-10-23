"""Version 2 boxes in the engine: a whole inner system runs as one node, on the outer run's meter.

The models are scripted doubles.
"""

from __future__ import annotations

from typing import Any

import pytest

from xmas import MAS, Agent, Box, Budget, ErrorKind, TokenUsage
from xmas.edges import Edge, EdgeKind
from xmas.llms.meta import CallMeta

from .test_steps import MetaLLM


class FlakyLLM(MetaLLM):
    """A scripted model whose answer ``TIMEOUT`` is a timeout instead."""

    def generate_with_meta(self, prompt: str, system_prompt: str = "") -> tuple[str, CallMeta]:
        """Give the next answer, or raise for ``TIMEOUT``."""
        text, meta = super().generate_with_meta(prompt, system_prompt)
        if text == "TIMEOUT":
            raise TimeoutError("the model took too long")
        return text, meta


def build(answers: list[str], **kwargs: Any) -> tuple[MAS, Box]:
    """Writer, then the box ``shorten`` (a one-agent inner system), then poster; a timeout routes to apology."""
    llm = FlakyLLM(answers)
    cutter = Agent("cutter", "Cut this to half its length: {page}", output="short", inputs=["page"])
    inner = MAS([cutter], inputs={"page": "text"}, llm=llm)
    box = Box("shorten", inner, reads={"page": "draft"}, writes={"short": "draft"})
    outer = MAS(
        [
            Agent("writer", "Write about {query}", output="draft"),
            Agent("poster", "Post {draft}", output="post", inputs=["draft"]),
            Agent("apology", "Sorry about {query}", output="post"),
        ],
        [
            ("writer", "shorten"),
            ("shorten", "poster"),
            Edge("shorten", "apology", EdgeKind.ERROR, errors=("timeout",)),
        ],
        boxes=[box],
        state={"draft": {"merge": "last"}, "post": {"merge": "last"}},
        llm=llm,
        **kwargs,
    )
    return outer, box


def test_box_maps_in_and_out_and_nests_records() -> None:
    outer, _ = build(["A long draft about cats.", "Cats.", "Posted."])
    trace = outer.run("cats")
    assert trace.error is None and trace.output == "Posted."

    # maps: the outer draft goes in as the inner page; the inner short comes back as the draft
    (box,) = trace["shorten"]
    assert box.kind == "box" and box.route == ("poster",)
    assert box.inputs == {"draft": "A long draft about cats."} and box.outputs == {"draft": "Cats."}
    assert trace["poster"][0].inputs == {"draft": "Cats."}
    assert box.inner is not None
    assert box.inner.query == '{"page": "A long draft about cats."}' and box.inner.output == "Cats."
    assert [r.rendered_prompt for r in box.inner.records] == ["Cut this to half its length: A long draft about cats."]
    assert box.cost_usd == pytest.approx(0.01) and box.tokens == TokenUsage(10, 2)
    assert box.usage == box.inner.records[0].usage

    # nested records: the trace keeps them inside the box, walk() names the path
    assert [r.agent for r in trace.records] == ["writer", "shorten", "poster"]
    assert [(path, r.agent) for path, r in trace.walk()] == [
        ((), "writer"),
        ((), "shorten"),
        (("shorten",), "cutter"),
        ((), "poster"),
    ]
    data = trace.to_dict()["records"][1]
    assert data["kind"] == "box" and [r["agent"] for r in data["inner"]["records"]] == ["cutter"]
    assert "inner" not in trace.to_dict()["records"][0]

    # a box inside a box: the path names both
    again, _ = build(["A long draft about cats.", "Cats.", "Posted.", "Echoed."])
    wrapper = MAS(
        [Agent("echo", "Echo {post}", output="echoed", inputs=["post"])],
        [("all", "echo")],
        boxes=[Box("all", again, reads={"query": "query"}, writes={"post": "post"})],
        llm=again.llm,
    )
    nested = wrapper.run("cats")
    assert nested.error is None and nested.output == "Echoed."
    assert [(path, r.agent) for path, r in nested.walk()] == [
        ((), "all"),
        (("all",), "writer"),
        (("all",), "shorten"),
        (("all", "shorten"), "cutter"),
        (("all",), "poster"),
        ((), "echo"),
    ]

    # one budget: the outer max_calls counts the inner call
    limited, _ = build(["A long draft about cats.", "Cats.", "Posted."], budget=Budget(max_calls=2))
    failed = limited.run("cats")
    assert failed.error_kind == ErrorKind.BUDGET
    assert [r.agent for r in failed.records] == ["writer", "shorten"]
    assert failed.health is not None and failed.health.budget["calls"] == {"spent": 2, "limit": 2}

    # an inner failure leaves the box with its kind, and the outer `on error` routes it
    routed_mas, _ = build(["A long draft about cats.", "TIMEOUT", "Sorry."])
    routed = routed_mas.run("cats")
    assert routed.error is None and routed.output == "Sorry."
    (failed_box,) = routed["shorten"]
    assert failed_box.kind == "box" and failed_box.error_kind == ErrorKind.TIMEOUT
    assert failed_box.error == "TimeoutError: the model took too long" and failed_box.route == ("apology",)
    assert [r.agent for r in routed.records] == ["writer", "shorten", "apology"]
    assert failed_box.inner is not None and failed_box.inner.error == "TimeoutError: the model took too long"
    assert [(r.agent, r.error_kind) for r in failed_box.inner.records] == [("cutter", ErrorKind.TIMEOUT)]


def test_a_failed_box_keeps_its_inner_trace_and_cost() -> None:
    """The inner system's paid calls before its failure stay in the box's failed record, and in the trace's cost."""
    llm = FlakyLLM(["A draft.", "Cut.", "TIMEOUT", "Sorry."])
    inner = MAS(
        [
            Agent("cutter", "Cut {page}", output="short", inputs=["page"]),
            Agent("polish", "Polish {short}", output="final", inputs=["short"]),
        ],
        [("cutter", "polish")],
        inputs={"page": "text"},
        llm=llm,
    )
    outer = MAS(
        [Agent("writer", "Write {query}", output="draft"), Agent("apology", "Sorry {query}", output="post")],
        [("writer", "shorten"), Edge("shorten", "apology", EdgeKind.ERROR, errors=("timeout",))],
        boxes=[Box("shorten", inner, reads={"page": "draft"}, writes={"final": "post"})],
        state={"post": {"merge": "last"}},
        llm=llm,
    )
    for _ in range(2):  # a second run on the same inner system shows only its own records
        llm.answers = ["A draft.", "Cut.", "TIMEOUT", "Sorry."]
        trace = outer.run("cats")
        assert trace.error is None and trace.output == "Sorry."
        (box,) = trace["shorten"]
        assert box.inner is not None and [r.agent for r in box.inner.records] == ["cutter", "polish"]
        assert box.inner.records[1].error_kind == ErrorKind.TIMEOUT and box.inner.error_kind == ErrorKind.TIMEOUT
        assert box.cost_usd == pytest.approx(0.01) and box.tokens == TokenUsage(10, 2)
        assert trace.cost_usd == pytest.approx(0.03)  # writer, cutter and apology: what the meter paid
        assert [(path, r.agent) for path, r in trace.walk()][1:4] == [
            ((), "shorten"),
            (("shorten",), "cutter"),
            (("shorten",), "polish"),
        ]
