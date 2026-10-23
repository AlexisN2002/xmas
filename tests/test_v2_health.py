"""Health signals on hand-built records (plan test 13)."""

from __future__ import annotations

import json
from typing import Any

from xmas.budget import Budget, RunMeter
from xmas.errors import ErrorKind
from xmas.health import Flag, Health, HealthSettings, summarize
from xmas.trace import AgentRecord, Rejected


def rec(
    agent: str,
    step: int,
    output: str | None,
    *,
    prompt: str = "",
    inputs: dict[str, str] | None = None,
    error_kind: ErrorKind | None = None,
    **later: Any,
) -> AgentRecord:
    """Build a record; ``later`` sets the fields phase 2 added (rejected, error, route, capped)."""
    outputs = {} if output is None else {"out": output}
    return AgentRecord(agent, step, prompt, prompt, inputs or {}, outputs, error_kind=error_kind, **later)


RECORDS = (
    rec("w", 0, "a" * 9, prompt="Write {query}", inputs={"query": "cats"}),
    rec("judge", 1, '{"fits": false}', rejected=(Rejected("Sure.", "<root>: Invalid JSON"),), route=("w",)),
    rec("w", 2, "b" * 9),
    rec("judge", 3, '{"fits": false}', route=("w",)),
    rec("w", 4, "c" * 9),
    rec("judge", 5, '{"fits": true}', route=("pub",), capped=True),
    rec("w", 6, "d" * 27),  # 3 times the median of 9: not long
    rec("w", 7, "e" * 28),  # just beyond: long
    rec("w", 8, "f" * 3),  # the median divided by 3: not short
    rec("w", 9, "g" * 2),  # just under: short
    rec("pub", 10, " \n", prompt="Publish {draft} for {reader}", inputs={"draft": ""}),
    rec("count", 11, None, error_kind=ErrorKind.BAD_FORMAT, error="count: the answer was not JSON"),
    rec("few", 12, "x"),  # two samples: under min_samples, no median
    rec("few", 13, "y" * 100),
    rec("leaky", 14, "Done <|im_end|>"),
)


def test_signals() -> None:
    meter = RunMeter(Budget(max_calls=10, max_cost_usd=1.0))
    meter.calls, meter.cost_usd = 4, 0.5
    health = summarize(RECORDS, budget=None, meter=meter, settings=HealthSettings())

    assert [(flag.signal, flag.node, flag.step) for flag in health.flags] == [
        ("retried", "judge", 1),
        ("stuck", "judge", 3),
        ("long", "w", 7),
        ("short", "w", 9),
        ("empty", "pub", 10),
        ("unfilled", "pub", 10),
        ("bad_format", "count", 11),
        ("error_routed", "count", 11),
        ("leak", "leaky", 14),
    ]
    by_signal = {flag.signal: flag for flag in health.flags}
    assert by_signal["retried"] == Flag("retried", "judge", 1, "1 answer(s) rejected: <root>: Invalid JSON")
    assert by_signal["stuck"].detail == "the same output as step 1"
    assert by_signal["long"].detail == "28 characters, the node's median is 9"
    assert by_signal["unfilled"].detail == "placeholders without a value: reader"
    assert by_signal["error_routed"].detail == "bad_format: count: the answer was not JSON"
    assert by_signal["leak"].detail == "the output holds '<|im_end|>'"

    loops = {"judge -> w": {"taken": 2, "cap_hit": True}}
    assert health.loops == loops  # read from the records' routes
    assert summarize(RECORDS, while_edges=[("judge", "w"), ("w", "judge")]).loops == loops
    assert summarize(RECORDS, while_edges=[]).loops == {}
    assert health.budget == {"calls": {"spent": 4, "limit": 10}, "cost_usd": {"spent": 0.5, "limit": 1.0}}
    assert summarize(RECORDS, budget=Budget(max_wall_s=60), meter=meter).budget["wall_s"]["limit"] == 60
    assert summarize(RECORDS, budget=Budget(max_calls=1)).budget == {}  # no meter, nothing spent to report

    data = health.to_dict()
    assert list(data) == ["flags", "loops", "budget"]
    assert data["flags"][0] == {"signal": "retried", "node": "judge", "step": 1, "detail": by_signal["retried"].detail}
    assert json.loads(json.dumps(data)) == data

    # thresholds: min_samples at its edge, a stricter length factor, an extra leak pattern
    two = summarize(RECORDS, settings=HealthSettings(min_samples=2))
    assert ("short", "few", 12) in [(flag.signal, flag.node, flag.step) for flag in two.flags]
    wide = summarize(RECORDS, settings=HealthSettings(length_factor=4.5))
    assert not {flag.signal for flag in wide.flags} & {"long", "short"}
    assert [flag.detail for flag in summarize([rec("a", 0, "<think>hm</think> yes")]).flags] == [
        "the output holds '<think>'"
    ]
    secret = [rec("a", 0, "the SECRET is out")]
    assert summarize(secret).flags == ()
    assert [flag.detail for flag in summarize(secret, settings=HealthSettings(leak=("SECRET",))).flags] == [
        "the output holds 'SECRET'"
    ]

    healthy = summarize([rec("w", 0, "a draft", prompt="Write {query}", inputs={"query": "cats"}), rec("j", 1, "ok")])
    assert healthy == Health()
    assert healthy.to_dict() == {}
