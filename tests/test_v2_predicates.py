"""Predicates: the grammar, the built-ins, and what is refused (plan test 8)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from xmas.predicates import Context, PredicateError, parse_predicate

VALUES: dict[str, Any] = {
    "verdict": {"fits": False, "action": "shorten", "score": 7, "problems": ["long", "vague"]},
    "draft": "A short draft.",
    "count": 0,
    "nothing": None,
    "flags": {"a": True, "b": [False]},
    "ones": {"a": 1, "b": [0]},
    "same": {"b": [False], "a": True},
}


def is_long(name: str, args: tuple[Any, ...]) -> bool:
    assert name == "is_long"
    return len(args[0]) > 10


CTX = Context(VALUES, {"judge": 2, "writer": 3}, "judge", cost_usd=0.25, elapsed_s=None, call=is_long)

EVALUATE = [
    ("not verdict.fits", True, {"verdict"}),
    ("verdict.action == 'shorten' and verdict.score >= 7", True, {"verdict"}),
    ('verdict.action in ["publish", "hold"]', False, {"verdict"}),
    ("'vague' in verdict.problems and 'x' not in draft", True, {"verdict", "draft"}),
    ("visits < 3 and visits(writer) == 3", True, set()),
    ("cost <= 0.25 and not (elapsed > 10)", True, set()),
    ("exists(verdict.score) and not exists(verdict.missing) and exists(nothing)", True, {"verdict", "nothing"}),
    ("len(draft) == 14 and len(verdict.problems) == 2 and len(nothing) == null", True, {"draft", "verdict", "nothing"}),
    ("verdict.missing > 1 or verdict.missing < 1 or draft > 3", False, {"verdict", "draft"}),
    ("verdict.missing == null and nothing == null and verdict.deep.deeper == null", True, {"verdict", "nothing"}),
    ("count or nothing or verdict.fits or verdict.missing", False, {"count", "nothing", "verdict"}),
    ("true == 1 or false == 0 or [1, 'a'] != [1, 'a']", False, set()),
    ("flags != ones and flags == same", True, {"flags", "ones", "same"}),  # a bool never equals a number, nested too
    ("is_long(draft) and not is_long('short')", True, {"draft"}),
    ("verdict.score > -1.5 and verdict.action < 'z' and (count == 0 or count == 1)", True, {"verdict", "count"}),
]

REFUSED = [
    ("__import__('os')", "names starting '__' belong to xmas"),
    ("frobnicate(draft)", "unknown function 'frobnicate'"),
    ("verdict.fits; draft", "unexpected character ';' at position 12"),
    ("len(draft", "predicate ends early, expected ')' at position 9"),
    ("(" * 33 + "true" + ")" * 33, "nests deeper than 32 at position 32"),
    ("x == " + "'a' or x == " * 50 + "'b'", "characters long, 500 at most"),
]


@pytest.mark.parametrize(
    ("text", "expected", "roots"),
    [*EVALUATE, *((text, problem, None) for text, problem in REFUSED)],
    ids=range(len(EVALUATE) + len(REFUSED)),
)
def test_grammar(text: str, expected: bool | str, roots: set[str] | None) -> None:
    if isinstance(expected, str):  # refused when parsed, or (an unknown function) when evaluated
        with pytest.raises(PredicateError, match=re.escape(expected)) as raised:
            parse_predicate(text).evaluate(Context(VALUES))
        assert isinstance(raised.value, ValueError)
        if "position" in expected:
            assert raised.value.position == int(expected.rsplit(" ", 1)[1])
        return
    predicate = parse_predicate(text)
    assert predicate.evaluate(CTX) is expected
    assert set(predicate.roots()) == roots
    assert predicate.functions() == ({"is_long"} if "is_long" in text else set())
    assert str(predicate) == text
