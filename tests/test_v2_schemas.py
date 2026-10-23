"""Answer forms: the README dialect parsed, refused with every problem named, and its exact format block."""

from __future__ import annotations

from typing import Any

import pytest

from xmas.schemas import RETRY_NOTE, FormError, compile_form

VERDICT: dict[str, Any] = {
    "fits": {"type": "bool", "means": "the slide makes its claim at a glance"},
    "problems": {"type": "list of text", "means": "one line per problem"},
    "action": {"type": "text", "one of": ["post", "hold"]},
    "checks": {"type": "list of object", "fields": {"comment": {"type": "int"}, "reason": {"type": "text"}}},
}
VERDICT_BLOCK = """\
Answer with one JSON object and nothing else. Its fields:
- fits (true or false): the slide makes its claim at a glance
- problems (list of text): one line per problem
- action (one of "post", "hold")
- checks (list of object)
  - comment (integer)
  - reason (text)"""

NESTED: dict[str, Any] = {
    "slide": {
        "type": "object",
        "means": "the slide as drawn",
        "fields": {
            "title": {"type": "text"},
            "size": {
                "type": "object",
                "fields": {"w": {"type": "number"}, "unit": {"type": "text", "one of": ["px", "pt"], "optional": True}},
            },
        },
    },
    "score": {"type": "int"},
}
NESTED_BLOCK = """\
Answer with one JSON object and nothing else. Its fields:
- slide (object): the slide as drawn
  - title (text)
  - size (object)
    - w (number)
    - unit (one of "px", "pt", optional)
- score (integer)"""

CASES = {
    "good": (
        VERDICT,
        '{"fits": true, "problems": [], "action": "post", "checks": [{"comment": 1, "reason": "ok"}], "extra": 1}',
        {"fits": True, "problems": [], "action": "post", "checks": [{"comment": 1, "reason": "ok"}], "extra": 1},
        None,
    ),
    "fenced": (
        VERDICT,
        '  ```json\n{"fits": "yes", "problems": ["x"], "action": "hold", "checks": []}\n```\n',
        {"fits": True, "problems": ["x"], "action": "hold", "checks": []},
        None,
    ),
    "prose": (VERDICT, 'Sure! {"fits": true}', None, "<root>: Invalid JSON: expected value at line 1 column 1"),
    "wrong": (
        VERDICT,
        '{"fits": true, "problems": "none", "action": "prefer A", "checks": [{"comment": "one"}]}',
        None,
        "problems: Input should be a valid array; action: Input should be 'post' or 'hold'; "
        "checks.0.comment: Input should be a valid integer, unable to parse string as an integer; "
        "checks.0.reason: Field required",
    ),
    "nested object": (
        NESTED,
        '{"slide": {"title": "Results", "size": {"w": "1920"}}, "score": "2"}',
        {"slide": {"title": "Results", "size": {"w": 1920.0, "unit": None}}, "score": 2},
        None,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_answer_form_dialect(case: str) -> None:
    fields, answer, value, problem = CASES[case]
    form = compile_form(fields)

    if problem is None:
        assert form.parse(answer) == value
    else:
        with pytest.raises(FormError) as caught:
            form.parse(answer)
        assert caught.value.problem == problem
        assert RETRY_NOTE.format(problem=caught.value.problem) == (
            f"Your previous answer did not match the required format: {problem}\nAnswer again with only the JSON."
        )

    if fields is VERDICT:
        assert form.format_block() == VERDICT_BLOCK
        assert form.paths() == ("fits", "problems", "action", "checks")
        assert form.choices("action") == ("post", "hold")
        assert form.choices("fits") is None
    else:
        assert form.format_block() == NESTED_BLOCK
        assert form.paths() == ("slide", "slide.title", "slide.size", "slide.size.w", "slide.size.unit", "score")
        assert form.choices("slide.size.unit") == ("px", "pt")
        with pytest.raises(FormError, match=r"slide\.title: Input should be a valid string; score: Field required"):
            form.parse('```\n{"slide": {"title": 1, "size": {"w": 1}}}\n```')
