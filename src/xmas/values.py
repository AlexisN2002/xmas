"""Typed values: the type words of spec version 2, and how values render and check.

A value in a run's state is plain data: text is a ``str``, json is plain JSON
data (``dict`` with ``str`` keys, ``list``, ``str``, ``int``, ``float``,
``bool``, ``None``). The type lives in the system, never on the value, so a
version 1 state of strings stays a valid version 2 state of text.

:func:`parse_type` reads every type word of the format, but only text and json
run: :func:`xmas.check.check_system` refuses the others before a value of
theirs is rendered or checked. This is a leaf module: it imports no
other xmas module.
"""

from __future__ import annotations

import json
import math
import typing
from dataclasses import dataclass
from typing import Any

KINDS = ("text", "json", "image", "audio", "document", "table", "file")
"""Every value kind of the format; ``list of <kind>`` builds on them."""

_LIST = "list"
LIST_PREFIX = "list of "
"""How a list type starts, in value types and in answer forms alike."""


@dataclass(frozen=True)
class ValueType:
    """The type of one value: ``kind`` is one of :data:`KINDS`, or ``"list"`` with ``item`` its items' type."""

    kind: str
    item: ValueType | None = None

    def __post_init__(self) -> None:
        """Refuse unknown kinds and an item on anything but a list."""
        if self.kind == _LIST:
            if self.item is None:
                raise ValueError("A list type needs an item type.")
        elif self.kind not in KINDS:
            raise ValueError(f"Unknown value type {self.kind!r}; known: {', '.join(KINDS)}, list of <type>.")
        elif self.item is not None:
            raise ValueError(f"Only a list has an item type, not {self.kind!r}.")

    def __str__(self) -> str:
        """Return the type as the spec writes it, such as ``list of text``."""
        return f"{LIST_PREFIX}{self.item}" if self.item is not None else self.kind


TEXT = ValueType("text")
"""A ``str``."""

JSON = ValueType("json")
"""Plain JSON data."""


def parse_type(text: str) -> ValueType:
    """Read a type as the spec writes it, such as ``"json"`` or ``"list of text"``; ValueError if it names none."""
    if not isinstance(text, str):
        raise ValueError(f"A value type is a string such as 'text', not {text!r}.")
    words = text.strip()
    if words.startswith(LIST_PREFIX):
        return ValueType(_LIST, parse_type(words[len(LIST_PREFIX) :]))
    return ValueType(words)


def type_of_hint(hint: Any) -> ValueType:
    """Return the value type of a callable's parameter or return value.

    ``str`` (``Annotated`` unwrapped) is text and anything else is json, so a
    ``str | None`` return is json (``None`` is a json value). A JSON-schema
    mapping, as a tool describes its parameters, is text for
    ``{"type": "string"}`` and for a string that may be null (a ``str | None``
    parameter), and json otherwise.
    """
    if isinstance(hint, dict):
        kinds: set[Any] = set()
        for option in hint.get("anyOf") or [hint]:
            kind = option.get("type") if isinstance(option, dict) else None
            kinds.update(kind if isinstance(kind, list) else [kind])
        return TEXT if "string" in kinds and kinds <= {"string", "null"} else JSON
    if typing.get_origin(hint) is typing.Annotated:
        hint = typing.get_args(hint)[0]
    return TEXT if hint is str else JSON


def render_value(value: Any, vtype: ValueType, *, present: bool = True) -> str:
    """Render a value as the text a prompt, a record or a trace output shows.

    Text renders as itself; json renders as one line of
    ``json.dumps(value, ensure_ascii=False)`` with the keys in the order the
    value has them, so ``0``, ``false``, ``[]`` and a json ``null`` show as
    JSON. A missing value renders as ``""``: the caller says so with
    ``present=False``, or passes ``None`` for a text value.
    """
    if not present or (value is None and vtype == TEXT):
        return ""
    if vtype == TEXT:
        return value if isinstance(value, str) else str(value)
    return json.dumps(value, ensure_ascii=False)


def check_value(value: Any, vtype: ValueType) -> str | None:
    """Return ``None`` when a value fits its type, else one line such as ``"not JSON at counts.words: set"``."""
    if vtype == TEXT:
        return None if isinstance(value, str) else f"expected text, got {type(value).__name__}"
    return _json_problem(value, "")


def _json_problem(value: Any, path: str) -> str | None:
    """Return where ``value`` stops being plain JSON data, or ``None``."""
    where = f" at {path}" if path else ""
    if value is None or isinstance(value, (str, bool, int)):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"not JSON{where}: {value!r}"
    if isinstance(value, list):
        for index, item in enumerate(value):
            problem = _json_problem(item, f"{path}.{index}" if path else str(index))
            if problem is not None:
                return problem
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"not JSON{where}: key {key!r} is not a string"
            problem = _json_problem(item, f"{path}.{key}" if path else key)
            if problem is not None:
                return problem
        return None
    return f"not JSON{where}: {type(value).__name__}"
