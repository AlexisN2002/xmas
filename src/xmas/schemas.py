"""Answer forms: the JSON a json agent must answer with, checked without vendor help.

A form is written in the README's dialect, one entry per field::

    fits: {type: bool, means: the draft makes one claim}
    action: {type: text, one of: [publish, shorten]}
    checks: {type: list of object, fields: {comment: {type: int}, reason: {type: text, optional: true}}}

``type`` is ``text``, ``bool``, ``int``, ``number``, ``json`` or ``object``, or
``list of <type>``; ``fields`` describes an object (or a list's objects);
``one of`` lists the allowed values (of a list's items, for a list); ``means``
is what the model is told; ``optional`` lets the field be left out. Fields the
form does not name are kept. Coercion is pydantic's lax mode, so ``"yes"`` is
``true`` and ``"3"`` is ``3``.

:func:`compile_form` turns a form into an :class:`AnswerForm`, whose
:meth:`~AnswerForm.parse` reads an answer and whose
:meth:`~AnswerForm.format_block` is the text appended to the agent's prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model

from xmas.errors import ErrorKind
from xmas.values import LIST_PREFIX

RETRY_NOTE = "Your previous answer did not match the required format: {problem}\nAnswer again with only the JSON."
"""Appended to the prompt of the one retry; ``{problem}`` is the first problem."""

_BASE: dict[str, Any] = {"text": str, "bool": bool, "int": int, "number": float, "json": Any}
_WORDS = {"text": "text", "bool": "true or false", "int": "integer", "number": "number", "json": "any JSON"}
_OBJECT = "object"
_KEYS = frozenset({"type", "means", "optional", "one of", "fields"})
_FENCE = re.compile(r"\A```[\w-]*[ \t]*\n(.*)\n[ \t]*```\Z", re.S)
_ANY_JSON: TypeAdapter[Any] = TypeAdapter(Any)


class FormError(ValueError):
    """An answer does not match its form.

    Attributes:
        problem: One line naming every problem, such as
            ``"action: Input should be 'post' or 'hold'; checks.0.reason: Field required"``.
    """

    kind = ErrorKind.BAD_FORMAT

    def __init__(self, problem: str) -> None:
        """Keep the problem as the message."""
        super().__init__(problem)
        self.problem = problem


class AnswerForm:
    """A compiled answer form; build it with :func:`compile_form`.

    Attributes:
        fields: The form as given (a plain copy), or ``None`` for any JSON.
    """

    fields: dict[str, dict[str, Any]] | None

    def __init__(self, fields: Mapping[str, Mapping[str, Any]] | None) -> None:
        """Check the form and build its pydantic model.

        Raises:
            ValueError: If the form breaks the dialect; the message names the field.
        """
        self.fields = None if fields is None else _plain(fields, "")
        self._model = None if self.fields is None else _model(self.fields, "answer")

    def parse(self, text: str) -> Any:
        """Read an answer.

        Whitespace and at most one surrounding Markdown code fence are stripped;
        the rest must be JSON matching the form. Without a form any JSON passes.

        Args:
            text: The model's answer.

        Returns:
            The value: for a form, an object with the form's fields first (an
            optional field left out is ``None``), then any extra fields.

        Raises:
            FormError: If the answer is not JSON or does not match the form.
        """
        body = text.strip()
        if match := _FENCE.match(body):
            body = match.group(1)
        try:
            if self._model is None:
                return _ANY_JSON.validate_json(body)
            return self._model.model_validate_json(body).model_dump(mode="json", by_alias=True)
        except ValidationError as error:
            raise FormError(
                "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in error.errors())
            ) from error

    def format_block(self) -> str:
        """Return the format instructions appended to the agent's prompt.

        For the form ``{fits: {type: bool, means: it fits}}`` this is::

            Answer with one JSON object and nothing else. Its fields:
            - fits (true or false): it fits

        A field with ``one of`` reads ``(one of "a", "b")``, an optional one
        adds ``, optional``, and an object's fields follow on lines indented two
        spaces more. Without a form it is
        ``Answer with one JSON value and nothing else.``
        """
        if self.fields is None:
            return "Answer with one JSON value and nothing else."
        if not self.fields:
            return "Answer with one JSON object and nothing else."
        return "\n".join(["Answer with one JSON object and nothing else. Its fields:", *_lines(self.fields, "")])

    def paths(self) -> tuple[str, ...]:
        """Return every field path in the form, nested objects as ``a.b``.

        Fields of a list's objects are not paths, since a path cannot index a
        list. Without a form there are none.
        """
        return () if self.fields is None else tuple(_paths(self.fields, ""))

    def choices(self, path: str) -> tuple[Any, ...] | None:
        """Return a field's ``one of`` values, or ``None`` when it has none or is not in the form.

        Args:
            path: A path from :meth:`paths`.
        """
        fields = self.fields
        spec: dict[str, Any] | None = None
        for name in path.split("."):
            spec = None if fields is None else fields.get(name)
            if spec is None:
                return None
            fields = spec.get("fields") if spec["type"] == _OBJECT else None
        return None if spec is None or "one of" not in spec else tuple(spec["one of"])

    def __repr__(self) -> str:
        """Return an unambiguous representation."""
        return f"AnswerForm({self.fields!r})"


def compile_form(fields: Mapping[str, Mapping[str, Any]] | None) -> AnswerForm:
    """Compile an answer form.

    Args:
        fields: The form in the README's dialect; ``None`` accepts any JSON and
            ``{}`` any JSON object.

    Returns:
        The compiled form.

    Raises:
        ValueError: If the form breaks the dialect; the message names the field.
    """
    return AnswerForm(fields)


def _plain(fields: Mapping[Any, Any], where: str) -> dict[str, dict[str, Any]]:
    """Check one level of a form and return a plain copy of it."""
    if not isinstance(fields, Mapping):
        raise ValueError(f"Answer form{where}: fields must be a mapping, got {type(fields).__name__}.")
    out: dict[str, dict[str, Any]] = {}
    for name, spec in fields.items():
        path = f"{where}.{name}" if where else str(name)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Answer form: field name {name!r} is not a non-empty string.")
        if not isinstance(spec, Mapping):
            raise ValueError(f"Answer form field {path!r}: expected a mapping such as {{type: text}}, got {spec!r}.")
        unknown = set(spec) - _KEYS
        if unknown:
            raise ValueError(f"Answer form field {path!r}: unknown keys {sorted(unknown)}; known: {sorted(_KEYS)}.")
        kind = spec.get("type")
        base = _item_kind(kind)
        if base is None:
            raise ValueError(
                f"Answer form field {path!r}: unknown type {kind!r}; "
                "known: text, bool, int, number, json, object, list of <type>."
            )
        copy: dict[str, Any] = {"type": kind}
        if "means" in spec:
            copy["means"] = str(spec["means"])
        if "optional" in spec:
            if not isinstance(spec["optional"], bool):
                raise ValueError(f"Answer form field {path!r}: optional must be true or false.")
            copy["optional"] = spec["optional"]
        if "one of" in spec:
            choices = spec["one of"]
            if not isinstance(choices, list) or not choices or base in (_OBJECT, "json"):
                raise ValueError(f"Answer form field {path!r}: one of needs a non-empty list on a plain type.")
            copy["one of"] = list(choices)
        if "fields" in spec:
            if base != _OBJECT:
                raise ValueError(f"Answer form field {path!r}: only an object has fields, not {kind!r}.")
            copy["fields"] = _plain(spec["fields"], path)
        out[name] = copy
    return out


def _item_kind(kind: Any) -> str | None:
    """Return the base type under any ``list of`` words, or ``None`` if unknown."""
    while isinstance(kind, str) and kind.startswith(LIST_PREFIX):
        kind = kind[len(LIST_PREFIX) :].strip()
    return kind if isinstance(kind, str) and (kind in _BASE or kind == _OBJECT) else None


def _model(fields: dict[str, dict[str, Any]], name: str) -> type[BaseModel]:
    """Build the pydantic model of one level, fields aliased as in ``tools._build_model``."""
    made: dict[str, Any] = {}
    for index, (key, spec) in enumerate(fields.items()):
        default = None if spec.get("optional") else ...
        extra = {"description": spec["means"]} if "means" in spec else {}
        made[f"f{index}"] = (_annotation(spec, spec["type"], f"{name}.{key}"), Field(default, alias=key, **extra))
    model: type[BaseModel] = create_model(name, __config__=ConfigDict(extra="allow"), **made)
    return model


def _annotation(spec: dict[str, Any], kind: str, name: str) -> Any:
    """Return the Python type of a field whose (remaining) type words are ``kind``."""
    if kind.startswith(LIST_PREFIX):
        return list[_annotation(spec, kind[len(LIST_PREFIX) :].strip(), name)]  # type: ignore[misc]
    if "one of" in spec:
        return Literal[tuple(spec["one of"])]
    if kind == _OBJECT:
        return _model(spec["fields"], name) if "fields" in spec else dict[str, Any]
    return _BASE[kind]


def _phrase(spec: dict[str, Any], kind: str) -> str:
    """Return the words for a field's type, such as ``list of text`` or ``one of "a", "b"``."""
    if kind.startswith(LIST_PREFIX):
        return LIST_PREFIX + _phrase(spec, kind[len(LIST_PREFIX) :].strip())
    if "one of" in spec:
        return "one of " + ", ".join(json.dumps(choice, ensure_ascii=False) for choice in spec["one of"])
    return _OBJECT if kind == _OBJECT else _WORDS[kind]


def _lines(fields: dict[str, dict[str, Any]], indent: str) -> list[str]:
    """Return the format block's lines for one level of a form."""
    lines: list[str] = []
    for key, spec in fields.items():
        words = _phrase(spec, spec["type"]) + (", optional" if spec.get("optional") else "")
        means = f": {spec['means']}" if "means" in spec else ""
        lines.append(f"{indent}- {key} ({words}){means}")
        if "fields" in spec:
            lines.extend(_lines(spec["fields"], indent + "  "))
    return lines


def _paths(fields: dict[str, dict[str, Any]], prefix: str) -> list[str]:
    """Return the paths of one level of a form and of its nested objects."""
    paths: list[str] = []
    for key, spec in fields.items():
        path = prefix + key
        paths.append(path)
        if spec["type"] == _OBJECT and "fields" in spec:
            paths.extend(_paths(spec["fields"], path + "."))
    return paths
