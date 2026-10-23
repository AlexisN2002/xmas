"""Predicates: the conditions on ``when``, ``while`` and ``switch`` edges.

A predicate is a small expression over the run's values, such as
``not verdict.fits and visits < 3``. It is parsed once, when the system is built,
into a tree of tuples, and evaluated after the deciding node ran. Nothing is ever
passed to ``eval``: the tokenizer knows numbers, quoted strings without escapes,
names and ``== != <= >= < > ( ) [ ] , .``, and any other character is refused
with its position.

Grammar::

    expr := conj ('or' conj)*
    conj := neg ('and' neg)*
    neg  := 'not' neg | comp
    comp := atom [op atom]          op: == != < <= > >= in, not in
    atom := number | string | true | false | null | '[' literals ']'
          | name '(' args ')' | path | '(' expr ')'
    path := name ('.' name)*

Built-ins: ``exists(path)``, ``len(x)``, ``visits`` (the deciding node's visits),
``visits(node)``, ``cost`` (USD so far) and ``elapsed`` (seconds so far). Any other
call names a registered callable that returns a bool. A missing path is ``None``;
an ordering comparison with ``None``, or between a number and a string, is false;
a bare path is read by its truthiness.

This is a leaf module: it imports no other xmas module.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

MAX_LENGTH = 500
"""The longest predicate text, in characters."""

MAX_DEPTH = 32
"""The deepest nesting of ``not``, brackets and calls."""

BUILTINS = frozenset({"exists", "len", "visits", "cost", "elapsed"})
"""Names that are built-ins, never values or registered callables."""

_KEYWORDS = frozenset({"and", "or", "not", "in", "true", "false", "null"})
_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">=", "in", "not in"})
_TOKEN = re.compile(
    r"""(?P<space>\s+)
    |(?P<number>-?\d+(?:\.\d+)?)
    |(?P<string>"[^"]*"|'[^']*')
    |(?P<name>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<op>==|!=|<=|>=|<|>|\(|\)|\[|\]|,|\.)""",
    re.VERBOSE,
)
_MISSING = object()


class PredicateError(ValueError):
    """A predicate that cannot be parsed or evaluated.

    A ``ValueError``, like every other refusal of a system's shape, because
    ``check_system`` parses each predicate when the system is built; evaluation
    raises it only for a call no registered callable answers.

    Attributes:
        position: The zero-based character position of the problem, or ``None``
            when it is not about one place in the text.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        """Keep the message and the position."""
        super().__init__(message if position is None else f"{message} at position {position}")
        self.position = position


@dataclass(frozen=True)
class Context:
    """What a predicate can see when it is evaluated.

    Attributes:
        values: The run's values, by name; json values are plain JSON data.
        visits: How often each node has run, the deciding node's current visit
            counted.
        node: The deciding node, which bare ``visits`` refers to.
        cost_usd: USD spent so far, or ``None`` when unknown.
        elapsed_s: Wall seconds so far, or ``None`` when unknown.
        call: Calls a registered callable by name with its arguments and returns
            its bool, or ``None`` when no callable is registered.
    """

    values: Mapping[str, Any]
    visits: Mapping[str, int] = field(default_factory=dict)
    node: str = ""
    cost_usd: float | None = None
    elapsed_s: float | None = None
    call: Callable[[str, tuple[Any, ...]], bool] | None = None


@dataclass(frozen=True)
class Predicate:
    """A parsed predicate.

    Attributes:
        text: The text it was parsed from.
        tree: The syntax tree, as nested tuples: ``("lit", value)``,
            ``("path", names)``, ``("call", name, args)``, ``("exists", names)``,
            ``("len", arg)``, ``("visits", node or None)``, ``("cost",)``,
            ``("elapsed",)``, ``("not", arg)``, ``("and", args)``,
            ``("or", args)`` and ``("cmp", op, left, right)``.
    """

    text: str
    tree: tuple[Any, ...]

    def roots(self) -> frozenset[str]:
        """Return the value names the predicate reads: the first name of every path."""
        return frozenset(path[0] for path in self.paths())

    def paths(self) -> frozenset[tuple[str, ...]]:
        """Return every path the predicate reads, as tuples of names."""
        return frozenset(node[1] for node in _walk(self.tree) if node[0] in ("path", "exists"))

    def functions(self) -> frozenset[str]:
        """Return the registered callables the predicate calls; built-ins are not listed."""
        return frozenset(node[1] for node in _walk(self.tree) if node[0] == "call")

    def calls(self) -> frozenset[tuple[str, int]]:
        """Return each call of a registered callable as ``(name, number of arguments)``."""
        return frozenset((node[1], len(node[2])) for node in _walk(self.tree) if node[0] == "call")

    def evaluate(self, ctx: Context) -> bool:
        """Return whether the predicate holds in ``ctx``.

        Args:
            ctx: The values, visits, spend and callables to evaluate against.

        Returns:
            The predicate's truth value.

        Raises:
            PredicateError: If a called function is not registered or its
                result is not a bool.
        """
        return bool(_eval(self.tree, ctx))

    def __str__(self) -> str:
        """Return the text it was parsed from."""
        return self.text


def parse_predicate(text: str) -> Predicate:
    """Parse a predicate.

    Args:
        text: The predicate, at most 500 characters.

    Returns:
        The parsed predicate.

    Raises:
        PredicateError: If the text is empty, too long, too deep, holds a
            character outside the grammar, or does not parse.
    """
    if len(text) > MAX_LENGTH:
        raise PredicateError(f"predicate is {len(text)} characters long, {MAX_LENGTH} at most")
    parser = _Parser(text, _tokenize(text))
    if not parser.tokens:
        raise PredicateError("predicate is empty")
    tree = parser.expr()
    if parser.index < len(parser.tokens):
        _, value, position = parser.tokens[parser.index]
        raise PredicateError(f"unexpected {value!r}", position)
    return Predicate(text, tree)


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    """Split ``text`` into ``(kind, text, position)`` tokens, spaces dropped."""
    tokens: list[tuple[str, str, int]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise PredicateError(f"unexpected character {text[position]!r}", position)
        kind = match.lastgroup or ""
        if kind != "space":
            tokens.append((kind, match.group(), position))
        position = match.end()
    return tokens


class _Parser:
    """Recursive descent over the tokens, one method per grammar rule."""

    def __init__(self, text: str, tokens: list[tuple[str, str, int]]) -> None:
        """Start before the first token."""
        self.text = text
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    def peek(self) -> str | None:
        """Return the next token's text, or ``None`` at the end."""
        return self.tokens[self.index][1] if self.index < len(self.tokens) else None

    def take(self, expected: str | None = None) -> tuple[str, str, int]:
        """Consume the next token, which must be ``expected`` when given."""
        if self.index >= len(self.tokens):
            want = "a value" if expected is None else repr(expected)
            raise PredicateError(f"predicate ends early, expected {want}", len(self.text))
        token = self.tokens[self.index]
        if expected is not None and token[1] != expected:
            raise PredicateError(f"expected {expected!r}, got {token[1]!r}", token[2])
        self.index += 1
        return token

    def enter(self, position: int) -> None:
        """Count one level of nesting, refusing the 33rd."""
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise PredicateError(f"predicate nests deeper than {MAX_DEPTH}", position)

    def expr(self) -> tuple[Any, ...]:
        """Parse ``conj ('or' conj)*``."""
        parts = [self.conj()]
        while self.peek() == "or":
            self.take()
            parts.append(self.conj())
        return parts[0] if len(parts) == 1 else ("or", tuple(parts))

    def conj(self) -> tuple[Any, ...]:
        """Parse ``neg ('and' neg)*``."""
        parts = [self.neg()]
        while self.peek() == "and":
            self.take()
            parts.append(self.neg())
        return parts[0] if len(parts) == 1 else ("and", tuple(parts))

    def neg(self) -> tuple[Any, ...]:
        """Parse ``'not' neg | comp``."""
        if self.peek() != "not":
            return self.comp()
        self.enter(self.take()[2])
        inner = self.neg()
        self.depth -= 1
        return ("not", inner)

    def comp(self) -> tuple[Any, ...]:
        """Parse ``atom [op atom]``, reading ``not in`` as one operator."""
        left = self.atom()
        op = self.peek()
        if op == "not" and self.index + 1 < len(self.tokens) and self.tokens[self.index + 1][1] == "in":
            self.take()
            op = "not in"
        elif op not in _COMPARISONS:
            return left
        self.take()
        return ("cmp", op, left, self.atom())

    def atom(self) -> tuple[Any, ...]:
        """Parse a literal, a list, a call, a path or a bracketed expression."""
        kind, value, position = self.take()
        if kind == "number":
            return ("lit", float(value) if "." in value else int(value))
        if kind == "string":
            return ("lit", value[1:-1])
        if value == "(":
            self.enter(position)
            inner = self.expr()
            self.take(")")
            self.depth -= 1
            return inner
        if value == "[":
            self.enter(position)
            items = self.literals()
            self.depth -= 1
            return ("lit", items)
        if kind != "name":
            raise PredicateError(f"unexpected {value!r}", position)
        if value in ("true", "false", "null"):
            return ("lit", {"true": True, "false": False, "null": None}[value])
        if value in _KEYWORDS:
            raise PredicateError(f"unexpected {value!r}", position)
        if value.startswith("__"):
            raise PredicateError(f"names starting '__' belong to xmas: {value!r}", position)
        if self.peek() == "(":
            return self.call(value, position)
        if value in ("cost", "elapsed"):
            return (value,)
        if value == "visits":
            return ("visits", None)
        if value in BUILTINS:
            raise PredicateError(f"{value} needs an argument in brackets", position)
        return ("path", self.path(value))

    def literals(self) -> list[Any]:
        """Parse the rest of ``'[' literal (',' literal)* ']'``."""
        items: list[Any] = []
        while self.peek() not in ("]", None):
            if items:
                self.take(",")
            kind, value, position = self.take()
            if kind == "number":
                items.append(float(value) if "." in value else int(value))
            elif kind == "string":
                items.append(value[1:-1])
            elif value in ("true", "false", "null"):
                items.append({"true": True, "false": False, "null": None}[value])
            else:
                raise PredicateError(f"a list holds literals only, got {value!r}", position)
        self.take("]")
        return items

    def path(self, first: str) -> tuple[str, ...]:
        """Parse the rest of ``name ('.' name)*``."""
        names = [first]
        while self.peek() == ".":
            self.take()
            kind, value, position = self.take()
            if kind != "name" or value in _KEYWORDS:
                raise PredicateError(f"expected a name after '.', got {value!r}", position)
            names.append(value)
        return tuple(names)

    def call(self, name: str, position: int) -> tuple[Any, ...]:
        """Parse ``'(' args ')'`` after ``name``, checking the built-ins' arguments."""
        self.enter(self.take("(")[2])
        args: list[tuple[Any, ...]] = []
        while self.peek() not in (")", None):
            if args:
                self.take(",")
            args.append(self.expr())
        self.take(")")
        self.depth -= 1
        if name in ("cost", "elapsed"):
            raise PredicateError(f"{name} takes no arguments", position)
        if name in BUILTINS and len(args) != 1:
            raise PredicateError(f"{name}() takes one argument, got {len(args)}", position)
        if name == "len":
            return ("len", args[0])
        if name in ("exists", "visits") and (args[0][0] != "path" or (name == "visits" and len(args[0][1]) != 1)):
            what = "a path" if name == "exists" else "a node name"
            raise PredicateError(f"{name}() takes {what}", position)
        if name == "exists":
            return ("exists", args[0][1])
        if name == "visits":
            return ("visits", args[0][1][0])
        return ("call", name, tuple(args))


def _walk(tree: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Return every subtree of ``tree``, ``tree`` included."""
    head = tree[0]
    if head in ("not", "len"):
        children: tuple[tuple[Any, ...], ...] = (tree[1],)
    elif head in ("and", "or"):
        children = tree[1]
    elif head == "cmp":
        children = (tree[2], tree[3])
    elif head == "call":
        children = tree[2]
    else:
        children = ()
    found = [tree]
    for child in children:
        found.extend(_walk(child))
    return found


def _eval(tree: tuple[Any, ...], ctx: Context) -> Any:
    """Return the value of one subtree in ``ctx``."""
    head = tree[0]
    if head == "lit":
        return tree[1]
    if head == "path":
        value = _resolve(tree[1], ctx.values)
        return None if value is _MISSING else value
    if head == "exists":
        return _resolve(tree[1], ctx.values) is not _MISSING
    if head == "len":
        value = _eval(tree[1], ctx)
        return len(value) if isinstance(value, str | list | tuple | dict) else None
    if head == "visits":
        return ctx.visits.get(ctx.node if tree[1] is None else tree[1], 0)
    if head == "cost":
        return ctx.cost_usd
    if head == "elapsed":
        return ctx.elapsed_s
    if head == "not":
        return not _eval(tree[1], ctx)
    if head == "and":
        return all(_eval(part, ctx) for part in tree[1])
    if head == "or":
        return any(_eval(part, ctx) for part in tree[1])
    if head == "cmp":
        return _compare(tree[1], _eval(tree[2], ctx), _eval(tree[3], ctx))
    name, args = tree[1], tuple(_eval(arg, ctx) for arg in tree[2])
    if ctx.call is None:
        raise PredicateError(f"unknown function {name!r}")
    result = ctx.call(name, args)
    if not isinstance(result, bool):
        raise PredicateError(f"callable {name!r} returned {type(result).__name__}, not bool")
    return result


def _resolve(path: tuple[str, ...], values: Mapping[str, Any]) -> Any:
    """Follow ``path`` through ``values`` and nested mappings, or return ``_MISSING``."""
    current: Any = values
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            return _MISSING
        current = current[name]
    return current


def _compare(op: str, left: Any, right: Any) -> bool:
    """Apply a comparison: ordering only between two numbers or two strings."""
    if op == "==":
        return _equal(left, right)
    if op == "!=":
        return not _equal(left, right)
    if op in ("in", "not in"):
        if isinstance(right, str):
            found = isinstance(left, str) and left in right
        elif isinstance(right, list | tuple):
            found = any(_equal(left, item) for item in right)
        elif isinstance(right, Mapping):
            found = isinstance(left, str) and left in right
        else:
            found = False
        return found if op == "in" else not found
    numbers = all(isinstance(side, int | float) and not isinstance(side, bool) for side in (left, right))
    if not numbers and not (isinstance(left, str) and isinstance(right, str)):
        return False
    if op == "<":
        return bool(left < right)
    if op == "<=":
        return bool(left <= right)
    if op == ">":
        return bool(left > right)
    return bool(left >= right)


def _equal(left: Any, right: Any) -> bool:
    """Compare like JSON: a bool never equals a number, and lists and objects compare item by item."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    return bool(left == right)
