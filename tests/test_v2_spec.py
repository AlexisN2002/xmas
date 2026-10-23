"""Spec version 2: the worked example round-trips, the check names every problem, describe is plain data.

These run on construction, serialization and ``describe()`` alone; running a
version 2 system is the engine's part and is tested in ``test_v2_values.py``.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import MockLLM
from tests.test_tools import ToolMockLLM
from xmas import END, MAS, START, Agent
from xmas.check import check_system
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.graph import Join, classify_joins
from xmas.health import HealthSettings
from xmas.nodes import Box, Step
from xmas.spec import validate_spec
from xmas.tools import Tool, tool


@tool
def count_words(draft: str) -> dict[str, int]:
    """Count the words of a draft."""
    return {"words": len(draft.split())}


@tool
def publish_text(text: str) -> str:
    """Publish a text."""
    return text


@tool
def repeat(text: str, times: int) -> str:
    """Repeat a text."""
    return text * times


class RemoteTools:
    """A tool source standing in for an MCP server."""

    url = "https://tools.example.com/mcp"

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        return [Tool("web_fetch", "Fetch a page.", schema)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer every call with the same page."""
        return "page"


HERE = __name__

# Section 3's worked example, phase 2a subset: no boxes, conditions, frozen or
# health, so the writer no longer reads the verdict (only a while edge brings it
# back) and publish joins the judge with the writer instead of a switch.
EXAMPLE = f"""
version: 2
agents:
- name: writer
  model: text-model
  writes: {{draft: text}}
  tools: [web_fetch]
- name: judge
  reads: [draft, counts]
  writes: {{verdict: json}}
  schema:
    fits: {{type: bool, means: the draft makes one claim in under 300 words}}
    action: {{type: text, one of: [publish, shorten], means: what should happen next}}
steps:
- name: count
  uses: count_words
  writes: {{count_words: counts}}
- name: publish
  uses: publish_text
  reads: {{text: draft}}
  writes: {{publish_text: published}}
edges:
- writer -> count -> judge -> publish
- writer -> publish
state: {{draft: {{merge: last}}}}
callables:
  count_words: {{kind: python, import: '{HERE}:count_words', reads: {{draft: text}}, writes: {{count_words: json}}}}
  publish_text: {{kind: python, import: '{HERE}:publish_text', reads: {{text: text}}, writes: {{publish_text: text}}}}
  web_fetch: {{kind: mcp, server: 'https://tools.example.com/mcp', reads: {{url: text}}, writes: {{web_fetch: text}}}}
prompts:
  writer: 'Answer {{query}} in a short draft.'
  judge: "Judge this draft: {{draft}}\\nIts word counts: {{counts}}"
"""

# The edges of the full worked example, conditions included (phase 2b runs them).
FULL_EDGES = """
- writer -> count -> judge
- judge -> writer: {when: not verdict.fits, max_visits: 3}
- judge -> publish: {switch: verdict.action}
- judge -> shorten: {switch: verdict.action}
- judge -> apology: else
- shorten -> publish
- shorten -> apology: {on error: any}
- count -> apology: {on error: [timeout, outage]}
"""


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE)
    return data


def load(data: dict[str, Any]) -> MAS:
    return MAS.from_dict(
        data, llms={"text-model": ToolMockLLM()}, tools=[count_words, publish_text, repeat, RemoteTools()]
    )


def test_worked_example_round_trips_and_v1_is_untouched(tmp_path: Path) -> None:
    data = example()
    mas = load(copy.deepcopy(data))
    assert (mas.version, list(mas.agents), list(mas.steps)) == (2, ["writer", "judge"], ["count", "publish"])
    assert mas.agents["judge"].output_type == "json"
    assert mas.steps["publish"] == Step("publish", "publish_text", {"text": "draft"}, {"publish_text": "published"})
    assert mas.edges[0] == Edge("writer", "count") and len(mas.edges) == 4
    assert (mas.merges, {k: str(v) for k, v in mas.inputs.items()}) == ({"draft": "last"}, {"query": "text"})
    assert mas.info is not None and mas.info.joins == {"publish": Join("join", (("judge",), ("writer",)), (None, None))}
    assert mas.info.writers["counts"] == ("count",) and str(mas.info.types["verdict"]) == "json"

    spec = mas.to_dict()
    assert spec == data
    assert list(spec) == ["version", "agents", "steps", "edges", "state", "callables", "prompts"]
    again = load(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec  # idempotent
    assert mas.copy().to_dict() == spec and mas.copy().structure == mas.structure
    reformed = mas.copy()
    reformed.agents["judge"].schema = {"fits": {"type": "bool"}}
    assert reformed.structure != mas.structure

    # The edge grammar of the full example parses and dumps back as written.
    full = yaml.safe_load(FULL_EDGES)
    edges = parse_edges(full)
    assert edges[2] == Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3)
    assert edges[3].case == "publish" and edges[8].errors == ("timeout", "outage")
    assert dump_edges(edges) == full and dump_edges(parse_edges(dump_edges(edges))) == full
    other_forms = [
        {"a -> b": {"switch": "v.k", "case": "go"}},
        {"a -> c": {"when": "v.n > 2"}},
        {"b -> c": {"on error": "outage"}},
    ]
    assert [edge.kind for edge in parse_edges(other_forms)] == [EdgeKind.SWITCH, EdgeKind.WHEN, EdgeKind.ERROR]
    assert dump_edges(parse_edges(other_forms))[2] == {"b -> c": {"on error": ["outage"]}}
    with pytest.raises(ValueError, match=r"\.mas bundle"):
        mas.to_yaml(tmp_path / "v2.yaml")

    # Version 1 keeps its four keys, its path and its pinned refusals.
    v1 = MAS([Agent("a", "Do {query}", "x"), Agent("b", "Check {x}", "y", inputs=["x"])], [("a", "b")], llm=MockLLM())
    assert (v1.version, v1.steps, v1.info) == (1, {}, None)
    v1_spec = v1.to_dict()
    assert set(v1_spec) == {"version", "agents", "edges", "default_model"}
    assert MAS.from_dict(dict(v1_spec)).to_dict() == v1_spec
    assert v1.copy().structure == v1.structure and len(v1.structure) == 2
    with pytest.raises(ValueError, match="Unknown spec keys"):
        MAS.from_dict({**v1_spec, "steps": []})
    with pytest.raises(ValueError, match="Unsupported spec version"):
        validate_spec({**v1_spec, "version": 2})


def _broken(change: str) -> dict[str, Any]:
    """Return the example with one problem put in."""
    data = example()
    judge, steps = data["agents"][1], data["steps"]
    if change == "unread input":
        judge["reads"].append("notes")
        data["prompts"]["judge"] += " {notes}"
        steps[0]["writes"] = {"count_words": "len"}  # and a second problem in the same error
    elif change == "two writers":
        steps.append({"name": "again", "uses": "publish_text", "writes": {"publish_text": "draft"}})
        data["state"] = {}
        data["edges"].append("judge -> again -> publish")
    elif change == "placeholder not a read":
        data["prompts"]["writer"] += " {verdict}"
    elif change == "unknown uses":
        steps[0]["uses"] = "count_letters"
        del data["callables"]["count_words"]
    elif change == "text into an int parameter":
        steps.append({"name": "echo", "uses": "repeat", "reads": {"times": "draft"}})
        data["edges"] = ["writer -> count -> judge -> echo -> publish", "writer -> publish"]
        del data["callables"]
    elif change == "image":
        data["inputs"] = {"query": "text", "photo": "image"}
    elif change == "reserved name":
        data["agents"][0]["writes"] = {"len": "text"}
    elif change == "bad rename":
        steps[1]["reads"] = {"txt": "draft"}
    elif change == "v2 cycle":
        data["edges"].append("judge -> count")
    elif change == "missing prompt":
        del data["prompts"]["judge"]
    elif change == "callables mismatch":
        data["callables"]["count_words"]["writes"] = {"count_words": "text"}
    return data


CASES = {
    "unread input": [
        r"3 problems",
        r"'judge' reads 'notes', which is neither an input",
        r"'count' writes 'len', which is a reserved name",
    ],
    "two writers": [r"value 'draft' has 2 writers \('writer', 'again'\)"],
    "placeholder not a read": [r"agent 'writer': prompt placeholders \['verdict'\] are not among its reads"],
    "unknown uses": [r"Step 'count' uses unknown callable 'count_letters'"],
    "text into an int parameter": [r"step 'echo' passes text 'draft' to parameter 'times' \(integer\)"],
    "image": [r"input 'photo' has type 'image', which arrives in phase 3"],
    "reserved name": [r"'writer' writes 'len', which is a reserved name"],
    "bad rename": [r"step 'publish': reads renames 'txt', which is not a parameter of 'publish_text'"],
    "v2 cycle": [r"a loop needs a while edge: the edge judge -> count closes a cycle"],
    "inputs vs run": [r"runs as a box, or with run_state in phase 3"],
    "missing prompt": [r"Agent 'judge' has no prompt in the 'prompts' section"],
    "callables mismatch": [r"callable 'count_words' writes \{'count_words': 'text'\} in the spec"],
}


@pytest.mark.parametrize("change", list(CASES))
def test_check_names_every_dataflow_problem(change: str) -> None:
    if change == "inputs vs run":
        mas = MAS([Agent("cutter", "Cut {page}", "short", inputs=["page"])], llm=MockLLM(), inputs={"page": "text"})
        with pytest.raises(ValueError) as info:
            mas.run("x")
    else:
        with pytest.raises(ValueError) as info:
            load(_broken(change))
    for phrase in CASES[change]:
        assert re.search(phrase, str(info.value)), (phrase, str(info.value))


def test_describe_is_plain_data() -> None:
    mas = load(example())
    described = mas.describe()
    assert list(described) == ["version", "inputs", "nodes", "edges", "joins", "groups", "id"]
    assert json.loads(json.dumps(described)) == described
    assert described["nodes"] == [
        {
            "name": "writer",
            "kind": "agent",
            "model": "text-model",
            "reads": {"query": "text"},
            "writes": {"draft": "text"},
            "tools": ["web_fetch"],
            "frozen": False,
        },
        {
            "name": "count",
            "kind": "step",
            "uses": "count_words",
            "reads": {"draft": "text"},
            "writes": {"counts": "json"},
        },
        {
            "name": "judge",
            "kind": "agent",
            "model": None,
            "reads": {"draft": "text", "counts": "json"},
            "writes": {"verdict": "json"},
            "tools": [],
            "frozen": False,
            "schema": example()["agents"][1]["schema"],
        },
        {
            "name": "publish",
            "kind": "step",
            "uses": "publish_text",
            "reads": {"draft": "text"},
            "writes": {"published": "text"},
        },
    ]
    assert described["edges"] == [
        {"from": "__start__", "to": "writer", "kind": "plain", "implicit": True},
        {"from": "writer", "to": "count", "kind": "plain"},
        {"from": "count", "to": "judge", "kind": "plain"},
        {"from": "judge", "to": "publish", "kind": "plain"},
        {"from": "writer", "to": "publish", "kind": "plain"},
        {"from": "publish", "to": "__end__", "kind": "plain", "implicit": True},
    ]
    assert (described["inputs"], described["joins"], described["groups"]) == (
        {"query": "text"},
        {"publish": "join"},
        [],
    )
    assert described["id"].startswith("sha256:") and len(described["id"]) == 71

    # Prompts are not part of the id; a node's reads are.
    edited = mas.copy(prompts={"writer": "Answer {query} at length."})
    assert edited.describe()["id"] == described["id"]
    assert load(dict(mas.to_dict())).describe() == described
    rewired = mas.copy(edges=[("writer", "count"), ("count", "judge"), ("judge", "publish")])
    assert rewired.describe()["id"] != described["id"]

    # A version 1 system describes too, its marker loop as a loop edge.
    v1 = MAS(
        [Agent("solver", "Solve {query}", "answer"), Agent("checker", "Check {answer}", "verdict", inputs=["answer"])],
        [(START, "solver"), ("solver", "checker"), ("checker", "solver"), ("checker", END)],
        llm=MockLLM(),
    )
    old = v1.describe()
    assert json.loads(json.dumps(old)) == old and old["version"] == 1
    assert {"from": "checker", "to": "solver", "kind": "loop", "when": "<continue> in output", "max_visits": 3} in old[
        "edges"
    ]
    assert [node["kind"] for node in old["nodes"]] == ["agent", "agent"]


# Section 3's worked example in full: conditions, a box, frozen and health. The
# box's inner input is `page`, since the type words are reserved value names.
@tool(name="count_words", writes={"counts": "json"})
def counted(draft: str) -> dict[str, int]:
    """Count the words of a draft."""
    return {"words": len(draft.split())}


@tool(name="publish_text", writes={"published": "text"})
def published(text: str) -> str:
    """Publish a text."""
    return text


@tool(name="apology_text", writes={"published": "text"})
def apologized(query: str) -> str:
    """Apologize for not answering."""
    return f"Sorry, no answer to {query}."


FULL = f"""
version: 2
agents:
- name: writer
  model: text-model
  reads: [query, verdict]
  writes: {{draft: text}}
  tools: [web_fetch]
- name: judge
  reads: [draft, counts]
  writes: {{verdict: json}}
  schema:
    fits: {{type: bool, means: the draft makes one claim in under 300 words}}
    action: {{type: text, one of: [publish, shorten], means: what should happen next}}
steps:
- {{name: count, uses: count_words}}
- {{name: publish, uses: publish_text, reads: {{text: draft}}}}
- {{name: apology, uses: apology_text}}
boxes:
- name: shorten
  reads: {{page: draft}}
  writes: {{short: draft}}
  system:
    version: 2
    inputs: {{page: text}}
    agents: [{{name: cutter, reads: [page], writes: {{short: text}}}}]
    prompts: {{cutter: 'Cut this to half its length: {{page}}'}}
edges:
- writer -> count -> judge
- 'judge -> writer': {{when: not verdict.fits, max_visits: 3}}
- 'judge -> publish': {{switch: verdict.action}}
- 'judge -> shorten': {{switch: verdict.action}}
- 'judge -> apology': else
- shorten -> publish
- 'shorten -> apology': {{on error: any}}
- 'count -> apology': {{on error: [timeout, outage]}}
state: {{draft: {{merge: last}}}}
frozen: [judge]
health: {{length_factor: 4}}
callables:
  count_words: {{kind: python, import: '{HERE}:counted', reads: {{draft: text}}, writes: {{counts: json}}}}
  publish_text: {{kind: python, import: '{HERE}:published', reads: {{text: text}}, writes: {{published: text}}}}
  apology_text: {{kind: python, import: '{HERE}:apologized', reads: {{query: text}}, writes: {{published: text}}}}
  web_fetch: {{kind: mcp, server: 'https://tools.example.com/mcp', reads: {{url: text}}, writes: {{web_fetch: text}}}}
prompts:
  writer: 'Answer {{query}} in a short draft. The last verdict, empty at first: {{verdict}}'
  judge: "Judge this draft: {{draft}}\\nIts word counts: {{counts}}"
"""


def full() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(FULL)
    return data


def load_full(data: dict[str, Any]) -> MAS:
    return MAS.from_dict(
        data, llms={"text-model": ToolMockLLM()}, tools=[counted, published, apologized, RemoteTools()]
    )


def test_full_worked_example_round_trips_and_checks() -> None:
    data = full()
    mas = load_full(copy.deepcopy(data))
    assert (list(mas.steps), list(mas.boxes), sorted(mas.frozen)) == (
        ["count", "publish", "apology"],
        ["shorten"],
        ["judge"],
    )
    assert mas.health == HealthSettings(length_factor=4)
    assert mas.while_edges == (("judge", "writer"),)
    box = mas.boxes["shorten"]
    assert (box.reads, box.writes, list(box.system.agents)) == ({"page": "draft"}, {"short": "draft"}, ["cutter"])
    assert mas.edges[3] == Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action", case="publish")
    assert mas.edges[-1] == Edge("count", "apology", EdgeKind.ERROR, errors=("timeout", "outage"))
    assert mas.info is not None and mas.info.writers["published"] == ("publish", "apology")  # alternatives
    assert mas.info.joins == {
        "publish": Join("merge", (("judge", "shorten"),)),
        "apology": Join("merge", (("judge", "shorten", "count"),)),
        END: Join("merge", (("publish", "apology"),)),
    }
    assert {s: t for s, t in mas.graph.edges if s == START} == {START: "writer"}  # the loop's first node

    spec = mas.to_dict()
    assert spec == data
    assert list(spec) == [
        "version", "agents", "steps", "boxes", "edges", "state", "frozen", "health", "callables", "prompts"
    ]  # fmt: skip
    again = load_full(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec and again.structure == mas.structure  # idempotent
    copied = mas.copy()
    assert (copied.to_dict(), copied.frozen, copied.health, copied.boxes) == (spec, mas.frozen, mas.health, mas.boxes)

    # Frozen and health stay out of the structure; a box's inner system does not.
    assert load_full({**data, "frozen": [], "health": {}}).structure == mas.structure
    inner = full()
    inner["boxes"][0]["system"]["agents"][0]["model"] = "text-model"
    assert load_full(inner).structure != mas.structure


def test_describe_shows_conditions_boxes_joins_and_frozen() -> None:
    mas = load_full(full())
    described = mas.describe()
    assert json.loads(json.dumps(described)) == described
    assert [(n["name"], n["kind"], n.get("frozen")) for n in described["nodes"]] == [
        ("writer", "agent", False),
        ("count", "step", None),
        ("judge", "agent", True),
        ("shorten", "box", None),
        ("publish", "step", None),
        ("apology", "step", None),
    ]
    box = described["nodes"][3]
    assert (box["reads"], box["writes"]) == ({"draft": "text"}, {"draft": "text"})
    assert box["system"] == mas.boxes["shorten"].system.describe()
    assert box["system"]["inputs"] == {"page": "text"} and box["system"]["nodes"][0]["name"] == "cutter"
    edges = described["edges"]
    assert edges[0] == {"from": START, "to": "writer", "kind": "plain", "implicit": True}
    assert {"from": "judge", "to": "writer", "kind": "while", "when": "not verdict.fits", "max_visits": 3} in edges
    assert {"from": "judge", "to": "publish", "kind": "switch", "switch": "verdict.action", "case": "publish"} in edges
    assert {"from": "judge", "to": "apology", "kind": "else"} in edges
    assert {"from": "count", "to": "apology", "kind": "error", "errors": ["timeout", "outage"]} in edges
    assert {"from": "apology", "to": END, "kind": "plain", "implicit": True} in edges
    assert described["joins"] == {"publish": "merge", "apology": "merge", END: "merge"}
    unfrozen = load_full({**full(), "frozen": []}).describe()
    assert unfrozen["id"] != described["id"] and unfrozen["nodes"][2]["frozen"] is False


# Plan S4b's nine shapes: (nodes, edges as (source, target, kind)) -> what each node with several inputs is.
_KINDS: dict[str, dict[str, Any]] = {
    "plain": {},
    "when": {"kind": EdgeKind.WHEN, "when": "flag"},
    "switch": {"kind": EdgeKind.SWITCH, "switch": "verdict.action"},
    "else": {"kind": EdgeKind.ELSE},
    "while": {"kind": EdgeKind.WHILE, "when": "flag", "max_visits": 3},
    "error": {"kind": EdgeKind.ERROR, "errors": ("any",)},
}
SHAPES: dict[str, tuple[str, list[tuple[str, str, str]], dict[str, str] | str]] = {
    "1 loop with an exit": (
        "writer render measure judge assemble sign_off",
        [("writer", "render", "plain"), ("render", "measure", "plain"), ("measure", "judge", "plain"),
         ("judge", "writer", "while"), ("judge", "assemble", "else"), ("assemble", "sign_off", "plain")],
        {},
    ),
    "2 when/else diamond": (
        "a b c m", [("a", "b", "when"), ("a", "c", "else"), ("b", "m", "plain"), ("c", "m", "plain")], {"m": "merge"}
    ),
    "3 parallel diamond": (
        "a b c j", [("a", "b", "plain"), ("a", "c", "plain"), ("b", "j", "plain"), ("c", "j", "plain")],
        {"j": "join, 2 groups"},
    ),
    "4 on-error fallback into a join": (
        "f p1 p2 fb j",
        [("f", "p1", "plain"), ("f", "p2", "plain"), ("p1", "j", "plain"), ("p1", "fb", "error"),
         ("fb", "j", "plain"), ("p2", "j", "plain")],
        {"j": "join, 2 groups"},
    ),
    "5 ambiguous": (
        "f b x d e j",
        [("f", "b", "plain"), ("f", "x", "plain"), ("x", "d", "when"), ("x", "e", "else"),
         ("b", "j", "plain"), ("d", "j", "plain")],
        "node 'j' is neither a merge nor a join: d -> j does not run on every pass that runs b -> j",
    ),
    "6 loop exit joins a parallel chain": (
        "w c a b m s1 s3 fin",
        [("w", "c", "plain"), ("c", "a", "when"), ("c", "b", "else"), ("a", "m", "plain"), ("b", "m", "plain"),
         ("m", "w", "while"), ("m", "fin", "else"), ("s1", "s3", "plain"), ("s3", "fin", "plain")],
        {"m": "merge", "fin": "join, 2 groups"},
    ),
    "7 parallel join with an inner diamond": (
        "f b x y z w j",
        [("f", "b", "plain"), ("f", "x", "plain"), ("x", "y", "when"), ("x", "z", "else"),
         ("y", "w", "plain"), ("z", "w", "plain"), ("w", "j", "plain"), ("b", "j", "plain")],
        {"w": "merge", "j": "join, 2 groups"},
    ),
    "8 switch to two terminals": (
        "judge post hold", [("judge", "post", "switch"), ("judge", "hold", "else")], {END: "merge"}
    ),
    "9 worked example": (
        "writer count judge publish shorten apology",
        [("writer", "count", "plain"), ("count", "judge", "plain"), ("judge", "writer", "while"),
         ("judge", "publish", "switch"), ("judge", "shorten", "switch"), ("judge", "apology", "else"),
         ("shorten", "publish", "plain"), ("shorten", "apology", "error"), ("count", "apology", "error")],
        {"publish": "merge", "apology": "merge", END: "merge"},
    ),
}  # fmt: skip


@pytest.mark.parametrize("shape", list(SHAPES))
def test_join_shapes_classify_as_quoted(shape: str) -> None:
    names, triples, expected = SHAPES[shape]
    edges = [Edge(source, target, **_KINDS[kind]) for source, target, kind in triples]
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=re.escape(expected)):
            classify_joins(names.split(), edges)
        return
    joins = classify_joins(names.split(), edges)
    assert {n: j.kind if j.kind == "merge" else f"join, {len(j.groups)} groups" for n, j in joins.items()} == expected
    relays = {name: join.relays for name, join in joins.items() if join.kind == "join"}
    assert relays == {
        "3 parallel diamond": {"j": (None, None)},
        "4 on-error fallback into a join": {"j": ("__in__.j.0", None)},  # p1's ok route or fb
        "6 loop exit joins a parallel chain": {"fin": ("__in__.fin.0", None)},  # m's else route
        "7 parallel join with an inner diamond": {"j": (None, None)},
    }.get(shape, {})


def _small(edges: list[Any], *agents: tuple[str, str, list[str]]) -> MAS:
    """Build a system of agents ``(name, write, reads)`` whose prompts use every read."""
    return MAS(
        [Agent(name, " ".join(f"{{{r}}}" for r in reads), write, inputs=reads) for name, write, reads in agents],
        edges=edges,
        llm=MockLLM(),
        version=2,
    )


def _operator_case(change: str) -> MAS:
    """Build the full example with one 2b problem put in, or a small system that has one."""
    data = full()
    edges: list[Any] = data["edges"]
    if change == "while closes no loop":
        edges.append({"count -> publish": {"when": "counts", "max_visits": 2}})
    elif change == "else without a decision":
        edges.append({"writer -> apology": "else"})
    elif change == "decision without an else":
        edges.remove({"judge -> apology": "else"})
        edges.append("judge -> apology")
    elif change == "case outside one of":
        edges[3] = {"judge -> shorten": {"switch": "verdict.action", "case": "trim"}}
    elif change == "case twice":
        edges[3] = {"judge -> shorten": {"switch": "verdict.action", "case": "publish"}}
    elif change == "switch on no path":
        edges[2] = {"judge -> publish": {"switch": "verdict.action == 'publish'"}}
    elif change == "budget routed":
        edges[-1] = {"count -> apology": {"on error": ["timeout", "budget"]}}
    elif change == "unknown error kind":
        edges[-1] = {"count -> apology": {"on error": ["flood"]}}
    elif change == "any beside kinds":
        edges[-1] = {"count -> apology": {"on error": ["any", "timeout"]}}
    elif change == "root not readable":
        edges[1] = {"judge -> writer": {"when": "not verdict.fits and published", "max_visits": 3}}
    elif change == "field not in the form":
        edges[1] = {"judge -> writer": {"when": "not verdict.fit", "max_visits": 3}}
    elif change == "field of text":
        edges[1] = {"judge -> writer": {"when": "draft.size > 3", "max_visits": 3}}
    elif change == "unknown function":
        edges[1] = {"judge -> writer": {"when": "not verdict.fits and approved(draft)", "max_visits": 3}}
    elif change == "bad predicate":
        edges[1] = {"judge -> writer": {"when": "not verdict.fits ==", "max_visits": 3}}
    elif change == "box passes the wrong type":
        data["boxes"][0]["reads"] = {"page": "counts"}
    elif change == "frozen ghost":
        data["frozen"] = ["judge", "ghost"]
    elif change == "reads a failed node's write":
        error = Edge("b", "f", EdgeKind.ERROR, errors=("timeout",))
        return _small([("a", "b"), ("b", "c"), error], ("a", "draft", ["query"]), ("b", "verdict", ["draft"]),
                      ("c", "done", ["verdict"]), ("f", "fixed", ["verdict"]))  # fmt: skip
    elif change == "neither merge nor join":
        when, other = Edge("x", "d", EdgeKind.WHEN, when="flag"), Edge("x", "e", EdgeKind.ELSE)
        return _small([("f", "b"), ("f", "x"), when, other, ("b", "j"), ("d", "j")], ("f", "o1", ["query"]),
                      ("b", "o2", ["query"]), ("x", "flag", ["query"]), ("d", "o3", ["query"]),
                      ("e", "o4", ["query"]), ("j", "o5", ["query"]))  # fmt: skip
    elif change == "parallel terminals":
        return _small([("a", "b"), ("a", "c")], ("a", "o1", ["query"]), ("b", "o2", ["query"]), ("c", "o3", ["query"]))
    elif change == "leaves a loop on every pass":
        loop = Edge("c", "w", EdgeKind.WHILE, when="flag", max_visits=2)
        return _small([("w", "c"), loop, Edge("c", "fin", EdgeKind.ELSE), ("w", "fin")], ("w", "o1", ["query"]),
                      ("c", "flag", ["query"]), ("fin", "o2", ["query"]))  # fmt: skip
    return load_full(data)


OPERATOR_CASES = {
    "while closes no loop": r"the while edge count -> publish closes no loop; make it a when edge",
    "else without a decision": r"node 'writer' has 0 when, while or switch edges and 1 else edges",
    "decision without an else": r"node 'judge' has 3 when, while or switch edges and 0 else edges",
    "case outside one of": r"edge judge -> shorten: case 'trim' is not one of \['publish', 'shorten'\]",
    "case twice": r"node 'judge' switches to case 'publish' twice",
    "switch on no path": r"edge judge -> publish: switch takes a path such as verdict.action",
    "budget routed": r"edge count -> apology: on error 'budget' is never routed; the run stops on it",
    "unknown error kind": r"edge count -> apology: on error 'flood' is no error kind",
    "any beside kinds": r"edge count -> apology: on error 'any' stands alone",
    "root not readable": r"edge judge -> writer decides on 'published', which is not readable at 'judge'",
    "field not in the form": r"edge judge -> writer reads verdict.fit, but 'judge''s answer form has no field 'fit'",
    "field of text": r"edge judge -> writer reads the field 'size' of 'draft', which is not json",
    "unknown function": r"edge judge -> writer calls 'approved', which is no registered callable",
    "bad predicate": r"edge judge -> writer: .* at position 19",
    "box passes the wrong type": r"box 'shorten' passes json 'counts' to its text input 'page'",
    "frozen ghost": r"frozen names 'ghost', which is no node",
    "reads a failed node's write": r"'f' runs only when 'b' fails, so it cannot read 'verdict'",
    "neither merge nor join": r"node 'j' is neither a merge nor a join: d -> j does not run on every pass that runs b",
    "parallel terminals": r"several nodes feed '__end__' \(\['b', 'c'\]\) on one pass",
    "leaves a loop on every pass": r"node 'fin' cannot wait for w -> fin, which leaves the loop w \.\. c on every pass",
}


@pytest.mark.parametrize("change", list(OPERATOR_CASES))
def test_check_names_every_operator_problem(change: str) -> None:
    with pytest.raises(ValueError) as info:
        _operator_case(change)
    message = str(info.value)
    assert len(re.findall(OPERATOR_CASES[change], message)) == 1, message


def test_box_maps_and_containment() -> None:
    data = full()
    with pytest.raises(ValueError, match=r"reads must map exactly the inner inputs \['page'\], got \['text'\]"):
        load_full({**data, "boxes": [{**data["boxes"][0], "reads": {"text": "draft"}}]})
    with pytest.raises(ValueError, match=r"writes must map inner writes, some of \['short'\], got \['long'\]"):
        load_full({**data, "boxes": [{**data["boxes"][0], "writes": {"long": "draft"}}]})
    v1 = MAS([Agent("a", "Do {query}", "x")], llm=MockLLM())
    with pytest.raises(ValueError, match="the system in a box must be version 2"):
        Box("old", v1, {"query": "query"}, {"x": "x"})

    mas = load_full(data)
    inner = mas.boxes["shorten"].system
    nested = MAS([], boxes=[Box("outer", mas, {"query": "page"}, {"published": "short"})], inputs={"page": "text"})
    assert nested.describe()["nodes"][0]["system"]["nodes"][3]["system"] == inner.describe()  # boxes nest
    inner.boxes["again"] = Box("again", nested, {"page": "page"}, {"short": "page2"})  # a system put inside itself
    with pytest.raises(ValueError, match="box 'again' contains the system it sits in"):
        check_system(inner)
