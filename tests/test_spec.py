"""Spec version 2: the worked example round-trips, the check names every problem, describe is plain data.

These run on construction, serialization and ``describe()`` alone; running a
version 2 system is tested in ``test_steps.py`` and ``test_operators.py``.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from xmas import END, MAS, START, Agent
from xmas.check import check_system
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.health import HealthSettings
from xmas.nodes import Box, Step
from xmas.routes import Join
from xmas.tools import Tool, ToolSource, tool

from .conftest import MockLLM, apology_text, count_words, publish_text


@tool
def repeat(text: str, times: int) -> str:
    """Repeat a text."""
    return text * times


class RemoteTools(ToolSource):
    """A tool source standing in for an MCP server."""

    url = "https://tools.example.com/mcp"

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        return [Tool("web_fetch", "Fetch a page.", schema)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer every call with the same page."""
        return "page"


# The worked example: conditions, a box, frozen and health. The box's inner
# input is `page`, since the type words are reserved value names.
EXAMPLE = """
version: 2
agents:
- name: writer
  model: text-model
  reads: [query, verdict]
  writes: {draft: text}
  tools: [web_fetch]
- name: judge
  reads: [draft, counts]
  writes: {verdict: json}
  schema:
    fits: {type: bool, means: the draft makes one claim in under 300 words}
    action: {type: text, one of: [publish, shorten], means: what should happen next}
steps:
- {name: count, uses: count_words}
- {name: publish, uses: publish_text, reads: {text: draft}}
- {name: apology, uses: apology_text}
boxes:
- name: shorten
  reads: {page: draft}
  writes: {short: draft}
  system:
    version: 2
    inputs: {page: text}
    agents: [{name: cutter, reads: [page], writes: {short: text}}]
    prompts: {cutter: 'Cut this to half its length: {page}'}
edges:
- writer -> count -> judge
- 'judge -> writer': {when: not verdict.fits, max_visits: 3}
- 'judge -> publish': {switch: verdict.action}
- 'judge -> shorten': {switch: verdict.action}
- 'judge -> apology': else
- shorten -> publish
- 'shorten -> apology': {on error: any}
- 'count -> apology': {on error: [timeout, outage]}
state: {draft: {merge: last}}
frozen: [judge]
health: {length_factor: 4}
callables:
  count_words: {kind: python, import: 'tests.conftest:count_words', reads: {draft: text}, writes: {counts: json}}
  publish_text: {kind: python, import: 'tests.conftest:publish_text', reads: {text: text}, writes: {published: text}}
  apology_text: {kind: python, import: 'tests.conftest:apology_text', reads: {query: text}, writes: {published: text}}
  web_fetch: {kind: mcp, server: 'https://tools.example.com/mcp', reads: {url: text}, writes: {web_fetch: text}}
prompts:
  writer: 'Answer {query} in a short draft. The last verdict, empty at first: {verdict}'
  judge: "Judge this draft: {draft}\\nIts word counts: {counts}"
"""


def example() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE)
    return data


def load(data: dict[str, Any]) -> MAS:
    return MAS.from_dict(
        copy.deepcopy(data),
        llms={"text-model": MockLLM(turns=[])},
        tools=[count_words, publish_text, apology_text, repeat, RemoteTools()],
    )


def test_the_worked_example_round_trips() -> None:
    data = example()
    mas = load(data)
    assert (mas.version, list(mas.agents), list(mas.steps), list(mas.boxes), sorted(mas.frozen)) == (
        2,
        ["writer", "judge"],
        ["count", "publish", "apology"],
        ["shorten"],
        ["judge"],
    )
    assert mas.agents["judge"].output_type == "json"
    assert mas.steps["publish"] == Step("publish", "publish_text", {"text": "draft"})
    renamed = Step("count", "count_words", writes={"counts": "tally"})
    assert renamed.to_dict() == {"name": "count", "uses": "count_words", "writes": {"counts": "tally"}}
    assert mas.health == HealthSettings(length_factor=4)
    assert mas.while_edges == (("judge", "writer"),)
    box = mas.boxes["shorten"]
    assert (box.reads, box.writes, list(box.system.agents)) == ({"page": "draft"}, {"short": "draft"}, ["cutter"])
    assert mas.edges[0] == Edge("writer", "count")
    assert mas.edges[3] == Edge("judge", "publish", EdgeKind.SWITCH, switch="verdict.action", case="publish")
    assert mas.edges[-1] == Edge("count", "apology", EdgeKind.ERROR, errors=("timeout", "outage"))
    assert (mas.merges, {k: str(v) for k, v in mas.inputs.items()}) == ({"draft": "last"}, {"query": "text"})
    assert (
        mas.info is not None and mas.info.writers["counts"] == ("count",) and str(mas.info.types["verdict"]) == "json"
    )
    assert mas.info.writers["published"] == ("publish", "apology")  # alternatives
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
    again = load(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec and again.structure == mas.structure  # idempotent
    copied = mas.copy()
    assert (copied.to_dict(), copied.frozen, copied.health, copied.boxes) == (spec, mas.frozen, mas.health, mas.boxes)
    assert copied.structure == mas.structure

    # A new answer form, or a box's inner system, changes the structure; frozen and health do not.
    reformed = mas.copy()
    reformed.agents["judge"].schema = {"fits": {"type": "bool"}}
    assert reformed.structure != mas.structure
    assert load({**data, "frozen": [], "health": {}}).structure == mas.structure
    inner = example()
    inner["boxes"][0]["system"]["agents"][0]["model"] = "text-model"
    assert load(inner).structure != mas.structure


def test_a_version_2_system_is_not_saved_as_one_yaml_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.mas bundle"):
        load(example()).to_yaml(tmp_path / "v2.yaml")


def test_the_edge_grammar_parses_and_dumps_back_as_written() -> None:
    written = example()["edges"]
    edges = parse_edges(written)
    assert edges[2] == Edge("judge", "writer", EdgeKind.WHILE, when="not verdict.fits", max_visits=3)
    assert edges[3].case == "publish" and edges[8].errors == ("timeout", "outage")
    assert dump_edges(edges) == written and dump_edges(parse_edges(dump_edges(edges))) == written

    other_forms = [
        {"a -> b": {"switch": "v.k", "case": "go"}},
        {"a -> c": {"when": "v.n > 2"}},
        {"b -> c": {"on error": "outage"}},
    ]
    assert [edge.kind for edge in parse_edges(other_forms)] == [EdgeKind.SWITCH, EdgeKind.WHEN, EdgeKind.ERROR]
    assert dump_edges(parse_edges(other_forms))[2] == {"b -> c": {"on error": ["outage"]}}
    assert dump_edges(parse_edges(["a -> b -> c", "a -> c"])) == ["a -> b -> c", "a -> c"]  # two chains


def test_a_version_1_system_has_no_version_2_parts() -> None:
    v1 = MAS(
        [Agent("solver", "Solve {query}", "answer"), Agent("checker", "Check {answer}", "verdict", inputs=["answer"])],
        [(START, "solver"), ("solver", "checker"), ("checker", "solver"), ("checker", END)],
        llm=MockLLM(),
    )
    assert (v1.version, v1.steps, v1.info) == (1, {}, None)
    assert v1.copy().structure == v1.structure
    v2_lines = ("step ", "edge ", "form ", "input ", "merge ", "box ")
    assert not any(line.startswith(v2_lines) for line in v1.structure.splitlines())

    described = v1.describe()  # its marker loop is a loop edge
    assert json.loads(json.dumps(described)) == described and described["version"] == 1
    loop = {"from": "checker", "to": "solver", "kind": "loop", "when": "<continue> in output", "max_visits": 3}
    assert loop in described["edges"]
    assert [node["kind"] for node in described["nodes"]] == ["agent", "agent"]


EXAMPLE_DATA = example()
AGENTS, STEPS, EDGES = EXAMPLE_DATA["agents"], EXAMPLE_DATA["steps"], EXAMPLE_DATA["edges"]
PROMPTS, CALLABLES = EXAMPLE_DATA["prompts"], EXAMPLE_DATA["callables"]
WRITER, JUDGE = AGENTS
COUNT, PUBLISH, APOLOGY = STEPS


@pytest.mark.parametrize(
    ("change", "phrases"),
    [
        pytest.param(
            {
                "agents": [WRITER, {**JUDGE, "reads": [*JUDGE["reads"], "notes"]}],
                "prompts": {**PROMPTS, "judge": PROMPTS["judge"] + " {notes}"},
                "steps": [{**COUNT, "writes": {"counts": "len"}}, PUBLISH, APOLOGY],  # a second problem
            },
            [
                r"3 problems",
                r"'judge' reads 'notes', which is neither an input",
                r"'count' writes 'len', which is a reserved name",
            ],
            id="unread input",
        ),
        pytest.param({"state": {}}, [r"value 'draft' has 2 writers \('writer', 'shorten'\)"], id="two writers"),
        pytest.param(
            {"prompts": {**PROMPTS, "writer": PROMPTS["writer"] + " {counts}"}},
            [r"agent 'writer': prompt placeholders \['counts'\] are not among its reads"],
            id="placeholder not a read",
        ),
        pytest.param(
            {
                "steps": [{**COUNT, "uses": "count_letters"}, PUBLISH, APOLOGY],
                "callables": {name: entry for name, entry in CALLABLES.items() if name != "count_words"},
            },
            [r"Step 'count' uses unknown callable 'count_letters'"],
            id="unknown uses",
        ),
        pytest.param(
            {
                "steps": [*STEPS, {"name": "echo", "uses": "repeat", "reads": {"times": "draft"}}],
                "edges": [*EDGES, "publish -> echo"],
            },
            [r"step 'echo' passes text 'draft' to parameter 'times' \(integer\)"],
            id="text into an int parameter",
        ),
        pytest.param(
            {"inputs": {"query": "text", "photo": "image"}},
            [r"input 'photo' has type 'image', which arrives in phase 3"],
            id="image",
        ),
        pytest.param(
            {"agents": [{**WRITER, "writes": {"len": "text"}}, JUDGE]},
            [r"'writer' writes 'len', which is a reserved name"],
            id="reserved name",
        ),
        pytest.param(
            {"steps": [COUNT, {**PUBLISH, "reads": {"txt": "draft"}}, APOLOGY]},
            [r"step 'publish': reads renames 'txt', which is not a parameter of 'publish_text'"],
            id="bad rename",
        ),
        pytest.param(
            {"edges": [*EDGES, "publish -> count"]},
            [r"a loop needs a while edge: the edge publish -> count closes a cycle"],
            id="cycle without a while edge",
        ),
        pytest.param(
            {"prompts": {"writer": PROMPTS["writer"]}},
            [r"Agent 'judge' has no prompt in the 'prompts' section"],
            id="missing prompt",
        ),
        pytest.param(
            {"callables": {**CALLABLES, "count_words": {**CALLABLES["count_words"], "writes": {"counts": "text"}}}},
            [r"callable 'count_words' writes \{'counts': 'text'\} in the spec"],
            id="callables mismatch",
        ),
    ],
)
def test_check_names_every_dataflow_problem(change: dict[str, Any], phrases: list[str]) -> None:
    with pytest.raises(ValueError) as info:
        load({**EXAMPLE_DATA, **change})
    for phrase in phrases:
        assert re.search(phrase, str(info.value)), (phrase, str(info.value))


def edge(index: int, replacement: Any) -> dict[str, list[Any]]:
    """The example's edges with the one at ``index`` replaced."""
    return {"edges": [replacement if i == index else item for i, item in enumerate(EDGES)]}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"edges": [*EDGES, {"count -> publish": {"when": "counts", "max_visits": 2}}]},
            r"the while edge count -> publish closes no loop; make it a when edge",
            id="while closes no loop",
        ),
        pytest.param(
            {"edges": [*EDGES, {"writer -> apology": "else"}]},
            r"node 'writer' has 0 when, while or switch edges and 1 else edges",
            id="else without a decision",
        ),
        pytest.param(
            edge(4, "judge -> apology"),
            r"node 'judge' has 3 when, while or switch edges and 0 else edges",
            id="decision without an else",
        ),
        pytest.param(
            edge(3, {"judge -> shorten": {"switch": "verdict.action", "case": "trim"}}),
            r"edge judge -> shorten: case 'trim' is not one of \['publish', 'shorten'\]",
            id="case outside one of",
        ),
        pytest.param(
            edge(3, {"judge -> shorten": {"switch": "verdict.action", "case": "publish"}}),
            r"node 'judge' switches to case 'publish' twice",
            id="case twice",
        ),
        pytest.param(
            edge(2, {"judge -> publish": {"switch": "verdict.action == 'publish'"}}),
            r"edge judge -> publish: switch takes a path such as verdict.action",
            id="switch on no path",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["timeout", "budget"]}}),
            r"edge count -> apology: on error 'budget' is never routed; the run stops on it",
            id="budget routed",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["flood"]}}),
            r"edge count -> apology: on error 'flood' is no error kind",
            id="unknown error kind",
        ),
        pytest.param(
            edge(7, {"count -> apology": {"on error": ["any", "timeout"]}}),
            r"edge count -> apology: on error 'any' stands alone",
            id="any beside kinds",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits and published", "max_visits": 3}}),
            r"edge judge -> writer decides on 'published', which is not readable at 'judge'",
            id="root not readable",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fit", "max_visits": 3}}),
            r"edge judge -> writer reads verdict.fit, but 'judge''s answer form has no field 'fit'",
            id="field not in the form",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "draft.size > 3", "max_visits": 3}}),
            r"edge judge -> writer reads the field 'size' of 'draft', which is not json",
            id="field of text",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits and approved(draft)", "max_visits": 3}}),
            r"edge judge -> writer calls 'approved', which is no registered callable",
            id="unknown function",
        ),
        pytest.param(
            edge(1, {"judge -> writer": {"when": "not verdict.fits ==", "max_visits": 3}}),
            r"edge judge -> writer: .* at position 19",
            id="bad predicate",
        ),
        pytest.param(
            {"boxes": [{**EXAMPLE_DATA["boxes"][0], "reads": {"page": "counts"}}]},
            r"box 'shorten' passes json 'counts' to its text input 'page'",
            id="box passes the wrong type",
        ),
        pytest.param({"frozen": ["judge", "ghost"]}, r"frozen names 'ghost', which is no node", id="frozen ghost"),
    ],
)
def test_check_names_every_operator_problem(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as info:
        load({**EXAMPLE_DATA, **change})
    assert len(re.findall(message, str(info.value))) == 1, str(info.value)


@pytest.mark.parametrize(
    ("edges", "agents", "message"),
    [
        pytest.param(
            [("a", "b"), ("b", "c"), Edge("b", "f", EdgeKind.ERROR, errors=("timeout",))],
            [
                ("a", "draft", ["query"]),
                ("b", "verdict", ["draft"]),
                ("c", "done", ["verdict"]),
                ("f", "fixed", ["verdict"]),
            ],
            r"'f' runs only when 'b' fails, so it cannot read 'verdict'",
            id="reads a failed node's write",
        ),
        pytest.param(
            [
                ("f", "b"),
                ("f", "x"),
                Edge("x", "d", EdgeKind.WHEN, when="flag"),
                Edge("x", "e", EdgeKind.ELSE),
                ("b", "j"),
                ("d", "j"),
            ],
            [
                ("f", "o1", ["query"]),
                ("b", "o2", ["query"]),
                ("x", "flag", ["query"]),
                ("d", "o3", ["query"]),
                ("e", "o4", ["query"]),
                ("j", "o5", ["query"]),
            ],
            r"node 'j' is neither a merge nor a join: d -> j does not run on every pass that runs b",
            id="neither merge nor join",
        ),
        pytest.param(
            [("a", "b"), ("a", "c")],
            [("a", "o1", ["query"]), ("b", "o2", ["query"]), ("c", "o3", ["query"])],
            r"several nodes feed '__end__' \(\['b', 'c'\]\) on one pass",
            id="parallel terminals",
        ),
        pytest.param(
            [
                ("w", "c"),
                Edge("c", "w", EdgeKind.WHILE, when="flag", max_visits=2),
                Edge("c", "fin", EdgeKind.ELSE),
                ("w", "fin"),
            ],
            [("w", "o1", ["query"]), ("c", "flag", ["query"]), ("fin", "o2", ["query"])],
            r"node 'fin' cannot wait for w -> fin, which leaves the loop w \.\. c on every pass",
            id="leaves a loop on every pass",
        ),
    ],
)
def test_check_names_every_operator_problem_of_a_small_system(
    edges: list[Any], agents: list[tuple[str, str, list[str]]], message: str
) -> None:
    """Agents are ``(name, write, reads)``, and each prompt uses every read."""
    with pytest.raises(ValueError) as info:
        MAS(
            [Agent(name, " ".join(f"{{{r}}}" for r in reads), write, inputs=reads) for name, write, reads in agents],
            edges=edges,
            llm=MockLLM(),
            version=2,
        )
    assert len(re.findall(message, str(info.value))) == 1, str(info.value)


def test_describe_is_plain_data_that_names_every_node_and_edge() -> None:
    mas = load(example())
    described = mas.describe()
    assert json.loads(json.dumps(described)) == described
    assert set(described) == {"version", "inputs", "nodes", "edges", "joins", "groups", "id"}
    assert (described["version"], described["inputs"], described["groups"]) == (2, {"query": "text"}, [])
    assert described["joins"] == {"publish": "merge", "apology": "merge", END: "merge"}
    assert described["id"].startswith("sha256:") and len(described["id"]) == 71

    writer, count, judge, box, publish, apology = described["nodes"]
    assert writer == {
        "name": "writer",
        "kind": "agent",
        "model": "text-model",
        "reads": {"query": "text", "verdict": "json"},
        "writes": {"draft": "text"},
        "tools": ["web_fetch"],
        "frozen": False,
    }
    assert count == {
        "name": "count",
        "kind": "step",
        "uses": "count_words",
        "reads": {"draft": "text"},
        "writes": {"counts": "json"},
    }
    assert judge == {
        "name": "judge",
        "kind": "agent",
        "model": None,
        "reads": {"draft": "text", "counts": "json"},
        "writes": {"verdict": "json"},
        "tools": [],
        "frozen": True,
        "schema": JUDGE["schema"],
    }
    assert (box["name"], box["kind"]) == ("shorten", "box")
    assert (box["reads"], box["writes"]) == ({"draft": "text"}, {"draft": "text"})
    assert box["system"] == mas.boxes["shorten"].system.describe()
    assert box["system"]["inputs"] == {"page": "text"} and box["system"]["nodes"][0]["name"] == "cutter"
    assert [(node["name"], node["kind"]) for node in (publish, apology)] == [("publish", "step"), ("apology", "step")]

    edges = described["edges"]
    assert edges[0] == {"from": START, "to": "writer", "kind": "plain", "implicit": True}
    assert {"from": "writer", "to": "count", "kind": "plain"} in edges
    assert {"from": "judge", "to": "writer", "kind": "while", "when": "not verdict.fits", "max_visits": 3} in edges
    assert {"from": "judge", "to": "publish", "kind": "switch", "switch": "verdict.action", "case": "publish"} in edges
    assert {"from": "judge", "to": "apology", "kind": "else"} in edges
    assert {"from": "count", "to": "apology", "kind": "error", "errors": ["timeout", "outage"]} in edges
    assert {"from": "apology", "to": END, "kind": "plain", "implicit": True} in edges


def test_the_describe_id_follows_wiring_and_frozen_but_not_prompts() -> None:
    mas = load(example())
    described = mas.describe()
    assert load(mas.to_dict()).describe() == described
    assert mas.copy(prompts={"writer": "Answer {query} at length. {verdict}"}).describe()["id"] == described["id"]
    rewired = mas.copy(edges=mas.edges[:-1])  # count's on error edge gone
    assert rewired.describe()["id"] != described["id"]
    unfrozen = load({**example(), "frozen": []}).describe()
    assert unfrozen["id"] != described["id"] and unfrozen["nodes"][2]["frozen"] is False


def test_a_box_maps_its_inner_inputs_and_never_contains_itself() -> None:
    data = example()
    with pytest.raises(ValueError, match=r"reads must map exactly the inner inputs \['page'\], got \['text'\]"):
        load({**data, "boxes": [{**data["boxes"][0], "reads": {"text": "draft"}}]})
    with pytest.raises(ValueError, match=r"writes must map inner writes, some of \['short'\], got \['long'\]"):
        load({**data, "boxes": [{**data["boxes"][0], "writes": {"long": "draft"}}]})
    v1 = MAS([Agent("a", "Do {query}", "x")], llm=MockLLM())
    with pytest.raises(ValueError, match="the system in a box must be version 2"):
        Box("old", v1, {"query": "query"}, {"x": "x"})

    mas = load(data)
    inner = mas.boxes["shorten"].system
    nested = MAS([], boxes=[Box("outer", mas, {"query": "page"}, {"published": "short"})], inputs={"page": "text"})
    assert nested.describe()["nodes"][0]["system"]["nodes"][3]["system"] == inner.describe()  # boxes nest
    inner.boxes["again"] = Box("again", nested, {"page": "page"}, {"short": "page2"})  # a system put inside itself
    with pytest.raises(ValueError, match="box 'again' contains the system it sits in"):
        check_system(inner)
