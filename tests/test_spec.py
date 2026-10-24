"""Spec version 2: the worked examples round-trip, the check names every problem, describe is plain data.

These run on construction, serialization and ``describe()`` alone; running a
version 2 system is tested in ``test_steps.py``, ``test_operators.py`` and
``test_chains.py``.
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
from xmas.chains import ForEach
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
            [r"input 'photo' has type 'image'; image values are not available yet"],
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
    assert set(described) == {"version", "inputs", "nodes", "edges", "joins", "id"}
    assert (described["version"], described["inputs"]) == (2, {"query": "text"})
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


@tool(writes={"measure": "json"})
def render_page(draft: str) -> dict[str, int]:
    """Measure a drawn page."""
    return {"words": len(draft.split())}


@tool(writes={"deck": "json"})
def assemble_deck(draft: list[str], verdict: list[Any], caption: list[str]) -> dict[str, Any]:
    """Put the pages and captions of a deck together."""
    return {"pages": draft, "verdicts": verdict, "captions": caption}


# Two chains, one per slide and one per figure; END joins the deck with the notes.
DECK = """
version: 2
inputs: {plan: json, house: text}
outputs: {deck: json, notes: text}
agents:
- {name: drafter, model: writer-model,
   for each: {item: page, in: plan.slides, at once: 4, key: id, item budget: {cost_usd: 0.05}},
   reads: {slide: page, rules: house, verdict: verdict}, writes: {draft: text}}
- {name: checker, model: judge-model, prompt: judge, for each: page in plan.slides,
   reads: {planned: page, shown: measure}, writes: {verdict: json},
   schema: {fits: {type: bool, means: the page makes its claim at a glance}, fix: {type: text}}}
- {name: captioner, model: writer-model, for each: {item: figure, in: plan.figures, key: id},
   reads: {figure: figure, rules: house}, writes: {caption: text}}
- {name: figure_checker, model: judge-model, prompt: judge, for each: figure in plan.figures,
   reads: {planned: figure, shown: caption}, writes: {figure_verdict: json},
   schema: {fits: {type: bool}, fix: {type: text}}}
- {name: noter, model: writer-model, reads: [plan], writes: {notes: text}}
steps:
- {name: render, uses: render_page, for each: page in plan.slides, retry: {outage: 1}}
- {name: assemble, uses: assemble_deck}
edges:
- drafter -> render -> checker
- 'checker -> drafter': {when: not verdict.fits, max_visits: 2}
- 'checker -> assemble': else
- captioner -> figure_checker -> assemble
callables:
  render_page: {kind: python, import: 'tests.test_spec:render_page', reads: {draft: text}, writes: {measure: json}}
  assemble_deck: {kind: python, import: 'tests.test_spec:assemble_deck',
                  reads: {draft: json, verdict: json, caption: json}, writes: {deck: json}}
prompts:
  drafter: "Draft slide {slide} in this house style: {rules}\\nThe last verdict, empty at first: {verdict}"
  judge: "Does the page make the claim of its plan?\\nPlan: {planned}\\nAs drawn: {shown}"
  captioner: "Caption figure {figure} in this house style: {rules}"
  noter: "Write speaker notes for this plan: {plan}"
"""
DECK_DATA: dict[str, Any] = yaml.safe_load(DECK)


def load_deck(data: dict[str, Any] | None = None) -> MAS:
    return MAS.from_dict(
        copy.deepcopy(DECK_DATA if data is None else data),
        llms={"writer-model": MockLLM(), "judge-model": MockLLM()},
        tools=[render_page, assemble_deck],
    )


def test_the_deck_example_round_trips() -> None:
    mas = load_deck()
    assert (mas.version, list(mas.each), mas.prompt_names, mas.retry) == (
        2,
        ["drafter", "checker", "captioner", "figure_checker", "render"],
        {"checker": "judge", "figure_checker": "judge"},
        {"render": {"outage": 1}},
    )
    assert mas.each["drafter"] == ForEach("page", "plan.slides", 4, "id", {"cost_usd": 0.05})
    assert mas.agents["drafter"].reads == {"slide": "page", "rules": "house", "verdict": "verdict"}
    assert mas.agents["checker"].prompt == mas.agents["figure_checker"].prompt == DECK_DATA["prompts"]["judge"]
    info = mas._checked()
    drafter, captioner = info.chains["drafter"], info.chains["captioner"]
    assert (sorted(drafter.nodes), drafter.each.root, captioner.nodes) == (
        ["checker", "drafter", "render"],
        "plan",
        ("captioner", "figure_checker"),
    )
    words = {name: str(vtype) for name, vtype in drafter.inputs.items()}
    assert words == {"page": "json", "house": "text"} and set(captioner.inputs) == {"figure", "house"}
    assert {name: str(vtype) for name, vtype in drafter.writes.items()} == {
        "draft": "text",
        "verdict": "json",
        "measure": "json",
    }
    assert {name: str(info.types[name]) for name in ("draft", "verdict", "measure", "caption", "figure_verdict")} == {
        "draft": "list of text",
        "verdict": "list of json",
        "measure": "list of json",
        "caption": "list of text",
        "figure_verdict": "list of json",
    }
    assert Edge("checker", END, EdgeKind.ELSE) in drafter.system.edges  # the exit ends one item's run
    assert drafter.system.resolve_llm(mas.agents["checker"]) is mas.llms["judge-model"]  # through the outer system
    assert info.edges == (Edge("drafter", "assemble"), Edge("captioner", "assemble"))
    assert info.joins == {
        "assemble": Join("join", (("drafter",), ("captioner",)), (None, None)),
        END: Join("join", (("noter",), ("assemble",)), (None, None)),
    }
    assert set(info.graph.successors("checker")) == {"drafter", "assemble"}  # the full topology, for describe

    spec = mas.to_dict()
    assert spec == DECK_DATA
    assert list(spec) == ["version", "inputs", "outputs", "agents", "steps", "edges", "callables", "prompts"]
    assert list(spec["agents"][1]) == ["name", "model", "prompt", "for each", "reads", "writes", "schema"]
    assert list(spec["steps"][0]) == ["name", "uses", "for each", "retry"]
    again = load_deck(json.loads(json.dumps(spec)))
    assert again.to_dict() == spec and again.structure == mas.structure  # idempotent

    # A prompt given to one agent of a shared prompt reaches the others; a split made by hand does not dump.
    shared = "Is this claim made? {planned} {shown}"
    copied = mas.copy(prompts={"checker": shared})
    assert copied.agents["figure_checker"].prompt == shared and copied.to_dict()["prompts"]["judge"] == shared
    copied.agents["checker"].prompt = "Split {planned} {shown}"
    with pytest.raises(ValueError, match="sharing the prompt 'judge' hold different texts"):
        copied.to_dict()

    described = mas.describe()
    assert json.loads(json.dumps(described)) == described and described["outputs"] == {"deck": "json", "notes": "text"}
    assert described["groups"] == [
        {
            "name": "drafter",
            "for each": "page in plan.slides",
            "nodes": ["drafter", "render", "checker"],
            "at once": 4,
            "key": "id",
            "item budget": {"cost_usd": 0.05},
        },
        {
            "name": "captioner",
            "for each": "figure in plan.figures",
            "nodes": ["captioner", "figure_checker"],
            "key": "id",
        },
    ]
    nodes = {node["name"]: node for node in described["nodes"]}
    assert nodes["drafter"]["for each"] == "page in plan.slides" and "for each" not in nodes["noter"]
    assert (
        nodes["checker"]["placeholders"] == {"planned": "page", "shown": "measure"}
        and "placeholders" not in nodes["noter"]
    )
    assert nodes["checker"]["reads"] == {"page": "json", "measure": "json"}  # one item's types
    assert nodes["assemble"]["reads"] == {"draft": "list of text", "verdict": "list of json", "caption": "list of text"}
    assert nodes["render"]["retry"] == {"outage": 1} and described["joins"] == {"assemble": "join", END: "join"}

    def variant(name: str, **fields: Any) -> MAS:
        data = copy.deepcopy(DECK_DATA)
        for entry in data["agents"] + data["steps"]:
            entry.update(fields if entry["name"] == name else {})
        return load_deck(data)

    head = DECK_DATA["agents"][0]["for each"]
    assert variant("drafter", **{"for each": {**head, "at once": 2}}).structure != mas.structure
    assert variant("drafter", **{"for each": {**head, "item budget": {"wall_s": 60}}}).structure == mas.structure
    assert variant("noter", reads={"plan": "plan", "house": "house"}).structure != mas.structure
    assert variant("render", retry={"timeout": 2}).structure != mas.structure
    assert mas.copy(prompts={"noter": "Notes on {plan}"}).structure == mas.structure


def chain_of(form: Any) -> MAS:
    """A one-agent chain over ``plan.slides`` whose list of drafts is the output."""
    return MAS(
        [Agent("writer", "Write {page}", "draft", inputs=["page"])],
        inputs={"plan": "json"},
        outputs={"draft": "list of text"},
        each={"writer": form},
        llm=MockLLM(),
    )


@pytest.mark.parametrize(
    ("form", "want"),
    [
        pytest.param("page in plan.slides", ForEach("page", "plan.slides"), id="short"),
        pytest.param({"item": "page", "in": "plan.slides"}, ForEach("page", "plan.slides"), id="long"),
        pytest.param(" page \t in  plan.slides ", ForEach("page", "plan.slides"), id="spaces"),
        pytest.param(
            {"item": "page", "in": "plan.slides", "at once": 4, "key": "id", "item budget": {"wall_s": 120}},
            ForEach("page", "plan.slides", 4, "id", {"wall_s": 120}),
            id="options",
        ),
        pytest.param(
            "page in plan.slides[0]", r"'page in plan.slides\[0\]' must read '<item> in <list>'", id="indexing"
        ),
        pytest.param("page plan.slides", r"must read '<item> in <list>'", id="no in"),
        pytest.param("in in plan.slides", r"chain 'writer': the item 'in' is a reserved name", id="reserved item"),
        pytest.param(
            {"item": "page", "in": "plan.slides", "key": "id", "at once": 0},
            r"at once must be a positive integer, got 0",
            id="round trip",
        ),
    ],
)
def test_for_each_grammar(form: Any, want: ForEach | str) -> None:
    if isinstance(want, str):
        with pytest.raises(ValueError, match=want):
            chain_of(form)
        return
    each = ForEach.parse(form)
    assert each == want and (each.root, each.fields) == ("plan", ("slides",)) and str(each) == "page in plan.slides"
    assert ForEach.parse(each.to_spec()) == each and chain_of(each).each["writer"] == each
    assert each.to_spec() == (form if each.options else "page in plan.slides")
    if each.options:  # the round trip of a dumped long form, the options left out when none is set
        assert chain_of(form).to_dict()["agents"][0]["for each"] == form
        assert ForEach.parse({"item": "page", "in": "plan.slides", "key": "id"}).to_spec() == {
            "item": "page",
            "in": "plan.slides",
            "key": "id",
        }


def deck(**sections: Any) -> dict[str, Any]:
    """The deck with some top-level sections replaced."""
    return {**copy.deepcopy(DECK_DATA), **sections}


def entries(section: str, name: str, **fields: Any) -> dict[str, list[dict[str, Any]]]:
    """The deck's ``section`` with the fields of entry ``name`` replaced, ``None`` dropping one."""
    changed = []
    for entry in copy.deepcopy(DECK_DATA[section]):
        if entry["name"] == name:
            entry = {key: value for key, value in {**entry, **fields}.items() if value is not None}
        changed.append(entry)
    return {section: changed}


def over(path: str, key: str | None = None) -> dict[str, Any]:
    """The deck with the slide chain running over ``path``."""
    head = {"item": "page", "in": path, **({"key": key} if key else {})}
    agents = entries("agents", "drafter", **{"for each": head})["agents"]
    agents[1]["for each"] = f"page in {path}"
    return deck(agents=agents, steps=entries("steps", "render", **{"for each": f"page in {path}"})["steps"])


def boxed_chain() -> MAS:
    """A chain whose one node is a box holding a chain of its own."""
    inner = chain_of("page in plan.slides")
    return MAS(
        [],
        boxes=[Box("pages", inner, {"plan": "deck"}, {"draft": "drafts"})],
        inputs={"decks": "json"},
        outputs={"drafts": "list of list of text"},
        each={"pages": "deck in decks"},
    )


def small(**settings: Any) -> MAS:
    """Two agents, ``a -> b``, with the given settings."""
    agents = [Agent("a", "Do {query}", "x"), Agent("b", "Check {x}", "y", inputs=["x"])]
    return MAS(agents, edges=[("a", "b")], llm=MockLLM(), **settings)


@pytest.mark.parametrize(
    ("build", "phrase"),
    [
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], "noter -> checker"])),
            r"chain \['drafter', 'checker', 'render'\] \(page in plan.slides\) is entered at \['checker', 'drafter'\]",
            id="two heads",
        ),
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], "render -> noter"])),
            r"chain 'drafter' leaves to \['assemble', 'noter'\]; a chain leaves to one node",
            id="exits to two targets",
        ),
        pytest.param(
            lambda: load_deck(deck(edges=[*DECK_DATA["edges"], {"render -> assemble": {"on error": "any"}}])),
            r"chain 'drafter': the error edge render -> assemble leaves it; route it inside the chain",
            id="an error edge leaving",
        ),
        pytest.param(
            boxed_chain, r"chain 'pages': box 'pages' holds a for each chain; chains do not nest", id="nested"
        ),
        pytest.param(
            lambda: load_deck(deck(inputs={"plan": "json", "house": "text", "page": "json"})),
            r"chain 'drafter': the item 'page' is an input or a written value too",
            id="item is an input",
        ),
        pytest.param(
            lambda: load_deck(over("spec.slides")),
            r"'drafter' reads 'spec', which is neither an input nor written by any node",
            id="root unreadable",
        ),
        pytest.param(
            lambda: load_deck(over("house")),
            r"chain 'drafter': for each page in house runs over text 'house'; a chain runs over json or a list",
            id="text root",
        ),
        pytest.param(
            lambda: load_deck(over("titles", key="id") | {"inputs": {**DECK_DATA["inputs"], "titles": "list of text"}}),
            r"chain 'drafter': key 'id' needs json items, not text",
            id="key on text items",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", writes={"draft": "text"}))),
            r"chain 'drafter': 'noter' writes 'draft', which the chain writes; a value written in a chain has",
            id="chain write written outside",
        ),
        pytest.param(
            lambda: load_deck(
                deck(
                    **entries("agents", "checker", **{"for each": {"item": "page", "in": "plan.slides", "at once": 2}})
                )
            ),
            r"chain 'drafter': 'checker' sets at once, which only the head 'drafter' sets",
            id="option off the head",
        ),
        pytest.param(
            lambda: small(each={"a": "q in query"}, version=1),
            r"for each, retry, shared prompts, read maps, conditional edges and typed writes need version 2",
            id="each in version 1",
        ),
        pytest.param(
            lambda: MAS(
                [Agent("writer", "Write {page}", "draft", inputs=["page"])],
                inputs={"plan": "json"},
                each={"writer": "page in plan"},
                llm=MockLLM(),
            ),
            r"chain 'writer': it feeds the end, which needs declared outputs",
            id="chain into the end without outputs",
        ),
        pytest.param(
            lambda: load_deck(deck(outputs={**DECK_DATA["outputs"], "summary": "text"})),
            r"output 'summary' is neither an input nor written by any node",
            id="output not written",
        ),
        pytest.param(
            lambda: load_deck(deck(outputs={"deck": "json", "notes": "json"})),
            r"output 'notes' is declared json but is text",
            id="output type",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("steps", "render", retry={"money": 1}))),
            r"chain 'drafter': 'render' retries on 'money'; a node retries on \['outage', 'rate_limited', 'timeout'\]",
            id="retry money",
        ),
        pytest.param(
            lambda: MAS(
                [],
                boxes=[Box("b", chain_of("page in plan.slides"), {"plan": "plan"}, {"draft": "d"})],
                inputs={"plan": "json"},
                retry={"b": {"outage": 1}},
            ),
            r"box 'b' takes no retry; retry inside its system",
            id="retry on a box",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "checker", prompt="judges"))),
            r"Agent 'checker': prompt 'judges' names no entry of 'prompts'",
            id="prompt names no entry",
        ),
        pytest.param(
            lambda: load_deck().copy(
                prompts={"checker": "A {planned} {shown}", "figure_checker": "B {planned} {shown}"}
            ),
            r"agents \['checker', 'figure_checker'\] share the prompt 'judge' but hold different texts",
            id="shared texts differ",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", reads=["plan", "error"]))),
            r"'noter' reads 'error', which only a node an on error edge reaches can read",
            id="error read without an error edge",
        ),
        pytest.param(
            lambda: load_deck(deck(**entries("agents", "noter", writes={"run": "text"}))),
            r"'noter' writes 'run', which is a reserved name",
            id="a node writing run",
        ),
    ],
)
def test_check_names_every_chain_problem(build: Any, phrase: str) -> None:
    with pytest.raises(ValueError) as info:
        build()
    assert re.search(phrase, str(info.value)), str(info.value)


@pytest.mark.xfail(strict=True, reason="the engine renders a prompt from its read map (runtime._agent_body)")
def test_agent_read_maps_and_dicts() -> None:
    drafter = Agent("drafter", "Draft {slide} in {rules}", "draft", inputs={"slide": "page", "rules": "house"})
    assert (drafter.reads, drafter.inputs, drafter.renamed) == (
        {"slide": "page", "rules": "house"},
        ["page", "house"],
        True,
    )
    data = drafter.to_dict()
    assert data["inputs"] == {"slide": "page", "rules": "house"} and Agent.from_dict(data) == drafter
    assert drafter.copy(inputs=["page"]).reads == {"page": "page"}
    assert drafter.copy(inputs={"slide": "page"}).to_dict()["inputs"] == {"slide": "page"}
    assert Agent("a", "{v} {w}", "x", inputs={"v": "v", "w": "v"}).inputs == ["v"]  # values once each, in order
    assert Agent("a", "{query}", "x", inputs={"query": "query"}).to_dict()["inputs"] == ["query"]  # identity is a list
    with pytest.raises(ValueError, match="Unknown agent keys: \\['reads'\\]"):
        Agent.from_dict({**data, "reads": {"slide": "page"}})
    with pytest.raises(ValueError, match="'inputs' to be a list of strings or a map of names to names"):
        Agent.from_dict({**data, "inputs": {"slide": 3}})
    v1 = Agent("solver", "Solve {query}", "answer")
    assert list(v1.to_dict()) == ["name", "prompt", "inputs", "output", "tools", "model"]
    v1.inputs = ["query", "hint"]
    assert v1.reads == {"query": "query", "hint": "hint"} and not v1.renamed

    # A renamed read makes a system version 2; a placeholder outside the map is a problem.
    reader = Agent("reader", "Read {text}", "said", inputs={"text": "query"})
    with pytest.raises(ValueError, match="read maps"):
        MAS([reader], llm=MockLLM(), version=1)
    with pytest.raises(ValueError, match=r"prompt placeholders \['query'\] are not among its reads \['text'\]"):
        MAS([reader.copy(prompt="Read {query}")], llm=MockLLM())
    with pytest.raises(ValueError, match="takes the prompt 'reader', another agent's name"):
        MAS([reader, Agent("twin", "Read {query}", "heard")], llm=MockLLM(), prompt_names={"twin": "reader"})

    llm = MockLLM(lambda prompt: prompt.upper())
    trace = MAS([reader], llm=llm).run("cats")
    assert trace["reader"][0].inputs == {"text": "cats"} and llm.calls == ["Read cats"]  # keyed by placeholder
