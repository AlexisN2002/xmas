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
from xmas.edges import Edge, EdgeKind, dump_edges, parse_edges
from xmas.nodes import Step
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
    assert mas.info is not None and mas.info.joins == {"publish": ("judge", "writer")}
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
    conditional = {
        **data,
        "edges": [*data["edges"], {"judge -> writer": {"when": "not verdict.fits", "max_visits": 3}}],
    }
    with pytest.raises(ValueError, match="a while edge arrives in phase 2b"):
        load(conditional)
    with pytest.raises(ValueError, match=r"\['boxes', 'frozen'\] arrive in phase 2b"):
        load({**data, "boxes": [], "frozen": ["judge"]})
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
    "v2 cycle": [r"a loop needs a while edge, phase 2b: the edge judge -> count closes a cycle"],
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
