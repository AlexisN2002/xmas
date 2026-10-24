"""The ``.mas`` bundle: save and load as a folder or a zip, prompt files, nested boxes, callables, the lock.

A bundle names models and tool servers and holds no address or key; loading it
checks the model names against the configured catalogue and refuses runs while
one is missing.
"""

from __future__ import annotations

import copy
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from promptolution.llms.base_llm import BaseLLM

from xmas import MAS, Agent
from xmas import bundle as mas_bundle
from xmas.bundle import LOCK, Lock, compare, dump_prompt, lock_of, parse_prompt, read
from xmas.config import Config, ModelNameWarning, ToolServer
from xmas.llms.api_llm import APILLM
from xmas.mas import ModelEntry, model_entry
from xmas.nodes import Step
from xmas.tools import Tool, ToolSource, tool
from xmas.tools.mcp import MCPTools

from .conftest import DECK, PLAN, MockLLM, assemble_deck, deck, deck_llms, render_page
from .test_spec import EXAMPLE, RemoteTools

NO_CATALOGUE = Config()
"""A configuration with no endpoint and no catalogue: model names are not checked, and nothing goes to the network."""

DECK_FILES = [
    "callables.yaml",
    "prompts/captioner.md",
    "prompts/drafter.md",
    "prompts/judge.md",
    "prompts/noter.md",
    "xmas.lock",
    "xmas.yaml",
]


def records(mas: MAS) -> list[tuple[str, str, str]]:
    trace = mas.run_state({"plan": PLAN, "house": "plain"})
    assert trace.error is None
    return sorted((r.agent, r.rendered_prompt, r.raw_response) for r in trace.records)


def mtimes(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def example_mas() -> MAS:
    return MAS.from_dict(
        yaml.safe_load(EXAMPLE),
        llms={"text-model": MockLLM(turns=[])},
        tools=[*_conftest_tools(), RemoteTools()],
    )


def _conftest_tools() -> list[Tool]:
    from .conftest import apology_text, count_words, publish_text

    return [count_words, publish_text, apology_text]


def test_the_deck_saves_and_loads_as_a_bundle(tmp_path: Path) -> None:
    made = deck()
    path = made.save(tmp_path / "deck.mas")
    assert path == made.bundle == tmp_path / "deck.mas"
    assert sorted(read(path)) == DECK_FILES
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    spec = made.to_dict()
    assert "prompts" not in wiring and "callables" not in wiring
    assert wiring == {key: value for key, value in spec.items() if key not in ("prompts", "callables")}
    assert yaml.safe_load((path / "callables.yaml").read_text(encoding="utf-8")) == spec["callables"]
    assert (path / "prompts/judge.md").read_text(encoding="utf-8") == spec["prompts"]["judge"] + "\n"  # no header

    loaded = MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)  # callables imported by their paths
    assert loaded.bundle == path and loaded.unresolved == {}
    assert loaded.to_dict() == spec
    assert loaded.describe()["id"] == made.describe()["id"]
    assert records(loaded) == records(deck())

    lock, before = (path / LOCK).read_bytes(), mtimes(path)
    assert loaded.save(path) == path
    assert (mtimes(path), (path / LOCK).read_bytes()) == (before, lock)  # nothing rewritten
    parsed = Lock.from_yaml(lock.decode("utf-8"))
    assert sorted(parsed.files) == [name for name in DECK_FILES if name != LOCK]
    assert sorted(parsed.code) == ["assemble_deck", "render_page"]
    assert lock.startswith(b"# written by MAS.save; do not edit\n")


def test_a_zipped_bundle_is_the_same_bundle(tmp_path: Path) -> None:
    first = deck().save(tmp_path / "a.mas", zipped=True)
    second = deck().save(tmp_path / "b.mas", zipped=True)
    assert first.read_bytes() == second.read_bytes()  # byte-reproducible
    folder = deck().save(tmp_path / "c.mas")
    assert read(first) == read(folder)
    loaded = MAS.load(first, llms=deck_llms(), config=NO_CATALOGUE)
    assert loaded.to_dict() == deck().to_dict()
    assert records(loaded) == records(deck())
    before = first.stat().st_mtime_ns
    loaded.save(first, zipped=True)
    assert first.stat().st_mtime_ns == before  # same bytes, not rewritten

    nested = tmp_path / "nested.mas"
    with zipfile.ZipFile(nested, "w") as archive:
        for name, data in read(folder).items():
            archive.writestr(f"deck.mas/{name}", data)
    assert read(nested) == read(folder)  # entries under one top folder
    with zipfile.ZipFile(first, "a") as archive:
        archive.writestr("examples/warmup.yaml", b"examples: []\n")
    deck().save(first, zipped=True)
    assert read(first)["examples/warmup.yaml"] == b"examples: []\n"  # carried over, and locked
    assert "examples/warmup.yaml" in Lock.from_yaml(read(first)[LOCK].decode("utf-8")).files

    with pytest.raises(ValueError, match=r"is a zipped bundle; save\(path, zipped=True\)"):
        deck().save(first)
    with pytest.raises(ValueError, match=r"is a bundle folder"):
        deck().save(folder, zipped=True)
    (tmp_path / "junk.mas").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match=r"neither a bundle folder nor a zipped bundle"):
        read(tmp_path / "junk.mas")
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "absent.mas")


def _zip_with(path: Path, name: str, data: bytes = b"x", mode: int = stat.S_IFREG | 0o644) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xmas.yaml", b"version: 1\n")
        info = zipfile.ZipInfo(name)
        info.external_attr = mode << 16
        archive.writestr(info, data)
    return path


@pytest.mark.parametrize(
    ("name", "mode", "message"),
    [
        ("../escape.md", stat.S_IFREG | 0o644, "points outside the bundle"),
        ("/etc/passwd", stat.S_IFREG | 0o644, "points outside the bundle"),
        ("prompts/link.md", stat.S_IFLNK | 0o777, "is a symlink"),
    ],
)
def test_a_zip_entry_that_leaves_the_bundle_is_refused(tmp_path: Path, name: str, mode: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read(_zip_with(tmp_path / "bad.mas", name, mode=mode))


def test_too_big_and_symlinked_bundles_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = deck().save(tmp_path / "deck.mas")
    zipped = deck().save(tmp_path / "deck.zip.mas", zipped=True)
    monkeypatch.setattr(mas_bundle, "MAX_BYTES", 100)
    for path in (folder, zipped):
        with pytest.raises(ValueError, match=r"holds more than 100 bytes"):
            read(path)
    monkeypatch.undo()
    (folder / "examples").mkdir()
    (folder / "examples" / "link.yaml").symlink_to(folder / "xmas.yaml")
    with pytest.raises(ValueError, match=r"examples/link.yaml is a symlink"):
        read(folder)


@pytest.mark.parametrize(
    ("text", "schema", "file"),
    [
        ("Judge {draft}", {"fits": {"type": "bool"}}, "---\nschema:\n  fits:\n    type: bool\n---\nJudge {draft}\n"),
        ("---\n---\nA rule line", None, "---\n---\n---\n---\nA rule line\n"),
        ("Answer {query}\n", None, "Answer {query}\n\n"),
        ("", None, "\n"),
        ("--- not a header", None, "--- not a header\n"),
    ],
)
def test_a_prompt_file_is_its_text_and_one_newline(text: str, schema: dict[str, Any] | None, file: str) -> None:
    assert dump_prompt(text, schema) == file
    assert parse_prompt(file, "prompts/p.md") == (text, schema)


def test_a_prompt_file_without_a_final_newline_is_read_as_is() -> None:
    assert parse_prompt("Answer {query}", "prompts/p.md") == ("Answer {query}", None)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("---\nreads: [x]\nschema: {a: {type: text}}\n---\nHi", r"holds only 'schema', got \['reads', 'schema'\]"),
        ("---\nschema: {a: 1}\nHi\n", "opened on line 1 is never closed by a '---' line"),
        ("---\nschema: [a]\n---\nHi\n", "'schema' maps answer fields to their forms"),
        ("---\n- a\n---\nHi\n", "holds only 'schema', got list"),
        ("---\nschema: {a\n---\nHi\n", "the front matter is not YAML"),
    ],
)
def test_bad_front_matter_is_refused_naming_the_file(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        parse_prompt(raw, "prompts/judge.md")
    assert str(info.value).startswith("prompts/judge.md: ")


def _shared_judges(fix: dict[str, Any]) -> MAS:
    spec = yaml.safe_load(DECK)
    spec["agents"][3]["schema"] = copy.deepcopy(spec["agents"][1]["schema"]) | fix
    return MAS.from_dict(spec, llms=dict(deck_llms()), tools=[render_page, assemble_deck])


def test_a_shared_schema_moves_into_the_prompt_file_only_when_every_sharer_agrees(tmp_path: Path) -> None:
    agreeing = _shared_judges({})
    path = agreeing.save(tmp_path / "agree.mas")
    text, schema = parse_prompt((path / "prompts/judge.md").read_text(encoding="utf-8"), "judge")
    assert schema == agreeing.agents["checker"].schema and text == agreeing.prompts()["checker"]
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    assert all("schema" not in agent for agent in wiring["agents"])
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).to_dict() == agreeing.to_dict()

    wiring["agents"][1]["schema"] = {"fits": {"type": "bool"}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Agent 'checker' has a schema in both prompts/judge.md and xmas.yaml"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)

    differing = _shared_judges({"fix": {"type": "text", "means": "another"}})
    path = differing.save(tmp_path / "differ.mas")
    assert parse_prompt((path / "prompts/judge.md").read_text(encoding="utf-8"), "judge")[1] is None
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).to_dict() == differing.to_dict()


def test_prompt_files_are_checked_against_the_agents(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    (path / "prompts/noter.md").write_text("Write notes for {plan} and {budget}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"prompts/noter.md: prompt placeholders \['budget'\] are not among"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "prompts/noter.md").write_text("Write notes for {plan}\n", encoding="utf-8")
    (path / "prompts/ghost.md").write_text("Boo\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"prompts/ghost.md belongs to no agent"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "prompts/ghost.md").unlink()
    (path / "prompts/noter.md").unlink()
    with pytest.raises(ValueError, match=r"Agent 'noter' has no prompt file prompts/noter.md"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    for text, message in [
        ("version: 2\nprompts: {noter: x}\n", "xmas.yaml holds prompts"),
        ("- a list\n", "xmas.yaml must hold a mapping"),
        ("version: [2\n", "xmas.yaml is not YAML"),
    ]:
        (path / "xmas.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "xmas.yaml").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match=r"xmas.yaml is not UTF-8 text"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    (path / "xmas.yaml").unlink()
    with pytest.raises(ValueError, match=r"The bundle has no xmas.yaml"):
        mas_bundle.spec_of(read(path))


class Offered(ToolSource):
    """A tool source with no name that offers ``web_fetch``."""

    def tools(self) -> list[Tool]:
        """Offer the remote tool."""
        return RemoteTools().tools()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Answer with a page."""
        return "page"


def test_boxes_nest_and_mcp_names_its_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = example_mas()
    path = made.save(tmp_path / "example.mas")
    names = sorted(read(path))
    assert {"boxes/shorten.mas/xmas.yaml", "boxes/shorten.mas/prompts/cutter.md"} <= set(names)
    assert not any(name.startswith("boxes/") and name.endswith(LOCK) for name in names)
    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    assert wiring["boxes"][0]["system"] == "boxes/shorten.mas"
    callables = yaml.safe_load((path / "callables.yaml").read_text(encoding="utf-8"))
    assert callables["web_fetch"] == {
        "kind": "mcp",
        "server": "tools",
        "reads": {"url": "text"},
        "writes": {"web_fetch": "text"},
    }
    llms: dict[str, BaseLLM] = {"text-model": MockLLM(turns=[])}
    loaded = MAS.load(path, llms=llms, tools=[RemoteTools()], config=NO_CATALOGUE)
    assert loaded.to_dict() == made.to_dict() and loaded.describe()["id"] == made.describe()["id"]

    made_servers: list[tuple[str, str | None, str | None, dict[str, Any]]] = []

    class Served(Offered):
        """A stand-in for MCPTools that records how it was made."""

        def __init__(self, url: str, key: str | None = None, *, writes: Any = None, name: str | None = None) -> None:
            """Record the address, key, name and writes."""
            self.name = name
            made_servers.append((url, key, name, dict(writes)))

    monkeypatch.setattr("xmas.tools.mcp.MCPTools", Served)
    served = Config(tools={"tools": ToolServer("https://tools.example.com/mcp", "tool-key")})
    assert MAS.load(path, llms=llms, config=served).to_dict() == made.to_dict()
    assert made_servers == [
        ("https://tools.example.com/mcp", "tool-key", "tools", {"web_fetch": {"web_fetch": "text"}})
    ]
    with pytest.raises(ValueError, match=r"add \[tools.tools\] url to .*, or pass tools=\[...\] offering them"):
        MAS.load(path, llms=llms, config=NO_CATALOGUE)
    del callables["web_fetch"]["server"]
    (path / "callables.yaml").write_text(yaml.safe_dump(callables), encoding="utf-8")
    with pytest.raises(ValueError, match=r"remote tool with no server: pass tools=.*configure \[tools.<name>\] url"):
        MAS.load(path, llms=llms, config=NO_CATALOGUE)

    wiring["boxes"][0]["system"] = "../shorten.mas"
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring), encoding="utf-8")
    with pytest.raises(ValueError, match=r"'../shorten.mas' is outside the bundle"):
        MAS.load(path, llms=llms, tools=[RemoteTools()], config=NO_CATALOGUE)
    spec = made.to_dict()
    spec["boxes"][0]["system"] = "boxes/shorten.mas"
    with pytest.raises(ValueError, match=r"names its system 'boxes/shorten.mas': MAS.load reads a bundle's boxes"):
        MAS.from_dict(spec, llms=llms, tools=[*_conftest_tools(), RemoteTools()])
    spec = made.to_dict()
    spec["callables"]["web_fetch"]["server"] = "https://tools.example.com/mcp"
    with pytest.raises(ValueError, match=r"its address goes to the configuration"):
        MAS.from_dict(spec, llms=llms, tools=[*_conftest_tools(), RemoteTools()])

    plain = MAS.from_dict(yaml.safe_load(EXAMPLE) | {"callables": {}}, llms=llms, tools=[*_conftest_tools(), Offered()])
    assert plain.to_dict()["callables"]["web_fetch"] == {
        "kind": "mcp",
        "reads": {"url": "text"},
        "writes": {"web_fetch": "text"},
    }
    named = MCPTools("https://tools.example.com/mcp", "k", name="web")
    assert (named.name, repr(named)) == ("web", "MCPTools('https://tools.example.com/mcp', name='web')")
    with pytest.raises(ValueError, match=r"is a name such as 'web'"):
        MCPTools("https://tools.example.com/mcp", "k", name="https://tools.example.com")


class OfflineMCP(MCPTools):
    """MCP tools that list one tool without connecting."""

    def tools(self) -> list[Tool]:
        """Offer one remote tool."""
        schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        return [Tool("web_fetch", "Fetch a page.", schema)]


def test_no_url_or_key_in_any_bundle_file(tmp_path: Path) -> None:
    remote = OfflineMCP("https://tools.example.com/mcp", "sk-test", name="tools")
    gateway = APILLM(model="text-model", base_url="https://gateway.example/v1", api_key="sk-test")
    fetcher = Agent("fetcher", prompt="Read {query}", output="page", tools=["web_fetch"], model="text-model")
    online = MAS([fetcher], llms={"text-model": gateway}, tools=[remote], version=2)
    saved = [
        deck().save(tmp_path / "deck.mas"),
        example_mas().save(tmp_path / "example.mas"),
        online.save(tmp_path / "online.mas"),
    ]
    zipped = online.save(tmp_path / "online.zip.mas", zipped=True)
    assert online.to_dict()["callables"]["web_fetch"]["server"] == "tools"
    for path in [*saved, zipped]:
        for name, data in read(path).items():
            for word in (b"http", b"://", b"api_key", b"sk-test"):
                assert word not in data, (path.name, name, word)


def test_version_1_and_the_one_agent_case(tmp_path: Path) -> None:
    def solver_llm() -> MockLLM:
        return MockLLM(lambda prompt: f"solved {len(prompt)}")

    made = MAS([Agent("solver", prompt="Solve {query}\nShow the working.", output="prediction")], llm=solver_llm())
    path = made.save(tmp_path / "solver.mas")
    assert sorted(read(path)) == ["prompts/solver.md", "xmas.lock", "xmas.yaml"]
    loaded = MAS.load(path, llm=solver_llm(), config=NO_CATALOGUE)
    assert (loaded.version, loaded.to_dict(), loaded.encode()) == (1, made.to_dict(), made.encode())
    assert loaded.predict(made.encode(), ["a", "b"]) == made.predict(made.encode(), ["a", "b"])
    assert loaded.run("q").records == made.run("q").records

    with pytest.raises(ValueError, match=r"named name.mas, not 'solver.yaml'; to_yaml\(path\) writes"):
        made.save(tmp_path / "solver.yaml")
    made.to_yaml(tmp_path / "solver.yaml")
    assert MAS.from_yaml(tmp_path / "solver.yaml", llm=solver_llm()).to_dict() == made.to_dict()
    with pytest.raises(ValueError, match=r"\.mas bundle \(name\.mas/\): save\(path\) writes it"):
        deck().to_yaml(tmp_path / "deck.yaml")

    MAS(list(made.agents.values()), llm=solver_llm(), default_model="solver-model").save(path)
    by_name = MAS.load(path, llms={"solver-model": solver_llm()}, config=NO_CATALOGUE)
    assert by_name.run("q").output == made.run("q").output

    (path / "prompts/solver.md").write_text("---\nschema: {a: {type: text}}\n---\nSolve {query}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a version 1 agent has no answer form"):
        MAS.load(path, llm=solver_llm(), config=NO_CATALOGUE)


def test_save_rewrites_only_what_changed(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    callables = (path / "callables.yaml").read_text(encoding="utf-8")
    (path / "callables.yaml").write_text("# the deck's code\n" + callables, encoding="utf-8")
    (path / "examples").mkdir()
    (path / "examples/warmup.yaml").write_text("version: 1\nexamples: []\n", encoding="utf-8")
    (path / "optimize.yaml").write_text("tune: [prompts]  # only the prompts\n", encoding="utf-8")
    deck().save(path)  # the lock takes the new files in
    before = mtimes(path)
    assert sorted(Lock.from_yaml((path / LOCK).read_text(encoding="utf-8")).files) == sorted(set(before) - {LOCK})

    loaded = MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)
    loaded.copy(prompts={"noter": "Write short notes for {plan}"}).save(path)
    after = mtimes(path)
    assert sorted(name for name in after if after[name] != before[name]) == ["prompts/noter.md", "xmas.lock"]
    assert (path / "prompts/noter.md").read_text(encoding="utf-8") == "Write short notes for {plan}\n"
    assert (path / "callables.yaml").read_text(encoding="utf-8").startswith("# the deck's code\n")
    assert (path / "optimize.yaml").read_text(encoding="utf-8") == "tune: [prompts]  # only the prompts\n"

    spec = yaml.safe_load(DECK)
    spec["agents"] = [agent for agent in spec["agents"] if agent["name"] != "noter"]
    del spec["prompts"]["noter"], spec["outputs"]["notes"]
    MAS.from_dict(spec, llms=dict(deck_llms()), tools=[render_page, assemble_deck]).save(path)
    assert not (path / "prompts/noter.md").exists() and (path / "examples/warmup.yaml").exists()

    (path / "callables.yaml").write_text("render_page: [unclosed\n", encoding="utf-8")
    example_mas().save(path)  # another system: the deck's files go, a box comes
    assert {"boxes/shorten.mas/xmas.yaml", "callables.yaml"} <= set(read(path))
    assert not (path / "prompts/drafter.md").exists()
    deck().save(path)
    assert not (path / "boxes").exists() and "render_page:" in (path / "callables.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"Prompt name '\.\./evil' cannot be a file name in a bundle"):
        MAS([Agent("noter", "Notes on {query}", "notes")], llm=MockLLM(), prompt_names={"noter": "../evil"}).save(path)

    foreign = tmp_path / "foreign.mas"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match=r"holds no xmas.yaml; a save writes only into a bundle"):
        deck().save(foreign)

    (path / "optimize.yaml").write_text("tune: [weights]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"optimize.yaml tune is a list of \['prompts', 'models', 'edges'\]"):
        MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE)


@pytest.mark.parametrize(
    ("optimize", "message"),
    [
        ("seed: 3\n", r"unknown keys \['seed'\]"),
        ("task: {examples: warmup, import: 'a:b'}\n", "task is {examples: <file>} or {import"),
        ("optimizer: {steps: 3}\n", "optimizer is a mapping with a 'name'"),
    ],
)
def test_optimize_yaml_is_checked(optimize: str, message: str) -> None:
    files = {"optimize.yaml": optimize.encode()}
    with pytest.raises(ValueError, match=message):
        mas_bundle.read_optimize(files)
    good = {"task": {"examples": "warmup"}, "tune": ["prompts", "models"], "optimizer": {"name": "capo"}}
    assert mas_bundle.read_optimize({"optimize.yaml": yaml.safe_dump(good).encode()}) == good
    assert mas_bundle.read_optimize({}) == mas_bundle.read_optimize({"optimize.yaml": b""}) == {}


STEPS_MODULE = '''
from xmas.tools import tool


@tool
def shout(text: str) -> str:
    """Shout a text."""
    return text.upper()


def pad(text: str, width: int) -> str:
    """Pad a text to a width with dots."""
    return text.ljust(width, ".")
'''

STEPS_BUNDLE = {
    "xmas.yaml": """\
version: 2
outputs: {padded: text}
steps:
- {name: loud, uses: shout, reads: {text: query}}
- {name: wide, uses: pad, reads: {text: shout}}
edges: [loud -> wide]
""",
    "callables.yaml": """\
shout: {kind: python, import: 'stepsmod:shout', reads: {text: text}, writes: {shout: text}}
pad:   # padded to ten
  kind: python
  import: 'stepsmod:pad'
  params: {width: 10}
  reads: {text: text}
  writes: {padded: text}
""",
}


def _steps_bundle(root: Path, monkeypatch: pytest.MonkeyPatch, callables: str | None = None) -> Path:
    (root / "stepsmod.py").write_text(STEPS_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    path = root / "steps.mas"
    path.mkdir(exist_ok=True)
    (path / "xmas.yaml").write_text(STEPS_BUNDLE["xmas.yaml"], encoding="utf-8")
    (path / "callables.yaml").write_text(callables or STEPS_BUNDLE["callables.yaml"], encoding="utf-8")
    return path


def test_callables_load_by_import_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _steps_bundle(tmp_path, monkeypatch)
    mas = MAS.load(path, config=NO_CATALOGUE)
    assert mas.run_state({"query": "hi"}).outputs == {"padded": "HI........"}
    item, _ = mas.resolve_callable(mas.steps["wide"])
    assert (item.params, sorted(item.reads), item(text="a")) == ({"width": 10}, ["text"], "a.........")
    assert mas.to_dict()["callables"]["pad"]["params"] == {"width": 10}
    assert mas.to_dict()["callables"]["pad"]["import"] == "stepsmod:pad"

    @tool(name="shout")
    def quiet(text: str) -> str:
        """Say a text quietly."""
        return text.lower()

    supplied = MAS.load(path, tools=[quiet], config=NO_CATALOGUE)
    assert supplied.run_state({"query": "Hi"}).outputs == {"padded": "hi........"}

    mismatch = STEPS_BUNDLE["callables.yaml"].replace(
        "reads: {text: text}, writes: {shout", "reads: {text: json}, writes: {shout"
    )
    _steps_bundle(tmp_path, monkeypatch, mismatch)
    with pytest.raises(ValueError, match=r"callable 'shout' reads {'text': 'json'} in the spec"):
        MAS.load(path, config=NO_CATALOGUE)
    for callables, message in [
        (
            STEPS_BUNDLE["callables.yaml"] + "  note: wraps the old pad\n",
            "unknown keys \\['note'\\].*use a YAML comment",
        ),
        (
            STEPS_BUNDLE["callables.yaml"].replace("stepsmod:shout", "stepsmod:yell"),
            "'stepsmod:yell' cannot be imported",
        ),
        (
            STEPS_BUNDLE["callables.yaml"].replace("'stepsmod:shout'", "stepsmod"),
            "needs import: 'package.module:function'",
        ),
        (STEPS_BUNDLE["callables.yaml"].replace("{width: 10}", "{height: 10}"), r"has no parameters \['height'\]"),
        (
            STEPS_BUNDLE["callables.yaml"].replace(
                "kind: python\n  import: 'stepsmod:pad'", "kind: mcp\n  server: web"
            ),
            "binds params, which only a python callable takes",
        ),
    ]:
        _steps_bundle(tmp_path, monkeypatch, callables)
        with pytest.raises(ValueError, match=message):
            MAS.load(path, config=NO_CATALOGUE)
    (path / "xmas.yaml").write_text(STEPS_BUNDLE["xmas.yaml"] + "callables: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"The callables are in both callables.yaml and xmas.yaml"):
        MAS.load(path, config=NO_CATALOGUE)


def test_a_bound_tool_keeps_its_params_out_of_the_model_view() -> None:
    @tool
    def scale(value: int, factor: int) -> int:
        """Scale a value."""
        return value * factor

    bound = scale.bind(factor=3)
    assert (bound.parameters["required"], sorted(bound.reads), bound(value=2)) == (["value"], ["value"], 6)
    assert bound.validate({"value": "4"}) == {"value": 4, "factor": 3}
    foreign = Tool("look", "Look.", {"type": "object", "properties": {"a": {}, "b": {}}, "additionalProperties": False})
    assert foreign.bind(b=1).validate({"a": 2}) == {"a": 2, "b": 1}
    assert scale.renamed("grow").name == "grow"


def test_lock_compare_and_code_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = deck().save(tmp_path / "deck.mas")
    lock = Lock.from_yaml((path / LOCK).read_text(encoding="utf-8"))
    assert compare(lock, read(path), mas_bundle.code_hashes(deck())).clean
    (path / "prompts/noter.md").write_text("Other notes for {plan}\n", encoding="utf-8")
    (path / "examples").mkdir()
    (path / "examples/warmup.yaml").write_text("examples: []\n", encoding="utf-8")
    (path / "prompts/captioner.md").unlink()
    diff = compare(lock, read(path))
    assert (diff.changed, diff.added, diff.missing, diff.code_changed) == (
        ("prompts/noter.md",),
        ("examples/warmup.yaml",),
        ("prompts/captioner.md",),
        (),
    )
    assert not diff.clean
    assert lock_of(read(path), {}).bundle != lock.bundle
    for text in ("version: 2\n", "[unclosed\n", "version: 1\nbundle: x\nfiles: []\n"):
        with pytest.raises(ValueError, match=r"is not a lock written by MAS.save"):
            Lock.from_yaml(text)

    steps = _steps_bundle(tmp_path, monkeypatch)
    mas = MAS.load(steps, config=NO_CATALOGUE)
    mas.save(steps)
    saved = Lock.from_yaml((steps / LOCK).read_text(encoding="utf-8"))
    assert sorted(saved.code) == ["pad", "shout"] and all(saved.code.values())
    module = tmp_path / "stepsmod.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace("text.upper()", "text.upper() + '!'"), encoding="utf-8"
    )
    import importlib
    import sys

    importlib.reload(sys.modules["stepsmod"])
    changed = MAS.load(steps, config=NO_CATALOGUE)
    assert compare(saved, read(steps), mas_bundle.code_hashes(changed)).code_changed == ("shout",)

    namespace: dict[str, Any] = {}
    exec("def echo(query):\n    return query\n", namespace)  # no source file to read
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    echo = Tool("echo", "Echo.", schema, fn=namespace["echo"], writes={"echo": "text"})
    unread = MAS([], steps=[Step("e", "echo")], tools=[echo], llm=MockLLM())
    assert mas_bundle.code_hashes(unread) == {"echo": None}
    assert compare(Lock("x", {}, {"echo": "old"}), {}, {"echo": None}).code_changed == ()  # no source: skipped


def test_model_names_warn_and_refuse_runs(tmp_path: Path) -> None:
    path = deck().save(tmp_path / "deck.mas")
    near = Config(catalogue=("writer-model", "judge-models"))
    llms = deck_llms()
    with pytest.warns(
        ModelNameWarning, match=r"model 'judge-model' \(agents checker, figure_checker\) is not in"
    ) as caught:
        mas = MAS.load(path, llms=llms, config=near)
    assert [(w.message.name, w.message.suggestions) for w in caught if isinstance(w.message, ModelNameWarning)] == [
        ("judge-model", ("judge-models", "writer-model"))
    ]
    assert mas.unresolved == {"judge-model": ("judge-models", "writer-model")}
    for call in (
        lambda: mas.run_state({"plan": PLAN, "house": "plain"}),
        lambda: mas.run("q"),
        lambda: mas.run_batch(["q"]),
        lambda: mas.predict("p", ["q"]),
        lambda: mas("q"),
    ):
        with pytest.raises(
            ValueError, match=r"'judge-model' \(closest: judge-models, writer-model\)\. Set the name in models:"
        ):
            call()
    assert all(not model.calls for model in llms.values())
    assert MAS.load(path, llms=deck_llms(), config=near, resolve=False).unresolved == {}
    assert MAS.load(path, llms=deck_llms(), config=NO_CATALOGUE).unresolved == {}

    wiring = yaml.safe_load((path / "xmas.yaml").read_text(encoding="utf-8"))
    wiring["models"] = {"judge-model": {"model": "judge-models", "accepts": []}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    entry = MAS.load(path, llms=deck_llms(), config=near)
    assert entry.unresolved == {} and entry.models == {"judge-model": ModelEntry(model="judge-models")}
    assert entry.to_dict()["models"] == {"judge-model": {"model": "judge-models"}}
    assert records(entry) == records(deck())

    rebound = MAS.load(path, llms=deck_llms(), config=near, overlay={"noter": "judge-model"})
    assert rebound.agents["noter"].model == "judge-model"
    with pytest.raises(ValueError, match=r"The overlay names unknown agents \['ghost'\]"):
        MAS.load(path, llms=deck_llms(), config=near, overlay={"ghost": "judge-model"})
    with pytest.raises(ValueError, match=r"No model object for \['judge-model'\]: pass llms="):
        MAS.load(path, llms={"writer-model": MockLLM()}, config=near)
    for bad in ("", "https://gateway.example/v1/kimi", 3):
        with pytest.raises(ValueError, match=r"model is a catalogue id such as 'kimi-k3'"):
            model_entry({"model": bad}, "models entry 'x'")
    wiring["models"] = {"judge-model": {"model": ["judge-models"]}}
    (path / "xmas.yaml").write_text(yaml.safe_dump(wiring, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"needs 'model' to be a catalogue id"):
        MAS.load(path, llms=deck_llms(), config=near)
