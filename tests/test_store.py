"""The run store: LangGraph's saver interface, the swap with ``SqliteSaver``, run and question rows, the allowlist."""

from __future__ import annotations

import importlib.util
import logging
import operator
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.event_hooks import SerdeEvent, register_serde_event_listener
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from xmas.errors import ErrorKind
from xmas.health import Flag, Health
from xmas.media import Ref
from xmas.store import Store, open_store
from xmas.trace import AgentRecord, Event, Rejected, TokenUsage, ToolCall, Trace

# The tables as langgraph-checkpoint-sqlite 3.1.2 creates them (SqliteSaver.setup).
REFERENCE_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
"""
HAS_SQLITE_SAVER = importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None

INNER = Trace(query="q", output="o", records=(AgentRecord("inner", 0, "p", "p", {}, {"o": "o"}),), graph_id="g1")
RECORD = AgentRecord(
    "writer",
    0,
    "Write {query}",
    "Write q",
    {"query": "q"},
    {"draft": "d"},
    tool_calls=(ToolCall("search", {"q": [1, 2]}, "found"),),
    usage=TokenUsage(3, 4, cached_tokens=1),
    tokens=TokenUsage(3, 4),
    error_kind=ErrorKind.LENGTH_CUT,
    rejected=(Rejected("bad", "not json"),),
    route=("ask",),
    inner=INNER,
    latency_s=0.5,
)
PICTURE = Ref("/tmp/slide-1.png", "image/png", 3, "ab" * 32)
HEALTH = Health(flags=(Flag("stuck", "writer", 0, "same answer twice"),))
EVENT = Event("end", "writer", "r1", 0.1, status="ok")


@dataclass(frozen=True)
class Unlisted:
    """A class no allowlist names."""

    value: int


class State(TypedDict):
    """A run's records, other xmas values, and the answer."""

    records: Annotated[list[Any], operator.add]
    extras: dict[str, Any]
    answer: Any


def work(state: State) -> dict[str, Any]:
    return {"records": [RECORD], "extras": {"picture": PICTURE, "health": HEALTH, "event": EVENT}}


def ask(state: State) -> dict[str, Any]:
    return {"answer": interrupt({"question": "r1/ask/1"})}


def graph(saver: BaseCheckpointSaver[Any]) -> Any:
    builder = StateGraph(State)
    builder.add_node("work", work)
    builder.add_node("ask", ask)
    builder.add_edge(START, "work")
    builder.add_edge("work", "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=saver)


CONFIG: Any = {"configurable": {"thread_id": "r1"}}


def pause(saver: BaseCheckpointSaver[Any]) -> None:
    out = graph(saver).invoke({"records": [], "extras": {}, "answer": None}, CONFIG, durability="exit")
    assert [i.value for i in out["__interrupt__"]] == [{"question": "r1/ask/1"}]


def resume(saver: BaseCheckpointSaver[Any]) -> dict[str, Any]:
    compiled = graph(saver)
    assert compiled.get_state(CONFIG).next == ("ask",)
    out: dict[str, Any] = compiled.invoke(Command(resume={"approved": True}), CONFIG, durability="exit")
    return out


@pytest.fixture
def serde_events() -> Iterator[list[SerdeEvent]]:
    seen: list[SerdeEvent] = []
    unregister = register_serde_event_listener(seen.append)
    yield seen
    unregister()


def test_a_paused_run_resumes_from_the_file_in_a_new_store(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "runs.db"
    first = Store(path)
    pause(first)
    first.close()
    store = Store(str(path))
    assert repr(store) == f"Store({str(path)!r})"
    snapshot = graph(store).get_state(CONFIG)
    assert snapshot.values["records"] == [RECORD]
    assert snapshot.values["records"][0].tool_calls == RECORD.tool_calls
    assert snapshot.values["records"][0].inner.records == INNER.records
    assert snapshot.values["records"][0].latency_s == 0.5
    assert snapshot.values["extras"] == {"picture": PICTURE, "health": HEALTH, "event": EVENT}
    out = resume(store)
    assert out["answer"] == {"approved": True}
    assert out["records"] == [RECORD]
    history = list(store.list(CONFIG))
    assert len(history) == 2  # exit durability: one checkpoint at the pause, one at the end
    latest, paused = history
    assert store.get_tuple(CONFIG) == latest
    assert store.get_tuple(paused.config) == paused
    assert latest.parent_config == paused.config
    assert paused.pending_writes  # the interrupt and the finished node's writes
    assert list(store.list(CONFIG, before=latest.config)) == [paused]
    assert list(store.list(None, limit=1)) == [latest]
    assert list(store.list(paused.config)) == [paused]
    assert list(store.list(CONFIG, filter={"source": "loop", "step": latest.metadata["step"]})) == [latest]
    assert list(store.list(CONFIG, filter={"source": None})) == []
    assert list(store.list(CONFIG, filter={"source": ["loop"]})) == []
    with pytest.raises(ValueError, match="metadata filter key"):
        list(store.list(CONFIG, filter={"source') OR 1=1 --": "x"}))
    store.delete_thread("r1")
    assert store.get_tuple(CONFIG) is None


def test_a_store_without_a_path_lives_in_memory() -> None:
    store = Store()
    assert store.path is None
    assert repr(store) == "Store(in memory)"
    pause(store)
    assert resume(store)["answer"] == {"approved": True}
    assert store.get_tuple({"configurable": {"thread_id": "other"}}) is None


def test_channel_versions_count_up_in_the_sqlite_saver_format() -> None:
    store = Store()
    first = store.get_next_version(None, None)
    assert first.startswith(f"{1:032}.")
    assert store.get_next_version(first, None).startswith(f"{2:032}.")
    assert store.get_next_version(7, None).startswith(f"{8:032}.")


def table_info(path: Path | str, table: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(f"PRAGMA table_info({table})").fetchall()


def test_checkpoint_tables_match_the_sqlite_saver_schema(tmp_path: Path) -> None:
    Store(tmp_path / "xmas.db").close()
    with sqlite3.connect(tmp_path / "reference.db") as conn:
        conn.executescript(REFERENCE_DDL)
    for table in ("checkpoints", "writes"):
        assert table_info(tmp_path / "xmas.db", table) == table_info(tmp_path / "reference.db", table)
    with sqlite3.connect(tmp_path / "xmas.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


@pytest.mark.skipif(not HAS_SQLITE_SAVER, reason="langgraph-checkpoint-sqlite is not installed (not a dependency)")
def test_swaps_with_langgraph_sqlite_saver_both_ways(tmp_path: Path, serde_events: list[SerdeEvent]) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found,unused-ignore]

    def sqlite_saver(path: Path, store: Store) -> Any:
        return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=store.serde)

    # Paused under xmas, resumed under SqliteSaver.
    store = Store(tmp_path / "a.db")
    pause(store)
    out = resume(sqlite_saver(tmp_path / "a.db", store))
    assert out["answer"] == {"approved": True} and out["records"] == [RECORD]
    assert len(list(store.list(CONFIG))) == 2  # xmas reads what SqliteSaver wrote

    # Paused under SqliteSaver, resumed under xmas.
    pause(sqlite_saver(tmp_path / "b.db", store))
    other = Store(tmp_path / "b.db")
    out = resume(other)
    assert out["answer"] == {"approved": True} and out["records"] == [RECORD]
    for table in ("checkpoints", "writes"):
        assert table_info(tmp_path / "a.db", table) == table_info(tmp_path / "b.db", table)
    assert not [e for e in serde_events if e["kind"] == "msgpack_unregistered_allowed"]


def test_runs_and_questions_tables(tmp_path: Path) -> None:
    store = Store(tmp_path / "runs.db")
    assert [c[1] for c in table_info(tmp_path / "runs.db", "xmas_runs")] == [
        "run_id", "system_id", "status", "created_at", "updated_at", "paused", "totals",
    ]  # fmt: skip
    assert [c[1] for c in table_info(tmp_path / "runs.db", "xmas_questions")] == [
        "question_id", "run_id", "node", "form", "shown", "asked_at", "answered_at", "answer", "who", "deadline",
        "payload",
    ]  # fmt: skip

    started = store.save_run("r_1", "sys-a", "running")
    assert started.paused == () and started.created_at == started.updated_at
    paused = store.save_run("r_1", "sys-a", "paused", paused=["r_1/approve/1"])
    assert paused.created_at == started.created_at and paused.updated_at >= started.updated_at
    assert paused.paused == ("r_1/approve/1",)
    store.save_run("r2", "sys-b", "running")
    assert [r.run_id for r in store.runs()] == ["r_1", "r2"]
    assert store.runs("paused") == (paused,)
    assert store.run("nope") is None

    asked = store.save_question(
        "r_1/approve/1", "r_1", "approve", "approve", {"draft": "d"}, "2026-10-01T08:00:00+00:00"
    )
    assert asked.answer is None and asked.who is None and asked.answered_at is None
    store.save_question("r_1/edit/1", "r_1", "edit", "edit", {})
    with pytest.raises(ValueError, match="already saved"):
        store.save_question("r_1/edit/1", "r_1", "edit", "edit", {})
    answered = store.answer("r_1/approve/1", {"approved": True, "option": "Ship it"}, who="tom")
    assert answered.answer == {"approved": True, "option": "Ship it"} and answered.who == "tom"
    assert answered.answered_at is not None and answered.shown == {"draft": "d"}
    timed_out = store.answer("r_1/edit/1", {"changed": False}, answered_at="2026-10-25T08:00:00+00:00")
    assert timed_out.who is None and timed_out.answered_at == "2026-10-25T08:00:00+00:00"
    assert [q.question_id for q in store.questions("r_1")] == ["r_1/approve/1", "r_1/edit/1"]
    assert store.question("nope") is None
    with pytest.raises(KeyError, match="nope"):
        store.answer("nope", {})

    reopened = Store(tmp_path / "runs.db")
    assert reopened.run("r_1") == paused and reopened.questions("r_1") == store.questions("r_1")

    checkpoint: Any = {"id": "c1", "channel_values": {}}
    for thread in ("r_1", "r_1/each/drafter/1/0", "rx1/box/b/1", "r2"):
        store.put({"configurable": {"thread_id": thread, "checkpoint_ns": ""}}, checkpoint, {}, {})
    store.delete("r_1")
    assert store.run("r_1") is None and store.questions("r_1") == ()
    with sqlite3.connect(tmp_path / "runs.db") as conn:
        threads = {row[0] for row in conn.execute("SELECT thread_id FROM checkpoints")}
    assert threads == {"rx1/box/b/1", "r2"}  # '_' matched literally, other runs kept


def test_open_store_reads_xmas_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XMAS_RUNS", raising=False)
    assert open_store().path is None
    monkeypatch.setenv("XMAS_RUNS", "")
    assert open_store().path is None
    monkeypatch.setenv("XMAS_RUNS", str(tmp_path / "env.db"))
    assert open_store().path == tmp_path / "env.db"
    assert (tmp_path / "env.db").is_file()
    assert open_store(str(tmp_path / "given.db")).path == tmp_path / "given.db"


def test_xmas_types_load_without_the_unregistered_type_warning(
    caplog: pytest.LogCaptureFixture, serde_events: list[SerdeEvent]
) -> None:
    store = Store()
    values = {"record": RECORD, "picture": PICTURE, "health": HEALTH, "event": EVENT, "kind": ErrorKind.MONEY}
    with caplog.at_level(logging.DEBUG):
        assert store.serde.loads_typed(store.serde.dumps_typed(values)) == values
        pause(store)
        resume(store)
    assert serde_events == []
    assert not [r for r in caplog.records if "unregistered" in r.getMessage()]

    # LangGraph's default serializer reports the same record as unregistered.
    plain = JsonPlusSerializer()
    plain.loads_typed(plain.dumps_typed(RECORD))
    assert {(e["kind"], e["name"]) for e in serde_events} >= {("msgpack_unregistered_allowed", "AgentRecord")}

    # A class outside the allowlist is refused: it comes back as its plain fields.
    serde_events.clear()
    assert store.serde.loads_typed(store.serde.dumps_typed(Unlisted(3))) == {"value": 3}
    assert [(e["kind"], e["name"]) for e in serde_events] == [("msgpack_blocked", "Unlisted")]
