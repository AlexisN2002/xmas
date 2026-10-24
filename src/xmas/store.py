"""Durable runs: LangGraph checkpoints plus xmas's run and question rows in one SQLite database.

:class:`Store` is a LangGraph checkpoint saver on the standard library's
``sqlite3``. Its ``checkpoints`` and ``writes`` tables are the ones
``langgraph-checkpoint-sqlite``'s ``SqliteSaver`` creates, column for column,
filled through the same serializer calls, so either saver opens a file the
other wrote. Beside them, ``xmas_runs`` holds one row per run and
``xmas_questions`` one row per question a run put to a person.

Where the store lives is configuration, never a spec key: :func:`open_store`
reads the ``XMAS_RUNS`` environment variable, and a store without a path lives
in memory, so its pauses resume in the same process only.

Reading a checkpoint back builds LangGraph's safe types and :data:`XMAS_TYPES`
only, xmas classes with their tuples restored; any other class in a checkpoint
comes back as its plain fields. Hand :attr:`Store.serde` to another saver to
read xmas values the same way.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from xmas.errors import ErrorKind
from xmas.health import Flag, Health
from xmas.media import Ref
from xmas.trace import AgentRecord, Event, Rejected, TokenUsage, ToolCall, Trace

XMAS_TYPES: tuple[type, ...] = (AgentRecord, TokenUsage, ToolCall, Rejected, Trace, Event, ErrorKind, Health, Flag, Ref)
"""The xmas classes a checkpoint may hold besides LangGraph's own safe types."""

_SCHEMA = """
PRAGMA journal_mode=WAL;
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
CREATE TABLE IF NOT EXISTS xmas_runs (
    run_id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    paused TEXT NOT NULL DEFAULT '[]',
    totals TEXT
);
CREATE TABLE IF NOT EXISTS xmas_questions (
    question_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node TEXT NOT NULL,
    form TEXT NOT NULL,
    shown TEXT NOT NULL DEFAULT '{}',
    asked_at TEXT NOT NULL,
    answered_at TEXT,
    answer TEXT,
    who TEXT,
    deadline TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS xmas_questions_run ON xmas_questions (run_id);
"""
_CHECKPOINT_COLUMNS = "thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata"
_FILTER_KEY = re.compile(r"^[a-zA-Z0-9_.-]+$")


class _Serializer(JsonPlusSerializer):
    """LangGraph's serializer, giving xmas classes back their tuples, which msgpack stores as lists."""

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Decode as LangGraph does, then restore the tuple fields of xmas classes."""
        return _tuples(super().loads_typed(data))


def _tuples(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _tuples(item)
    elif isinstance(value, list):
        value[:] = [_tuples(item) for item in value]
    elif isinstance(value, XMAS_TYPES) and is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in fields(value):
            item = getattr(value, field.name)
            fixed = _tuples(item)
            if isinstance(fixed, list) and str(field.type).startswith("tuple["):
                fixed = tuple(fixed)
            if fixed is not item:
                changes[field.name] = fixed
        return replace(value, **changes) if changes else value
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunRow:
    """One run as the store knows it.

    Attributes:
        run_id: The run's id, also its checkpoint thread.
        system_id: The ``describe()`` id of the system that ran it.
        status: A run status, such as ``running``, ``paused`` or ``stopped``.
        created_at: When the row was first saved, ISO UTC.
        updated_at: When it was last saved, ISO UTC.
        paused: The ids of the questions the run waits on.
        totals: What the run had used when it was last saved, as
            :meth:`~xmas.budget.RunMeter.totals` gives it, so a resumed run's
            budget counts on; ``None`` until saved.
    """

    run_id: str
    system_id: str
    status: str
    created_at: str
    updated_at: str
    paused: tuple[str, ...] = ()
    totals: dict[str, Any] | None = None


@dataclass(frozen=True)
class QuestionRow:
    """One question a run put to a person, and its answer once given.

    Attributes:
        question_id: The question's id.
        run_id: The run that asked it.
        node: The human node that asked it.
        form: The node's form, such as ``approve``.
        shown: What the person was shown, by name.
        asked_at: When it was asked, ISO UTC.
        answered_at: When it was answered, ISO UTC; ``None`` while open.
        answer: The answer; ``None`` while open.
        who: Who answered; ``None`` while open or for a default taken on timeout.
        deadline: When the question's default is taken, ISO UTC; ``None`` without a timeout.
        payload: The whole question as :meth:`~xmas.human.Question.to_dict` gives it.
    """

    question_id: str
    run_id: str
    node: str
    form: str
    shown: dict[str, str]
    asked_at: str
    answered_at: str | None = None
    answer: dict[str, Any] | None = None
    who: str | None = None
    deadline: str | None = None
    payload: dict[str, Any] | None = None


class Store(BaseCheckpointSaver[str]):
    """A checkpoint saver and the run and question rows beside it, in one SQLite database.

    One connection serves every thread, each statement under one lock. Async
    methods raise, as ``SqliteSaver``'s do.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, serde: SerializerProtocol | None = None) -> None:
        """Open the database file at ``path``, created with its folder if missing, or one in memory when ``None``."""
        super().__init__(serde=serde or _Serializer(allowed_msgpack_modules=XMAS_TYPES))
        self.path = Path(path).expanduser() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path or ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:
        """Name the file, or say it is in memory."""
        return f"Store({str(self.path)!r})" if self.path else "Store(in memory)"

    # LangGraph's saver interface

    def _tuple(self, row: Sequence[Any]) -> CheckpointTuple:
        thread_id, checkpoint_ns, checkpoint_id, parent_id, type_, checkpoint, metadata = row
        writes = self._execute(
            "SELECT task_id, channel, type, value, task_path, idx FROM writes"
            " WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
            (thread_id, checkpoint_ns, checkpoint_id),
        )
        writes.sort(key=lambda w: writes_sort_key(w[4], w[0], w[5]))

        def config(checkpoint_id: str) -> RunnableConfig:
            ids = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}
            return {"configurable": ids}

        return CheckpointTuple(
            config(checkpoint_id),
            self.serde.loads_typed((type_, checkpoint)),
            cast(CheckpointMetadata, json.loads(metadata) if metadata is not None else {}),
            config(parent_id) if parent_id else None,
            [(task, channel, self.serde.loads_typed((t, value))) for task, channel, t, value, _, _ in writes],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the checkpoint ``config`` names, else its thread's latest."""
        ids = config["configurable"]
        key = (str(ids["thread_id"]), ids.get("checkpoint_ns", ""))
        if checkpoint_id := get_checkpoint_id(config):
            rows = self._execute(
                f"SELECT {_CHECKPOINT_COLUMNS} FROM checkpoints"
                " WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                (*key, checkpoint_id),
            )
        else:
            rows = self._execute(
                f"SELECT {_CHECKPOINT_COLUMNS} FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"
                " ORDER BY checkpoint_id DESC LIMIT 1",
                key,
            )
        return self._tuple(rows[0]) if rows else None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield the matching checkpoints, newest first; ``filter`` matches metadata values."""
        wheres: list[str] = []
        params: list[Any] = []
        if config is not None:
            ids = config["configurable"]
            wheres.append("thread_id = ?")
            params.append(str(ids["thread_id"]))
            if ids.get("checkpoint_ns") is not None:
                wheres.append("checkpoint_ns = ?")
                params.append(ids["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = ?")
                params.append(checkpoint_id)
        for key, value in (filter or {}).items():
            if not _FILTER_KEY.match(key):
                raise ValueError(f"A metadata filter key is letters, digits, '_', '.' and '-', not {key!r}")
            if value is None:
                wheres.append(f"json_extract(CAST(metadata AS TEXT), '$.{key}') IS NULL")
                continue
            wheres.append(f"json_extract(CAST(metadata AS TEXT), '$.{key}') = ?")
            plain = isinstance(value, str | int | float)
            params.append(value if plain else json.dumps(value, separators=(",", ":")))
        if before is not None:
            wheres.append("checkpoint_id < ?")
            params.append(get_checkpoint_id(before))
        sql = f"SELECT {_CHECKPOINT_COLUMNS} FROM checkpoints"
        sql += (" WHERE " + " AND ".join(wheres) if wheres else "") + " ORDER BY checkpoint_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self._execute(sql, params):
            yield self._tuple(row)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,  # noqa: ARG002
    ) -> RunnableConfig:
        """Save ``checkpoint`` as the child of the one ``config`` names."""
        ids = config["configurable"]
        type_, blob = self.serde.dumps_typed(checkpoint)
        meta = json.dumps(get_checkpoint_metadata(config, metadata), ensure_ascii=False).encode("utf-8", "ignore")
        self._execute(
            f"INSERT OR REPLACE INTO checkpoints ({_CHECKPOINT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(ids["thread_id"]),
                ids["checkpoint_ns"],
                checkpoint["id"],
                ids.get("checkpoint_id"),
                type_,
                blob,
                meta,
            ),
        )
        return {
            "configurable": {
                "thread_id": ids["thread_id"],
                "checkpoint_ns": ids["checkpoint_ns"],
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        """Save a task's pending writes against the checkpoint ``config`` names."""
        ids = config["configurable"]
        verb = "REPLACE" if all(channel in WRITES_IDX_MAP for channel, _ in writes) else "IGNORE"
        key = (str(ids["thread_id"]), str(ids["checkpoint_ns"]), str(ids["checkpoint_id"]), task_id, task_path)
        rows = [
            (*key, WRITES_IDX_MAP.get(channel, idx), channel, *self.serde.dumps_typed(value))
            for idx, (channel, value) in enumerate(writes)
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                f"INSERT OR {verb} INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx,"
                " channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes."""
        with self._lock, self._conn:
            for table in ("checkpoints", "writes"):
                self._conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (str(thread_id),))

    def get_next_version(self, current: str | int | None, channel: None) -> str:  # noqa: ARG002
        """Return the next channel version, in ``SqliteSaver``'s format: a padded count, a dot, a random tail."""
        count = 0 if current is None else current if isinstance(current, int) else int(current.split(".")[0])
        return f"{count + 1:032}.{random.random():016}"

    # Runs and questions

    def save_run(
        self,
        run_id: str,
        system_id: str,
        status: str,
        paused: Sequence[str] = (),
        totals: Mapping[str, Any] | None = None,
    ) -> RunRow:
        """Create or update a run's row; ``created_at`` stays as first saved, ``totals`` as last given."""
        now = _now()
        self._execute(
            "INSERT INTO xmas_runs VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET"
            " system_id = excluded.system_id, status = excluded.status, updated_at = excluded.updated_at,"
            " paused = excluded.paused, totals = COALESCE(excluded.totals, totals)",
            (run_id, system_id, status, now, now, json.dumps(list(paused)), _json(totals)),
        )
        return cast(RunRow, self.run(run_id))

    def run(self, run_id: str) -> RunRow | None:
        """Return a run's row, or ``None`` for an unknown id."""
        rows = self._execute("SELECT * FROM xmas_runs WHERE run_id = ?", (run_id,))
        return _run_row(rows[0]) if rows else None

    def runs(self, status: str | None = None) -> tuple[RunRow, ...]:
        """Return every run's row, or those with ``status``, oldest first."""
        where, params = (" WHERE status = ?", (status,)) if status is not None else ("", ())
        return tuple(_run_row(r) for r in self._execute(f"SELECT * FROM xmas_runs{where} ORDER BY created_at", params))

    def save_question(
        self,
        question_id: str,
        run_id: str,
        node: str,
        form: str,
        shown: Mapping[str, str],
        asked_at: str | None = None,
        *,
        deadline: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> QuestionRow:
        """Record a question as asked, ``asked_at`` now unless given; an id already saved is refused."""
        try:
            self._execute(
                "INSERT INTO xmas_questions (question_id, run_id, node, form, shown, asked_at, deadline, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    run_id,
                    node,
                    form,
                    json.dumps(dict(shown)),
                    asked_at or _now(),
                    deadline,
                    _json(payload),
                ),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"question {question_id!r} is already saved") from None
        return cast(QuestionRow, self.question(question_id))

    def answer(
        self, question_id: str, answer: Mapping[str, Any], who: str | None = None, answered_at: str | None = None
    ) -> QuestionRow:
        """Record a question's answer, ``answered_at`` now unless given; raises ``KeyError`` for an unknown id."""
        rows = self._execute(
            "UPDATE xmas_questions SET answer = ?, who = ?, answered_at = ? WHERE question_id = ? RETURNING *",
            (json.dumps(dict(answer)), who, answered_at or _now(), question_id),
        )
        if not rows:
            raise KeyError(f"no question {question_id!r}")
        return _question_row(rows[0])

    def question(self, question_id: str) -> QuestionRow | None:
        """Return a question's row, or ``None`` for an unknown id."""
        rows = self._execute("SELECT * FROM xmas_questions WHERE question_id = ?", (question_id,))
        return _question_row(rows[0]) if rows else None

    def questions(self, run_id: str) -> tuple[QuestionRow, ...]:
        """Return a run's questions in the order asked."""
        rows = self._execute("SELECT * FROM xmas_questions WHERE run_id = ? ORDER BY asked_at, rowid", (run_id,))
        return tuple(_question_row(r) for r in rows)

    def due(self, now: datetime | None = None) -> tuple[RunRow, ...]:
        """Return the paused runs with an open question whose deadline has passed at ``now``, oldest first."""
        moment = now or datetime.now(UTC)
        rows = self._execute(
            "SELECT DISTINCT run_id, deadline FROM xmas_questions WHERE answer IS NULL AND deadline IS NOT NULL"
        )
        late = {run_id for run_id, deadline in rows if datetime.fromisoformat(deadline) <= moment}
        return tuple(run for run in self.runs("paused") if run.run_id in late)

    def prune_runs(self, older_than_days: float) -> tuple[str, ...]:
        """Delete the runs that ended (not running or paused) over ``older_than_days`` ago; return their ids."""
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
        ended = [
            run.run_id
            for run in self.runs()
            if run.status not in ("running", "paused") and datetime.fromisoformat(run.updated_at) < cutoff
        ]
        for run_id in ended:
            self.delete(run_id)
        return tuple(ended)

    def delete(self, run_id: str) -> None:
        """Delete a run's row, its questions, and its threads: ``run_id`` and every ``run_id/...``."""
        prefix = run_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM xmas_runs WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM xmas_questions WHERE run_id = ?", (run_id,))
            for table in ("checkpoints", "writes"):
                self._conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ? OR thread_id LIKE ? ESCAPE '\\'",
                    (run_id, prefix),
                )


def _json(value: Mapping[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(dict(value))


def _run_row(row: Sequence[Any]) -> RunRow:
    run_id, system_id, status, created_at, updated_at, paused, totals = row
    loaded = json.loads(totals) if totals is not None else None
    return RunRow(run_id, system_id, status, created_at, updated_at, tuple(json.loads(paused)), loaded)


def _question_row(row: Sequence[Any]) -> QuestionRow:
    question_id, run_id, node, form, shown, asked_at, answered_at, answer, who, deadline, payload = row
    return QuestionRow(
        question_id,
        run_id,
        node,
        form,
        json.loads(shown),
        asked_at,
        answered_at,
        json.loads(answer) if answer is not None else None,
        who,
        deadline,
        json.loads(payload) if payload is not None else None,
    )


def open_store(location: str | None = None) -> Store:
    """Open the store at ``location``, else at ``XMAS_RUNS``; in memory when neither names a file."""
    location = location if location is not None else os.environ.get("XMAS_RUNS")
    return Store(location or None)
