"""Configuration: where models and tools are served, and which model names exist there.

A saved system names its models and tool servers; it never holds an address or a key.
Those come from here, first from the arguments a caller passes, then from the
environment, then from a TOML file, ``$XMAS_CONFIG`` or else
``${XDG_CONFIG_HOME:-~/.config}/xmas/config.toml``. A missing file is no error.

.. code-block:: toml

    [endpoint]
    base_url = "http://127.0.0.1:8000/v1"
    api_key_env = "GATEWAY_KEY"        # or api_key = "..."
    catalogue = ["kimi-k3"]            # optional; else GET {base_url}/models
    stream = false

    [models."kimi-k3"]
    price = {input_per_m = 0.6, output_per_m = 2.5}
    stream = true

    [tools.web]
    url = "https://tools.example.com/mcp"
    key_env = "WEB_TOOLS_KEY"          # or key = "..."

    [store]
    media = "~/.cache/xmas/store"
    runs = "~/.cache/xmas/runs.sqlite"

The environment variables are ``XMAS_BASE_URL``, ``XMAS_API_KEY``, ``XMAS_MODELS`` (a
comma list that stands for the catalogue, so names check offline), ``XMAS_STORE``
(media) and ``XMAS_RUNS`` (the run store). A tool server without a key leaves it to
``XMAS_TOOLS_KEY``, which :class:`~xmas.tools.mcp.MCPTools` reads itself.

The catalogue is the list of model ids the endpoint serves. :func:`read_catalogue`
takes a configured list, else asks the endpoint and keeps its answer for
:data:`CATALOGUE_TTL_S` per endpoint and key; offline, a kept answer of any age
stands in. :func:`resolve_names` checks names against it: a miss is a
:class:`ModelNameWarning` naming the closest ids, never a substitute. No key appears
in a ``repr``, a message or the kept answer, and nothing here runs on import.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import stat
import time
import tomllib
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from xmas.llms.meta import TokenPrice

CONFIG_ENV = "XMAS_CONFIG"
"""The environment variable naming the configuration file."""

MODELS_ENV = "XMAS_MODELS"
"""The environment variable holding a comma list of model ids, used as the catalogue."""

CATALOGUE_TTL_S = 300.0
"""How long an endpoint's answer to ``GET /models`` is used before it is asked again."""

FETCH_TIMEOUT_S = 5.0
"""How long :func:`read_catalogue` waits for the endpoint."""

_SCALARS: dict[str, dict[str, type]] = {
    "endpoint": {"base_url": str, "api_key": str, "api_key_env": str, "catalogue": list, "stream": bool},
    "model": {"price": dict, "stream": bool},
    "price": {"input_per_m": float, "output_per_m": float, "cached_input_per_m": float},
    "tool": {"url": str, "key": str, "key_env": str},
    "store": {"media": str, "runs": str},
}
_TABLES = ("endpoint", "models", "tools", "store")
_SHAPES = {str: "a string", bool: "true or false", float: "a number", dict: "a table", list: "a list of model ids"}


def _words(value: list[Any]) -> bool:
    """Whether every item is a string."""
    return all(isinstance(item, str) for item in value)


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the configuration file's path: ``$XMAS_CONFIG``, else ``config.toml`` under the user's config folder."""
    environ = os.environ if environ is None else environ
    if environ.get(CONFIG_ENV):
        return Path(environ[CONFIG_ENV]).expanduser()
    return Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config").expanduser() / "xmas" / "config.toml"


@dataclass(frozen=True)
class ToolServer:
    """A tool server a callable names by ``server:``; its key never shows in a ``repr``.

    Attributes:
        url: The server's address.
        key: Its bearer key, or ``None`` to leave it to ``XMAS_TOOLS_KEY``.
    """

    url: str
    key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Config:
    """Endpoint, key, catalogue, prices, streaming, tool servers and stores, as :meth:`load` resolved them.

    Attributes:
        base_url: The OpenAI-compatible endpoint, or ``None`` when none is configured.
        api_key: Its key; never shown in a ``repr``.
        catalogue: Model ids configured to stand for the endpoint's list, or ``None`` to ask the endpoint.
        stream: Whether calls stream, for models without their own setting.
        streams: Per model id, whether its calls stream.
        prices: Per model id, the price used when a response reports no cost.
        tools: Tool servers by the name callables use.
        media_store: Folder for stored media, or ``None`` for the default.
        run_store: File of the run store, or ``None`` for the default.
        source: The file that was read, or ``None`` when there was none.
    """

    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    catalogue: tuple[str, ...] | None = None
    stream: bool = False
    streams: Mapping[str, bool] = field(default_factory=dict)
    prices: Mapping[str, TokenPrice] = field(default_factory=dict)
    tools: Mapping[str, ToolServer] = field(default_factory=dict)
    media_store: Path | None = None
    run_store: Path | None = None
    source: Path | None = None

    @classmethod
    def load(
        cls, path: str | os.PathLike[str] | None = None, environ: Mapping[str, str] | None = None, **given: Any
    ) -> Config:
        """Resolve the configuration: ``given`` fields that are not ``None``, then the environment, then the file.

        Args:
            path: The file to read, instead of :func:`config_path`.
            environ: The environment to read, instead of ``os.environ``.
            **given: Field values that win over everything else.

        Raises:
            TypeError: For a ``given`` name that is not a field.
            ValueError: For a file that is not TOML, or holds a key or a value it does not take; the message names it.
        """
        environ = os.environ if environ is None else environ
        for name in given:
            if name == "source" or name not in {f.name for f in fields(cls)}:
                raise TypeError(f"Config.load takes no argument {name!r}")
        file = Path(path).expanduser() if path is not None else config_path(environ)
        values = _read_file(file, environ) if file.is_file() else {}
        listed = [name.strip() for name in environ.get(MODELS_ENV, "").split(",") if name.strip()]
        from_env: dict[str, Any] = {
            "base_url": environ.get("XMAS_BASE_URL"),
            "api_key": environ.get("XMAS_API_KEY"),
            "catalogue": tuple(listed) if listed else None,
            "media_store": Path(environ["XMAS_STORE"]).expanduser() if environ.get("XMAS_STORE") else None,
            "run_store": Path(environ["XMAS_RUNS"]).expanduser() if environ.get("XMAS_RUNS") else None,
        }
        for layer in (from_env, given):
            values.update({key: value for key, value in layer.items() if value is not None and value != ""})
        if values.get("base_url"):
            values["base_url"] = str(values["base_url"]).rstrip("/")
        if values.get("catalogue") is not None:
            values["catalogue"] = tuple(values["catalogue"])
        return cls(**values, source=file if file.is_file() else None)

    def stream_of(self, model: str) -> bool:
        """Whether calls to ``model`` stream: its own setting, else :attr:`stream`."""
        return self.streams.get(model, self.stream)


def _read_file(path: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    """Parse the file into :class:`Config` fields, naming the key of any problem."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not a TOML file ({exc})") from None

    def checked(table: Any, kind: str, where: str) -> dict[str, Any]:
        if not isinstance(table, dict):
            raise ValueError(f"{path}: {where} is a table, not {type(table).__name__}")
        allowed = _SCALARS[kind]
        for key, value in table.items():
            if key not in allowed:
                raise ValueError(f"{path}: unknown key '{where}.{key}'; {where} takes {', '.join(allowed)}")
            want = allowed[key]
            fits = isinstance(value, (int, float)) if want is float else isinstance(value, want)
            if isinstance(value, bool) is not (want is bool) or not fits or (key == "catalogue" and not _words(value)):
                raise ValueError(f"{path}: {where}.{key} is {_SHAPES[want]}, not {type(value).__name__}")
        for key, alias in (("api_key", "api_key_env"), ("key", "key_env")):
            if key in table and alias in table:
                raise ValueError(f"{path}: {where} takes {key} or {alias}, not both")
        return table

    for key in data:
        if key not in _TABLES:
            raise ValueError(f"{path}: unknown table '{key}'; the file takes {', '.join(_TABLES)}")
    endpoint = checked(data.get("endpoint", {}), "endpoint", "endpoint")
    store = checked(data.get("store", {}), "store", "store")
    values: dict[str, Any] = {
        "base_url": endpoint.get("base_url"),
        "api_key": endpoint.get("api_key")
        or (environ.get(endpoint["api_key_env"]) if "api_key_env" in endpoint else None),
        "catalogue": endpoint.get("catalogue"),
        "stream": endpoint.get("stream", False),
        "media_store": Path(store["media"]).expanduser() if "media" in store else None,
        "run_store": Path(store["runs"]).expanduser() if "runs" in store else None,
    }
    prices, streams = {}, {}
    for model, entry in _subtables(data, "models", path).items():
        where = f"models.{json.dumps(model)}"
        checked(entry, "model", where)
        if "stream" in entry:
            streams[model] = entry["stream"]
        if "price" in entry:
            price = checked(entry["price"], "price", f"{where}.price")
            missing = [key for key in ("input_per_m", "output_per_m") if key not in price]
            if missing:
                raise ValueError(f"{path}: {where}.price needs {' and '.join(missing)}")
            prices[model] = TokenPrice(**{key: float(value) for key, value in price.items()})
    tools = {}
    for name, entry in _subtables(data, "tools", path).items():
        where = f"tools.{name}"
        checked(entry, "tool", where)
        if "url" not in entry:
            raise ValueError(f"{path}: {where} needs url")
        secret = entry.get("key") or (environ.get(entry["key_env"]) if "key_env" in entry else None)
        tools[name] = ToolServer(entry["url"], secret)
    if ("api_key" in endpoint or any("key" in entry for entry in data.get("tools", {}).values())) and _shared(path):
        warnings.warn(f"{path} holds a key and others can read it; chmod 600 {path}", UserWarning, stacklevel=3)
    return {**{k: v for k, v in values.items() if v is not None}, "prices": prices, "streams": streams, "tools": tools}


def _subtables(data: Mapping[str, Any], table: str, path: Path) -> dict[str, Any]:
    """Return ``data[table]``, a table of tables such as ``[models."<id>"]``."""
    entries = data.get(table, {})
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: {table} is a table, not {type(entries).__name__}")
    return entries


def _shared(path: Path) -> bool:
    """Whether the group or others may read ``path``."""
    return bool(path.stat().st_mode & (stat.S_IRGRP | stat.S_IROTH))


def _shown(url: str) -> str:
    """Return ``url`` without credentials, query or fragment, fit for a message."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


@dataclass(frozen=True)
class Catalogue:
    """The model ids an endpoint serves, and where they came from.

    Attributes:
        ids: The ids, or ``None`` when no list could be had and names go unchecked.
        source: ``"configured"``, ``"endpoint"``, ``"cache"`` or ``"none"``.
        note: Why the list is old or missing, for a report line.
    """

    ids: tuple[str, ...] | None
    source: str
    note: str = ""


def _cache_file(base_url: str, api_key: str | None) -> Path:
    """Return the file that keeps the answer of ``base_url`` for ``api_key``; the key enters only hashed."""
    folder = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser() / "xmas"
    keyed = hashlib.sha256((api_key or "").encode()).hexdigest()
    return folder / f"catalogue-{hashlib.sha256(f'{base_url}\n{keyed}'.encode()).hexdigest()[:16]}.json"


def _kept(file: Path) -> tuple[float, tuple[str, ...]] | None:
    """Return the kept answer's time and ids, or ``None`` when there is none readable."""
    try:
        kept = json.loads(file.read_text(encoding="utf-8"))
        return float(kept["fetched"]), tuple(str(name) for name in kept["ids"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_catalogue(
    config: Config,
    *,
    refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] = time.time,
) -> Catalogue:
    """Return the catalogue: the configured list, else a fresh kept answer, else ``GET {base_url}/models``.

    Args:
        config: Where to look.
        refresh: Ask the endpoint even when a kept answer is fresh.
        transport: The HTTP transport, for tests.
        clock: The time in seconds since the epoch, for tests.

    Returns:
        The ids; when the endpoint cannot be reached, a kept answer of any age, else ``ids=None`` with a note.
    """
    if config.catalogue is not None:
        return Catalogue(tuple(config.catalogue), "configured")
    if not config.base_url:
        return Catalogue(None, "none", "no endpoint; model names not checked")
    file = _cache_file(config.base_url, config.api_key)
    kept = _kept(file)
    now = clock()
    if kept is not None and not refresh and 0 <= now - kept[0] < CATALOGUE_TTL_S:
        return Catalogue(kept[1], "cache")
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        with httpx.Client(transport=transport, timeout=FETCH_TIMEOUT_S) as client:
            response = client.get(f"{config.base_url}/models", headers=headers)
            response.raise_for_status()
            ids = tuple(str(entry["id"]) for entry in response.json()["data"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        why = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
        if kept is not None:
            date = datetime.fromtimestamp(kept[0], UTC).strftime("%Y-%m-%d %H:%M UTC")
            return Catalogue(kept[1], "cache", f"cache of {date} ({why})")
        return Catalogue(
            None,
            "none",
            f"the catalogue of {_shown(config.base_url)} could not be read ({why}); model names not checked",
        )
    with contextlib.suppress(OSError):  # an unwritable cache only costs the next call a fetch
        file.parent.mkdir(parents=True, exist_ok=True)
        temp = file.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps({"fetched": now, "ids": list(ids)}), encoding="utf-8")
        temp.replace(file)
    return Catalogue(ids, "endpoint")


def _bare(name: str) -> str:
    """Return ``name`` lower-cased, without a ``vendor/`` prefix."""
    return name.rsplit("/", 1)[-1].lower()


def closest(name: str, ids: Iterable[str], n: int = 3) -> list[str]:
    """Return up to ``n`` ids nearest ``name``: equal but for case and vendor prefix, else difflib's closest."""
    by_bare: dict[str, list[str]] = {}
    for candidate in ids:
        by_bare.setdefault(_bare(candidate), []).append(candidate)
    if _bare(name) in by_bare:
        return by_bare[_bare(name)][:n]
    picks = difflib.get_close_matches(_bare(name), list(by_bare), n=n, cutoff=0.6)
    return [candidate for bare in picks for candidate in by_bare[bare]][:n]


def misses(names: Iterable[str], catalogue: Catalogue) -> dict[str, tuple[str, ...]]:
    """Return each name not in the catalogue with its closest ids; nothing when the catalogue is unknown."""
    if catalogue.ids is None:
        return {}
    known = set(catalogue.ids)
    return {name: tuple(closest(name, catalogue.ids)) for name in dict.fromkeys(names) if name not in known}


class ModelNameWarning(UserWarning):
    """A model name the catalogue does not hold; ``name`` and ``suggestions`` say which and what is near."""

    def __init__(self, message: str, name: str, suggestions: tuple[str, ...]) -> None:
        """Keep the message, the name and the closest ids."""
        super().__init__(message)
        self.name = name
        self.suggestions = suggestions


@dataclass(frozen=True)
class Resolution:
    """The outcome of :func:`resolve_names`.

    Attributes:
        catalogue: The catalogue the names were checked against.
        suggestions: Each missing name with its closest ids, possibly none.
        warnings: One :class:`ModelNameWarning` per missing name, for the caller to issue.
    """

    catalogue: Catalogue
    suggestions: Mapping[str, tuple[str, ...]]
    warnings: tuple[ModelNameWarning, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        """The names not in the catalogue."""
        return tuple(self.suggestions)

    @property
    def checked(self) -> bool:
        """Whether there was a catalogue to check against."""
        return self.catalogue.ids is not None


def resolve_names(
    names: Iterable[str] | Mapping[str, str], config: Config, *, catalogue: Catalogue | None = None
) -> Resolution:
    """Check model ids against the catalogue, reading it unless one is given.

    Args:
        names: Model ids, or a mapping of ids to who binds them (``"agent editor"``), named in the warning.
        config: Where the catalogue comes from.
        catalogue: A catalogue already read.

    Returns:
        The missing names with suggestions and their warnings; nothing missing when the catalogue is unknown.
    """
    catalogue = read_catalogue(config) if catalogue is None else catalogue
    found = misses(names, catalogue)
    labels = names if isinstance(names, Mapping) else {}
    where = (
        f"the catalogue of {_shown(config.base_url)}"
        if catalogue.source != "configured" and config.base_url
        else f"the configured catalogue ({MODELS_ENV} or endpoint.catalogue)"
    )
    issued = []
    for name, near in found.items():
        who = f" ({labels[name]})" if labels.get(name) else ""
        tail = f"closest: {', '.join(near)}" if near else "nothing close"
        issued.append(ModelNameWarning(f"model {name!r}{who} is not in {where}; {tail}", name, near))
    return Resolution(catalogue, found, tuple(issued))
