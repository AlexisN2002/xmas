"""Media values by reference: pictures, sound, documents, tables and files.

A media value in a run's state, records and traces is a :class:`Ref`, never
its bytes: where the bytes are, their media type, and, for a local file, the
size and sha256 they had when referenced. Reading a ref checks that sha256, so
a file that changed since is refused rather than read.

Bytes that xmas keeps itself (a model's picture, a tool's blob, a step's
returned file) go into a content-addressed store, :func:`store`, so the same
bytes are stored once and a rewrite is a new ref. Its location is
configuration: the ``XMAS_STORE`` environment variable, else
``~/.cache/xmas/store``; never a spec key.

:data:`Image`, :data:`Audio`, :data:`Document`, :data:`Table` and :data:`File`
type a callable's parameters, so a step's tool schema names the kind it reads.
:class:`Capabilities` says which kinds a model takes and gives besides text.

This is a leaf module: it imports no other xmas module.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Protocol
from urllib.parse import urlsplit

from pydantic_core import core_schema

if TYPE_CHECKING:  # pragma: no cover
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue

MEDIA_KINDS = ("image", "audio", "document", "table", "file")
"""The value kinds a :class:`Ref` holds; ``file`` takes any ref."""

DOCUMENT_TYPES = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.presentation",
        "text/markdown",
        "text/html",
        "text/plain",
    }
)
"""Media types of the ``document`` kind."""

TABLE_TYPES = frozenset(
    {
        "text/csv",
        "text/tab-separated-values",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }
)
"""Media types of the ``table`` kind."""

MAX_PART_BYTES = 20 * 1024 * 1024
"""The largest media value sent to a model in one content part."""

_OCTET = "application/octet-stream"
_REMOTE = ("http://", "https://")
_SHA_CHUNK = 1 << 20


def kind_of(media_type: str) -> str:
    """Return the kind of a media type: ``image/*``, ``audio/*``, a document or table type, else ``file``."""
    base = media_type.split(";", 1)[0].strip().lower()
    family = base.split("/", 1)[0]
    if family in ("image", "audio"):
        return family
    if base in DOCUMENT_TYPES:
        return "document"
    if base in TABLE_TYPES:
        return "table"
    return "file"


def fits_kind(ref: Ref, kind: str) -> bool:
    """Whether ``ref`` is a value of ``kind``; every ref is a ``file``."""
    return kind == "file" or ref.kind == kind


def _guess_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or _OCTET


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Ref:
    """A media value: where its bytes are and what they are.

    Attributes:
        location: An absolute file path, or an ``http(s)`` URL that xmas passes on but never fetches.
        media_type: Such as ``image/png``; it decides the :attr:`kind`.
        size: The byte count when referenced, if known.
        sha256: The hex digest when referenced; :meth:`read_bytes` checks it.
    """

    location: str
    media_type: str
    size: int | None = None
    sha256: str | None = None

    @property
    def name(self) -> str:
        """The last segment of the location, such as ``slide-3.png``."""
        return PurePosixPath(urlsplit(self.location).path).name if self.remote else Path(self.location).name

    @property
    def kind(self) -> str:
        """The value kind of :attr:`media_type`, one of :data:`MEDIA_KINDS`."""
        return kind_of(self.media_type)

    @property
    def remote(self) -> bool:
        """Whether the location is an ``http(s)`` URL."""
        return self.location.startswith(_REMOTE)

    @classmethod
    def from_path(cls, path: str | os.PathLike[str], media_type: str | None = None) -> Ref:
        """Reference a file where it is, recording its size and sha256; the type is guessed from its name if not given.

        Raises:
            ValueError: If ``path`` is not a file.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"{os.fspath(path)} is not a file")
        return cls(
            str(resolved),
            media_type or _guess_type(resolved.name),
            resolved.stat().st_size,
            _sha256_of(resolved),
        )

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str, name: str = "") -> Ref:
        """Keep bytes in :func:`store` and reference the stored copy; ``name`` lends its extension."""
        return store().put(data, media_type, name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ref:
        """Read a ref from :meth:`to_dict`'s form.

        Raises:
            ValueError: For unknown keys, a missing location or media type, or a field of the wrong type.
        """
        names = [field.name for field in fields(cls)]
        unknown = [key for key in data if key not in names]
        if unknown:
            raise ValueError(f"A media reference takes {', '.join(names)}; unknown: {', '.join(map(str, unknown))}")
        for key in ("location", "media_type"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"A media reference needs {key} as a non-empty string, got {data.get(key)!r}")
        size, sha256 = data.get("size"), data.get("sha256")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ValueError(f"A media reference's size is a byte count, got {size!r}")
        if sha256 is not None and not isinstance(sha256, str):
            raise ValueError(f"A media reference's sha256 is a hex string, got {sha256!r}")
        return cls(data["location"], data["media_type"], size, sha256)

    def to_dict(self) -> dict[str, Any]:
        """Return the set fields, in field order, as plain JSON data."""
        return {field.name: value for field in fields(self) if (value := getattr(self, field.name)) is not None}

    def read_bytes(self) -> bytes:
        """Return the bytes.

        Raises:
            ValueError: For a remote ref, or a file whose sha256 changed since it was referenced.
        """
        if self.remote:
            raise ValueError(f"{self.name} is remote ({self.location}); xmas passes URLs on and reads no network")
        data = Path(self.location).read_bytes()
        if self.sha256 is not None and hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"{self.name} changed since it was referenced ({self.location})")
        return data

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Validate a ``Ref`` as itself and a mapping through :meth:`from_dict`; serialize through :meth:`to_dict`."""
        return core_schema.no_info_plain_validator_function(
            _validate_ref,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda ref: ref.to_dict()),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Describe a ref inline, marked as a ``file``; :class:`Kind` narrows the mark."""
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "media_type": {"type": "string"},
                "size": {"type": ["integer", "null"]},
                "sha256": {"type": ["string", "null"]},
            },
            "required": ["location", "media_type"],
            "additionalProperties": False,
            "x-xmas-kind": "file",
        }


def _validate_ref(value: Any) -> Ref:
    if isinstance(value, Ref):
        return value
    if isinstance(value, Mapping):
        return Ref.from_dict(value)
    raise ValueError(f"expected a media reference, got {type(value).__name__}")


class Store(Protocol):
    """Where xmas keeps bytes it was handed."""

    def put(self, data: bytes, media_type: str, name: str = "") -> Ref:
        """Keep ``data`` and return its ref; ``name`` lends its extension."""
        ...  # pragma: no cover


class FileStore:
    """A content-addressed store in a directory: ``<root>/<sha[:2]>/<sha><ext>``, each file written once.

    The extension comes from ``name``, else from the media type. A file is
    written through a temporary file and an atomic rename, so parallel puts of
    the same bytes leave one complete file.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Keep bytes under ``root``, created on the first put."""
        self.root = Path(root).expanduser()

    def put(self, data: bytes, media_type: str, name: str = "") -> Ref:
        """Store ``data`` unless its file is already there, and return its ref."""
        if not isinstance(data, bytes):
            raise TypeError(f"A store keeps bytes, not {type(data).__name__}")
        sha256 = hashlib.sha256(data).hexdigest()
        extension = PurePosixPath(name).suffix.lower() or mimetypes.guess_extension(media_type) or ""
        target = self.root / sha256[:2] / f"{sha256}{extension}"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{sha256}.", suffix=".part")
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(data)
                Path(temporary).replace(target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        return Ref(str(target), media_type, len(data), sha256)

    def __repr__(self) -> str:
        """Name the root."""
        return f"FileStore({str(self.root)!r})"


def store() -> Store:
    """Return the store ``XMAS_STORE`` names, else ``~/.cache/xmas/store``; read on every call."""
    return FileStore(os.environ.get("XMAS_STORE") or Path.home() / ".cache" / "xmas" / "store")


@dataclass(frozen=True)
class Kind:
    """A schema marker: ``Annotated[Ref, Kind("image")]`` describes a parameter reading images."""

    kind: str

    def __post_init__(self) -> None:
        """Refuse a kind that is not a media kind."""
        if self.kind not in MEDIA_KINDS:
            raise ValueError(f"Kind takes one of {', '.join(MEDIA_KINDS)}, not {self.kind!r}")

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Mark the ref's schema with ``"x-xmas-kind"``."""
        return {**handler.resolve_ref_schema(handler(schema)), "x-xmas-kind": self.kind}


Image = Annotated[Ref, Kind("image")]
"""A picture: ``image/*``."""

Audio = Annotated[Ref, Kind("audio")]
"""Sound: ``audio/*``."""

Document = Annotated[Ref, Kind("document")]
"""A document: one of :data:`DOCUMENT_TYPES`."""

Table = Annotated[Ref, Kind("table")]
"""A table: one of :data:`TABLE_TYPES`."""

File = Annotated[Ref, Kind("file")]
"""Any ref."""

ACCEPTABLE = ("text", "json", "image", "audio")
"""The kinds a model may accept."""

PRODUCIBLE = ("text", "json", "image")
"""The kinds a model may produce."""

_ALWAYS = ("text", "json")
_KEYS = {"accepts": ("accept", ACCEPTABLE), "produces": ("produce", PRODUCIBLE)}


@dataclass(frozen=True)
class Capabilities:
    """What a model takes and gives besides text and json, in the words a ``models:`` entry states.

    Attributes:
        accepts: Kinds it reads, from :data:`ACCEPTABLE`.
        produces: Kinds it answers with, from :data:`PRODUCIBLE`.
    """

    accepts: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    def takes(self, kind: str) -> bool:
        """Whether the model reads ``kind``; text and json always."""
        return kind in _ALWAYS or kind in self.accepts

    def gives(self, kind: str) -> bool:
        """Whether the model answers with ``kind``; text and json always."""
        return kind in _ALWAYS or kind in self.produces

    @classmethod
    def parse(cls, entry: Mapping[str, Any], where: str) -> Capabilities:
        """Read a ``models:`` entry such as ``{"accepts": ["image"]}``; ``where`` names it in errors.

        Raises:
            ValueError: For a key other than accepts and produces, a value that is not a list of words, or a word a
                model cannot accept or produce.
        """
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} is a mapping with accepts and produces, not {type(entry).__name__}")
        for key in entry:
            if key not in _KEYS:
                raise ValueError(
                    f"{where} has key {key!r}; it takes accepts and produces "
                    "(model parameters come with the .mas bundle)"
                )
        words: dict[str, tuple[str, ...]] = {}
        for key, (verb, allowed) in _KEYS.items():
            given = entry.get(key, [])
            if not isinstance(given, (list, tuple)) or not all(isinstance(word, str) for word in given):
                raise ValueError(f"{where} {key} is a list of words such as [image], not {given!r}")
            for word in given:
                if word not in allowed:
                    media = ", ".join(kind for kind in allowed if kind not in _ALWAYS)
                    raise ValueError(f"{where} {key} {word!r}; a model may {verb} {media}")
            words[key] = tuple(given)
        return cls(words["accepts"], words["produces"])

    def to_spec(self) -> dict[str, list[str]]:
        """Return the ``models:`` entry: the stated keys only."""
        return {key: list(value) for key, value in (("accepts", self.accepts), ("produces", self.produces)) if value}


TEXT_ONLY = Capabilities()
"""A model that reads and writes text and json only."""
