"""Smoke tests for the packaging shell.

These intentionally avoid framework internals: they assert that the package is
importable, typed and versioned, so CI is meaningful before the core exists.
"""

import importlib.metadata
import re
from pathlib import Path

import xmas

SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].+)?$")


def test_package_is_importable() -> None:
    assert xmas.__name__ == "xmas"


def test_version_is_semver() -> None:
    assert SEMVER.match(xmas.__version__), f"not a semver string: {xmas.__version__!r}"


def test_installed_metadata_matches_dunder_version() -> None:
    """The build single-sources the version from ``xmas.__version__``."""
    assert importlib.metadata.version("xmas") == xmas.__version__


def test_package_ships_a_py_typed_marker() -> None:
    """Without this marker, type checkers ignore xmas in downstream projects."""
    marker = Path(xmas.__file__).parent / "py.typed"
    assert marker.is_file()
