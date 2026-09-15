from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


pytestmark = pytest.mark.smoke


def test_pyproject_exposes_the_two_console_scripts() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    assert scripts == {
        "graph-hypoth-orchestrate": "src.cli:main",
        "graph-hypoth-synthesist": "src.synthesist_run:main",
    }
