"""Normalize relation labels supplied with redundant graph arrow notation."""

from __future__ import annotations

from typing import Any


def normalize_relation_label(value: Any) -> str:
    """Strip surrounding arrow syntax while preserving words and internal hyphens."""
    return str(value or "").strip().strip("-<>=→← \t\r\n")
