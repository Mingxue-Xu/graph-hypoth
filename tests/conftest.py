from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import pytest


# Model-identifier fixtures never use a real vendor slug. Three registers exist
# because they name three different kinds of value:
#
#   1. ``x/<role>-model`` — OpenRouter-shaped catalog slugs (``x/`` is a fake
#      vendor namespace). Used wherever ``provider: openrouter`` is exercised.
#      Ids compared against each other must stay pairwise distinct or the
#      role-fallback assertions stop discriminating.
#   2. ``claude-model-id`` / ``codex-model-id`` (with ``-alt`` / ``-aux``
#      variants) — model names handed to ``claude --model`` / ``codex --model``.
#      Deliberately not ``x/…``: a CLI model name is not a catalog slug, so the
#      slug shape would be misleading here.
#   3. ``gpt-test`` / ``claude-test`` / ``probe-no-model-call`` — pre-existing
#      ids that are already obviously fake. Frozen: renaming them churns the
#      diff without changing intent.
#
# ``test-provider`` and ``TEST_MODEL_API_KEY`` are the neutral stand-ins for an
# arbitrary provider string and an arbitrary key-env name in passthrough tests.
# Key-stripping and redaction tests keep the real key names they assert on.


_ARXIV_LOCK = threading.Lock()
_ARXIV_LAST_CALL_AT = 0.0
_ARXIV_PACING_SECONDS = 3.0
_T = TypeVar("_T")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: live-network test (any provider)")
    config.addinivalue_line("markers", "live_exa: requires EXA_API_KEY")
    config.addinivalue_line("markers", "live_arxiv: makes real arXiv calls")
    config.addinivalue_line(
        "markers",
        "live_openalex: live OpenAlex API test (polite pool; OPENALEX_API_KEY optional)",
    )
    config.addinivalue_line(
        "markers",
        "live_llm_provider: live LLM provider swap contract tests",
    )
    config.addinivalue_line(
        "markers",
        "live_openrouter: live OpenRouter/CAMEL provider test",
    )
    config.addinivalue_line(
        "markers",
        "live_openai_compatible: live OpenAI-wire-compatible endpoint (self-hosted or proxy) test",
    )
    config.addinivalue_line(
        "markers",
        "drift_watchdog: catches upstream API drift",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if _live_tests_requested(config):
        return

    skip_live = pytest.mark.skip(
        reason=(
            "live tests are opt-in; use -m live/-m drift_watchdog or invoke the "
            "test file directly"
        )
    )
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


def _live_tests_requested(config: pytest.Config) -> bool:
    return _live_marker_expression_requested(config) or _direct_test_file_requested(config)


def _live_marker_expression_requested(config: pytest.Config) -> bool:
    expression = (getattr(config.option, "markexpr", "") or "").strip()
    if not expression:
        return False

    requested_markers = {
        "live",
        "live_exa",
        "live_arxiv",
        "live_openalex",
        "live_llm_provider",
        "live_openrouter",
        "live_openai_compatible",
        "drift_watchdog",
    }
    requested_pattern = "|".join(sorted(requested_markers))
    positive_expression = re.sub(
        rf"\bnot\s+\(?\s*({requested_pattern})\s*\)?",
        "",
        expression,
    )
    tokens = re.findall(r"\b[\w_]+\b", positive_expression)
    if not requested_markers.intersection(tokens):
        return False

    return True


def _direct_test_file_requested(config: pytest.Config) -> bool:
    for arg in getattr(config, "args", ()) or ():
        target = str(arg).split("::", 1)[0]
        if target.endswith(".py"):
            return True
    return False


@pytest.fixture
def paced_arxiv_call() -> Callable[[Callable[[], _T]], _T]:
    def run(call: Callable[[], _T]) -> _T:
        global _ARXIV_LAST_CALL_AT

        with _ARXIV_LOCK:
            elapsed = time.monotonic() - _ARXIV_LAST_CALL_AT
            if _ARXIV_LAST_CALL_AT and elapsed < _ARXIV_PACING_SECONDS:
                time.sleep(_ARXIV_PACING_SECONDS - elapsed)
            try:
                return call()
            finally:
                _ARXIV_LAST_CALL_AT = time.monotonic()

    return run
