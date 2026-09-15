from __future__ import annotations

import logging
import subprocess
import sys
import textwrap


def test_camel_context_window_filter_is_narrow() -> None:
    # Pytest may replace the root logger's filters after module collection, so
    # install at assertion time as well as exercising camel_adapter's import.
    import src.camel_adapter  # noqa: F401
    from src import _warning_filters

    _warning_filters._install_camel_context_window_filter()
    warning_filter = next(
        item
        for item in logging.getLogger().filters
        if getattr(item, "name", None) == "graph_hypoth_camel_context_window_filter"
    )
    suppressed = logging.LogRecord(
        "camel.models",
        logging.WARNING,
        __file__,
        1,
        "Unknown model 'vendor/new': context window size not defined. "
        "Defaulting to 999_999_999.",
        (),
        None,
    )
    unrelated = logging.LogRecord(
        "camel.models",
        logging.WARNING,
        __file__,
        1,
        "A different warning",
        (),
        None,
    )

    assert warning_filter.filter(suppressed) is False
    assert warning_filter.filter(unrelated) is True


def test_import_installs_only_the_camel_context_window_filter() -> None:
    script = textwrap.dedent(
        """
        import io
        import logging

        import src.camel_adapter

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.WARNING)

        logging.warning(
            "Unknown model 'vendor/new': context window size not defined. "
            "Defaulting to 999_999_999."
        )
        logging.warning("Different warning should remain visible")

        handler.flush()
        print(stream.getvalue(), end="")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "context window size not defined" not in completed.stdout
    assert "Different warning should remain visible" in completed.stdout
