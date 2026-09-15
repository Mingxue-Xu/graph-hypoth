from __future__ import annotations

import logging
import re


_CAMEL_UNKNOWN_CONTEXT_WINDOW_WARNING = re.compile(
    r"^Unknown model '.+': context window size not defined\. "
    r"Defaulting to 999_999_999\.$"
)
_CAMEL_CONTEXT_WINDOW_FILTER_NAME = "graph_hypoth_camel_context_window_filter"


class _CamelContextWindowFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        # ``logging.Filter.__init__`` assigns an empty instance ``name``;
        # replace it with the stable marker used for idempotent installation.
        self.name = _CAMEL_CONTEXT_WINDOW_FILTER_NAME

    def filter(self, record: logging.LogRecord) -> bool:
        return _CAMEL_UNKNOWN_CONTEXT_WINDOW_WARNING.match(record.getMessage()) is None


def _install_camel_context_window_filter() -> None:
    logger = logging.getLogger()
    if any(
        getattr(existing_filter, "name", None) == _CAMEL_CONTEXT_WINDOW_FILTER_NAME
        for existing_filter in logger.filters
    ):
        return
    logger.addFilter(_CamelContextWindowFilter())


_install_camel_context_window_filter()
