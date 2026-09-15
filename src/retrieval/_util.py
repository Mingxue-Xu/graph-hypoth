from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    """Duck-typed accessor: works on a dict, an object, or None (-> default).

    Consolidated from six byte-identical `_get` copies across sources.py,
    service.py, coherence.py, resilience.py, tool_registry.py, fulltext.py.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def dump(
    value: Any,
    *,
    tuples: bool = False,
    str_keys: bool = False,
    callables: bool = False,
    exclude_computed: bool = False,
    dict_fallback: bool = True,
) -> Any:
    """Best-effort JSON-friendly dump for runtime-event/cache payloads.

    Defaults reproduce sources.py's original `_dump`. Callers pass flags to
    reproduce their own prior behavior byte-for-byte:
      - service.py:      tuples=True, str_keys=True, callables=True
      - tool_registry.py: tuples=True, str_keys=True, exclude_computed=True,
                          dict_fallback=False
    """
    if hasattr(value, "model_dump"):
        kwargs: dict[str, Any] = {"mode": "json", "exclude_none": True}
        if exclude_computed:
            kwargs["exclude_computed_fields"] = True
        return value.model_dump(**kwargs)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    list_types = (list, tuple) if tuples else (list,)
    if isinstance(value, list_types):
        return [
            dump(
                item,
                tuples=tuples,
                str_keys=str_keys,
                callables=callables,
                exclude_computed=exclude_computed,
                dict_fallback=dict_fallback,
            )
            for item in value
        ]
    if isinstance(value, dict):
        key_fn = str if str_keys else (lambda key: key)
        return {
            key_fn(key): dump(
                item,
                tuples=tuples,
                str_keys=str_keys,
                callables=callables,
                exclude_computed=exclude_computed,
                dict_fallback=dict_fallback,
            )
            for key, item in value.items()
        }
    if callables and callable(value):
        return None
    if dict_fallback and hasattr(value, "__dict__"):
        return {
            key: dump(
                item,
                tuples=tuples,
                str_keys=str_keys,
                callables=callables,
                exclude_computed=exclude_computed,
                dict_fallback=dict_fallback,
            )
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return value
