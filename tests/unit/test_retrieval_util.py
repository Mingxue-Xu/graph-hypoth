from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.retrieval._util import dump, get_attr_or_key
from src.retrieval.models import SearchPaperFilters


# ---------------------------------------------------------------------------
# get_attr_or_key — was six byte-identical `_get` copies (sources/service/
# coherence/resilience/tool_registry/fulltext).
# ---------------------------------------------------------------------------


def test_get_attr_or_key_none_value_returns_default() -> None:
    assert get_attr_or_key(None, "name") is None
    assert get_attr_or_key(None, "name", "fallback") == "fallback"


def test_get_attr_or_key_dict_uses_get() -> None:
    assert get_attr_or_key({"name": "x"}, "name") == "x"
    assert get_attr_or_key({}, "name", "fallback") == "fallback"


def test_get_attr_or_key_object_uses_getattr() -> None:
    class Obj:
        name = "x"

    assert get_attr_or_key(Obj(), "name") == "x"
    assert get_attr_or_key(Obj(), "missing", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# dump — was three modules' `_dump` with semantic drift (sources.py,
# service.py, tool_registry.py). Fixtures below capture each module's
# pre-consolidation behavior as literals, keyed by the flag combination that
# reproduces it.
# ---------------------------------------------------------------------------


def _noop() -> None:  # module-level function: has an empty __dict__.
    pass


@dataclass
class _DataclassWithCallable:
    name: str
    handler: Callable[[], None]


class _PlainObj:
    def __init__(self) -> None:
        self.a = 1
        self.b = "x"
        self._hidden = "nope"
        self.func = _noop


def test_dump_dataclass_uses_asdict_regardless_of_flags() -> None:
    # asdict() deep-copies field values; functions are copy-atomic, so the
    # callable attribute survives untouched — same for every module.
    value = _DataclassWithCallable(name="x", handler=_noop)
    expected = {"name": "x", "handler": _noop}
    assert dump(value) == expected
    assert dump(value, tuples=True, str_keys=True, callables=True) == expected
    assert (
        dump(value, tuples=True, str_keys=True, exclude_computed=True, dict_fallback=False)
        == expected
    )


def test_dump_pydantic_model_default_includes_computed_field() -> None:
    # sources.py / service.py behavior: no exclude_computed_fields passed.
    filters = SearchPaperFilters(highlight_query="hi")
    assert dump(filters) == {
        "highlight_query": "hi",
        "verify_quotes": True,
        "request_kind": "search",
    }


def test_dump_pydantic_model_exclude_computed_drops_computed_field() -> None:
    # tool_registry.py behavior: exclude_computed_fields=True.
    filters = SearchPaperFilters(highlight_query="hi")
    assert dump(filters, exclude_computed=True) == {
        "highlight_query": "hi",
        "verify_quotes": True,
    }


def test_dump_list_default_no_tuple_recursion() -> None:
    # sources.py behavior: only `list` is recursed; a tuple falls through
    # every branch (no __dict__) and is returned unchanged.
    assert dump([1, {"a": 1}]) == [1, {"a": 1}]
    tup = (1, 2)
    assert dump(tup) is tup


def test_dump_tuples_true_recurses_into_list() -> None:
    # service.py / tool_registry.py behavior: list | tuple both recursed,
    # a tuple becomes a plain list.
    assert dump((1, {"a": 1}), tuples=True) == [1, {"a": 1}]


def test_dump_dict_str_keys_flag() -> None:
    assert dump({5: "v"}) == {5: "v"}
    assert dump({5: "v"}, str_keys=True) == {"5": "v"}


def test_dump_plain_object_dict_fallback_filters_underscore_and_callables() -> None:
    # sources.py / service.py behavior: dict_fallback=True (default) walks
    # vars(), dropping `_`-prefixed and callable attributes.
    assert dump(_PlainObj()) == {"a": 1, "b": "x"}


def test_dump_plain_object_dict_fallback_false_returns_unchanged() -> None:
    # tool_registry.py behavior: no __dict__ branch at all.
    obj = _PlainObj()
    assert dump(obj, dict_fallback=False) is obj


def test_dump_bare_callable_three_module_behaviors() -> None:
    # A bare top-level callable (module function, empty __dict__) is where
    # all three modules' _dump disagreed:
    #  - sources.py (dict_fallback=True, callables=False): falls into the
    #    __dict__ branch, whose vars() is empty -> {}.
    #  - service.py (callables=True): short-circuits to None before the
    #    __dict__ branch is reached.
    #  - tool_registry.py (dict_fallback=False): no __dict__ branch, no
    #    callable check -> returned unchanged.
    assert dump(_noop) == {}
    assert dump(_noop, tuples=True, str_keys=True, callables=True) is None
    assert (
        dump(_noop, tuples=True, str_keys=True, exclude_computed=True, dict_fallback=False)
        is _noop
    )
