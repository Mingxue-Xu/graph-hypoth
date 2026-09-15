"""Reader-translation seam ( Critic Panel): rewrite reader cards into the reader's own
vocabulary, with every {original -> familiar} equivalence verified by a SECOND
(critical-tier) call. A wrong analogy is worse than jargon: any rejection after one
retry ships the IDENTITY card. All guards below are deterministic (no LLM)."""

from __future__ import annotations

import json

from src.research_profile import (
    LexiconPaper,
    LexiconTerm,
    ReaderLexicon,
)


class _FakeBackend:
    """Queue of canned completions; records every messages list it was run with."""

    def __init__(self, *contents):
        self._contents = list(contents)
        self.calls = []

    def run(self, messages, tools=None):
        self.calls.append(messages)
        content = self._contents.pop(0) if self._contents else ""
        return {"choices": [{"message": {"content": content}}]}


def _lexicon():
    return ReaderLexicon(
        home_field="tensor factorization for LLM compression",
        familiar_terms=[
            LexiconTerm(term="rank truncation", gloss="dropping small singular values")
        ],
        analogy_domains=["numerical linear algebra"],
        papers=[LexiconPaper(title="TensorGPT", year=2024, venue="arXiv")],
        source="local paper",
        generated="2026-07-02",
    )


def _card():
    return {
        "headline": "Quantization noise may act like weak priors in belief propagation.",
        "what_might_be_happening": (
            "Lloyd-Max quantization adds noise that behaves like extra uncertainty."
        ),
        "why_it_matters": "It could justify cheap message compression.",
        "how_to_check": "Compare quantized and full-precision runs.",
        "what_would_change_our_mind": "No difference at coarse quantization.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [
            {
                "term": "Lloyd-Max quantization",
                "plain_meaning": "A method that picks quantization levels to minimize error.",
                "source": "Quantized message passing",
            }
        ],
    }


def _translated(**overrides):
    card = {
        "headline": "Rounding noise may act like weak priors in belief propagation.",
        "what_might_be_happening": (
            "Optimal rounding (Lloyd-Max quantization) adds noise that behaves like "
            "extra uncertainty, the way rank truncation drops detail."
        ),
        "why_it_matters": "It could justify cheap message compression.",
        "how_to_check": "Compare quantized and full-precision runs.",
        "what_would_change_our_mind": "No difference at coarse quantization.",
        "status": "This is a testable idea, not a proven conclusion.",
        "key_terms": [
            {
                "term": "Lloyd-Max quantization",
                "plain_meaning": (
                    "A way of choosing rounding levels so the average squared "
                    "rounding error is as small as possible."
                ),
                "source": "Quantized message passing",
                "familiar_gloss": "like rank truncation, it keeps what matters most",
            }
        ],
        "translations": [
            {
                "original": "Lloyd-Max quantization",
                "familiar": "optimal rounding",
                "first_use_rendering": "optimal rounding (Lloyd-Max quantization)",
                "rationale": "both pick levels minimizing squared error",
            }
        ],
    }
    card.update(overrides)
    return card


_VERDICT_CONFIRMED = {
    "verdicts": [
        {"original": "Lloyd-Max quantization", "verdict": "confirmed", "note": "same method"}
    ]
}
_VERDICT_APPROXIMATE = {
    "verdicts": [
        {
            "original": "Lloyd-Max quantization",
            "verdict": "approximate",
            "note": "loses the fixed-rate qualification",
        }
    ]
}
_VERDICT_REJECTED = {
    "verdicts": [
        {
            "original": "Lloyd-Max quantization",
            "verdict": "rejected",
            "note": "different objective",
        }
    ]
}


def _translator(tb, vb):
    from src.reader_translation import LLMReaderTranslator

    return LLMReaderTranslator(tb, verifier_backend=vb)


# --- identity / passthrough ----------------------------------------------------------------
def test_translate_all_identity_without_translator_or_lexicon():
    from src.reader_translation import translate_all

    elabs = {"h1": _card()}
    assert translate_all(None, elabs, lexicon=_lexicon()) is elabs
    tb, vb = _FakeBackend(), _FakeBackend()
    assert translate_all(_translator(tb, vb), elabs, lexicon=None) is elabs
    assert tb.calls == [] and vb.calls == []


def test_empty_card_passes_through_with_zero_backend_calls():
    tb, vb = _FakeBackend(), _FakeBackend()
    translator = _translator(tb, vb)
    empty = {}
    assert translator.translate(empty, lexicon=_lexicon()) is empty
    no_headline = {"key_terms": []}
    assert translator.translate(no_headline, lexicon=_lexicon()) is no_headline
    assert tb.calls == [] and vb.calls == []


# --- prompt content -------------------------------------------------------------------------
def test_translator_prompt_carries_lexicon_card_definitions_quotes():
    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    _translator(tb, vb).translate(
        _card(),
        lexicon=_lexicon(),
        concept_definitions={"Lloyd-Max quantization": "chooses levels minimizing MSE"},
        source_quotes={
            "quantized message passing": "we apply Lloyd-Max quantization to messages"
        },
    )
    user = tb.calls[0][1]["content"]
    assert "tensor factorization for LLM compression" in user  # home_field
    assert "rank truncation" in user  # familiar term
    assert "numerical linear algebra" in user  # analogy domain
    assert "TensorGPT" in user  # reader's paper
    assert "Quantization noise may act like weak priors" in user  # the card itself
    assert "chooses levels minimizing MSE" in user  # concept definition
    assert "we apply Lloyd-Max quantization to messages" in user  # verbatim quote


def test_verifier_receives_definition_and_quote_per_pair():
    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    _translator(tb, vb).translate(
        _card(),
        lexicon=_lexicon(),
        concept_definitions={"Lloyd-Max quantization": "chooses levels minimizing MSE"},
        source_quotes={
            "quantized message passing": "we apply Lloyd-Max quantization to messages"
        },
    )
    assert len(vb.calls) == 1  # ONE batched completion per card
    user = vb.calls[0][1]["content"]
    assert "Lloyd-Max quantization" in user and "optimal rounding" in user
    assert "both pick levels minimizing squared error" in user  # rationale
    assert "chooses levels minimizing MSE" in user  # definition
    assert "we apply Lloyd-Max quantization to messages" in user  # verbatim quote


# --- acceptance + record schema --------------------------------------------------------------
def test_confirmed_translation_ships_with_records_and_pre_translation():
    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert out["headline"] == "Rounding noise may act like weak priors in belief propagation."
    records = out["translations"]
    assert len(records) == 1
    # FROZEN record schema (downstream consumers depend on these exact keys).
    assert set(records[0]) == {
        "original", "familiar", "first_use_rendering", "faithfulness", "rationale",
    }
    assert records[0]["faithfulness"] == "confirmed"
    assert records[0]["first_use_rendering"] == "optimal rounding (Lloyd-Max quantization)"
    assert out["pre_translation"]["headline"] == original["headline"]  # originals verbatim
    assert "pre_translation" not in original  # input card untouched


def test_approximate_translation_ships_with_mark():
    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_APPROXIMATE))
    out = _translator(tb, vb).translate(_card(), lexicon=_lexicon())
    assert out["translations"][0]["faithfulness"] == "approximate"
    assert out["headline"].startswith("Rounding noise")  # approximate still ships


def test_key_terms_keep_original_canonical_term():
    renamed = _translated()
    renamed["key_terms"] = [
        {
            "term": "optimal rounding",  # translator (wrongly) renamed the canonical entry
            "plain_meaning": "A way of choosing rounding levels minimizing error.",
            "source": "Quantized message passing",
            "familiar_gloss": "like rank truncation, it keeps what matters most",
        }
    ]
    tb = _FakeBackend(json.dumps(renamed))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    out = _translator(tb, vb).translate(_card(), lexicon=_lexicon())
    assert out["key_terms"][0]["term"] == "Lloyd-Max quantization"  # canonical restored
    assert out["key_terms"][0]["familiar_gloss"] == (
        "like rank truncation, it keeps what matters most"
    )


# --- rejection / retry / identity -------------------------------------------------------------
def test_rejected_pair_retries_once_then_identity():
    tb = _FakeBackend(json.dumps(_translated()), json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_REJECTED), json.dumps(_VERDICT_REJECTED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert len(tb.calls) == 2 and len(vb.calls) == 2  # exactly ONE retry
    assert "failed the faithfulness check" in tb.calls[1][1]["content"]  # excluded pair named
    for key in ("headline", "what_might_be_happening", "key_terms"):
        assert out[key] == original[key]  # identity card: no unfaithful analogy ships
    assert out["translations"][0]["faithfulness"] == "rejected"  # attempt stays auditable
    assert "pre_translation" not in out


def test_rejected_then_confirmed_retry_ships():
    retry = _translated(
        headline="Quantization noise may act like weak priors in belief propagation.",
    )
    tb = _FakeBackend(json.dumps(_translated()), json.dumps(retry))
    vb = _FakeBackend(json.dumps(_VERDICT_REJECTED), json.dumps(_VERDICT_CONFIRMED))
    out = _translator(tb, vb).translate(_card(), lexicon=_lexicon())
    assert out["what_might_be_happening"].startswith("Optimal rounding")
    faiths = sorted(r["faithfulness"] for r in out["translations"])
    assert faiths == ["confirmed", "rejected"]  # retry verdict + first attempt kept auditable


# --- deterministic guards ---------------------------------------------------------------------
def test_first_use_guard_requires_original_term_in_prose():
    missing = _translated(
        what_might_be_happening="Optimal rounding adds noise that behaves like uncertainty."
    )
    tb = _FakeBackend(json.dumps(missing))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    # the original term appears ONLY in key_terms (canonical restore) — that never satisfies
    # the first-use policy: the guard scans prose, so this is still an identity card.
    assert out["headline"] == original["headline"]
    assert out["what_might_be_happening"] == original["what_might_be_happening"]


def test_first_use_guard_accepts_original_term_inside_list_fields():  # verify-r regression
    translated = _translated(
        what_might_be_happening="Optimal rounding adds noise that behaves like uncertainty."
    )
    translated["next_steps"] = [
        "Compare against plain Lloyd-Max quantization at matched bitwidths.",
        "Measure posterior error on both.",
    ]
    tb = _FakeBackend(json.dumps(translated))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    out = _translator(tb, vb).translate(_card(), lexicon=_lexicon())
    # the original term lives in a LIST field (next_steps) — the r-run guard wrongly
    # reverted this shape; it must ship.
    assert out["what_might_be_happening"].startswith("Optimal rounding")
    assert out["translations"][0]["faithfulness"] == "confirmed"


def test_headline_guard_restores_original_over_20_words():
    long_headline = " ".join(["word"] * 21) + " may matter."
    tb = _FakeBackend(json.dumps(_translated(headline=long_headline)))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert out["headline"] == original["headline"]  # restored deterministically
    assert out["what_might_be_happening"].startswith("Optimal rounding")  # rest kept


def test_malformed_translator_output_is_identity_with_no_verifier_call():
    tb = _FakeBackend("not json at all")
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert out == original
    assert vb.calls == []  # nothing to verify


def test_malformed_verifier_output_is_identity():
    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend("garbage")
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert out == original  # unverifiable -> reject-when-in-doubt -> identity


def test_translator_with_no_pairs_is_identity():
    no_pairs = _translated(translations=[])
    tb = _FakeBackend(json.dumps(no_pairs))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    original = _card()
    out = _translator(tb, vb).translate(original, lexicon=_lexicon())
    assert out == original  # rewritten prose without declared pairs is unauditable
    assert vb.calls == []


# --- redundancy-fix: passage-cap (shared PASSAGE_PROMPT_MAX_CHARS) ----------------------------
def test_quote_max_chars_shares_the_graph_config_passage_cap():
    from src import graph_config_defaults as gcd
    from src.reader_translation import _QUOTE_MAX_CHARS

    assert _QUOTE_MAX_CHARS == gcd.PASSAGE_PROMPT_MAX_CHARS


def test_quote_for_source_truncates_long_quotes_to_the_shared_cap():  # byte-identical regression
    from src.reader_translation import _quote_for_source

    long_quote = "w" * 5000
    index = {"Quantized Message Passing": long_quote}
    quote = _quote_for_source("Quantized Message Passing", index)
    assert quote == long_quote[:4000]
    assert len(quote) == 4000


# --- source_quote_index -----------------------------------------------------------------------
def test_source_quote_index_maps_lowered_titles_to_quotes():
    from src.reader_translation import source_quote_index

    class _Ev:
        def __init__(self, title, quote):
            self.title = title
            self.quote = quote

    evidence = [
        _Ev("Quantized Message Passing", "we quantize messages"),
        {"title": "Loopy BP Study", "quote": "loopy graphs converge"},
        _Ev("", "no title, skipped"),
        _Ev("Quantized Message Passing", "duplicate title, first wins"),
    ]
    idx = source_quote_index(evidence)
    assert idx == {
        "quantized message passing": "we quantize messages",
        "loopy bp study": "loopy graphs converge",
    }


def test_translate_all_translates_each_card():
    from src.reader_translation import translate_all

    tb = _FakeBackend(json.dumps(_translated()))
    vb = _FakeBackend(json.dumps(_VERDICT_CONFIRMED))
    out = translate_all(_translator(tb, vb), {"h1": _card(), "h2": {}}, lexicon=_lexicon())
    assert out["h1"]["translations"][0]["faithfulness"] == "confirmed"
    assert out["h2"] == {}  # empty card passthrough inside the fan-out
