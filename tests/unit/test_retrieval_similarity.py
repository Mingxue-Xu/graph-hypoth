"""Unit tests for the shared deterministic similarity primitives.

`retrieval/similarity.py` is the single source of truth for pure scoring primitives reused by the
coherence and association paths. Tests use fixed inputs with no model load or network access.
"""

from __future__ import annotations

import math

import pytest

from src.retrieval import scoring_defaults as sd
from src.retrieval import similarity as sim
from src.state import RetrievedEvidence


def _evidence(evidence_id: str, *, source_id=None, metadata=None) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        source="openalex",
        source_id=source_id,
        title="t",
        quote="",
        relevance="r",
        retrieved_by="builder",
        tool_call_id="tc",
        rank=1,
        trust_tier="indexed_metadata",
        metadata=metadata or {},
    )


# --- clip01: canonical shared clamp helper -----------------------------------------------

def test_clip01_passes_through_unit_interval() -> None:
    assert sim.clip01(0.3) == pytest.approx(0.3, abs=1e-6)


def test_clip01_clamps_below_zero_and_above_one() -> None:
    assert sim.clip01(-0.2) == 0.0
    assert sim.clip01(1.5) == 1.0


def test_clip01_boundary_values_are_exact() -> None:
    assert sim.clip01(0.0) == 0.0
    assert sim.clip01(1.0) == 1.0


# --- Cosine clamp ----------------------------------------------------------------------

def test_cosine_identical_vectors_is_one() -> None:
    assert sim._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_pair_clamps_to_zero() -> None:
    # Clamping cosine to [0, 1] gives an orthogonal pair 0, not the 0.5 produced by rescaling.
    assert sim._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_negative_cosine_clamps_to_zero() -> None:  # negative-cosine edge case
    # Opposite vectors: cos = -1 -> clamp to 0, never negative.
    assert sim._cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_empty_or_mismatched_is_zero() -> None:
    assert sim._cosine([], [1.0]) == 0.0
    assert sim._cosine([1.0, 2.0], [1.0]) == 0.0
    assert sim._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero-norm guard


# --- Tokenization and normalization with stopwords ------------------------------------

def test_tokenize_lowercases_alnum_tokens() -> None:
    assert sim._tokenize("Statin Therapy, 2024!") == {"statin", "therapy", "2024"}
    assert sim._tokenize(None) == set()


def test_normalized_terms_subtracts_stopwords() -> None:
    # "the"/"of"/"and" are in the canonical scoring stopword list; content stays.
    terms = sim.normalized_terms("the therapy of statins and mortality")
    assert "the" not in terms and "of" not in terms and "and" not in terms
    assert {"therapy", "statins", "mortality"} <= terms
    # every removed token is a stopword from scoring_defaults (single source).
    assert "therapy" not in sd.STOPWORDS


# --- Jaccard and lexical similarity ------------------------------------------------------

def test_jaccard_of_token_sets() -> None:
    assert sim.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3, abs=1e-6)
    assert sim.jaccard(set(), {"a"}) == 0.0
    assert sim.jaccard({"a"}, {"a"}) == pytest.approx(1.0, abs=1e-6)


def test_lexical_similarity_is_jaccard_over_raw_tokens() -> None:
    # coherence-compat string form: Jaccard over RAW tokens (no stopword removal).
    assert sim._lexical_similarity("a b c", "b c d") == pytest.approx(2 / 4, abs=1e-6)
    assert sim._lexical_similarity("", "a") == 0.0


# --- Identifier normalization and citation overlap -----------------------------------

def test_normalize_identifier_strips_prefixes_and_lowercases() -> None:
    assert sim._normalize_identifier("https://doi.org/10.X/A") == "10.x/a"
    assert sim._normalize_identifier("https://openalex.org/W1") == "w1"
    assert sim._normalize_identifier(None) is None
    assert sim._normalize_identifier("   ") is None


def test_identifiers_of_spans_all_known_ids() -> None:
    item = _evidence(
        "ev_000001",
        source_id="10.X/A",
        metadata={"doi": "10.X/A", "openalex_id": "https://openalex.org/W1"},
    )
    item.external_ids = {"pmid": "12345"}
    assert sim.identifiers_of(item) == {"10.x/a", "w1", "12345"}


def test_references_of_normalized_from_metadata() -> None:
    item = _evidence("ev_000001", metadata={"references": ["https://doi.org/10.X/B", "W2", None]})
    assert sim.references_of(item) == {"10.x/b", "w2"}


def test_overlap_bibliographic_coupling(  #  coupling leg
) -> None:
    # Two papers sharing 2 of 3 union references -> coupling = 2/3.
    refs_a, refs_b = {"d1", "d2", "d9"}, {"d1", "d2"}
    assert sim.overlap(refs_a, set(), refs_b, set()) == pytest.approx(2 / 3, abs=1e-6)


def test_overlap_co_citation_is_one(  #  co-citation leg
) -> None:
    # A references B's id -> co-citation 1.0 dominates the (zero) coupling.
    assert sim.overlap({"w2"}, {"w1"}, set(), {"w2"}) == pytest.approx(1.0, abs=1e-6)


def test_overlap_disjoint_is_zero() -> None:
    assert sim.overlap({"d1"}, {"a"}, {"d2"}, {"b"}) == 0.0


def test_overlap_is_symmetric() -> None:
    a = sim.overlap({"w2"}, {"w1"}, set(), {"w2"})
    b = sim.overlap(set(), {"w2"}, {"w2"}, {"w1"})
    assert math.isclose(a, b, abs_tol=1e-6)
