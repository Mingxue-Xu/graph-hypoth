"""Presence and invariant tests for retrieval and verification scoring defaults.

These guards keep coefficient versions auditable, domain defaults separated, and
the stopword, synonym, and embedding-model inputs reproducible.
"""

import pytest

from src.retrieval import scoring_defaults as sd


def test_version_stamps_present_and_nonempty():
    # The three version stamps that make every stamped output recomputable.
    assert sd.SCORING_DEFAULTS_VERSION == "scoring-defaults-2026-08-30"
    assert isinstance(sd.STOPWORDS_VERSION, str) and sd.STOPWORDS_VERSION
    assert isinstance(sd.SYNONYM_LEXICON_VERSION, str) and sd.SYNONYM_LEXICON_VERSION


def test_stopwords_invariant_318():
    assert isinstance(sd.STOPWORDS, frozenset)
    assert len(sd.STOPWORDS) == 318  # exact sklearn English stopword count


def test_synonym_lexicon_is_frozen_rows():
    assert isinstance(sd.SYNONYM_LEXICON, dict)
    assert sd.SYNONYM_LEXICON  # non-empty seed so the channel is exercised day one
    assert all(isinstance(v, frozenset) for v in sd.SYNONYM_LEXICON.values())


def test_embedding_model_and_revision_pinned():
    # The adapter id `allenai/specter2` silently lexical-falls-back; pin `_base`.
    assert sd.EMBEDDING_MODEL == "allenai/specter2_base"
    assert isinstance(sd.EMBEDDING_REVISION, str) and sd.EMBEDDING_REVISION  # non-empty HF SHA


def test_doc_pinned_weight_sums_are_one():  # documented weights
    approx_one = pytest.approx(1.0, abs=1e-6)
    # combined association = semantic + citation + source_prior
    assert sum(sd.ASSOCIATION_WEIGHTS.values()) == approx_one
    # semantic = embedding + lexical
    assert sum(sd.SEMANTIC_WEIGHTS.values()) == approx_one
    # legacy claim-scoped relatedness = embedding + citation
    assert sum(sd.LEGACY_RELATEDNESS_WEIGHTS.values()) == approx_one
    # Conf POSITIVE coefficients sum to 1.0; the two penalties are negative offsets.
    conf_positives = sum(w for w in sd.CONF_WEIGHTS.values() if w > 0)
    assert conf_positives == approx_one


def test_audit_dict_surfaces_every_constant():  # reported in audit
    audit = sd.audit_dict()
    assert audit["scoring_defaults_version"] == sd.SCORING_DEFAULTS_VERSION
    # Every module-level constant MUST be surfaced in audit, EXCEPT the two bulky
    # bodies (STOPWORDS / SYNONYM_LEXICON), which are represented by their version
    # stamps so audit output stays compact yet reproducible.
    bulky_bodies = {"STOPWORDS", "SYNONYM_LEXICON"}
    module_constants = {n for n in vars(sd) if n.isupper()}
    for name in module_constants - bulky_bodies:
        assert name.lower() in audit, f"{name} is not surfaced in audit_dict()"
    # The bulky bodies are NOT dumped wholesale into audit.
    assert "stopwords" not in audit
    assert "synonym_lexicon" not in audit
