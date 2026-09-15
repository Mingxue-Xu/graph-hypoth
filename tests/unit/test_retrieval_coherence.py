from __future__ import annotations

import logging
from types import SimpleNamespace

from src.retrieval.coherence import OPENROUTER_API_BASE_URL
from src.retrieval import scoring_defaults as sd
from src.retrieval.coherence import (
    EvidenceEnricher,
    JudgeVerdict,
    _lexical_similarity,
    load_specter2_embedder,
)
from src.retrieval.ledger import EvidenceLedger
from src.state import RetrievedEvidence


def _config(**overrides):
    base = {
        "enabled": True,
        "embedding_enabled": True,
        "embedding_model": "allenai/specter2",
        "embedding_weight": 0.6,
        "citation_enabled": True,
        "citation_weight": 0.4,
        "judge_enabled": False,
        "judge_top_k": 5,
        "judge_model": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _evidence(evidence_id, *, title, quote="", source_id=None, metadata=None):
    return RetrievedEvidence(
        evidence_id=evidence_id,
        source="openalex",
        source_id=source_id,
        title=title,
        quote=quote,
        relevance="r",
        retrieved_by="builder",
        tool_call_id="t",
        rank=0,
        trust_tier="indexed_metadata",
        metadata=metadata or {},
    )


def test_enricher_default_embedding_model_is_loadable_base(monkeypatch) -> None:  # loadable default
    # When the config omits embedding_model the enricher must default to the loadable
    # `allenai/specter2_base`, NOT the bare PEFT-adapter id that silently lexical-falls-back.
    captured: dict[str, str] = {}

    def fake_loader(model_name: str, revision=None):
        captured["model"] = model_name
        return lambda texts: [[0.0] for _ in texts]

    monkeypatch.setattr(
        "src.retrieval.coherence.load_specter2_embedder", fake_loader
    )
    EvidenceEnricher(SimpleNamespace(embedding_enabled=True))

    assert captured["model"] == "allenai/specter2_base"
    assert captured["model"] == sd.EMBEDDING_MODEL  # must track scoring_defaults, not restate it


def test_load_specter2_embedder_pins_the_hf_revision(monkeypatch) -> None:  # revision pinning
    # The pinned HF revision (scoring_defaults.EMBEDDING_REVISION) MUST reach the
    # SentenceTransformer loader, not be silently dropped — else the embedding channel
    # can drift to an unpinned model snapshot and split from the citation/lexical legs.
    import sentence_transformers

    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            captured["model"] = model_name
            captured["revision"] = kwargs.get("revision")

        def encode(self, texts, normalize_embeddings=False):
            return [[0.0] for _ in texts]

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    load_specter2_embedder.cache_clear()
    try:
        embedder = load_specter2_embedder(sd.EMBEDDING_MODEL, revision=sd.EMBEDDING_REVISION)
        assert embedder is not None
        assert captured["revision"] == sd.EMBEDDING_REVISION
    finally:
        load_specter2_embedder.cache_clear()


def test_enricher_passes_config_embedding_revision_to_loader(monkeypatch) -> None:  # revision forwarding
    # The enricher MUST forward the config's pinned embedding_revision into the loader
    # (the revision was pinned as data on config but never threaded to the loader).
    captured: dict[str, object] = {}

    def fake_loader(model_name, revision=None):
        captured["model"] = model_name
        captured["revision"] = revision
        return lambda texts: [[0.0] for _ in texts]

    monkeypatch.setattr(
        "src.retrieval.coherence.load_specter2_embedder", fake_loader
    )
    EvidenceEnricher(
        SimpleNamespace(
            embedding_enabled=True,
            embedding_model=sd.EMBEDDING_MODEL,
            embedding_revision=sd.EMBEDDING_REVISION,
        )
    )
    assert captured["revision"] == sd.EMBEDDING_REVISION


def test_embedding_channel_uses_lexical_fallback_when_no_embedder() -> None:
    # No embedder injected and embedding disabled-load -> lexical similarity.
    enricher = EvidenceEnricher(_config(embedding_enabled=False), embedder=None)
    items = [
        _evidence("ev_000001", title="graph neural networks for materials"),
        _evidence("ev_000002", title="completely unrelated cooking recipe"),
    ]

    enriched = enricher.enrich("graph neural networks materials discovery", items)

    s1 = enriched[0].relatedness_score
    s2 = enriched[1].relatedness_score
    assert s1 is not None and s2 is not None
    assert s1 > s2  # topically closer item scores higher


def test_injected_embedder_drives_relatedness_score() -> None:
    # Deterministic stub embedder: claim and item0 identical vectors -> cosine 1.
    vectors = {
        "claim": [1.0, 0.0],
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
    }

    def embedder(texts):
        out = [vectors["claim"]]
        out.append(vectors["a"])
        out.append(vectors["b"])
        return out

    enricher = EvidenceEnricher(
        _config(citation_enabled=False, embedding_weight=1.0, citation_weight=0.0),
        embedder=embedder,
    )
    items = [_evidence("ev_000001", title="a"), _evidence("ev_000002", title="b")]

    enriched = enricher.enrich("claim", items)

    assert enriched[0].relatedness_score == 1.0
    assert enriched[1].relatedness_score == 0.0


def test_citation_channel_detects_bibliographic_coupling_model_free() -> None:
    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, embedding_weight=0.0, citation_weight=1.0),
        embedder=None,
    )
    # ev1 and ev2 share two references; ev3 shares none -> ev1/ev2 are neighbors.
    items = [
        _evidence("ev_000001", title="p1", metadata={"references": ["d1", "d2", "d9"]}),
        _evidence("ev_000002", title="p2", metadata={"references": ["d1", "d2"]}),
        _evidence("ev_000003", title="p3", metadata={"references": ["d7"]}),
    ]

    enriched = enricher.enrich("claim", items)

    by_id = {item.evidence_id: item for item in enriched}
    assert by_id["ev_000001"].relatedness_score > 0
    assert by_id["ev_000002"].citation_neighbors == ["ev_000001"]
    assert by_id["ev_000003"].relatedness_score == 0.0
    assert by_id["ev_000003"].citation_neighbors is None


def test_co_citation_within_openalex_namespace_scores_above_zero() -> None:
    # Two OpenAlex items that BOTH carry a DOI identity, but whose references are
    # stored as OpenAlex work-ID URLs (the adapter populates references from
    # `referenced_works`). A references B's openalex_id. Since identity must span
    # ALL known identifiers (DOI + openalex_id + source_id), B's openalex_id has
    # to match A's reference set even though the DOI is the "primary" id.
    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, embedding_weight=0.0, citation_weight=1.0),
        embedder=None,
    )
    items = [
        _evidence(
            "ev_000001",
            title="citing",
            source_id="10.x/a",
            metadata={
                "doi": "10.x/a",
                "openalex_id": "https://openalex.org/W1",
                "references": ["https://openalex.org/W2"],
            },
        ),
        _evidence(
            "ev_000002",
            title="cited",
            source_id="10.x/b",
            metadata={
                "doi": "10.x/b",
                "openalex_id": "https://openalex.org/W2",
                "references": [],
            },
        ),
    ]

    enriched = enricher.enrich("claim", items)

    by_id = {item.evidence_id: item for item in enriched}
    # The cited item (B) is referenced by A via its openalex_id -> co-citation > 0.
    assert by_id["ev_000002"].relatedness_score > 0
    # And symmetrically the citing item also sees the coupling.
    assert by_id["ev_000001"].relatedness_score > 0


def test_co_citation_within_doi_namespace_scores_above_zero() -> None:
    # DOI-based pair (Crossref-style): A references B's DOI. The reference is a
    # prefixed/upper-cased DOI URL, so identity and reference normalization must
    # be symmetric (strip https://doi.org/, lowercase) for the match to land.
    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, embedding_weight=0.0, citation_weight=1.0),
        embedder=None,
    )
    item_a = _evidence(
        "ev_000001",
        title="citing",
        source_id="10.X/A",
        metadata={"doi": "10.X/A", "references": ["https://doi.org/10.X/B"]},
    )
    item_a.external_ids = {"doi": "10.x/a"}
    item_b = _evidence(
        "ev_000002",
        title="cited",
        source_id="10.X/B",
        metadata={"doi": "10.X/B", "references": []},
    )
    item_b.external_ids = {"doi": "10.x/b"}

    enriched = enricher.enrich("claim", [item_a, item_b])

    by_id = {item.evidence_id: item for item in enriched}
    assert by_id["ev_000002"].relatedness_score > 0
    assert by_id["ev_000001"].relatedness_score > 0


def test_citation_neighbors_metadata_is_independent_copy() -> None:
    # citation_neighbors stored on the item and in metadata["r2"] must not alias
    # the same list object (mutating one would silently corrupt the other).
    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, embedding_weight=0.0, citation_weight=1.0),
        embedder=None,
    )
    items = [
        _evidence("ev_000001", title="p1", metadata={"references": ["d1", "d2"]}),
        _evidence("ev_000002", title="p2", metadata={"references": ["d1", "d2"]}),
    ]

    enriched = enricher.enrich("claim", items)

    item = enriched[0]
    assert item.citation_neighbors is not None
    assert item.citation_neighbors == item.metadata["r2"]["citation_neighbors"]
    assert item.citation_neighbors is not item.metadata["r2"]["citation_neighbors"]


def test_judge_empty_choices_does_not_raise_and_leaves_items_unjudged(
    caplog,
) -> None:
    # A judge whose underlying model returns empty choices / non-JSON must not
    # raise out of enrich; items stay un-judged and the failure is logged.
    def judge(claim, items):
        # Simulate _build_llm_judge's guarded behavior: an empty/malformed model
        # response must surface as a failure, not a silent empty map.
        raise IndexError("list index out of range")

    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, judge_top_k=1), embedder=None, judge=judge
    )
    items = [_evidence("ev_000001", title="t", quote="q")]

    with caplog.at_level(logging.WARNING):
        [enriched] = enricher.enrich("claim", items)

    assert enriched.relation_label is None
    assert any(
        record.levelno >= logging.WARNING and "judge" in record.getMessage().lower()
        for record in caplog.records
    )


def test_build_llm_judge_handles_empty_choices_and_bad_json(monkeypatch) -> None:
    # The model-backed judge must not raise on empty `choices` or non-JSON
    # content; it returns an empty verdict map in both cases.
    import src.retrieval.coherence as coherence_mod

    class _FakeCompletions:
        def __init__(self, response):
            self._response = response

        def create(self, **_kwargs):
            return self._response

    class _FakeClient:
        def __init__(self, response):
            self.chat = SimpleNamespace(completions=_FakeCompletions(response))

    def _build_with_response(response):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(
            "openai.OpenAI", lambda *a, **k: _FakeClient(response), raising=True
        )
        model_config = SimpleNamespace(model_id="m", api_key_env="OPENROUTER_API_KEY")
        return coherence_mod._build_llm_judge(model_config)

    item = _evidence("ev_000001", title="t", quote="q")

    empty_choices = SimpleNamespace(choices=[])
    judge_empty = _build_with_response(empty_choices)
    assert judge_empty is not None
    assert judge_empty("claim", [item]) == {}

    bad_json = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
    )
    judge_bad = _build_with_response(bad_json)
    assert judge_bad is not None
    assert judge_bad("claim", [item]) == {}


def test_build_llm_judge_uses_shared_openrouter_base_url(
    monkeypatch,
) -> None:
    # The builder uses the module's one OpenRouter URL constant.
    import src.retrieval.coherence as coherence_mod

    captured: dict[str, str] = {}

    class _FakeClient:
        def __init__(self, *, api_key, base_url):
            captured["base_url"] = base_url

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", _FakeClient, raising=True)
    model_config = SimpleNamespace(model_id="m", api_key_env="OPENROUTER_API_KEY")
    coherence_mod._build_llm_judge(model_config)

    assert captured["base_url"] == OPENROUTER_API_BASE_URL


def test_build_llm_judge_warns_on_valid_json_wrong_shape(monkeypatch, caplog) -> None:
    # Valid JSON that does not match the verdict schema (e.g. a dict without a
    # "verdicts" list) must return {} AND log a warning, so a systematic schema
    # mismatch is observable instead of being silently dropped.
    import logging

    import src.retrieval.coherence as coherence_mod

    class _FakeCompletions:
        def __init__(self, response):
            self._response = response

        def create(self, **_kwargs):
            return self._response

    class _FakeClient:
        def __init__(self, response):
            self.chat = SimpleNamespace(completions=_FakeCompletions(response))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    wrong_shape = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"foo": "bar"}'))]
    )
    monkeypatch.setattr(
        "openai.OpenAI", lambda *a, **k: _FakeClient(wrong_shape), raising=True
    )
    model_config = SimpleNamespace(model_id="m", api_key_env="OPENROUTER_API_KEY")
    judge = coherence_mod._build_llm_judge(model_config)
    assert judge is not None

    item = _evidence("ev_000001", title="t", quote="q")
    with caplog.at_level(logging.WARNING):
        assert judge("claim", [item]) == {}
    assert any("no usable verdicts" in record.message for record in caplog.records)


def test_injected_judge_sets_relation_fields_on_top_k_only() -> None:
    def judge(claim, items):
        # Only label the single highest-ranked item.
        top = items[0]
        return {
            top.evidence_id: JudgeVerdict(
                relation_label="supports",
                domain_context="materials-science usage of 'stress'",
                synonym_links=["strain", "load"],
            )
        }

    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, judge_top_k=1), embedder=None, judge=judge
    )
    items = [
        _evidence("ev_000001", title="graph neural networks materials", quote="q1"),
        _evidence("ev_000002", title="unrelated", quote="q2"),
    ]

    enriched = enricher.enrich("graph neural networks materials", items)

    by_id = {item.evidence_id: item for item in enriched}
    labelled = by_id["ev_000001"]
    assert labelled.relation_label == "supports"
    assert labelled.domain_context == "materials-science usage of 'stress'"
    assert labelled.synonym_links == ["strain", "load"]


def test_lexical_similarity_basic() -> None:
    assert _lexical_similarity("alpha beta", "alpha beta") == 1.0
    assert _lexical_similarity("alpha beta", "gamma delta") == 0.0


def test_enrichment_survives_ledger_readmission_round_trip() -> None:
    # The service enriches evidence before the graph workflow readmits it into a fresh ledger
    # through add_retrieved -> _source_shadow -> add_many. Enrichment fields survive that
    # round-trip (they ride metadata["r2"], not the dropped top-level fields).
    def judge(claim, items):
        return {
            items[0].evidence_id: JudgeVerdict(
                relation_label="supports", domain_context="materials context"
            )
        }

    enricher = EvidenceEnricher(
        _config(embedding_enabled=False, judge_top_k=1), embedder=None, judge=judge
    )
    items = [
        _evidence(
            "ev_000001",
            title="graph neural networks",
            quote="q1",
            source_id="10.x/gnn",
            metadata={"references": ["d1", "d2"], "doi": "10.x/gnn"},
        ),
        _evidence(
            "ev_000002",
            title="other",
            quote="q2",
            source_id="10.x/other",
            metadata={"references": ["d1", "d2"], "doi": "10.x/other"},
        ),
    ]
    enriched = enricher.enrich("graph neural networks", items)

    fresh = EvidenceLedger(
        trust_policy={"openalex": "indexed_metadata"},
        trust_tier_priority={"indexed_metadata": 80},
        source_order=["openalex"],
    )
    fresh.add_retrieved(
        items=enriched,
        query="graph neural networks",
        retrieved_by="builder",
        tool_call_id="t",
    )

    readmitted = {item.title: item for item in fresh.all_evidence()}
    gnn = readmitted["graph neural networks"]
    assert gnn.relation_label == "supports"  # survived the re-admit round-trip
    assert gnn.domain_context == "materials context"
    assert gnn.relatedness_score is not None
    # neighbors stored by stable identity (doi), so they survive id renumbering
    assert gnn.citation_neighbors == ["10.x/other"]

    assert gnn.relation_label == "supports"
    assert gnn.domain_context == "materials context"
