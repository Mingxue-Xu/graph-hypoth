import pytest

from src.retrieval.ledger import MAX_QUOTE_CHARS, EvidenceLedger
from src.retrieval.models import SourceResult
from src.state import RetrievedEvidence


TRUST_POLICY = {
    "arxiv": "authoritative_preprint",
    "exa": "web_research_paper",
    "fake": "deterministic_test",
}
TRUST_TIER_PRIORITY = {
    "authoritative_preprint": 100,
    "deterministic_test": 100,
    "web_research_paper": 60,
}


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        trust_policy=TRUST_POLICY,
        trust_tier_priority=TRUST_TIER_PRIORITY,
        source_order=["arxiv", "exa", "fake"],
    )


def _source_result(
    *,
    source: str = "arxiv",
    source_id: str | None = "2401.00001",
    title: str = "Evidence Grounded Debate",
    published_date: str | None = "2024-01-02T00:00:00+00:00",
    url: str | None = "https://arxiv.org/abs/2401.00001",
    text: str | None = "A concise quote about evidence grounded debate.",
    score: float | None = None,
) -> SourceResult:
    return SourceResult(
        source=source,
        source_id=source_id,
        title=title,
        authors=["Ada Lovelace"],
        published_date=published_date,
        url=url,
        text=text,
        score=score,
    )


def test_ledger_assigns_run_local_sequential_evidence_ids() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            _source_result(source_id="2401.00001", title="First"),
            _source_result(source_id="2401.00002", title="Second"),
            _source_result(source_id="2401.00003", title="Third"),
        ],
        query="multi-agent verification",
        retrieved_by="builder",
        tool_call_id="call-1",
    )

    assert [item.evidence_id for item in ledger.all_evidence()] == [
        "ev_000001",
        "ev_000002",
        "ev_000003",
    ]


def test_ledger_preserves_existing_ids_and_appends_next_id() -> None:
    existing = RetrievedEvidence(
        evidence_id="ev_000007",
        source="fake",
        source_id="existing",
        title="Existing",
        quote="existing quote",
        relevance="existing",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000001",
        rank=1,
        trust_tier="deterministic_test",
        metadata={"source_rank": 1},
    )
    ledger = EvidenceLedger(
        trust_policy={"fake": "deterministic_test"},
        trust_tier_priority={"deterministic_test": 100},
        source_order=["fake"],
        existing=[existing],
    )

    added = ledger.add_many(
        results=[
            SourceResult(
                source="fake",
                source_id="new",
                title="New",
                text="new quote",
                score=1.0,
            )
        ],
        query="query",
        retrieved_by="builder",
        tool_call_id="tool_000002",
    )

    assert existing.evidence_id == "ev_000007"
    assert added[0].evidence_id == "ev_000008"


def test_add_retrieved_ignores_call_local_ids_and_appends_provenance() -> None:
    ledger = _ledger()
    ledger.add_many(
        results=[_source_result(source_id="shared", title="Shared")],
        query="initial",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000001",
    )
    incoming = RetrievedEvidence(
        evidence_id="ev_000001",
        source="arxiv",
        source_id="debate-only",
        title="Debate Only",
        quote="debate quote",
        relevance="debate relevance",
        retrieved_by="builder",
        tool_call_id="tool_000002",
        score=0.7,
        rank=1,
        trust_tier="authoritative_preprint",
        metadata={"source_rank": 1},
    )

    added = ledger.add_retrieved(
        items=[incoming],
        query="debate",
        retrieved_by="builder",
        tool_call_id="tool_000002",
    )

    assert sorted(item.evidence_id for item in ledger.all_evidence()) == [
        "ev_000001",
        "ev_000002",
    ]
    assert added[0].evidence_id == "ev_000002"


def test_ledger_reindexes_existing_items_with_duplicate_call_local_ids() -> None:
    first = _ledger()
    first.add_many(
        results=[_source_result(source_id="paper-a", title="Paper A")],
        query="initial query",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000001",
    )
    second = _ledger()
    second.add_many(
        results=[_source_result(source_id="paper-b", title="Paper B")],
        query="debate query",
        retrieved_by="builder",
        tool_call_id="tool_000002",
    )

    merged = EvidenceLedger(
        trust_policy=TRUST_POLICY,
        trust_tier_priority=TRUST_TIER_PRIORITY,
        source_order=["arxiv", "exa", "fake"],
        existing=[*first.all_evidence(), *second.all_evidence()],
    )

    assert [item.evidence_id for item in merged.all_evidence()] == [
        "ev_000001",
        "ev_000002",
    ]


def test_add_retrieved_dedupes_existing_source_without_reusing_call_local_id() -> None:
    ledger = _ledger()
    ledger.add_many(
        results=[_source_result(source_id="shared", title="Shared")],
        query="initial",
        retrieved_by="retrieve_evidence",
        tool_call_id="tool_000001",
    )
    incoming = RetrievedEvidence(
        evidence_id="ev_000001",
        source="arxiv",
        source_id="shared",
        title="Shared",
        quote="same source quote",
        relevance="debate relevance",
        retrieved_by="builder",
        tool_call_id="tool_000002",
        score=0.7,
        rank=1,
        trust_tier="authoritative_preprint",
        metadata={"source_rank": 1},
    )

    admitted = ledger.add_retrieved(
        items=[incoming],
        query="debate",
        retrieved_by="builder",
        tool_call_id="tool_000002",
    )

    assert len(ledger.all_evidence()) == 1
    assert admitted[0].evidence_id == "ev_000001"
    assert [entry["tool_call_id"] for entry in admitted[0].metadata["provenance"]] == [
        "tool_000001",
        "tool_000002",
    ]


def test_ledger_dedupes_by_source_id_and_appends_provenance() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[_source_result(source_id="2401.00001", title="Original")],
        query="first query",
        retrieved_by="builder",
        tool_call_id="call-1",
    )
    returned = ledger.add_many(
        results=[_source_result(source_id="2401.00001", title="Updated")],
        query="second query",
        retrieved_by="verifier",
        tool_call_id="call-2",
    )

    evidence = ledger.all_evidence()
    assert len(evidence) == 1
    assert returned == evidence
    assert [entry["tool_call_id"] for entry in evidence[0].metadata["provenance"]] == [
        "call-1",
        "call-2",
    ]


def test_ledger_dedupes_by_url_when_source_id_is_missing() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            _source_result(
                source_id=None,
                url="https://example.org/paper/",
                title="First URL title",
            )
        ],
        query="first query",
        retrieved_by="builder",
        tool_call_id="call-1",
    )
    ledger.add_many(
        results=[
            _source_result(
                source_id=None,
                url="https://EXAMPLE.org/paper",
                title="Second URL title",
            )
        ],
        query="second query",
        retrieved_by="verifier",
        tool_call_id="call-2",
    )

    assert len(ledger.all_evidence()) == 1
    assert len(ledger.all_evidence()[0].metadata["provenance"]) == 2


def test_ledger_dedupes_by_normalized_title_and_publication_year_without_ids() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            _source_result(
                source_id=None,
                url=None,
                title="Evidence   Grounded Debate",
                published_date="2024-01-02T00:00:00+00:00",
            )
        ],
        query="first query",
        retrieved_by="builder",
        tool_call_id="call-1",
    )
    ledger.add_many(
        results=[
            _source_result(
                source_id=None,
                url=None,
                title=" evidence grounded debate ",
                published_date="2024-12-30",
            )
        ],
        query="second query",
        retrieved_by="verifier",
        tool_call_id="call-2",
    )

    assert len(ledger.all_evidence()) == 1
    assert len(ledger.all_evidence()[0].metadata["provenance"]) == 2


def _keyword_embedder(keyword: str):
    """Deterministic stub embedder: any text containing ``keyword`` maps to the
    claim's vector (cosine 1.0); everything else is orthogonal (cosine 0.0).
    Network-free, mirroring the coherence stub-embedder convention."""

    def embed(texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if keyword in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    return embed


def test_score_relevance_makes_relevant_low_trust_outrank_irrelevant_high_trust() -> None:
    # Relevance-first ranking: an on-topic web_research_paper (trust 60) must outrank
    # an off-topic authoritative_preprint (trust 100) once claim-relevance is scored.
    ledger = _ledger()
    ledger.add_many(
        results=[
            _source_result(
                source="arxiv",
                source_id="arxiv-offtopic",
                title="Medical Imaging Segmentation",
                url="https://arxiv.org/abs/2401.99999",
                text="radiology scan annotation pipeline",
            ),
            _source_result(
                source="exa",
                source_id="exa-ontopic",
                title="Efficient Transformer Vision-Language Models",
                url="https://example.com/efficient-transformer",
                text="transformer token pruning for efficiency",
            ),
        ],
        query="claim",
        retrieved_by="builder",
        tool_call_id="call-1",
    )
    # Trust-first before scoring: the arxiv preprint (tier 100) leads.
    assert ledger.all_evidence()[0].source_id == "arxiv-offtopic"

    ledger.score_relevance(_keyword_embedder("transformer"), claim="transformer efficiency")

    ranked = ledger.all_evidence()
    assert ranked[0].source_id == "exa-ontopic"
    assert ranked[0].metadata["claim_relevance"] == pytest.approx(1.0, abs=1e-6)
    assert ranked[1].metadata["claim_relevance"] == pytest.approx(0.0, abs=1e-6)


def test_rrf_score_breaks_ties_after_relevance_and_tier() -> None:
    # Two same-tier papers, no relevance scored: the one with the higher RRF score
    # (surfaced by more fan-out sub-queries) ranks first. Titles are arranged so the
    # plain title tiebreak would REVERSE the order — this passes only because RRF wins.
    def _ev(eid: str, sid: str, title: str, rrf: float) -> RetrievedEvidence:
        return RetrievedEvidence(
            evidence_id=eid,
            source="exa",
            source_id=sid,
            title=title,
            quote="q",
            relevance="r",
            retrieved_by="b",
            tool_call_id="c",
            rank=1,
            trust_tier="web_research_paper",
            metadata={
                "source_rank": 1,
                "rrf_score": rrf,
                "quote_selection": {"match_type": "highlight"},
            },
        )

    ledger = EvidenceLedger(
        trust_policy=TRUST_POLICY,
        trust_tier_priority=TRUST_TIER_PRIORITY,
        source_order=["arxiv", "exa", "fake"],
        existing=[
            _ev("ev_000001", "low", "AAA sorts first", 0.01),
            _ev("ev_000002", "high", "ZZZ sorts last", 0.05),
        ],
    )
    ledger.drop_empty_quote_stubs()  # no stubs here; just triggers a re-rank

    assert [item.source_id for item in ledger.all_evidence()] == ["high", "low"]


def test_drop_empty_quote_stubs_removes_no_source_text_records() -> None:
    # Records whose quote_selection.match_type is "no_source_text" carry no usable
    # text for the miner and must be dropped before the final pool is returned.
    ledger = EvidenceLedger(
        trust_policy=TRUST_POLICY,
        trust_tier_priority=TRUST_TIER_PRIORITY,
        source_order=["arxiv", "exa", "fake"],
        existing=[
            RetrievedEvidence(
                evidence_id="ev_000001",
                source="exa",
                source_id="has-quote",
                title="Has Quote",
                quote="a real extracted quote",
                relevance="r",
                retrieved_by="b",
                tool_call_id="c",
                rank=1,
                trust_tier="web_research_paper",
                metadata={"quote_selection": {"match_type": "highlight"}},
            ),
            RetrievedEvidence(
                evidence_id="ev_000002",
                source="exa",
                source_id="empty-stub",
                title="Empty Stub",
                quote="",
                relevance="r",
                retrieved_by="b",
                tool_call_id="c",
                rank=2,
                trust_tier="web_research_paper",
                metadata={"quote_selection": {"match_type": "no_source_text"}},
            ),
        ],
    )

    ledger.drop_empty_quote_stubs()

    assert [item.source_id for item in ledger.all_evidence()] == ["has-quote"]


def test_ledger_ranks_by_trust_tier_source_order_source_rank_and_score() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            _source_result(
                source="exa",
                source_id="exa-low-trust",
                title="Low Trust High Score",
                score=1.0,
            ),
            _source_result(
                source="fake",
                source_id="fake-evidence",
                title="Fake Same Trust Later Source",
                score=0.99,
            ),
            _source_result(
                source="arxiv",
                source_id="arxiv-lower-score",
                title="Arxiv Lower Score",
                score=0.2,
            ),
            _source_result(
                source="arxiv",
                source_id="arxiv-higher-score",
                title="Arxiv Higher Score",
                score=0.9,
            ),
        ],
        query="ranking query",
        retrieved_by="builder",
        tool_call_id="call-1",
    )

    assert [(item.rank, item.source_id) for item in ledger.all_evidence()] == [
        (1, "arxiv-higher-score"),
        (2, "arxiv-lower-score"),
        (3, "fake-evidence"),
        (4, "exa-low-trust"),
    ]


def test_ledger_quotes_are_compact_and_truncated() -> None:
    ledger = _ledger()
    # Exceed the cap with unpunctuated text so the truncate fallback (not the sentence
    # window) bounds the quote; assert against the constant so it tracks the configured cap.
    long_text = "First line\n\n" + ("word " * (MAX_QUOTE_CHARS // 4 + 100))

    ledger.add_many(
        results=[
            _source_result(source_id="long", title="Long Quote", text=long_text),
        ],
        query="quote query",
        retrieved_by="builder",
        tool_call_id="call-1",
    )

    quote = ledger.all_evidence()[0].quote
    assert "\n" not in quote
    assert "  " not in quote
    assert len(quote) <= MAX_QUOTE_CHARS


def test_ledger_prefers_clean_exa_highlight_and_records_selection() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            SourceResult(
                source="exa",
                source_id="paper-highlight",
                title="Highlight Paper",
                text=(
                    "Context before. The selected highlight gives direct evidence "
                    "for the claim. It is complete. Context after."
                ),
                summary="Summary fallback should not be selected.",
                metadata={
                    "exa_highlights": [
                        "Stitched locator [ ... ] is not a strict quote.",
                        "The selected highlight gives direct evidence for the claim. "
                        "It is complete.",
                    ],
                    "exa_highlight_scores": [0.1, 0.9],
                },
            )
        ],
        query="direct evidence claim",
        retrieved_by="builder",
        tool_call_id="call-highlight",
    )

    evidence = ledger.all_evidence()[0]
    assert evidence.quote == (
        "The selected highlight gives direct evidence for the claim. It is complete."
    )
    assert evidence.metadata["quote_selection"] == {
        "source": "highlight",
        "highlight_index": 1,
        "query": "direct evidence claim",
        "candidate_excerpt": (
            "The selected highlight gives direct evidence for the claim. It is complete."
        ),
        "verified_quote": (
            "The selected highlight gives direct evidence for the claim. It is complete."
        ),
        "verification_status": "accepted",
        "match_type": "exact",
    }


def test_ledger_selects_query_relevant_text_window_before_summary() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            SourceResult(
                source="exa",
                source_id="paper-text",
                title="Text Paper",
                text=(
                    "A background sentence talks about unrelated setup. "
                    "Compression lowers perplexity for language models. "
                    "Another unrelated sentence follows."
                ),
                summary="Generated summary should not be selected.",
                metadata={"exa_highlights": ["Noisy [ ... ] stitched locator."]},
            )
        ],
        query="compression perplexity",
        retrieved_by="builder",
        tool_call_id="call-text",
    )

    evidence = ledger.all_evidence()[0]
    assert evidence.quote == "Compression lowers perplexity for language models."
    assert evidence.metadata["quote_selection"] == {
        "source": "text",
        "highlight_index": None,
        "query": "compression perplexity",
        "candidate_excerpt": "Compression lowers perplexity for language models.",
        "verified_quote": "Compression lowers perplexity for language models.",
        "verification_status": "accepted",
        "match_type": "exact",
    }


def test_ledger_text_quote_skips_keyword_front_matter() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            SourceResult(
                source="exa",
                source_id="paper-keywords",
                title="Context-Adaptive Coding Paper",
                text=(
                    "Context-Adaptive Coding Paper\n"
                    "Ada Lovelace\n"
                    "Abstract\n"
                    "This paper studies neural compression systems.\n"
                    "Keywords: attention, arithmetic coding, huffman coding, "
                    "context-adaptive models\n"
                    "1. Introduction\n"
                    "The main text explains that adaptive probability estimates "
                    "change with the observed context."
                ),
                summary="Generated summary should not be selected.",
            )
        ],
        query="attention arithmetic coding huffman context-adaptive",
        retrieved_by="builder",
        tool_call_id="call-text",
    )

    evidence = ledger.all_evidence()[0]
    assert "Keywords:" not in evidence.quote
    assert evidence.quote == (
        "The main text explains that adaptive probability estimates change "
        "with the observed context."
    )


def test_ledger_ignores_title_only_highlight_without_source_text() -> None:
    ledger = _ledger()

    ledger.add_many(
        results=[
            SourceResult(
                source="exa",
                source_id="paper-title-only",
                title="How Does a Transformer Learn Compression?",
                text=None,
                summary=None,
                metadata={
                    "exa_highlights": ["How Does a Transformer Learn Compression?"],
                    "exa_highlight_query": "transformer compression",
                },
            )
        ],
        query="transformer compression",
        retrieved_by="builder",
        tool_call_id="call-title",
    )

    evidence = ledger.all_evidence()[0]
    assert evidence.quote == ""
    assert evidence.metadata["quote_selection"]["candidate_excerpt"] == ""


def test_ledger_marks_clean_highlight_verified_against_source_text() -> None:
    ledger = _ledger()

    added = ledger.add_many(
        results=[
            SourceResult(
                source="exa",
                source_id="paper-verified",
                title="Verified Highlight",
                text=(
                    "Introductory context. The exact verified excerpt supports the "
                    "claim. Additional context."
                ),
                metadata={
                    "exa_highlights": [
                        "The exact verified excerpt supports the claim."
                    ],
                    "exa_highlight_query": "verified excerpt",
                },
            )
        ],
        query="verified excerpt",
        retrieved_by="builder",
        tool_call_id="call-verified",
    )

    selection = added[0].metadata["quote_selection"]
    assert selection["source"] == "highlight"
    assert selection["verified_quote"] == "The exact verified excerpt supports the claim."
    assert selection["verification_status"] == "accepted"
    assert selection["match_type"] == "exact"


def test_ledger_does_not_accept_unverified_exa_highlight_as_quote() -> None:
    result = SourceResult(
        source="exa",
        source_id="paper-highlight-only",
        title="Highlight Only",
        text="The source text says only grounded statements.",
        metadata={
            "exa_highlights": ["An unsupported provider excerpt."],
            "exa_highlight_query": "unsupported provider excerpt",
        },
    )
    ledger = _ledger()

    [item] = ledger.add_many(
        results=[result],
        query="unsupported provider excerpt",
        retrieved_by="builder",
        tool_call_id="tool_1",
    )

    assert item.quote == "The source text says only grounded statements."
    selection = item.metadata["quote_selection"]
    assert selection["candidate_excerpt"] == "An unsupported provider excerpt."
    assert selection["candidate_verification_status"] == "rejected"
    assert selection["candidate_match_type"] == "not_found"
    assert selection["verification_status"] == "accepted"


def test_per_source_score_normalization_to_unit_interval() -> None:
    ledger = _ledger()
    ledger.add_many(
        results=[
            SourceResult(source="arxiv", source_id="a1", title="A1", text="a1", score=0.2),
            SourceResult(source="arxiv", source_id="a2", title="A2", text="a2", score=0.9),
            SourceResult(source="fake", source_id="f1", title="F1", text="f1", score=42.0),
        ],
        query="q",
        retrieved_by="test",
        tool_call_id="tool",
    )

    by_id = {item.source_id: item for item in ledger.all_evidence()}
    assert by_id["a1"].score == 0.0
    assert by_id["a2"].score == 1.0
    assert by_id["f1"].score == 1.0
    assert by_id["f1"].metadata["raw_score"] == 42.0


def test_equal_numeric_scores_normalize_to_one_for_that_source() -> None:
    ledger = _ledger()
    ledger.add_many(
        results=[
            SourceResult(source="arxiv", source_id="a1", title="A1", text="a1", score=0.5),
            SourceResult(source="arxiv", source_id="a2", title="A2", text="a2", score=0.5),
        ],
        query="q",
        retrieved_by="test",
        tool_call_id="tool",
    )

    assert {item.score for item in ledger.all_evidence()} == {1.0}


def test_mixed_none_and_numeric_scores_are_grouped_by_source() -> None:
    ledger = _ledger()
    ledger.add_many(
        results=[
            SourceResult(source="arxiv", source_id="a1", title="A1", text="a1", score=None),
            SourceResult(source="arxiv", source_id="a2", title="A2", text="a2", score=0.9),
            SourceResult(source="fake", source_id="f1", title="F1", text="f1", score=12.0),
        ],
        query="q",
        retrieved_by="test",
        tool_call_id="tool",
    )

    by_id = {item.source_id: item for item in ledger.all_evidence()}
    assert by_id["a1"].score is None
    assert by_id["a2"].score == 1.0
    assert by_id["f1"].score == 1.0


def _xsrc_ledger() -> EvidenceLedger:
    # Ledger spanning the cross-domain sources, with their real trust tiers.
    return EvidenceLedger(
        trust_policy={
            "arxiv": "authoritative_preprint",
            "europepmc": "peer_reviewed_oa",
            "openalex": "indexed_metadata",
            "crossref": "indexed_metadata",
        },
        trust_tier_priority={
            "authoritative_preprint": 100,
            "peer_reviewed_oa": 90,
            "indexed_metadata": 80,
        },
        source_order=["arxiv", "europepmc", "openalex", "crossref"],
    )


def test_ledger_merges_crossref_and_openalex_by_doi() -> None:
    ledger = _xsrc_ledger()
    crossref = SourceResult(
        source="crossref", source_id="10.1/x", title="Paper A (crossref)",
        external_ids={"doi": "10.1/x"}, text="grounded abstract A",
    )
    openalex = SourceResult(
        source="openalex", source_id="10.1/x", title="Paper A (openalex)",
        external_ids={"doi": "10.1/x"}, text="grounded abstract A",
    )
    ledger.add_many(results=[crossref, openalex], query="q", retrieved_by="b", tool_call_id="t")

    [item] = ledger.all_evidence()  # merged to one
    assert item.external_ids["doi"] == "10.1/x"
    assert len(item.metadata["provenance"]) == 2


def test_ledger_merges_openalex_and_europepmc_by_pmid_without_doi() -> None:
    ledger = _xsrc_ledger()
    openalex = SourceResult(
        source="openalex", source_id="W1", title="Bio Paper (openalex)",
        external_ids={"pmid": "12345"}, text="bio abstract",
    )
    europepmc = SourceResult(
        source="europepmc", source_id="MED/12345", title="Bio Paper (epmc)",
        external_ids={"pmid": "12345", "pmcid": "PMC9"}, text="bio abstract",
    )
    ledger.add_many(results=[openalex, europepmc], query="q", retrieved_by="b", tool_call_id="t")

    [item] = ledger.all_evidence()  # merged on shared PMID despite no DOI
    assert item.source == "europepmc"  # higher trust wins
    assert item.external_ids.get("pmcid") == "PMC9"  # identity map unioned
    assert len(item.metadata["provenance"]) == 2


def test_ledger_highest_trust_wins_regardless_of_arrival_order() -> None:
    ledger = _xsrc_ledger()
    crossref = SourceResult(  # lower trust, arrives first
        source="crossref", source_id="10.2/y", title="From Crossref",
        external_ids={"doi": "10.2/y"}, text="crossref quote",
    )
    europepmc = SourceResult(  # higher trust, arrives second
        source="europepmc", source_id="10.2/y", title="From EuropePMC",
        external_ids={"doi": "10.2/y"}, text="europepmc quote",
    )
    ledger.add_many(results=[crossref, europepmc], query="q", retrieved_by="b", tool_call_id="t")

    [item] = ledger.all_evidence()
    assert item.source == "europepmc"      # higher-trust fields adopted
    assert item.title == "From EuropePMC"
    assert item.evidence_id == "ev_000001"  # stable id preserved across the swap
    assert len(item.metadata["provenance"]) == 2


def test_ledger_does_not_merge_distinct_dois() -> None:
    ledger = _xsrc_ledger()
    a = SourceResult(source="crossref", source_id="10.1/a", title="A", external_ids={"doi": "10.1/a"})
    b = SourceResult(source="openalex", source_id="10.1/b", title="B", external_ids={"doi": "10.1/b"})
    ledger.add_many(results=[a, b], query="q", retrieved_by="b", tool_call_id="t")

    assert len(ledger.all_evidence()) == 2


def test_re_admit_preserves_external_ids() -> None:
    ledger = _xsrc_ledger()
    src = SourceResult(
        source="openalex", source_id="10.3/z", title="Z",
        external_ids={"doi": "10.3/z", "pmid": "777"}, text="z quote",
    )
    ledger.add_many(results=[src], query="q", retrieved_by="b", tool_call_id="t")
    [item] = ledger.all_evidence()
    assert item.external_ids == {"doi": "10.3/z", "pmid": "777"}

    fresh = _xsrc_ledger()  # run-level registry re-admission path
    fresh.add_retrieved(items=[item], query="q", retrieved_by="b", tool_call_id="t2")
    [readmitted] = fresh.all_evidence()
    assert readmitted.external_ids == {"doi": "10.3/z", "pmid": "777"}


def test_ledger_bridges_openalex_doi_pmid_to_pmid_only_europepmc() -> None:
    # OpenAlex carries {doi, pmid}; a later EuropePMC sighting carries only the
    # shared pmid (no DOI). They are the same paper and must merge to one item.
    ledger = _xsrc_ledger()
    openalex = SourceResult(
        source="openalex", source_id="W1", title="Bridged Paper (openalex)",
        external_ids={"doi": "10.9/bridge", "pmid": "123"}, text="bridge abstract",
    )
    europepmc = SourceResult(
        source="europepmc", source_id="MED/123", title="Bridged Paper (epmc)",
        external_ids={"pmid": "123"}, text="bridge abstract",
    )
    ledger.add_many(
        results=[openalex, europepmc], query="q", retrieved_by="b", tool_call_id="t"
    )

    [item] = ledger.all_evidence()  # merged via the shared PMID bridge
    assert item.source == "europepmc"  # higher trust wins
    assert item.external_ids.get("doi") == "10.9/bridge"  # identity map unioned
    assert [entry["source"] for entry in item.metadata["provenance"]] == [
        "openalex",
        "europepmc",
    ]


def test_highest_trust_merge_keeps_lower_trust_quote_and_score() -> None:
    # Lower-trust Exa hit has a real verified quote and a real score; the
    # higher-trust EuropePMC sighting that merges over it carries no text and
    # no score, so it must NOT wipe the existing quote/score.
    ledger = EvidenceLedger(
        trust_policy={"exa": "web_research_paper", "europepmc": "peer_reviewed_oa"},
        trust_tier_priority={"web_research_paper": 60, "peer_reviewed_oa": 90},
        source_order=["europepmc", "exa"],
    )
    exa = SourceResult(
        source="exa", source_id="exa-1", title="Scored Paper (exa)",
        external_ids={"doi": "10.7/keep"},
        text="A grounded sentence supporting the claim.", score=0.92,
    )
    europepmc = SourceResult(
        source="europepmc", source_id="MED/keep", title="Scored Paper (epmc)",
        external_ids={"doi": "10.7/keep"}, text=None, summary=None, score=None,
    )
    ledger.add_many(
        results=[exa, europepmc], query="grounded sentence claim",
        retrieved_by="b", tool_call_id="t",
    )

    [item] = ledger.all_evidence()
    assert item.source == "europepmc"  # higher-trust displayed fields adopted
    assert item.quote == "A grounded sentence supporting the claim."  # kept
    assert item.score == 1.0  # the Exa score survives (normalized, sole exa hit)
    assert item.metadata["raw_score"] == 0.92  # raw score not wiped to None


def test_highest_trust_merge_preserves_references_from_lower_trust_original() -> None:
    # Crossref carries metadata['references']; a higher-trust EuropePMC sighting
    # without references merges over it. The coherence layer reads references, so
    # the original list must survive the merge.
    ledger = _xsrc_ledger()
    crossref = SourceResult(
        source="crossref", source_id="10.5/refs", title="Cited Paper (crossref)",
        external_ids={"doi": "10.5/refs"}, text="cited abstract",
        metadata={"references": ["10.5/ref-a", "10.5/ref-b"]},
    )
    europepmc = SourceResult(
        source="europepmc", source_id="MED/refs", title="Cited Paper (epmc)",
        external_ids={"doi": "10.5/refs"}, text="cited abstract",
    )
    ledger.add_many(
        results=[crossref, europepmc], query="q", retrieved_by="b", tool_call_id="t"
    )

    [item] = ledger.all_evidence()
    assert item.source == "europepmc"  # higher trust wins
    assert item.metadata["references"] == ["10.5/ref-a", "10.5/ref-b"]  # preserved


def test_highest_trust_merge_unions_references_from_both_sources() -> None:
    # When BOTH sources carry references, the merge must UNION them (dedup, order
    # preserved) instead of letting the higher-trust source's list replace the
    # lower-trust one — the coherence layer reads the combined citation set.
    ledger = _xsrc_ledger()
    crossref = SourceResult(
        source="crossref", source_id="10.6/u", title="Paper (crossref)",
        external_ids={"doi": "10.6/u"}, text="abstract",
        metadata={"references": ["10.6/ref-a", "10.6/ref-b"]},
    )
    europepmc = SourceResult(
        source="europepmc", source_id="MED/u", title="Paper (epmc)",
        external_ids={"doi": "10.6/u"}, text="abstract",
        metadata={"references": ["10.6/ref-b", "10.6/ref-c"]},
    )
    ledger.add_many(
        results=[crossref, europepmc], query="q", retrieved_by="b", tool_call_id="t"
    )

    [item] = ledger.all_evidence()
    assert item.source == "europepmc"  # higher trust wins
    assert item.metadata["references"] == ["10.6/ref-a", "10.6/ref-b", "10.6/ref-c"]


def test_ledger_bridges_same_paper_across_sources_by_title_when_no_shared_id() -> None:
    # The same work from sources that share no external ID (arXiv carries only an
    # arxiv id; openalex/crossref carry a doi) AND whose titles vary by HTML markup +
    # year must still collapse to ONE record via the normalized title bridge.
    ledger = _xsrc_ledger()
    arxiv = SourceResult(
        source="arxiv", source_id="2307.00526", external_ids={"arxiv": "2307.00526"},
        title="TensorGPT: Efficient Compression of LLMs Using <i>Tensor-Train</i>",
        published_date="2023-07-01", url="https://arxiv.org/abs/2307.00526",
        text="Tensor-train compression of language models.",
    )
    openalex = SourceResult(
        source="openalex", source_id="W123", external_ids={"doi": "10.1/tensorgpt"},
        title="TensorGPT: Efficient Compression of LLMs Using Tensor-Train",
        published_date="2024-05-01", url="https://openalex.org/W123",
        text="Tensor-train compression of language models.",
    )
    crossref = SourceResult(
        source="crossref", source_id="10.1/tensorgpt", external_ids={"doi": "10.1/tensorgpt"},
        title="TensorGPT: Efficient Compression of LLMs Using Tensor-Train",
        published_date="2024-05-01", url="https://doi.org/10.1/tensorgpt",
        text="Tensor-train compression of language models.",
    )
    ledger.add_many(results=[arxiv], query="q", retrieved_by="b", tool_call_id="c1")
    ledger.add_many(results=[openalex], query="q", retrieved_by="b", tool_call_id="c2")
    ledger.add_many(results=[crossref], query="q", retrieved_by="b", tool_call_id="c3")

    items = ledger.all_evidence()
    assert len(items) == 1
    assert len(items[0].metadata["provenance"]) == 3


def test_ledger_does_not_bridge_distinct_versions_with_different_titles() -> None:
    # Conservative title bridging keeps V1 and V2 distinct when their titles differ and no shared
    # id, so the title bridge must NOT collapse them.
    ledger = _xsrc_ledger()
    v1 = SourceResult(
        source="arxiv", source_id="2403.07378", external_ids={"arxiv": "2403.07378"},
        title="SVD-LLM: Truncation-aware Singular Value Decomposition for LLM Compression",
        published_date="2024-03-01", text="v1",
    )
    v2 = SourceResult(
        source="arxiv", source_id="2408.07378", external_ids={"arxiv": "2408.07378"},
        title="SVD-LLM V2: Optimizing Singular Value Truncation for LLM Compression",
        published_date="2024-08-01", text="v2",
    )
    ledger.add_many(results=[v1], query="q", retrieved_by="b", tool_call_id="c1")
    ledger.add_many(results=[v2], query="q", retrieved_by="b", tool_call_id="c2")

    assert len(ledger.all_evidence()) == 2


def test_merge_keeps_richer_quote_when_higher_trust_source_has_shorter_quote() -> None:
    # A higher-trust source must not downgrade an existing richer quote (for example, an Exa
    # highlight / full passage) to a shorter metadata snippet. The rich quote survives.
    rich_text = (
        "Low-rank factorization compresses the weight matrices of large language "
        "models while preserving downstream accuracy across diverse benchmarks, and "
        "the compression ratio is governed by the retained singular values."
    )
    short_text = "See abstract."
    title = "Information-Theoretic Low-Rank Compression of Language Models"

    solo = _xsrc_ledger()
    solo.add_many(
        results=[SourceResult(
            source="openalex", source_id="W9", external_ids={"doi": "10.9/x"},
            title=title, text=rich_text,
        )],
        query="compression", retrieved_by="b", tool_call_id="s",
    )
    rich_quote = solo.all_evidence()[0].quote
    assert len(rich_quote) > len(short_text)  # sanity: the rich quote really is richer

    merged = _xsrc_ledger()
    merged.add_many(
        results=[SourceResult(
            source="openalex", source_id="W9", external_ids={"doi": "10.9/x"},
            title=title, text=rich_text,
        )],
        query="compression", retrieved_by="b", tool_call_id="c1",
    )
    merged.add_many(
        results=[SourceResult(
            source="arxiv", source_id="2401.9", external_ids={"doi": "10.9/x"},
            title=title, text=short_text,
        )],
        query="compression", retrieved_by="b", tool_call_id="c2",
    )

    [item] = merged.all_evidence()
    assert item.source == "arxiv"          # higher-trust displayed fields still adopted
    assert item.quote == rich_quote        # but the richer quote is retained, not downgraded


def test_ledger_stores_full_text_excerpt_for_appraiser() -> None:  # full-text appraisal input
    # A long body (a fetched full text, or an abstract longer than the 3-sentence quote
    # window) must be stored as `full_text_excerpt` so the appraiser sees the BODY, not a
    # 3-sentence / 500-char snippet — round-1 went `insufficient` on title/abstract-only.
    ledger = _ledger()
    body = (
        "Background. Low-rank factorization compresses weight matrices. "
        "Methods. We evaluate singular value truncation across ranks. "
        "Results. Compression preserves accuracy at moderate ratios. "
    ) * 12
    ledger.add_many(
        results=[_source_result(source_id="ft", title="Full Text Paper", text=body)],
        query="compression", retrieved_by="b", tool_call_id="c1",
    )

    ev = ledger.all_evidence()[0]
    excerpt = ev.metadata.get("full_text_excerpt")
    assert excerpt
    assert len(excerpt) > len(ev.quote)   # strictly more than the compact verified quote
    assert len(excerpt) > 500             # and beyond the old 500-char appraiser ceiling


def test_ledger_omits_full_text_excerpt_for_short_abstract() -> None:  # avoid redundant metadata
    # When the body adds nothing beyond the compact quote, no excerpt is stored (no bloat).
    ledger = _ledger()
    ledger.add_many(
        results=[_source_result(
            source_id="s", title="Short", text="One short sentence about compression.",
        )],
        query="compression", retrieved_by="b", tool_call_id="c1",
    )
    assert not ledger.all_evidence()[0].metadata.get("full_text_excerpt")


def test_ledger_quote_window_grows_past_three_sentences_when_relevant() -> None:
    # When the query-relevant content spans more than 3 sentences (a full-text body), the
    # quote window now grows past the old hard 3-sentence cap (MAX_QUOTE_SENTENCES) instead
    # of being clipped — the appraiser sees the whole relevant passage.
    ledger = _ledger()
    text = (
        "Low-rank compression reduces model size. "
        "Compression preserves accuracy at moderate rank. "
        "Higher compression increases rank truncation error. "
        "Compression with calibration recovers accuracy. "
        "Aggressive compression degrades rank-sensitive layers. "
        "Compression ratio trades accuracy for size."
    )
    ledger.add_many(
        results=[SourceResult(
            source="exa", source_id="ft", title="Compression Paper", text=text,
        )],
        query="compression accuracy rank",
        retrieved_by="builder", tool_call_id="c1",
    )

    quote = ledger.all_evidence()[0].quote
    assert "Low-rank compression reduces model size" in quote      # first relevant sentence
    assert "trades accuracy for size" in quote                     # ...through the last
    assert quote.count(".") >= 4                                   # > the old 3-sentence cap
