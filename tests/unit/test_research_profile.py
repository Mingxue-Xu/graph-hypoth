"""The researcher research-profile (the user-input file, separate from the system model config).

Carries the researcher's domain decisions — expertise/research-interest, the seed claim, priorities,
and the confirmation policy — and derives Research Synthesist prompt steering. The
confirm-policy resolver turns the auto-modes into a confirmed-id set deterministically; interactive
mode returns None so the CLI prompts.
"""

from __future__ import annotations


def _surfaced():
    return [
        {"candidate_id": "h5", "field_novelty": 0.70, "saturation": 0.25, "cross_concept": True, "common_sense": False},
        {"candidate_id": "h6", "field_novelty": 0.64, "saturation": 0.28, "cross_concept": True, "common_sense": False},
        {"candidate_id": "h3", "field_novelty": 0.55, "saturation": 0.42, "cross_concept": True, "common_sense": False},
        {"candidate_id": "h9", "field_novelty": 0.07, "saturation": 0.90, "cross_concept": False, "common_sense": True},
    ]


def test_profile_derives_synthesist_spec_from_expertise():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(
        claim="C", expertise="information theory applied to LLM compression",
        concepts=[{"term": "calibration data", "weight": 0.86}],
        priority_author="Example Researcher", k_judges=3,
    )
    spec = profile.synthesist_spec(judge_corpus="P1: a paper")
    assert spec.judge_reference_field == "information theory applied to LLM compression"
    assert "information theory applied to LLM compression" in spec.miner_focus  # templated focus
    assert "information theory applied to LLM compression" in spec.judgeability  # templated constraint
    assert spec.judge_corpus == "P1: a paper" and spec.k_judges == 3


def test_profile_round_trips_venue_preference(tmp_path):  # optional soft hint
    from src.research_profile import (
        ResearchProfile,
        load_research_profile,
    )

    path = tmp_path / "profile.yaml"
    path.write_text("claim: C\nvenue_preference:\n  - CVPR\n  - ACL\n", encoding="utf-8")
    assert load_research_profile(path).venue_preference == ["CVPR", "ACL"]
    # Optional: absent -> empty default, so existing profiles/runs are unaffected.
    assert ResearchProfile(claim="C").venue_preference == ""


def test_synthesist_spec_threads_normalized_venue_preference():
    from src.research_profile import ResearchProfile

    spec = ResearchProfile(claim="C", venue_preference=["CVPR", "ACL"]).synthesist_spec()
    assert spec.venue_preference == "CVPR, ACL"
    assert ResearchProfile(claim="C").synthesist_spec().venue_preference == ""


def test_profile_explicit_overrides_win_over_templates():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(
        claim="C", expertise="X", field="LLM compression (behavioral)",
        judgeability="must be benchmark-testable", miner_focus="favor behavioral outcomes", k_judges=5,
    )
    spec = profile.synthesist_spec()
    assert spec.judge_reference_field == "LLM compression (behavioral)"
    assert spec.judgeability == "must be benchmark-testable"
    assert spec.miner_focus == "favor behavioral outcomes"
    assert spec.k_judges == 5


def test_load_research_profile_from_yaml(tmp_path):
    from src.research_profile import load_research_profile

    path = tmp_path / "profile.yaml"
    path.write_text(
        "claim: compression hurts factuality\n"
        "expertise: LLM compression\n"
        "concepts:\n"
        "  - {term: calibration, weight: 0.85}\n"
        "priority_author: Example Researcher\n"
        "confirm:\n"
        "  mode: top_k\n"
        "  k: 3\n",
        encoding="utf-8",
    )
    profile = load_research_profile(path)
    assert profile.claim == "compression hurts factuality"
    assert [(concept.term, concept.weight) for concept in profile.concepts] == [
        ("calibration", 0.85)
    ]
    assert profile.confirm.mode == "top_k" and profile.confirm.k == 3
    assert profile.seed_kind == "claim"
    assert profile.seed_input == "compression hurts factuality"


def test_research_goal_yaml_aliases_to_claim(tmp_path):
    from src.research_profile import load_research_profile

    path = tmp_path / "profile.yaml"
    path.write_text(
        "research goal: >\n"
        "  Make fixed-point GBP cheaper while preserving posterior accuracy.\n"
        "expertise: quantized Gaussian belief propagation\n",
        encoding="utf-8",
    )

    profile = load_research_profile(path)
    assert profile.claim == "Make fixed-point GBP cheaper while preserving posterior accuracy.\n"
    assert profile.anchor() == profile.claim
    assert profile.seed_kind == "research_goal"
    assert profile.seed_input == profile.claim


def test_research_goal_python_aliases_to_claim():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(research_goal="make GBP cheaper", expertise="GBP quantization")
    assert profile.claim == "make GBP cheaper"
    assert profile.anchor() == "make GBP cheaper"
    assert profile.seed_kind == "research_goal"
    assert profile.seed_input == "make GBP cheaper"


def test_research_question_yaml_preserves_type_and_round_trips(tmp_path):
    from src.research_profile import ResearchProfile, load_research_profile

    path = tmp_path / "profile.yaml"
    path.write_text("research_question: When does pruning preserve facts?\n")
    profile = load_research_profile(path)
    assert profile.research_question == "When does pruning preserve facts?"
    assert profile.seed_kind == "research_question"
    assert profile.seed_input == profile.anchor() == profile.research_question
    assert ResearchProfile.model_validate(profile.model_dump()) == profile
    spec = profile.synthesist_spec()
    for prompt in (spec.miner_focus, spec.judgeability):
        assert "Research question: When does pruning preserve facts?" in prompt
        assert "not an established claim" in prompt or "not an established" in prompt


def test_research_question_rejects_ambiguous_inputs():
    import pytest

    from src.research_profile import ResearchProfile

    for competing in ({"claim": "Pruning preserves facts."}, {"research_goal": "Save cost."}):
        with pytest.raises(ValueError, match="choose one starting input"):
            ResearchProfile(research_question="When does pruning preserve facts?", **competing)
    with pytest.raises(ValueError, match="non-empty question"):
        ResearchProfile(research_question=" ")


def test_claim_wins_over_research_goal_aliases():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(
        claim="explicit claim", research_goal="research goal alias", expertise="GBP quantization"
    )
    assert profile.claim == "explicit claim"
    assert profile.seed_kind == "claim"
    assert profile.seed_input == "explicit claim"


def test_seed_kind_explicit_override_for_a_raw_message_entry_point():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(claim="raw user message text", seed_kind="raw_message")
    assert profile.seed_kind == "raw_message"
    assert profile.seed_input == "raw user message text"


def test_resolve_confirmed_ids_all_and_top_k_and_interactive():
    from src.research_profile import ConfirmPolicy, resolve_confirmed_ids

    surfaced = _surfaced()
    assert resolve_confirmed_ids(surfaced, ConfirmPolicy(mode="interactive")) is None
    assert resolve_confirmed_ids(surfaced, ConfirmPolicy(mode="all")) == ["h5", "h6", "h3", "h9"]
    assert resolve_confirmed_ids(surfaced, ConfirmPolicy(mode="top_k", k=2)) == ["h5", "h6"]


def test_resolve_confirmed_ids_threshold_predicate():
    from src.research_profile import ConfirmPolicy, resolve_confirmed_ids

    policy = ConfirmPolicy(
        mode="threshold", min_field_novelty=0.6, max_saturation=0.35, exclude_common_sense=True
    )
    # h5/h6 clear; h3 fails (saturation 0.42 > 0.35); h9 excluded (common_sense) and below novelty.
    assert resolve_confirmed_ids(_surfaced(), policy) == ["h5", "h6"]


def test_retrieval_query_is_focused_keywords_not_the_verbose_claim():
    # Keyword sources (OpenAlex/arXiv) return ~0 for the long natural-language claim; the
    # retrieval query must be topical keywords derived from expertise + priority terms.
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(
        claim="Low-rank factorization compresses an LLM by replacing W with the y = Wx surrogate, "
        "and so on for several hundred more characters of natural language prose.",
        expertise="geometry of low-rank LLM compression",
        concepts=[
            {"term": "low-rank factorization", "weight": 0.92},
            {"term": "singular value", "weight": 0.84},
        ],
    )
    q = profile.retrieval_query()
    assert "geometry of low-rank LLM compression" in q
    assert "low-rank factorization" in q
    assert "singular value" in q
    assert "y = Wx" not in q  # not the verbose claim
    assert len(q) < len(profile.claim)


def test_retrieval_query_falls_back_to_claim_without_expertise_or_priorities():
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(claim="just the seed claim")
    assert profile.retrieval_query() == "just the seed claim"


# --- Claimless discovery: emphasis determination without a claim -------------------------
def test_claim_is_optional_for_discovery_mode():  # run from interest without a claim
    from src.research_profile import ResearchProfile

    profile = ResearchProfile(expertise="geometry of low-rank LLM compression")
    assert profile.claim is None
    assert profile.seed_kind is None
    assert profile.seed_input == ""


def test_profile_requires_claim_or_lens():  # safety invariant: never run with no operand
    import pytest
    from pydantic import ValidationError

    from src.research_profile import ResearchProfile

    with pytest.raises(ValidationError):
        ResearchProfile()  # no claim, expertise, or field means there is no semantic anchor


def test_lens_combines_field_breadth_and_expertise_focus():
    from src.research_profile import ResearchProfile

    p = ResearchProfile(
        field="matrix and tensor factorization for LLM compression",
        expertise="geometric understanding of weight-matrix compression",
    )
    lens = p.lens()
    assert "matrix and tensor factorization for LLM compression" in lens  # breadth half
    assert "geometric understanding of weight-matrix compression" in lens  # focus half


def test_lens_dedupes_when_field_equals_expertise():
    from src.research_profile import ResearchProfile

    p = ResearchProfile(field="X compression", expertise="X compression")
    assert p.lens() == "X compression"


def test_anchor_is_claim_when_present_else_lens():
    from src.research_profile import ResearchProfile

    with_claim = ResearchProfile(claim="C", expertise="E")
    assert with_claim.anchor() == "C"  # a present claim is the anchor
    without_claim = ResearchProfile(field="F", expertise="E")
    assert without_claim.anchor() == without_claim.lens()  # without a claim, the lens is the anchor


def test_emphasis_policy_defaults_author_directed_and_parses_blended():
    from src.research_profile import ResearchProfile

    assert ResearchProfile(claim="C").emphasis_policy == "author_directed"  # default
    assert ResearchProfile(claim="C", emphasis_policy="blended").emphasis_policy == "blended"


def test_retrieval_query_without_claim_uses_lens_and_priority_terms():  # claimless retrieval
    from src.research_profile import ResearchProfile

    p = ResearchProfile(
        field="tensor factorization for LLM compression",
        expertise="geometric understanding of compression",
        concepts=[{"term": "tensor-train", "weight": 0.88}],
    )
    q = p.retrieval_query()
    assert q  # non-empty even with no claim (would have been None before)
    assert "tensor factorization for LLM compression" in q  # field breadth
    assert "geometric understanding of compression" in q  # expertise (focus half)
    assert "tensor-train" in q  # priority term


def test_synthesist_spec_steers_prompts_with_authored_priority_by_policy():  # authored-priority steering
    from src.research_profile import ResearchProfile

    p = ResearchProfile(
        expertise="LLM compression geometry",
        concepts=[
            {"term": "tensor-train", "weight": 0.9},
            {"term": "singular value", "weight": 0.8},
        ],
        emphasis_policy="author_directed",
    )
    spec = p.synthesist_spec()
    # the authored priority terms steer BOTH the miner focus and the proposer judgeability prompt.
    assert "tensor-train" in spec.miner_focus and "singular value" in spec.miner_focus
    assert "tensor-train" in spec.judgeability


def test_synthesist_spec_no_priority_leaves_steering_untouched():  # regression: inert without priority
    from src.research_profile import ResearchProfile

    p = ResearchProfile(claim="C", expertise="X", miner_focus="favor behavioral outcomes")
    assert p.synthesist_spec().miner_focus == "favor behavioral outcomes"  # no priority -> no steer added


def test_the_shipped_example_profile_loads_and_names_a_priority_author():
    # The README tells a first-time user to run exactly this file. Weighted `concepts`
    # with no `priority_author` abort the run at the priority stage, so the shipped
    # example has to carry one.
    from pathlib import Path

    from src.research_profile import load_research_profile

    profile = load_research_profile(Path("examples/research_profile.yaml"))

    assert profile.concepts
    assert profile.priority_author.strip()


def test_empty_yaml_string_fields_coerce_to_empty_not_none(tmp_path):  # hand-authored claimless profile
    from src.research_profile import load_research_profile

    path = tmp_path / "p.yaml"
    path.write_text(
        "expertise: green chemistry of deep eutectic solvents\n"
        "priority_author:\n"   # empty YAML scalar -> None; must coerce to "" for the str field
        "priority_focus:\n",
        encoding="utf-8",
    )
    profile = load_research_profile(path)
    assert profile.priority_author == "" and profile.priority_focus == ""


# Typed profile fields and reader lexicon.
def test_profile_parses_all_typed_fields(tmp_path):
    from src.research_profile import load_research_profile

    path = tmp_path / "profile.yaml"
    path.write_text(
        "field: probabilistic inference on factor graphs\n"
        "concepts:\n"
        "  - {term: belief propagation, weight: 0.9}\n"
        "  - {term: cumulants, weight: 0.7}\n"
        "target_outcomes:\n"
        "  - convergence guarantees\n"
        "methods:\n"
        "  - cumulant expansion\n"
        "interest: when Gaussian approximations are valid\n"
        "seed_papers:\n"
        "  - {title: A BP paper, url: 'https://example.org/bp'}\n"
        "exclude_terms:\n"
        "  - quantum\n"
        "reader_lexicon:\n"
        "  home_field: tensor factorization for LLM compression\n"
        "  familiar_terms:\n"
        "    - {term: rank truncation, gloss: dropping small singular values}\n"
        "  analogy_domains: [numerical linear algebra]\n"
        "  papers:\n"
        "    - {title: TensorGPT, year: 2024, venue: arXiv}\n"
        "  source: local paper\n"
        "  generated: '2026-07-02'\n",
        encoding="utf-8",
    )
    p = load_research_profile(path)
    assert p.field == "probabilistic inference on factor graphs"
    assert [c.term for c in p.concepts] == ["belief propagation", "cumulants"]
    assert p.target_outcomes == ["convergence guarantees"]
    assert p.methods == ["cumulant expansion"]
    assert p.interest == "when Gaussian approximations are valid"
    assert p.seed_papers[0].title == "A BP paper"
    assert p.seed_papers[0].url == "https://example.org/bp"
    assert p.exclude_terms == ["quantum"]
    lex = p.reader_lexicon
    assert lex is not None
    assert lex.home_field == "tensor factorization for LLM compression"
    assert lex.familiar_terms[0].term == "rank truncation"
    assert lex.familiar_terms[0].gloss == "dropping small singular values"
    assert lex.analogy_domains == ["numerical linear algebra"]
    assert lex.papers[0].title == "TensorGPT" and lex.papers[0].year == 2024
    assert lex.source == "local paper" and lex.generated == "2026-07-02"


def test_synthesist_spec_derives_from_field_concepts_and_interest():
    from src.research_profile import ResearchProfile

    p = ResearchProfile(
        field="factor-graph inference",
        concepts=[{"term": "belief propagation", "weight": 0.9}],
        interest="when Gaussian approximations hold",
    )
    spec = p.synthesist_spec()
    assert spec.judge_reference_field == "factor-graph inference"
    assert "factor-graph inference" in spec.miner_focus
    assert "belief propagation" in spec.miner_focus  # concept terms steer the focus
    assert "when Gaussian approximations hold" in spec.judgeability


def test_synthesist_spec_uses_expertise_fallback_templates():
    from src.research_profile import ResearchProfile

    spec = ResearchProfile(claim="C", expertise="LLM compression").synthesist_spec()
    assert spec.miner_focus == (
        "Favor the concepts, predictors, and measurable outcomes most relevant to "
        "LLM compression."
    )
    assert spec.judgeability == (
        "Every candidate's variables and outcome MUST be measurable or testable by a "
        "practitioner of LLM compression on a standard benchmark or ablation; a reader "
        "in that field must be able to judge whether the hypothesis is true."
    )
    assert spec.judge_reference_field == "LLM compression"


def test_lens_and_anchor_work_through_new_field_key():
    from src.research_profile import ResearchProfile

    p = ResearchProfile(field="F", expertise="E")
    assert p.lens() == "F. E"
    assert p.anchor() == "F. E"


def test_retrieval_query_uses_field_and_concept_terms():
    from src.research_profile import ResearchProfile

    p = ResearchProfile(
        field="factor graphs",
        expertise="belief propagation theory",
        concepts=[{"term": "cumulants", "weight": 0.8}],
    )
    q = p.retrieval_query()
    assert "factor graphs" in q and "belief propagation theory" in q and "cumulants" in q


def test_reader_lexicon_defaults_none():  # None => translation never runs
    from src.research_profile import ResearchProfile

    assert ResearchProfile(claim="C").reader_lexicon is None


def test_reader_lexicon_loads_from_profile_relative_path(tmp_path):
    from src.research_profile import load_research_profile

    (tmp_path / "lex.yaml").write_text(
        "home_field: tensor factorization\n"
        "familiar_terms:\n"
        "  - {term: SVD, gloss: singular value decomposition}\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "claim: C\nreader_lexicon: lex.yaml\n", encoding="utf-8"
    )
    p = load_research_profile(profile_path)
    assert p.reader_lexicon is not None
    assert p.reader_lexicon.home_field == "tensor factorization"
    assert p.reader_lexicon.familiar_terms[0].term == "SVD"


def test_reader_lexicon_loads_from_absolute_path(tmp_path):
    from src.research_profile import load_research_profile

    lex_path = tmp_path / "abs-lex.yaml"
    lex_path.write_text("home_field: F\n", encoding="utf-8")
    profile_path = tmp_path / "sub"
    profile_path.mkdir()
    profile_file = profile_path / "profile.yaml"
    profile_file.write_text(
        f"claim: C\nreader_lexicon: {lex_path}\n", encoding="utf-8"
    )
    assert load_research_profile(profile_file).reader_lexicon.home_field == "F"


def test_reader_lexicon_missing_path_raises_naming_attempts(tmp_path):
    import pytest

    from src.research_profile import load_research_profile

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "claim: C\nreader_lexicon: nowhere.yaml\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="nowhere.yaml"):
        load_research_profile(profile_path)


def test_example_reader_lexicon_loads_through_schema():
    from pathlib import Path

    import yaml

    from src.research_profile import ReaderLexicon

    artifact = (
        Path(__file__).resolve().parents[2]
        / "examples" / "reader-lexicons" / "example.yaml"
    )
    data = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    lex = ReaderLexicon.model_validate(data)
    assert lex.home_field and lex.source and lex.generated
    assert lex.familiar_terms and all(t.term for t in lex.familiar_terms)
    assert lex.analogy_domains and lex.papers


def test_synthesist_spec_threads_experiment_refine_rounds():
    from src.research_profile import ResearchProfile

    assert ResearchProfile(claim="C").synthesist_spec().experiment_refine_rounds == 1  # default 1
    assert ResearchProfile(claim="C", experiment_refine_rounds=0).synthesist_spec().experiment_refine_rounds == 0
    assert ResearchProfile(claim="C", experiment_refine_rounds=3).synthesist_spec().experiment_refine_rounds == 3


def test_experiment_refine_rounds_rejects_negative():
    import pytest
    from pydantic import ValidationError

    from src.research_profile import ResearchProfile

    with pytest.raises(ValidationError):
        ResearchProfile(claim="C", experiment_refine_rounds=-1)
