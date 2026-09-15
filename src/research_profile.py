"""The researcher-authored profile for a standalone Research Synthesist run.

Deliberately separate from the system model config (``config/evidence-evaluation.yaml``, which sets
the per-role LLM models, an operator concern). This file carries only the researcher's domain
decisions: their expertise and research interest, the seed claim, node priorities, and the
confirmation policy. From the expertise it derives the Research Synthesist prompt steering
(``SynthesistBuildSpec``):
the miner focus, the proposer judgeability constraint, and the judge reference-field — each
overridable by an explicit field. Pure config + deterministic derivation; no LLM, no network.

The typed profile fields include ``field``, weighted ``concepts``, ``target_outcomes``,
``methods``, ``interest``, ``seed_papers``, and ``exclude_terms``, plus a ``reader_lexicon``
describing what the reader already knows. The researcher-facing ``research_goal`` and
``research goal`` keys alias into ``claim``. ``research_question`` is a typed open
question; its text also occupies the legacy ``claim`` slot used by workflow APIs,
while ``seed_kind`` retains its question semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.graph_state_runtime import SynthesistBuildSpec


def _default_miner_focus(expertise: str) -> str:
    if not expertise:
        return ""
    return f"Favor the concepts, predictors, and measurable outcomes most relevant to {expertise}."


def _derived_judgeability(interest: str) -> str:
    """Build a judgeability sentence from the researcher's stated ``interest``."""
    if not interest:
        return ""
    return (
        f"Every candidate must speak to the researcher's stated interest — {interest} — and its "
        "variables and outcome MUST be measurable or testable so a reader can judge whether the "
        "hypothesis is true."
    )


def _normalize_venue(value: list[str] | str) -> str:
    """Render the optional ``venue_preference`` (list or free text) as a single string
    for prompt steering; empty when unset."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _default_judgeability(expertise: str) -> str:
    if not expertise:
        return ""
    return (
        f"Every candidate's variables and outcome MUST be measurable or testable by a practitioner "
        f"of {expertise} on a standard benchmark or ablation; a reader in that field must be able to "
        f"judge whether the hypothesis is true."
    )


class ConfirmPolicy(BaseModel):
    """Confirmation behavior for surfaced hypotheses.

    ``interactive`` pauses for input; other modes compute the confirmed set automatically.
    """

    mode: Literal["interactive", "all", "top_k", "threshold"] = "interactive"
    k: int | None = None
    min_field_novelty: float | None = None
    max_saturation: float | None = None
    require_cross_concept: bool = False
    exclude_common_sense: bool = False


class ConceptTerm(BaseModel):
    """One weighted concept of interest."""

    term: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class SeedPaper(BaseModel):
    title: str
    url: str = ""


class LexiconTerm(BaseModel):
    term: str
    gloss: str = ""


class LexiconPaper(BaseModel):
    title: str
    year: int | None = None
    venue: str = ""


class ReaderLexicon(BaseModel):
    """What THIS reader already knows — the vocabulary the reader-translation seam targets.
    Digested from the reader's own text (see ``docs/user-lexicon.md``) and
    human-reviewed before first use; staleness is degradation-only (a missing term means
    no translation, never a break)."""

    home_field: str = ""
    familiar_terms: list[LexiconTerm] = Field(default_factory=list)
    analogy_domains: list[str] = Field(default_factory=list)
    papers: list[LexiconPaper] = Field(default_factory=list)
    source: str = ""       # provenance, for example a researcher-provided bibliography
    generated: str = ""    # date string


class ResearchProfile(BaseModel):
    """The researcher-authored run profile (the only file the researcher touches).

    With a ``claim``, emphasis is a fixed point and the run converges on that claim. Without one,
    emphasis is a composed distribution over the field: the run anchors on ``lens()`` (``field``
    breadth + ``expertise`` focus) and ``emphasis_policy`` selects how authored priority, the lens,
    and corpus novelty combine to steer the generation.

    """

    model_config = ConfigDict(extra="forbid")

    claim: str | None = None  # absent => claimless discovery; the lens anchors instead
    research_question: str | None = None
    seed_input: str = ""  # Resolved seed text; mirrors `claim`, or "" for claimless discovery.
    seed_kind: Literal["claim", "research_question", "research_goal", "raw_message"] | None = None
    expertise: str = ""
    field: str = ""  # the judge reference field; defaults to ``expertise``
    judgeability: str = ""  # proposer constraint; templated from ``interest``/``expertise`` when empty
    miner_focus: str = ""  # miner steering; templated from ``field``/``expertise`` when empty
    emphasis_policy: Literal["author_directed", "blended"] = "author_directed"
    concepts: list[ConceptTerm] = Field(default_factory=list)
    priority_author: str = ""
    priority_focus: str = ""
    # Additional typed research inputs used by retrieval and prompt steering.
    target_outcomes: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    interest: str = ""
    seed_papers: list[SeedPaper] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)
    # The reader-translation vocabulary source; None => translation never runs.
    reader_lexicon: ReaderLexicon | None = None
    # OPTIONAL soft hint: preferred publication venues. Biases the retrieval planner and
    # the hypothesis-proposer framing toward these venues; never excludes other work.
    venue_preference: list[str] | str = ""
    k_judges: int = Field(default=3, ge=1, le=9)
    # Bounded Experiment Designer ↔ Experiment Validator refinement passes.
    experiment_refine_rounds: int = Field(default=1, ge=0)
    confirm: ConfirmPolicy = Field(default_factory=ConfirmPolicy)

    @model_validator(mode="before")
    @classmethod
    def _input_aliases(cls, data: Any) -> Any:
        """Map user-facing aliases onto model fields BEFORE field validation.

        ``research_goal`` and ``research goal`` map to ``claim``. The canonical key wins when both
        are present.

        It also derives ``seed_kind`` from the key that populated the seed: the literal ``claim``
        key wins; otherwise a ``research_goal`` /
        ``research goal`` alias -> ``"research_goal"``; an explicit ``seed_kind`` (e.g. a
        raw-message entry point that has no dedicated YAML key) always wins over the derivation.
        ``seed_input`` mirrors the resolved ``claim`` text and is empty for claimless discovery.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Keep the legacy claim slot as the workflow's text adapter; seed_kind preserves
        # that a question is open, not an asserted proposition. Accept our own model_dump
        # representation, but reject competing user inputs rather than silently picking one.
        question = data.get("research_question")
        if question is not None:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("research_question must be a non-empty question")
            question = question.strip()
            if any(alias in data for alias in ("research_goal", "research goal")) or (
                "claim" in data and not (
                    data.get("seed_kind") == "research_question" and data["claim"] == question
                )
            ):
                raise ValueError("choose one starting input: research_question, claim, or research_goal")
            if data.get("seed_kind") not in (None, "research_question"):
                raise ValueError("research_question requires seed_kind=research_question")
            data.update(research_question=question, claim=question,
                        seed_kind="research_question", seed_input=question)
        has_claim_key = "claim" in data
        has_goal_alias = any(alias in data for alias in ("research_goal", "research goal"))
        for alias in ("research_goal", "research goal"):
            if alias in data:
                value = data.pop(alias)
                if "claim" not in data:
                    data["claim"] = value
        if "seed_kind" not in data:
            if has_claim_key:
                data["seed_kind"] = "claim"
            elif has_goal_alias:
                data["seed_kind"] = "research_goal"
        if "seed_input" not in data:
            data["seed_input"] = data.get("claim") or ""
        return data

    @field_validator(
        "expertise", "field", "judgeability", "miner_focus",
        "priority_author", "priority_focus", "interest", mode="before",
    )
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        """An empty YAML scalar (e.g. ``priority_author:``) parses as None; coerce the optional
        string fields to "" so a hand-authored claimless profile with blank fields still loads."""
        return "" if value is None else value

    @field_validator(
        "concepts", "target_outcomes", "methods", "seed_papers", "exclude_terms",
        mode="before",
    )
    @classmethod
    def _none_to_empty_list(cls, value: object) -> object:
        """Same courtesy for the optional list fields: a blank YAML scalar means 'none'."""
        return [] if value is None else value

    @model_validator(mode="after")
    def _require_claim_or_lens(self) -> ResearchProfile:
        """Require either a claim or an ``expertise``/``field`` lens as the run anchor."""
        if not (self.claim or self.expertise or self.field):
            raise ValueError(
                "a profile needs a `claim`, `research_question`, or `expertise`/`field` for claimless "
                "discovery mode (nothing to anchor retrieval, mining, or scope on)"
            )
        return self

    def lens(self) -> str:
        """The claimless anchor: ``field`` breadth plus ``expertise`` focus, the two halves
        a claim otherwise fuses. Deduped so an equal field/expertise is not repeated."""
        out: list[str] = []
        for part in (self.field, self.expertise):
            part = part.strip()
            if part and part not in out:
                out.append(part)
        return ". ".join(out)

    def anchor(self) -> str:
        """Resolve the scope, mining, and proposal anchor from ``claim`` or ``lens()``."""
        return self.claim or self.lens()

    def retrieval_query(self) -> str:
        """A focused keyword query for the retrieval sources, NOT the verbose seed claim.

        Keyword backends (OpenAlex / arXiv / crossref) match terms and return ~nothing for a long
        natural-language claim full of math notation; exa's neural search is fine either way. So we
        send all sources a short topical query built from the researcher's ``field`` +
        ``expertise`` + the ``concepts`` term labels (the same lens that anchors a claimless
        run). Falls back to the claim when no field/expertise/priorities are set."""
        terms = [self.field, self.expertise, *(c.term for c in self.concepts)]
        query = " ".join(term.strip() for term in terms if term and term.strip())
        return query or (self.claim or "")

    def _emphasis_steer(self) -> str:
        """The authored-priority steering clause for the miner/proposer prompts (Tier B steer),
        framed by ``emphasis_policy``. Empty when no ``concepts`` are authored, so steering is
        inert by default (the prompts keep their expertise-templated form)."""
        terms = [c.term for c in self.concepts if c.term and c.term.strip()]
        if not terms:
            return ""
        listed = ", ".join(terms)
        if self.emphasis_policy == "author_directed":
            return f"Above all, prioritize the researcher's authored interests: {listed}."
        return f"Give extra weight to the researcher's authored interests: {listed}."

    def synthesist_spec(self, *, judge_corpus: str = "") -> SynthesistBuildSpec:
        """Derive the Research Synthesist seam steering. Explicit fields win; otherwise miner
        focus is keyed on ``field`` and judgeability on ``interest``. Expertise supplies the
        fallback templates. Authored ``concepts`` additionally steer the miner and proposer
        prompts according to ``emphasis_policy``. ``judge_corpus`` (the curated saturation
        reference) is supplied by the driver at run time."""
        steer = self._emphasis_steer()
        # Concept terms join through the steering clause below. Precedence is explicit field,
        # then the profile field, then expertise.
        miner_focus = (
            self.miner_focus
            or _default_miner_focus(self.field)
            or _default_miner_focus(self.expertise)
        )
        judgeability = (
            self.judgeability
            or _derived_judgeability(self.interest)
            or _default_judgeability(self.expertise)
        )
        if steer:
            miner_focus = f"{miner_focus} {steer}".strip()
            judgeability = f"{judgeability} {steer}".strip()
        if self.seed_kind == "research_question":
            question_context = (
                f"Research question: {self.seed_input} This is an open question, not an "
                "established claim. Propose competing mechanisms, conditions, failure boundaries, "
                "methods, or predictors that could answer it; do not assume its desired outcome holds."
            )
            miner_focus = f"{miner_focus} {question_context}".strip()
            judgeability = f"{judgeability} {question_context}".strip()
        return SynthesistBuildSpec(
            miner_focus=miner_focus,
            judgeability=judgeability,
            judge_reference_field=self.field or self.expertise,
            judge_corpus=judge_corpus,
            k_judges=self.k_judges,
            venue_preference=_normalize_venue(self.venue_preference),
            experiment_refine_rounds=self.experiment_refine_rounds,
        )


def _resolve_lexicon_path(spec: str, profile_path: Path) -> Path:
    """Resolve a ``reader_lexicon`` string value to a file: absolute as-is; otherwise
    profile-file-relative, then cwd-relative. Missing everywhere -> ValueError naming
    every attempted path (never a silent None)."""
    raw = Path(spec)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [profile_path.parent / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(c) for c in candidates)
    raise ValueError(f"reader_lexicon file not found; tried: {attempted}")


def load_research_profile(path: str | Path) -> ResearchProfile:
    """Load + validate a ``profile.yaml`` into a ``ResearchProfile``. A ``reader_lexicon``
    value may be an inline mapping OR a string path to a lexicon YAML (resolved here — the
    model itself stays I/O-free)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    lexicon_spec = data.get("reader_lexicon")
    if isinstance(lexicon_spec, str):
        lexicon_path = _resolve_lexicon_path(lexicon_spec, path)
        data["reader_lexicon"] = (
            yaml.safe_load(lexicon_path.read_text(encoding="utf-8")) or {}
        )
    return ResearchProfile(**data)


def resolve_confirmed_ids(
    surfaced: Sequence[Mapping[str, object]], policy: ConfirmPolicy
) -> list[str] | None:
    """Compute the confirmed-id set from the surfaced ranking per the policy. Returns
    ``None`` for ``interactive`` (the CLI prompts the owner); otherwise a deterministic subset.
    ``surfaced`` items expose ``candidate_id`` + ``field_novelty`` / ``saturation`` /
    ``cross_concept`` / ``common_sense`` (the ranking row fields)."""
    if policy.mode == "interactive":
        return None
    ids = [str(item["candidate_id"]) for item in surfaced]
    if policy.mode == "all":
        return ids
    if policy.mode == "top_k":
        return ids[: max(0, policy.k or 0)]
    out: list[str] = []
    for item in surfaced:
        field_novelty = float(item.get("field_novelty", 0.0))  # type: ignore[arg-type]
        saturation = float(item.get("saturation", 0.0))  # type: ignore[arg-type]
        if policy.min_field_novelty is not None and field_novelty < policy.min_field_novelty:
            continue
        if policy.max_saturation is not None and saturation > policy.max_saturation:
            continue
        if policy.require_cross_concept and not bool(item.get("cross_concept", False)):
            continue
        if policy.exclude_common_sense and bool(item.get("common_sense", False)):
            continue
        out.append(str(item["candidate_id"]))
    return out
