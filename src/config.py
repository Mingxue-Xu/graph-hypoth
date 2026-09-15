import ipaddress
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _coherence_embedding_model_default() -> str:
    # Deferred import: `src.retrieval` initializes
    # `retrieval.sources`, which imports THIS module — a top-level import here
    # would be circular. Safe because this only runs at CoherenceConfig()
    # instantiation, always after config.py itself has finished loading.
    from src.retrieval import scoring_defaults as sd

    return sd.EMBEDDING_MODEL


def _coherence_embedding_revision_default() -> str:
    from src.retrieval import scoring_defaults as sd

    return sd.EMBEDDING_REVISION


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Supported values: ``openrouter`` for direct LLM calls (one credential,
    # OPENROUTER_API_KEY, with base_url https://openrouter.ai/api/v1), and the
    # key-free local CLI providers ``claude-cli`` (default coding-agent mode) and
    # ``codex-cli`` (optional). The field stays an unconstrained ``str`` because it
    # is handed to CAMEL as ``model_platform``; an unsupported value fails there.
    provider: str = "openrouter"
    # For ``openrouter``, an OpenRouter catalog slug you choose:
    # https://openrouter.ai/models. For ``claude-cli`` / ``codex-cli``, the name
    # handed to ``claude --model`` / ``codex --model`` (not an OpenRouter slug).
    model_id: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    # Used by direct reasoning backends such as ``provider: claude-cli`` and
    # ``provider: codex-cli``. CAMEL providers currently ignore this field and
    # retain their existing defaults. The union of both CLI vocabularies is
    # accepted here and narrowed per provider below, because the two CLIs do not
    # offer the same levels: Codex has ``minimal``, Claude has ``max``.
    reasoning_effort: Literal[
        "minimal", "low", "medium", "high", "xhigh", "max"
    ] | None = None

    @model_validator(mode="after")
    def validate_reasoning_effort_for_provider(self) -> "ModelConfig":
        """Reject an effort level the selected CLI does not actually offer.

        This matters more for Claude than for Codex: ``codex exec`` fails closed
        on an unknown ``model_reasoning_effort`` under ``--strict-config``, but
        the Claude CLI only *warns* on an unrecognised ``--effort`` and silently
        falls back to its default, which would run a controlled comparison at an
        unknown setting.
        """
        supported = _CLI_REASONING_EFFORTS.get(self.provider)
        if supported is None or self.reasoning_effort is None:
            return self
        if self.reasoning_effort not in supported:
            raise ValueError(
                f"reasoning_effort {self.reasoning_effort!r} is not supported by "
                f"provider {self.provider!r}; use one of: "
                + ", ".join(sorted(supported))
            )
        return self


# Effort levels each direct CLI backend actually accepts. Kept next to
# ModelConfig so a config is rejected at load time rather than by the backend
# after a process has already been spawned.
_CLI_REASONING_EFFORTS: dict[str, frozenset[str]] = {
    "codex-cli": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
    "claude-cli": frozenset({"low", "medium", "high", "xhigh", "max"}),
}


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(ge=0.0, le=2.0)
    model: ModelConfig | None = None


# Standalone Research Synthesist path: every graph-state LLM role is independently configurable,
# Generative seams default to the builder model; appraisal and judging seams default to the
# skeptical-verifier model.
_GENERATIVE_ROLES = frozenset(
    {
        "builder", "extractor", "elaboration_writer", "reader_translator",
        # Research Synthesist folds concept mining, proposal, and revision into one thread.
        "research_synthesist",
        # Experiment Designer expands the confirmed hypothesis into a
        # grounded experiment plan — a builder-tier writing seam.
        "experiment_designer",
    }
)
_CRITICAL_ROLES = frozenset(
    {
        "skeptical_verifier", "evidence_reviewer", "translation_verifier",
        # Critic Panel folds novelty, logic, terminology, and meta-review into one panel.
        "critic_panel",
        # Experiment Validator grades the plan independently — a
        # skeptical-tier judging seam (never grades a plan it designed).
        "experiment_validator",
    }
)


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The two required base configurations. Graph-state workflow roles below
    # inherit from builder or skeptical_verifier unless explicitly overridden.
    builder: AgentConfig
    skeptical_verifier: AgentConfig
    # Evidence Reviewer evaluates coherence, methods, and causal evidence. It falls back to
    # ``skeptical_verifier`` when unset.
    evidence_reviewer: AgentConfig | None = None
    # Optional synthesis seams, each falling back to its tier default when unset.
    # Research Synthesist and Critic Panel roles use builder and skeptical tiers respectively.
    # Former mining, proposal, judgment, audit, and revision roles are folded into these two.
    research_synthesist: AgentConfig | None = None
    critic_panel: AgentConfig | None = None
    elaboration_writer: AgentConfig | None = None
    # Reader-translation seams: the translator writes; the
    # faithfulness verifier JUDGES (critical tier — never grades its own analogy).
    reader_translator: AgentConfig | None = None
    translation_verifier: AgentConfig | None = None
    # Experiment Designer writes the plan; Experiment Validator judges it independently.
    experiment_designer: AgentConfig | None = None
    experiment_validator: AgentConfig | None = None

    def for_role(self, role: str) -> AgentConfig:
        """The effective ``AgentConfig`` for a graph-state LLM role. An explicitly-configured role
        returns itself; an unset (or virtual, e.g. ``extractor``) role falls back to
        its base configuration — generative -> ``builder``, critical -> ``skeptical_verifier``. Optional
        role-specific blocks can be set independently without specifying every workflow role."""
        explicit = getattr(self, role, None)
        if isinstance(explicit, AgentConfig):
            return explicit
        if role in _GENERATIVE_ROLES:
            return self.builder
        if role in _CRITICAL_ROLES:
            return self.skeptical_verifier
        raise ValueError(f"unknown agent role: {role!r}")


class RetrievalRankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["trust_tier_then_source_rank"] = "trust_tier_then_source_rank"
    source_order: list[str] = Field(
        default_factory=lambda: ["arxiv", "exa", "codex_web"]
    )
    trust_tier_priority: dict[str, int] = Field(
        default_factory=lambda: {
            "authoritative_preprint": 100,
            "peer_reviewed_oa": 90,
            "indexed_metadata": 80,
            "web_research_paper": 60,
            "subagent_web_research": 55,
            "deterministic_test": 100,
        }
    )


class RetrievalToolBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_calls_per_run: int | None = Field(default=24, ge=0)
    max_calls_per_agent_round: int | None = Field(default=None, ge=0)
    timeout_seconds: float = Field(default=30, gt=0)


class ArxivSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_delay_seconds: float = Field(default=3, ge=0)
    single_connection: Literal[True] = True
    query_max_chars: int = Field(default=300, ge=1)
    mode: Literal["full_pdf", "metadata_only"] = "full_pdf"
    doc_content_chars_max: int | None = Field(default=4000, ge=1)
    load_max_docs: int = Field(default=100, ge=1)
    load_all_available_meta: bool = True
    continue_on_failure: bool = True


class ExaSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    require_api_key_env: str = "EXA_API_KEY"
    search_type: Literal["auto", "neural", "keyword"] = "auto"
    category: str = "research paper"
    text: bool = False
    highlights: bool = True
    highlight_query: str | None = None
    highlight_max_characters: int = Field(default=1200, ge=1)
    text_max_characters: int = Field(default=20000, ge=1)
    max_age_hours: int | None = Field(default=None, ge=-1)
    livecrawl_timeout: int | None = Field(default=None, ge=1)
    include_html_tags: bool = False
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    include_text: list[str] | None = None
    exclude_text: list[str] | None = None
    max_qps: float = Field(default=10, gt=0)
    max_cost_dollars_per_run: float = Field(default=0.25, ge=0)
    max_cost_dollars_per_call: float = Field(default=0.05, ge=0)


class CrossrefSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "https://api.crossref.org/works"
    mailto: str | None = None
    query_max_chars: int = Field(default=300, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)
    select: str | None = None


class OpenAlexSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "https://api.openalex.org/works"
    mailto: str | None = None
    # OpenAlex works keyless via the polite pool; the key is optional and only
    # raises rate limits. Set require_api_key=true to skip the source when the
    # env var is absent (graceful degrade to crossref/arxiv).
    require_api_key_env: str = "OPENALEX_API_KEY"
    require_api_key: bool = False
    # Swappable key source: "env" (BYO key) now, "proxy" (hosted tier, key
    # injected upstream) later. See credentials.resolve_api_key.
    key_source: Literal["env", "proxy"] = "env"
    query_max_chars: int = Field(default=300, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)
    select: str | None = None


class EuropePmcSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    fulltext_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    result_type: Literal["lite", "core"] = "core"
    query_max_chars: int = Field(default=300, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)


class ApifySourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "https://api.apify.com"
    require_api_key_env: str = "APIFY_API_TOKEN"
    # The Apify actor to run (e.g. a Google-Scholar / Semantic-Scholar scraper). Empty by
    # default so the source stays inert until an actor is chosen for the deployment; its
    # input/output schema is actor-specific (see ApifyPaperSource field mapping).
    actor_id: str = ""
    query_field: str = "query"           # actor-input field that carries the search query
    # static actor input; maxResults caps the actor's own scrape (max_items only pages
    # the dataset read — it does NOT bound how much the actor crawls).
    extra_input: dict[str, Any] = Field(default_factory=lambda: {"maxResults": 15})
    query_max_chars: int = Field(default=300, ge=1)
    max_items: int = Field(default=15, ge=1)
    # run-sync-get-dataset-items blocks for the entire actor run (a browser scrape
    # measured ~292s live); the old 60s default timed out every call -> circuit breaker.
    timeout_seconds: float = Field(default=400.0, gt=0)


class CodexWebSourceConfig(BaseModel):
    """Configuration for the opt-in Codex live-web retrieval source.

    Model availability is account-specific, so ``model_id`` has no default and
    is required only when the source is selected.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    reasoning_effort: Literal[
        "minimal", "low", "medium", "high", "xhigh"
    ] = "high"
    # A schema-constrained research turn may perform many native searches before
    # producing its final record set; keep it bounded but allow more time than a
    # normal completion role.
    timeout_seconds: float = Field(default=900.0, gt=0)
    query_max_chars: int = Field(default=2000, ge=1)
    max_records: int = Field(default=8, ge=1, le=50)
    max_calls_per_run: int = Field(default=8, ge=1)
    max_excerpt_chars: int = Field(default=4000, ge=1, le=20000)
    web_search_mode: Literal["live"] = "live"
    context_size: Literal["low", "medium", "high"] = "high"
    allowed_domains: list[str] | None = None
    # This is an invariant rather than a relaxation switch: schema-valid model
    # output is not accepted without an observed completed search event.
    require_web_search_event: Literal[True] = True

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_id must not be blank")
        return normalized

    @field_validator("allowed_domains")
    @classmethod
    def validate_allowed_domains(
        cls, values: list[str] | None
    ) -> list[str] | None:
        if values is None:
            return None
        normalized: list[str] = []
        for value in values:
            domain = value.strip().lower().rstrip(".")
            labels = domain.split(".")
            try:
                ipaddress.ip_address(domain)
            except ValueError:
                is_ip_address = False
            else:
                is_ip_address = True
            if (
                not domain
                or "." not in domain
                or is_ip_address
                or domain.endswith(
                    (".local", ".localhost", ".internal", ".home.arpa")
                )
                or any(char.isspace() for char in domain)
                or any(char in domain for char in ":/?#@")
                or labels[-1].isdigit()
                or any(
                    not label
                    or len(label) > 63
                    or label.startswith("-")
                    or label.endswith("-")
                    or not label.replace("-", "").isalnum()
                    for label in labels
                )
            ):
                raise ValueError(
                    "allowed_domains entries must be bare public domain names"
                )
            if domain not in normalized:
                normalized.append(domain)
        if not normalized:
            raise ValueError("allowed_domains must not be empty")
        return normalized


class ClaudeWebSourceConfig(BaseModel):
    """Configuration for the opt-in Claude live-web retrieval source.

    Mirrors :class:`CodexWebSourceConfig` field for field where the two
    transports agree, and diverges only where the CLIs do: the reasoning-effort
    vocabulary is Claude's, and the granted tool surface is named explicitly
    because the Claude CLI has no sandbox flag to fall back on.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    # Claude's own effort vocabulary: it has ``max`` and lacks Codex's
    # ``minimal``. An unrecognised value only warns and silently falls back to
    # the CLI default, so it must be rejected here.
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    timeout_seconds: float = Field(default=600.0, gt=0)
    query_max_chars: int = Field(default=2000, ge=1)
    max_records: int = Field(default=8, ge=1, le=50)
    max_calls_per_run: int = Field(default=8, ge=1)
    max_excerpt_chars: int = Field(default=4000, ge=1, le=20000)
    # The exhaustive built-in tool allowlist handed to ``claude --tools``. This
    # is the isolation boundary for this source, so it is restricted to the
    # read-only web tools and must always retain the search tool.
    web_tools: list[str] = Field(default_factory=lambda: ["WebFetch", "WebSearch"])
    # A headless turn cannot answer an approval prompt; with the tool surface
    # already reduced above, a blocking mode would only ever time out.
    permission_mode: Literal[
        "bypassPermissions", "dontAsk", "acceptEdits", "auto", "plan", "manual"
    ] = "bypassPermissions"
    allowed_domains: list[str] | None = None
    # This is an invariant rather than a relaxation switch: schema-valid model
    # output is not accepted without an observed WebSearch tool call.
    require_web_search_event: Literal[True] = True

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("model_id must not be blank")
        return normalized

    @field_validator("web_tools")
    @classmethod
    def validate_web_tools(cls, values: list[str]) -> list[str]:
        allowed = {"WebSearch", "WebFetch"}
        normalized: list[str] = []
        for value in values:
            name = value.strip()
            if name not in allowed:
                raise ValueError(
                    "web_tools may only grant read-only web tools "
                    f"({sorted(allowed)}); got {value!r}"
                )
            if name not in normalized:
                normalized.append(name)
        if "WebSearch" not in normalized:
            raise ValueError(
                "web_tools must include WebSearch; it is the observed proof that "
                "the turn actually searched"
            )
        return sorted(normalized)

    @field_validator("allowed_domains")
    @classmethod
    def validate_allowed_domains(
        cls, values: list[str] | None
    ) -> list[str] | None:
        # Delegate to the Codex source's policy so both live-web sources admit
        # exactly the same domains rather than drifting apart.
        return CodexWebSourceConfig.model_validate(
            {"allowed_domains": values}
        ).allowed_domains


class RetrievalFullTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    timeout_seconds: float = Field(default=15.0, gt=0)
    max_bytes: int = Field(default=20_000_000, ge=1)
    max_pages: int = Field(default=40, ge=1)


class CoherenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    embedding_enabled: bool = True
    # The bare `allenai/specter2` adapter ID cannot load and
    # silently lexical-falls-back; use the `_base` model at a pinned HF revision. Sourced
    # from retrieval.scoring_defaults so the two cannot drift.
    embedding_model: str = Field(default_factory=_coherence_embedding_model_default)
    embedding_revision: str = Field(default_factory=_coherence_embedding_revision_default)
    embedding_weight: float = Field(default=0.6, ge=0)
    citation_enabled: bool = True
    citation_weight: float = Field(default=0.4, ge=0)
    # The LLM judge is opt-in (reuses the OpenRouter key) so default runs stay
    # key-free and deterministic; it only labels the top-K candidates.
    judge_enabled: bool = False
    judge_top_k: int = Field(default=5, ge=1)
    judge_model: ModelConfig | None = None


class ResiliencePolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    max_attempts: int = Field(default=3, ge=1)
    backoff_base_seconds: float = Field(default=0.5, ge=0)
    backoff_max_seconds: float = Field(default=8.0, ge=0)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=60.0, ge=0)


class ResilienceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    default: ResiliencePolicyConfig = Field(default_factory=ResiliencePolicyConfig)
    # Long-running/paid subprocess sources get a single attempt. Retrying a ~5-min
    # actor or Codex live-web call would multiply wall-clock and account usage.
    per_source: dict[str, ResiliencePolicyConfig] = Field(
        default_factory=lambda: {
            "apify": ResiliencePolicyConfig(max_attempts=1),
            "codex_web": ResiliencePolicyConfig(max_attempts=1),
            "claude_web": ResiliencePolicyConfig(max_attempts=1),
        }
    )

    def policy_for(self, source: str) -> ResiliencePolicyConfig:
        return self.per_source.get(source, self.default)


class RetrievalArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    write_json: bool = True
    write_html: bool = True


class RetrievalSourceLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arxiv: ArxivSourceConfig = Field(default_factory=ArxivSourceConfig)
    exa: ExaSourceConfig = Field(default_factory=ExaSourceConfig)
    crossref: CrossrefSourceConfig = Field(default_factory=CrossrefSourceConfig)
    openalex: OpenAlexSourceConfig = Field(default_factory=OpenAlexSourceConfig)
    europepmc: EuropePmcSourceConfig = Field(default_factory=EuropePmcSourceConfig)
    apify: ApifySourceConfig = Field(default_factory=ApifySourceConfig)
    codex_web: CodexWebSourceConfig = Field(default_factory=CodexWebSourceConfig)
    claude_web: ClaudeWebSourceConfig = Field(default_factory=ClaudeWebSourceConfig)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    mode: Literal["source_only"] = "source_only"
    profile: str = "default"
    # Keep the unbounded LangChain arXiv loader opt-in. Crossref provides a
    # bounded, keyless metadata baseline for callers that instantiate the
    # config without a YAML override.
    sources: list[str] = Field(default_factory=lambda: ["crossref"])
    optional_sources: list[str] = Field(default_factory=lambda: ["exa"])
    final_top_k: int = Field(default=8, ge=1)
    per_source_top_k: int = Field(default=3, ge=1)
    fake_mode: bool = False
    cache_replay: bool = True
    ranking: RetrievalRankingConfig = Field(default_factory=RetrievalRankingConfig)
    tool_budget: RetrievalToolBudgetConfig = Field(
        default_factory=RetrievalToolBudgetConfig
    )
    artifacts: RetrievalArtifactConfig = Field(default_factory=RetrievalArtifactConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    fulltext: RetrievalFullTextConfig = Field(
        default_factory=RetrievalFullTextConfig
    )
    coherence: CoherenceConfig = Field(default_factory=CoherenceConfig)
    source_limits: RetrievalSourceLimitsConfig = Field(
        default_factory=RetrievalSourceLimitsConfig
    )
    trust_policy: dict[str, str] = Field(
        default_factory=lambda: {
            "arxiv": "authoritative_preprint",
            "crossref": "indexed_metadata",
            "openalex": "indexed_metadata",
            "europepmc": "peer_reviewed_oa",
            "exa": "web_research_paper",
            "apify": "web_research_paper",
            "codex_web": "subagent_web_research",
            "claude_web": "subagent_web_research",
            "fake": "deterministic_test",
        }
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "RetrievalConfig":
        valid_sources = {
            "arxiv",
            "exa",
            "crossref",
            "openalex",
            "europepmc",
            "codex_web",
            "claude_web",
            "fake",
        }
        configured_sources = set(self.sources) | set(self.optional_sources)
        unknown_sources = configured_sources - valid_sources
        if unknown_sources:
            raise ValueError(
                f"unsupported retrieval sources: {sorted(unknown_sources)}"
            )
        if self.fake_mode and (
            "fake" in self.sources or "fake" in self.optional_sources
        ):
            raise ValueError("fake source is selected by fake_mode, not by sources")
        if "codex_web" in configured_sources and not self.source_limits.codex_web.model_id:
            raise ValueError(
                "selected codex_web source requires "
                "source_limits.codex_web.model_id"
            )
        if (
            "claude_web" in configured_sources
            and not self.source_limits.claude_web.model_id
        ):
            raise ValueError(
                "selected claude_web source requires "
                "source_limits.claude_web.model_id"
            )
        claude_resilience = self.resilience.policy_for("claude_web")
        if (
            "claude_web" in configured_sources
            and self.resilience.enabled
            and claude_resilience.enabled
            and claude_resilience.max_attempts != 1
        ):
            raise ValueError(
                "selected claude_web source requires exactly one resilience attempt"
            )
        codex_resilience = self.resilience.policy_for("codex_web")
        if (
            "codex_web" in configured_sources
            and self.resilience.enabled
            and codex_resilience.enabled
            and codex_resilience.max_attempts != 1
        ):
            raise ValueError(
                "selected codex_web source requires exactly one resilience attempt"
            )
        unknown_order = [s for s in self.ranking.source_order if s not in valid_sources]
        if unknown_order:
            raise ValueError(
                f"unknown sources in ranking.source_order: {sorted(set(unknown_order))}"
            )
        # Fail loudly at load time instead of a ledger KeyError at runtime: every
        # selectable source must have a trust tier present in trust_tier_priority.
        for source in self.selected_source_names():
            tier = self.trust_policy.get(source)
            if tier is None:
                raise ValueError(f"source {source!r} has no trust_policy entry")
            if tier not in self.ranking.trust_tier_priority:
                raise ValueError(
                    f"trust tier {tier!r} for source {source!r} is missing from "
                    "ranking.trust_tier_priority"
                )
        return self

    def selected_source_names(self) -> list[str]:
        if not self.enabled:
            return []
        if self.fake_mode:
            return ["fake"]
        ordered: list[str] = []
        for source in (*self.sources, *self.optional_sources):
            if source not in ordered:
                ordered.append(source)
        return ordered


class OrchestrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: AgentsConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @model_validator(mode="after")
    def validate_model_configs(self) -> "OrchestrationConfig":
        missing_model_configs = [
            f"agents.{role_name}.model"
            for role_name, agent_config in {
                "builder": self.agents.builder,
                "skeptical_verifier": self.agents.skeptical_verifier,
            }.items()
            if agent_config.model is None
        ]
        if missing_model_configs:
            raise ValueError(
                "model config is required for every required base block: "
                + ", ".join(missing_model_configs)
            )
        return self


# Shared default used by every CLI entry point, including ``synthesist_run.py``.
# imports this instead of repeating the literal.
DEFAULT_CONFIG_PATH = Path("config/evidence-evaluation.yaml")


def load_config(path: Path | str) -> OrchestrationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return OrchestrationConfig.model_validate(data)
