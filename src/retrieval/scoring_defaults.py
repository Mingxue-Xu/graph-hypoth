"""Central scoring defaults for deterministic retrieval and evidence verification.

This module owns association and verification coefficients, thresholds, pinned model data, and
their version stamps. Scheduler, transaction, and extraction defaults live in
``graph_config_defaults``. Every scored output records ``SCORING_DEFAULTS_VERSION`` so it can be
recomputed from the same values.
"""

from __future__ import annotations

# Bump this version whenever a value below changes. Every evidence-signal bundle records it so
# association scores and verdict thresholds remain reproducible.
SCORING_DEFAULTS_VERSION = "scoring-defaults-2026-08-30"

# ---------------------------------------------------------------------------
# Embedding model and immutable model revision. The bare `allenai/specter2` adapter ID cannot load;
# use the `_base` model.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "allenai/specter2_base"
# Pinned Hugging Face revision for reproducible embedding output.
EMBEDDING_REVISION = "3447645e1def9117997203454fa4495937bfbd83"

# ---------------------------------------------------------------------------
# Vendored stopwords keep normalization hermetic, with no runtime nltk or sklearn download.
# Source: scikit-learn ENGLISH_STOP_WORDS (318 words).
# ---------------------------------------------------------------------------
STOPWORDS_VERSION = "stopwords-sklearn-en-2026-08-30"
STOPWORDS: frozenset[str] = frozenset({
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
    'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything',
    'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because',
    'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below',
    'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call',
    'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail',
    'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else',
    'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything',
    'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five',
    'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further',
    'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here',
    'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into',
    'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd',
    'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover',
    'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither',
    'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
    'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
    'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed',
    'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since',
    'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime',
    'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the',
    'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
    'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this',
    'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together',
    'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up',
    'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when',
    'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
    'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom',
    'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
    'yourself', 'yourselves'
})
assert len(STOPWORDS) == 318  # invariant: exact sklearn ENGLISH_STOP_WORDS set

# ---------------------------------------------------------------------------
# The scorer owns this deterministic synonym seed. ``lexicon_expand`` unions each term's row for
# synonym coverage and lexical anchors. Empty rows contribute zero coverage.
# Keys + values are norm()-style lowercase surface terms.
# ---------------------------------------------------------------------------
SYNONYM_LEXICON_VERSION = "synlex-seed-2026-08-30"
SYNONYM_LEXICON: dict[str, frozenset[str]] = {
    "myocardial infarction": frozenset({"heart attack", "mi", "cardiac infarction"}),
    "mortality": frozenset({"death", "deaths", "fatality", "fatalities"}),
    "incidence": frozenset({"new cases", "occurrence"}),
    "hypertension": frozenset({"high blood pressure", "elevated blood pressure"}),
    "type 2 diabetes": frozenset({"t2d", "type ii diabetes", "adult-onset diabetes"}),
    "physical activity": frozenset({"exercise", "aerobic exercise"}),
    "smoking": frozenset({"tobacco use", "cigarette smoking"}),
    "obesity": frozenset({"adiposity", "high bmi"}),
    "stroke": frozenset({"cerebrovascular accident", "cva"}),
    "cancer": frozenset({"carcinoma", "neoplasm", "malignancy", "tumor"}),
    "randomized controlled trial": frozenset({"rct", "randomised controlled trial"}),
    "confounding": frozenset({"confounder", "confounders", "confounding variable"}),
    "cardiovascular disease": frozenset({"cvd", "heart disease"}),
    "risk": frozenset({"hazard"}),
}

# ---------------------------------------------------------------------------
# Lexical association
# ---------------------------------------------------------------------------
# Hybrid additive weights inside the anchor gate.
LEXICAL_WEIGHTS = {"label": 0.5, "anchor": 0.3, "syn": 0.2}
# Anchor-coverage term weights: label=3, alias=2, otherwise 1.
ANCHOR_TERM_WEIGHTS = {"label": 3, "alias": 2, "other": 1}
# Pure-lexical coverage/Jaccard split retained for the legacy scoring form.
LEXICAL_COVERAGE_JACCARD = {"coverage": 0.8, "jaccard": 0.2}

# ---------------------------------------------------------------------------
# Combined association score
# ---------------------------------------------------------------------------
SEMANTIC_WEIGHTS = {"embedding": 0.8, "lexical": 0.2}              # semantic = 0.8*E + 0.2*L
ASSOCIATION_WEIGHTS = {"semantic": 0.55, "citation": 0.35, "source_prior": 0.10}

# ---------------------------------------------------------------------------
# Citation and graph-neighbor score
# max(direct, ADJACENT*neighbor, BATCH*batch); overlap = max(coupling, co_citation)
# ---------------------------------------------------------------------------
ADJACENT_TARGET_ATTENUATION = 0.5
BATCH_CITATION_ATTENUATION = 0.3

# ---------------------------------------------------------------------------
# Relevance gate and triage. The association gate is a weak sanity
# floor; relevance is rank-based via K_TOP (the primary selector).
# ---------------------------------------------------------------------------
THETA_ASSOC = 0.05   # Weak association floor; rank remains the primary selector.
K_TOP = 5

# ---------------------------------------------------------------------------
# Quote-quality weights consumed by evidence-strength scoring
# ---------------------------------------------------------------------------
QUOTE_WEIGHTS = {"span": 0.4, "context": 0.2, "target": 0.2, "status": 0.2, "summary": 0.5}

# ---------------------------------------------------------------------------
# Evidence-review sub-signal fusion
# ---------------------------------------------------------------------------
RELSCORE_WEIGHTS = {"entail": 0.5, "ctx": 0.2, "construct": 0.3, "contra": 0.5}
METHODS_WEIGHTS = {"design": 0.35, "adjust": 0.30, "measure": 0.20,
                   "pop": 0.15, "bias": 0.40}
SE_WEIGHTS = {"rel": 0.45, "method": 0.45, "quote": 0.10}

# ---------------------------------------------------------------------------
# Verification-confidence weights and saturation caps
# ---------------------------------------------------------------------------
CONF_WEIGHTS = {"top3": 0.55, "balance": 0.25, "breadth": 0.20,
                "open_risk": -0.25, "limits": -0.15}
K_CONF = 3            # Top-three evidence cap
K_BREADTH = 3         # Breadth saturation cap
K_RISK = 3            # Open-risk saturation cap
K_LIMITS = 3          # Limits saturation cap

# ---------------------------------------------------------------------------
# Verdict-selection thresholds. ``THETA_VERDICT`` gates abstention on every edge.
# ---------------------------------------------------------------------------
THETA_VERDICT = 0.5   # Minimum confidence required to avoid an insufficient verdict.
DELTA_MARGIN = 0.025  # Near-tie margin that produces a hold verdict.

# ---------------------------------------------------------------------------
# Claim-scoped relatedness retained for compatibility.
# ---------------------------------------------------------------------------
LEGACY_RELATEDNESS_WEIGHTS = {"embedding": 0.6, "citation": 0.4}

# Small numeric floor used by saturating ratios.
EPSILON = 1e-9

# ---------------------------------------------------------------------------
# Content-addressed identity derivation keeps extraction deltas idempotent. The helpers live in the
# delta and validator modules; the recipe is:
#   node_id = stable_hash_payload({type, norm(label), norm(definition)})
#   edge_id = stable_hash_payload({sorted(source_node_ids), sorted(target_node_ids),
#                                  direction, relation_type})
# ---------------------------------------------------------------------------


def audit_dict() -> dict[str, object]:
    """Flat, audit-surfaceable view of every scoring default.

    Excludes the bulky STOPWORDS/SYNONYM_LEXICON bodies — their version stamps stand
    in for them so audit output stays compact yet reproducible.
    """
    return {
        "scoring_defaults_version": SCORING_DEFAULTS_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "stopwords_version": STOPWORDS_VERSION,
        "synonym_lexicon_version": SYNONYM_LEXICON_VERSION,
        "lexical_weights": LEXICAL_WEIGHTS,
        "anchor_term_weights": ANCHOR_TERM_WEIGHTS,
        "lexical_coverage_jaccard": LEXICAL_COVERAGE_JACCARD,
        "semantic_weights": SEMANTIC_WEIGHTS,
        "association_weights": ASSOCIATION_WEIGHTS,
        "adjacent_target_attenuation": ADJACENT_TARGET_ATTENUATION,
        "batch_citation_attenuation": BATCH_CITATION_ATTENUATION,
        "theta_assoc": THETA_ASSOC,
        "k_top": K_TOP,
        "quote_weights": QUOTE_WEIGHTS,
        "relscore_weights": RELSCORE_WEIGHTS,
        "methods_weights": METHODS_WEIGHTS,
        "se_weights": SE_WEIGHTS,
        "conf_weights": CONF_WEIGHTS,
        "k_conf": K_CONF,
        "k_breadth": K_BREADTH,
        "k_risk": K_RISK,
        "k_limits": K_LIMITS,
        "theta_verdict": THETA_VERDICT,
        "delta_margin": DELTA_MARGIN,
        "legacy_relatedness_weights": LEGACY_RELATEDNESS_WEIGHTS,
        "epsilon": EPSILON,
    }
