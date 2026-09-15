"""Authored-priority emphasis composition for claimless discovery.

In claimless discovery the emphasis is a composed distribution over the field, not a fixed point.
The lens and corpus novelty/saturation already shape ranking. This module adds the component the
researcher authors directly — weighted concepts
— as a re-rank over the gate-eligible surfaced set, framed by ``emphasis_policy``:

  * ``author_directed`` (default) — authored priority is lexicographically primary; the existing
    RankScore breaks ties. The researcher's stated emphasis dominates (predictable, faithful).
  * ``blended`` — emphasis = ``w_author * author_priority + w_rank * RankScore`` (a tunable mix of
    stated interest and what the cycle's field-novelty/saturation ranking surfaced).

Safety invariant: with no authored concepts the re-rank is the identity and the existing
RankScore order is preserved untouched, so claim-mode and no-priority runs stay byte-identical
(this is the re-rank's analogue of the run path's ``no-priority => verify-all`` guard). The re-rank
NEVER gates a candidate out; it only re-orders the already-eligible set (the four Gate_h bars are
unchanged). Pure + deterministic (candidate_id final tie-break) — replay-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src import graph_config_defaults as gcd
from src.cycles.hypothesis import ScoredCandidate


def author_priority(
    node_labels: Sequence[str], authored_concepts: Sequence[tuple[str, float]]
) -> float:
    """The authored priority of a candidate: the maximum weight among authored concepts that
    match any of its node labels (case-insensitive substring — mirrors
    ``cycles.priority.build_priority_from_labels``). 0.0 when no term matches or no priority is
    authored."""
    best = 0.0
    labels_lower = [label.lower() for label in node_labels]
    for term, weight in authored_concepts:
        needle = term.lower().strip()
        if needle and any(needle in label for label in labels_lower):
            best = max(best, float(weight))
    return best


def rerank_by_emphasis(
    scored: Sequence[ScoredCandidate],
    *,
    authored_concepts: Sequence[tuple[str, float]],
    policy: str = "author_directed",
    weights: dict[str, float] | None = None,
) -> list[ScoredCandidate]:
    """Re-order the gate-eligible ranked set by authored-priority emphasis, per ``policy``.

    Identity when no concepts are authored. For ``blended`` the
    weights default to ``EMPHASIS_BLEND_WEIGHTS``. Stable and deterministic:
    ``candidate_id`` is the final tie-break, so the order is total and replay-safe.
    """
    items = list(scored)
    if not authored_concepts:
        return items  # No authored emphasis: preserve RankScore order.
    weights = weights if weights is not None else gcd.EMPHASIS_BLEND_WEIGHTS

    def ap(sc: ScoredCandidate) -> float:
        return author_priority([node.label for node in sc.candidate.new_nodes], authored_concepts)

    if policy == "blended":
        def key(sc: ScoredCandidate) -> tuple[float, str]:
            emphasis = weights["author"] * ap(sc) + weights["rank"] * sc.rank_score
            return (-emphasis, sc.candidate.candidate_id)
    else:  # author_directed: authored priority primary, RankScore as the tie-break
        def key(sc: ScoredCandidate) -> tuple[float, float, str]:
            return (-ap(sc), -sc.rank_score, sc.candidate.candidate_id)

    return sorted(items, key=key)


def emphasis_reranker(
    authored_concepts: Sequence[tuple[str, float]],
    policy: str = "author_directed",
    *,
    weights: dict[str, float] | None = None,
) -> Callable[[Sequence[ScoredCandidate]], list[ScoredCandidate]]:
    """Bind a profile's authored concepts and ``emphasis_policy`` into a reranker callable for
    the hypothesis cycle's ``reranker`` hook (``run_hypothesis_cycle`` /
    ``run_graph_state_workflow(expansion_reranker=...)``). Identity when no priority is authored."""

    def _rerank(scored: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
        return rerank_by_emphasis(
            scored, authored_concepts=authored_concepts, policy=policy, weights=weights
        )

    return _rerank
