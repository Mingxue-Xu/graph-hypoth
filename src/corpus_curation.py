"""Deterministic three-angle corpus curation for the standalone Research Synthesist run.

Over a retrieved full-text pool, classify each record into a configured angle by keyword, drop
records whose body is too thin for mining, keep the richest-bodied records per angle up to a quota,
and order the result angle-first. The module also formats miner passages with evidence markers and
the critic's saturation reference list.

Pure + deterministic — no LLM, no network, no embedder. The angle lexicons + quotas are the
researcher's steering (passed in), never hardcoded domain truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Record = Mapping[str, object]


@dataclass(frozen=True)
class CorpusAngle:
    """One curation angle: a name, the keyword lexicon that admits a record to it, and how many of
    the richest-bodied records to keep. Angle ORDER is priority — a record matching several angles
    lands in the first one listed, and the curated output is emitted angle-first."""

    name: str
    keywords: tuple[str, ...]
    quota: int


def _body(record: Record, body_key: str) -> str:
    return str(record.get(body_key) or "")


def _angle_of(text: str, angles: Sequence[CorpusAngle]) -> str | None:
    low = text.lower()
    for angle in angles:
        if any(kw.lower() in low for kw in angle.keywords):
            return angle.name
    return None


def curate_corpus(
    records: Sequence[Record],
    angles: Sequence[CorpusAngle],
    *,
    min_body_chars: int = 600,
    body_key: str = "body",
    title_key: str = "title",
) -> list[Record]:
    """Classify -> filter -> quota -> order. Records with a body shorter than ``min_body_chars`` are
    dropped (too thin to mine). Each remaining record is assigned to the FIRST angle whose lexicon
    matches its title+body; within an angle the richest-bodied records are kept up to its quota; the
    result is concatenated in angle order (deterministic: ties broken by the input order via a stable
    sort)."""
    buckets: dict[str, list[Record]] = {angle.name: [] for angle in angles}
    for record in records:
        body = _body(record, body_key)
        if len(body) < min_body_chars:
            continue
        name = _angle_of(f"{record.get(title_key, '')} {body}", angles)
        if name is not None:
            buckets[name].append(record)

    curated: list[Record] = []
    for angle in angles:
        ranked = sorted(buckets[angle.name], key=lambda r: len(_body(r, body_key)), reverse=True)
        curated.extend(ranked[: angle.quota])
    return curated


def corpus_passages(
    records: Sequence[Record],
    *,
    body_key: str = "body",
    evidence_id_key: str = "evidence_id",
    paper_id_key: str = "paper_id",
) -> list[str]:
    """Prefix every curated body with the IDs needed to attribute mined concepts."""
    passages: list[str] = []
    for record in records:
        marker = (
            f"[evidence_id={record.get(evidence_id_key, '')} "
            f"paper_id={record.get(paper_id_key, '')}]"
        )
        passages.append(f"{marker}\n{_body(record, body_key)}")
    return passages


def corpus_paper_list(records: Sequence[Record], *, title_key: str = "title") -> str:
    """The judge saturation reference set: the curated corpus as a numbered title list."""
    return "\n".join(
        f"P{i + 1}: {record.get(title_key, '')}" for i, record in enumerate(records)
    )
