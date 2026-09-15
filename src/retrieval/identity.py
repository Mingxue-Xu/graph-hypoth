"""Stable identities for retrieval-cache batches.

Evidence IDs such as ``ev_000001`` are local to one retrieval ledger/cache row.
Combining them with this batch ID makes a citation unambiguous across a run.
"""

from __future__ import annotations

from src.events import stable_hash_payload


def retrieval_batch_id(
    *, run_id: str, input_hash: str, retrieval_config_hash: str
) -> str:
    """Return the stable identity of one retrieval-cache primary-key tuple."""
    return stable_hash_payload(
        {
            "run_id": run_id,
            "input_hash": input_hash,
            "retrieval_config_hash": retrieval_config_hash,
        }
    )
