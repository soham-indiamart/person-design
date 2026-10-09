"""Deterministic, seller-level experiment assignment.

Hashing (experiment_id, seller_key) means a seller gets the same arm on every
call in the experiment, with no assignment table needed. The salt per experiment
keeps assignments in different experiments independent.
"""

from __future__ import annotations

import hashlib


def _unit(key: str) -> float:
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_arm(seller_key: str, experiment_id: str, allocation: dict[str, float]) -> str:
    """Return the arm for a seller. ``allocation`` maps arm -> share and must sum to 1."""
    total = sum(allocation.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"allocation must sum to 1, got {total}")
    u = _unit(f"{experiment_id}:{seller_key}")
    acc = 0.0
    for arm, share in sorted(allocation.items()):
        acc += share
        if u < acc:
            return arm
    return sorted(allocation)[-1]


def is_exploration_call(call_id: str, experiment_id: str, epsilon: float) -> bool:
    """Per-call exploration flag (treatment only): persona drawn at random from the plausible ones.

    Keeps the feedback loop from only ever observing outcomes of the matcher's own choices.
    """
    return _unit(f"explore:{experiment_id}:{call_id}") < epsilon
