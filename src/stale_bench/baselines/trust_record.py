"""Offer exactly when the record says available. Ignores the budget."""

from __future__ import annotations


class TrustRecord:
    def __init__(self, budget: float, meta: dict):
        pass

    def act(self, record, labels, ledger) -> float:
        return float(record.sampled_state)
