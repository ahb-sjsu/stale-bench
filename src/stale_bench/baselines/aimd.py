"""Additive increase on available labels, multiplicative decrease on unavailable ones.

A classic feedback controller. It reacts to outcomes but has no notion of
the budget, so whether it passes depends on the scenario.
"""

from __future__ import annotations


class AIMD:
    def __init__(self, budget: float, meta: dict, alpha: float = 0.01, beta: float = 0.5):
        self.q = budget
        self.alpha = alpha
        self.beta = beta

    def act(self, record, labels, ledger) -> float:
        for lab in labels:
            if lab.state:
                self.q = min(1.0, self.q + self.alpha)
            else:
                self.q *= self.beta
        return self.q
