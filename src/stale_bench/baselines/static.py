"""Ignore the record. Offer with one constant probability sized to the budget."""

from __future__ import annotations

import math


class StaticBudget:
    """Learns the unavailable fraction from its own labels.

    Its offers do not depend on the state, so its labels are an unbiased
    sample. It sizes the offer probability against an upper confidence bound
    on that fraction, aiming at 90% of the budget.
    """

    def __init__(self, budget: float, meta: dict, target: float = 0.9):
        self.goal = target * budget
        self.n = 0
        self.unavail = 0

    def act(self, record, labels, ledger) -> float:
        for lab in labels:
            self.n += 1
            self.unavail += lab.state == 0
        if self.n < 50:
            return self.goal  # cost is at most q whatever the state
        f = self.unavail / self.n
        f_ucb = min(1.0, f + 2.0 * math.sqrt(max(f * (1 - f), 0.25 / self.n) / self.n))
        return min(1.0, self.goal / max(f_ucb, 1e-9))
