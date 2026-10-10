"""Learned likelihood allocation over record cells.

A cell is the pair (sampled state, age, with ages above ``max_age`` pooled).
The policy counts how often each cell occurs and, from its own labels, how
often the unit turned out unavailable in that cell. Every ``refresh`` epochs
it spends the budget greedily on the cells least likely to be unavailable,
using upper confidence bounds for that likelihood. A fixed tenth of the
budget is spread over all cells so that every cell keeps getting labels.

Offers within a cell do not depend on the state, so each cell's labels are
an unbiased sample of that cell.
"""

from __future__ import annotations

import math
from collections import defaultdict


class LearnedAllocation:
    def __init__(
        self,
        budget: float,
        meta: dict,
        max_age: int = 6,
        refresh: int = 50,
        explore: float = 0.1,
        target: float = 0.9,
    ):
        self.budget = budget
        self.max_age = max_age
        self.refresh = refresh
        self.q_explore = explore * budget
        self.spend = (target - explore) * budget
        self.seen = defaultdict(int)
        self.n_lab = defaultdict(int)
        self.n_unavail = defaultdict(int)
        self.offer_cell = {}
        self.q = {}
        self.epochs = 0

    def cell(self, record):
        return (record.sampled_state, min(record.age, self.max_age))

    def _p0_ucb(self, c) -> float:
        n = self.n_lab[c]
        if n == 0:
            return 1.0
        p = (self.n_unavail[c] + 1) / (n + 2)
        return min(1.0, p + 2.0 * math.sqrt(p * (1 - p) / n))

    def _reallocate(self) -> None:
        total = sum(self.seen.values())
        q = {c: self.q_explore for c in self.seen}
        left = self.spend
        for c in sorted(self.seen, key=self._p0_ucb):
            cost_per_unit = (self.seen[c] / total) * self._p0_ucb(c)
            room = 1.0 - q[c]
            if cost_per_unit <= 0:
                q[c] = 1.0
                continue
            extra = min(room, left / cost_per_unit)
            q[c] += extra
            left -= extra * cost_per_unit
            if left <= 1e-12:
                break
        self.q = q

    def act(self, record, labels, ledger) -> float:
        if not ledger.last_offer:
            self.offer_cell.pop(record.epoch - 1, None)
        for lab in labels:
            c = self.offer_cell.pop(lab.offer_epoch)
            self.n_lab[c] += 1
            self.n_unavail[c] += lab.state == 0
        c = self.cell(record)
        self.seen[c] += 1
        self.epochs += 1
        if self.epochs % self.refresh == 0:
            self._reallocate()
        q = self.q.get(c, self.q_explore)
        # kept until the label arrives, or dropped next epoch if no offer was drawn
        self.offer_cell[record.epoch] = c
        return q


class GuardedAllocation(LearnedAllocation):
    """Strict-track variant. Never offers when one more charge would exceed the allowance."""

    def act(self, record, labels, ledger) -> float:
        q = super().act(record, labels, ledger)
        if ledger.charged + 1 > ledger.allowance:
            return 0.0
        return q
