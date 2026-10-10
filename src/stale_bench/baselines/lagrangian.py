"""Policy gradient with a learned price on unavailable-state offers.

One logistic offer probability per record cell. When a label arrives the
policy receives reward 1 if the unit was available and minus ``lam``
otherwise, and takes a REINFORCE step. Not offering earns exactly 0, so no
baseline is needed. ``lam`` rises when unavailable labels arrive faster
than the target rate and falls otherwise.
"""

from __future__ import annotations

import math
from collections import defaultdict


class LagrangianReinforce:
    def __init__(
        self,
        budget: float,
        meta: dict,
        max_age: int = 6,
        lr: float = 0.5,
        lr_price: float = 0.05,
        target: float = 0.85,
    ):
        self.max_age = max_age
        self.lr = lr
        self.lr_price = lr_price
        self.goal = target * budget
        start = math.log(budget / (1 - budget))
        self.theta = defaultdict(lambda: start)
        self.lam = 1.0
        self.offered = {}

    def act(self, record, labels, ledger) -> float:
        if not ledger.last_offer:
            self.offered.pop(record.epoch - 1, None)
        v = 0
        for lab in labels:
            c, q_then = self.offered.pop(lab.offer_epoch)
            r = 1.0 if lab.state else -self.lam
            v += lab.state == 0
            self.theta[c] = max(-8.0, min(8.0, self.theta[c] + self.lr * r * (1.0 - q_then)))
        self.lam = max(0.0, self.lam + self.lr_price * (v - self.goal))
        c = (record.sampled_state, min(record.age, self.max_age))
        q = 1.0 / (1.0 + math.exp(-self.theta[c]))
        self.offered[record.epoch] = (c, q)
        return q
