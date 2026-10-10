"""What a policy sees at each decision epoch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Record:
    """One delay-tagged capacity record, delivered just before the decision.

    ``sampled_state`` is the unit's state ``age`` epochs ago (1 = available,
    0 = unavailable). Timestamps are in seconds on the harness clock.
    """

    epoch: int
    unit: str
    sampled_state: int
    age: int
    t_source: float
    t_receive: float
    t_decide: float


@dataclass(frozen=True, slots=True)
class Label:
    """The realized state at an earlier offer, arriving late.

    Labels exist only for epochs where the policy offered a Job.
    """

    offer_epoch: int
    state: int
    arrive_epoch: int


@dataclass(frozen=True, slots=True)
class Ledger:
    """Prefix accounting before this epoch's decision.

    ``charged`` counts every offer whose label said unavailable plus every
    offer whose label has not arrived yet. The strict track requires
    ``charged`` after the decision to stay at or under ``allowance``.
    """

    charged: int
    allowance: float
    resolved_unavailable: int
    unresolved: int
    last_offer: int
    """1 if the harness turned the previous epoch's probability into an offer."""
