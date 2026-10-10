"""Test-only policies."""


class Spy:
    """Offers often and checks the harness's promises as it goes."""

    def __init__(self, budget, meta):
        assert set(meta) == {"scenario", "set", "budget", "burst", "dt"}
        self.t = -1
        self.offers = set()

    def act(self, record, labels, ledger):
        assert record.epoch == self.t + 1
        if ledger.last_offer:
            self.offers.add(record.epoch - 1)
        self.t = record.epoch
        assert 0 <= record.age <= record.epoch
        assert record.t_source <= record.t_receive <= record.t_decide
        for lab in labels:
            assert lab.arrive_epoch == record.epoch
            assert lab.offer_epoch in self.offers
            self.offers.discard(lab.offer_epoch)
        assert ledger.unresolved == len(self.offers)
        assert ledger.charged == ledger.resolved_unavailable + ledger.unresolved
        return 0.5


class Broken:
    def __init__(self, budget, meta):
        pass

    def act(self, record, labels, ledger):
        return 2.0
