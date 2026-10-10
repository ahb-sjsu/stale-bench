"""Example entry. Copy this file, rename the class, and open a pull request.

Rules in brief (see RULES.md):
- One file, Python 3.10+, NumPy and the standard library only.
- ``__init__(budget, meta)``. ``meta`` holds the scenario name, set, budget,
  burst and epoch length in seconds. It does not reveal the state process.
- ``act(record, labels, ledger)`` returns the probability of offering one Job
  this epoch. The harness draws the offer. Answer within 5 seconds.
- No files, network, or clock reads.

This entry offers only when the record says available, and stops when the
ledger shows no room left. Run it with

    stale-bench run entries/example_entry.py:Example --out results/example.json
"""


class Example:
    def __init__(self, budget, meta):
        self.budget = budget

    def act(self, record, labels, ledger):
        if ledger.charged + 1 > ledger.allowance:
            return 0.0
        return float(record.sampled_state)
