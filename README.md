# stale-bench

Replay harness for the **Stale-State Challenge** at STALE 2027, the First
International Workshop on Acting on Stale State in Networked Systems (held with
IEEE INFOCOM 2027, Honolulu, May 24, 2027, if the workshop is accepted).

A guest workload on a shared cluster must decide, every few seconds, whether to
offer one more Job. It cannot see the cluster directly. It sees a capacity record
that was sampled some time ago, and it learns whether an earlier offer landed on
free capacity only after a further delay. Offers made while capacity is free are
useful. Offers made while it is taken cost a budget that the guest must respect.
How much useful work can a policy get from old records without overspending?

## The task

At each epoch your policy receives

- a `Record` with the unit's state as sampled `age` epochs ago, plus source,
  receive and decision timestamps,
- the `Label`s that arrived this epoch, each giving the true state at one of your
  earlier offers,
- a `Ledger` with your running charge (unavailable outcomes plus unresolved
  offers), the allowance for this epoch, and whether the last epoch produced an
  offer.

It returns the probability of offering one Job. The harness draws the offer from
a seeded stream, so every decision can be recomputed later.

```python
class MyPolicy:
    def __init__(self, budget: float, meta: dict): ...
    def act(self, record, labels, ledger) -> float: ...
```

The capacity state is exogenous. Your offers do not change it. That is what lets
one recorded sequence be replayed against every entry. Settings where a guest's
own load changes cluster state are out of scope for the leaderboard and in scope
for workshop papers.

## Quick start

```bash
pip install -e ".[test]"
stale-bench run static                      # a built-in baseline
stale-bench run entries/example_entry.py:Example --isolate --log-dir logs/ex --out results/example.json
stale-bench audit results/example.json logs/ex
stale-bench leaderboard results/*.json
```

## Scenario sets

| Set | What it is | Scenarios |
|---|---|---|
| S1 | symmetric two-state chain, several flip rates, record ages and budgets | 6 |
| S2 | harder synthetic processes. Asymmetric availability, lognormal holding times, hidden fast and slow phases, labels 10 to 40 epochs late | 4 |
| S3 | controlled runs on a two-GPU workstation with synthetic competitor jobs (see `src/stale_bench/data/s3/README.md`) | 3 |

The public set uses seeds derived from scenario names, so anyone can reproduce
it. Final scoring uses the same scenarios with secret seeds and held-back S3
traces.

## Scoring

For each scenario, all runs are pooled.

- **Budget check.** Fail if a one-sided exact binomial test at level 0.01 rejects
  "the per-epoch rate of unavailable-state offers is at most the budget". Errors,
  timeouts and actions outside [0, 1] also fail the scenario.
- **Lift.** Useful offers per epoch divided by what the best record-ignoring
  policy would get at the same budget. A lift of 1 means the records bought
  nothing. A failed scenario scores 0.
- **Strict track.** Also require that after every decision, the charge stays at
  or under `budget * (epoch + 1) + burst`. Unresolved offers count as
  unavailable until their label arrives.

The score for a set is the mean over its scenarios.

## Baselines

| Name | Idea |
|---|---|
| `static` | constant offer probability sized from its own labels |
| `trust-record` | offer exactly when the record says available, ignoring the budget |
| `aimd` | additive increase on available labels, halving on unavailable ones |
| `allocation` | learns per-cell outcome rates and spends the budget on the safest cells |
| `lagrangian` | policy gradient with a learned price on unavailable outcomes |
| `guarded-allocation` | `allocation` that never breaks the strict-track prefix bound |

Public-set results for harness version 0.1.0. Each cell is the mean lift, with
the number of scenarios passed in brackets.

| Policy | S1 | S2 | S3 | Overall | Strict overall |
|---|---|---|---|---|---|
| `entries/example_entry.py` | 6.454 (6/6) | 9.221 (4/4) | 3.595 (3/3) | 6.645 (13/13) | 6.645 |
| `lagrangian` | 6.373 (6/6) | 8.229 (4/4) | 1.447 (3/3) | 5.807 (13/13) | 2.095 |
| `trust-record` | 5.084 (4/6) | 7.647 (3/4) | 3.943 (3/3) | 5.609 (10/13) | 0.385 |
| `allocation` | 5.194 (6/6) | 7.886 (4/4) | 0.319 (3/3) | 4.897 (13/13) | 4.620 |
| `guarded-allocation` | 5.194 (6/6) | 7.882 (4/4) | 0.319 (3/3) | 4.896 (13/13) | 4.896 |
| `static` | 0.808 (6/6) | 0.889 (4/4) | 0.447 (3/3) | 0.750 (13/13) | 0.373 |
| `aimd` | 0.141 (6/6) | 0.277 (4/4) | 0.209 (3/3) | 0.198 (13/13) | 0.195 |

The two-line example entry, which trusts the record and stops when the ledger
is full, beats every learned baseline here. The learners spend part of the
budget on exploration and on confidence margins, and the S3 traces (about 150
epochs each) are too short for them to recover that cost.

## Audit

`--log-dir` writes one compressed JSONL log per run with the state, record,
label delay, offer probability, random draw and offer for every epoch.
`stale-bench audit` recomputes every count from the logs alone and lists any
mismatch with the results file. Final leaderboard logs will be published.

## Entering

See [RULES.md](RULES.md). In short, add one file under `entries/`, open a pull
request, and CI scores it on the public set.

## License

Two licenses, split by what the file is.

| What | License | File |
|---|---|---|
| Code (harness, baselines, tools, tests) | MIT | `LICENSE` |
| Data (traces) and documentation | CC BY 4.0 | `LICENSE-TEXT` |
