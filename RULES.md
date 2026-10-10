# Stale-State Challenge rules (draft)

These rules are a draft and may change before the call for papers.

## Entries

1. One Python file under `entries/`, defining one policy class with
   `__init__(budget, meta)` and `act(record, labels, ledger)`.
2. Python 3.10 or later, the standard library, and NumPy. No other imports.
3. No file, network, subprocess or clock access. The policy may keep any state
   it likes in memory.
4. Each `act` call must return within 5 seconds. The whole public set should
   finish within one hour on a CI runner.
5. One class is scored per file. Teams may submit up to three files.
6. An entry is re-initialized for every run. Nothing carries over between runs.

## What a policy may use

The record, the labels, the ledger and `meta` (scenario name, set, budget,
burst, epoch length). Tuning to the public scenarios is allowed. The hidden
runs use new seeds and held-back traces, so tuning to public seeds will not
help.

## Scoring

Described in the README. The organizers run the hidden set once, at the freeze
date, in isolated mode (`--isolate`), and publish results, logs and the hidden
key afterwards so anyone can check the numbers.

## Dates (tentative)

| Event | Date |
|---|---|
| Harness and public set released | November 2026 |
| Leaderboard frozen, hidden scoring | April 30, 2027 |
| Results presented | May 24, 2027 |

## Presentation

Top entries are invited to present in the challenge session. A 2-page
description is optional for leaderboard entries and required for presentation.

## Organizer entries

Organizer baselines appear on the leaderboard but are not ranked.
