"""Convert controlled two-GPU runs into replay traces.

Input: JSON run files whose ``samples`` list holds per-epoch ``t`` and ``c``
(the number of competitor jobs on a two-slot GPU slice, 0 or 1). The binary
unit is the slot the competitor occupies, so the state is 1 - c.

Runs that produced the same competitor sequence are kept once. A fixed hash
split sends about 30% of sequences to the hidden set, written to a separate
directory that is never committed.

    python tools/convert_controlled_runs.py RUN_DIR [RUN_DIR ...] \
        --public src/stale_bench/data/s3 --hidden PRIVATE_DIR/s3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--public", required=True)
    p.add_argument("--hidden", required=True)
    p.add_argument("--hidden-percent", type=int, default=30)
    a = p.parse_args()

    seen = {}
    for d in a.run_dirs:
        for f in sorted(Path(d).glob("*.json")):
            run = json.loads(f.read_text())
            s = run.get("samples") or []
            if not s or "c" not in s[0]:
                continue
            cfg = run["cfg"]
            if cfg.get("C") != 2 or cfg.get("c_hi") != 1:
                continue
            state = [1 - int(x["c"]) for x in s]
            if set(state) - {0, 1}:
                continue
            h = hashlib.sha256(bytes(state)).hexdigest()[:12]
            seen.setdefault(h, (cfg["profile"], state, s))

    counts = {}
    for h, (profile, state, s) in sorted(seen.items()):
        hidden = int(h, 16) % 100 < a.hidden_percent
        root = Path(a.hidden if hidden else a.public) / profile
        root.mkdir(parents=True, exist_ok=True)
        with open(root / f"{h}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "state"])
            w.writerows(enumerate(state))
        key = ("hidden" if hidden else "public", profile)
        counts[key] = counts.get(key, 0) + 1
    for k, v in sorted(counts.items()):
        print(k, v)


if __name__ == "__main__":
    main()
