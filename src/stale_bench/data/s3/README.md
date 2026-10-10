# S3 controlled-run traces

Each CSV is one run, with columns `epoch,state` (1 = available).

The runs come from an organizer-owned workstation with two GPUs. A generator
started and stopped synthetic competitor jobs on one GPU slot according to a
schedule (`square`: periodic on/off, `poisson`: random arrivals with finite
lifetimes, `markov`: a two-state chain with several flip rates). The state is
1 when that slot had no competitor. The epoch length was about 1.15 s for
`square` and `poisson` and about 2.14 s for `markov`.

No other users' activity appears in these traces. Runs that produced an
identical competitor sequence appear once. About 30% of sequences are held
back for hidden scoring.

License: CC BY 4.0 (see `LICENSE-TEXT`).
