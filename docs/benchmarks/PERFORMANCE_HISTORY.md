# Performance History — 2^71 project

Centralized cumulative tracking of measured performance changes.

## Rules

1. Every entry is a **measured** result from
   `benchmarks/baseline/run_baseline.sh` records
   (`docs/benchmarks/data/*.json`). Estimated or projected gains are NOT
   recorded here — they live in
   `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md`.
2. Each entry references the exact commit, GPU, CUDA toolkit, and
   configuration.
3. "Improvement" is computed **cumulatively against the first baseline
   measured on the same GPU + driver + CUDA combination**, never by adding
   percentages of unrelated optimizations.
4. Correctness must be **Pass** for an entry to be Accepted.
5. Regressions are kept (status Rejected) with their measured numbers —
   they are evidence.

## Entries

| Commit | GPU | CUDA | Configuration | Baseline (Mkeys/s) | Updated (Mkeys/s) | Improvement | Correctness | Status |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | no entries yet (pending hardware) |

## Measurement record index

| UTC | Record file | Optimization PR | Notes |
|---|---|---|---|
| — | — | — | — |
