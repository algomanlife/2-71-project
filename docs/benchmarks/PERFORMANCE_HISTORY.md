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
| b3d13be | RTX 5090 | 13.4 | initial baseline (default config) | — | 7,800.4 (best observed¹) | — | Pass (19/19 + 528/528 proof + 12/12 grids) | Baseline |
| b3d13be | RTX 5090 | 13.4 | `512,64` (grid sweep best) | 7,800.4 | 8,133.0 (best observed¹) | +4.3% vs initial baseline (config change only, no code change) | Pass | Accepted (configuration finding) |
| b3d13be | RTX PRO 6000 | 13.4 | initial baseline (default config) | — | 7,074.8 (best observed¹) | — | Pass (12/12 grids) | Baseline |
| b3d13be | RTX PRO 6000 | 13.4 | `128,64` (grid sweep best) | 7,074.8 | 7,452.4 (best observed¹) | +5.3% vs initial baseline (config change only, no code change) | Pass | Accepted (configuration finding) |

¹ Best observed in early-termination known-answer runs — **not sustained
throughput, not medians**; see GRID_RESULTS.md interpretation rules. The
+4.3%/+5.3% figures compare single best-observed values of configuration
changes only (no code change between them) and must not be presented as
optimization gains.

No code-level optimization has been measured yet; the first entries will
come from the Phase-6 experiment program with the repeatable sweep
workflow.

## Measurement record index

| UTC | Record file | Optimization PR | Notes |
|---|---|---|---|
| 2026-11-05 | data/grid-sweep-2026-11-05.json | — (baseline + grid sweep) | maintainer-supplied runs; 12 grid configs x 2 GPUs; 256,128 no speed line |
