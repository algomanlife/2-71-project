# Optimization Results — Measured Only

**Status: NO lab optimization results exist yet.** The research lab and
its measurement workflow are implemented and compile-verified; runtime
executions on the RTX 5090 rig are pending. This file will contain only
measured, recorded results — never estimates (estimates live in
roadmaps/experiment definitions).

## Application (search pipeline) — measured history

See `docs/benchmarks/PERFORMANCE_HISTORY.md` (configuration findings
only; no code-level optimization measured yet).

## Research lab — measured results

| Date | Workload | Config | Median | Stdev | vs baseline | Record |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | no records yet (pending hardware run) |

## Rules

1. Entries reference machine-readable records
   (`research/cuda-lab/results/*.json` or `docs/benchmarks/data/*.json`).
2. Synthetic ops/s, EC ops/s, and key-search Mkeys/s are never mixed,
   compared, or summed.
3. Improvements require the 2×-pooled-stdev noise guard; inconclusive
   comparisons are labeled as such.
4. Regressions are recorded with their numbers.
