# Performance Experiments — CUDA Research Lab

Protocol (binding): for every experiment record baseline commit,
hypothesis, ONE isolated change, compile status, correctness result,
repeated measurements (median + stdev via
`research/cuda-lab/scripts/run_experiments.py`), comparison with
baseline, decision (keep/revert), and environment metadata. No claims
from single runs. These experiments run on STANDALONE lab workloads —
the search pipeline is not modified.

## Prepared experiments (runtime pending on the RTX 5090 rig)

| ID | Hypothesis | Change | Baseline | Decision rule |
|---|---|---|---|---|
| EXP-L1 | RDC=0 whole-program device code changes lab-kernel codegen/throughput | build lab with `-DCMAKE_CUDA_FLAGS=...` RDC off vs on (project EXP-01 covers the application binary similarly via `make RDC=0`) | lab record at commit, RDC=1 | keep only if median improvement > 2× pooled stdev AND correctness passes |
| EXP-L2 | Loop unrolling level changes modmul256 throughput | `#pragma unroll N` variants in k_modmul256 | first lab record | same guard |
| EXP-L3 | Launch configuration (blocks×threads) changes modmul256/local-rw throughput | sweep blocks∈{256,1024,4096}, threads∈{128,256,512} | first lab record | same guard |
| EXP-L4 | Local-memory vs register-resident access ratio quantifies BN-1 | compare `local-rw` vs `int64-mul` throughput per op | both from one record | descriptive, no keep/revert |
| EXP-L5 | Warp divergence cost (BN-5) | compare `divergent` vs `int32-mix` | one record | descriptive |
| EXP-L6 | Memory bandwidth ceiling (BN-6) | `global-rw` GB/s vs device spec | one record | descriptive |
| EXP-L7 | 10-Gops modmul256 target feasibility | run modmul256 at the best EXP-L3 config | — | report achieved Gops + variance; labeled synthetic (never key-search) |

## Search-application experiments (unchanged, see optimization/EXPERIMENTS.md)

EXP-01 (rdc A/B on the application) remains prepared; E1–E6 defined in
`docs/optimization/PERFORMANCE_BASELINE.md`.

## Record format

Each lab run writes JSON+CSV+Markdown into `research/cuda-lab/results/`
and `reports/` with: utc, GPU, driver, CUDA, commit, reps, per-bench
config, samples, median/min/max/stdev, throughput, notes. Comparisons:
`run_experiments.py --compare <baseline.json> <candidate.json>` applies
the 2×-pooled-stdev noise guard.
