# Performance Baseline & Optimization Program — 2^71 project

Established: 2026-11-05 on commit `b3d13be` (CUDA 13.4, both Blackwell
GPUs). Data: [`../benchmarks/GRID_RESULTS.md`](../benchmarks/GRID_RESULTS.md).

**Label discipline: all numbers below are best-observed values from
early-termination known-answer runs — not sustained throughput, not
medians.**

## Baseline

| GPU | SMs | Initial | Best observed | Best grid |
|---|---|---:|---:|---|
| RTX 5090 | 170 | 7,800.4 Mkeys/s | 8,133.0 Mkeys/s | `512,64` |
| RTX PRO 6000 Blackwell | 188 | 7,074.8 Mkeys/s | 7,452.4 Mkeys/s | `128,64` |

Per-key cost at best observed: ~0.123 ns/key (5090), ~0.134 ns/key
(PRO 6000).

## Bottleneck classification (evidence so far)

| Candidate | Evidence | Classification |
|---|---|---|
| Kernel stack frame / local memory (H1) | 16.7 KB frame compiles in (CI ptxas); 251 static LDL/STL; grid `1024,*` (16 KB frame) performs comparably to `128,*` (2 KB frame) in the sweep | **Ambiguous** — frame size alone does not predict throughput; runtime local-memory traffic (E1) is the arbiter |
| Register pressure / occupancy (H2) | spill 8/16 B at B≤512 (CI ptxas) | **Weakened** (compile evidence) |
| Launch geometry sensitivity | `512,128` dips ~12% on both GPUs; `1024,64` outlier on 5090 (−30%) | **Observed pattern, single-run values — hypothesis** |
| Hash pipeline (H3) | zero local memory in SASS | Not yet measurable without profiler |
| Kernel launch overhead | launches amortized over multi-second slices | **Low** (structural) |
| Setup phase | affine double-and-add with per-op inversion | One-time cost; irrelevant at these range sizes (33-bit ranges) |

## E-series experiments (hardware)

| ID | Experiment | Decision |
|---|---|---|
| E1 | Nsight Compute on canonical run: `l1tex__t_*_local`, stall reasons (`stalled_long_scoreboard`, `stalled_lg_throttle`), achieved occupancy, `sm__inst_executed` per pipe | classify local-memory vs compute bottleneck → gate O-05 |
| E2 | batch-size sweep (128/256/512) at fixed work, repeated, median | frame-vs-throughput correlation |
| E3 | `make NVCC_EXTRA=-DMAX_BATCH_SIZE=512` A/B | compiled-frame-only effect |
| E5 | `512,128` vs `512,64` repetition + variance | is the dip real? |
| E6 | nsys timeline: launch gaps, host poll overhead | H5 (10 ms poll) impact |

E1/E2 remain the decision gate for the O-05 restructuring (roadmap).

## Experiment protocol (Phase 6 — binding)

One mechanism per experiment; per experiment record: ID, git commit, GPU
model, driver, CUDA version, compiler flags, grid, kernel resource usage
(`make ptxas-info`), benchmark duration, repetition count, **median**
throughput, variance, correctness status, decision (keep/revert). Log:
[`EXPERIMENTS.md`](EXPERIMENTS.md). Tooling:
`benchmarks/sweep/grid_sweep.sh` (repeatable sweep + JSON) and
`benchmarks/profile/ncu_profile.sh` (E1 metrics).

Quality gates before acceptance: build passes; 19-test suite passes;
proof suite passes where feasible; deterministic outputs unchanged; no
new CUDA errors; benchmark reproducible (median of ≥3 runs, variance
recorded).
