# Baseline Results — 2^71 project

**Status: first measured results published (best-observed,
early-termination known-answer runs).** See the binding interpretation
rules in [`GRID_RESULTS.md`](GRID_RESULTS.md) — these are not sustained
throughput and not medians of repeated runs.

## Measured results (2026-11-05, commit `b3d13be`)

| Commit | GPU | CUDA | Config | Correctness | Best observed Mkeys/s | Record |
|---|---|---|---|---|---|---|
| b3d13be | RTX 5090 (170 SM) | 13.4 | `512,64` | key found, suite 19/19 | 8,133.0 | [grid-sweep-2026-11-05.json](data/grid-sweep-2026-11-05.json) |
| b3d13be | RTX PRO 6000 Blackwell (188 SM) | 13.4 | `128,64` | key found (12/12 grids) | 7,452.4 | [grid-sweep-2026-11-05.json](data/grid-sweep-2026-11-05.json) |
| b3d13be | RTX 5090 | 13.4 | initial baseline | key found | 7,800.4 | [grid-sweep-2026-11-05.json](data/grid-sweep-2026-11-05.json) |
| b3d13be | RTX PRO 6000 | 13.4 | initial baseline | key found | 7,074.8 | [grid-sweep-2026-11-05.json](data/grid-sweep-2026-11-05.json) |

Full 12-configuration grid sweep: [`GRID_RESULTS.md`](GRID_RESULTS.md).

**Label: "Best Observed — Grid Sweep"** (runs stopped when the known
reference key was found). Sustained-throughput baselines with repetition
and telemetry come from `benchmarks/sweep/grid_sweep.sh` in a later phase.

## Historical context (NOT a baseline of this project)

The original CUDACyclone project reported (via README, unverified in this
project, attribution in ATTRIBUTION.md):

| GPU | Grid | Speed | Source |
|---|---|---|---|
| RTX 4090 | 128,1024 | 6214 Mkeys/s | community report, original README |
| RTX 4060 | 512,512 | 1238 Mkeys/s | original author's report, original README |
| RTX 5090 | 128,256 | 8408 Mkeys/s (7 s sample) | original README example |

Retained for context only; produced by CUDACyclone builds with unknown
toolchain state and no recorded methodology. Not comparison points for
this project's optimization claims.

## Open measurement gaps

- Repeated runs with median + variance (Phase-7 sweep workflow)
- Power/clock telemetry during runs
- RTX PRO 6000 driver version
- Nsight Compute profiling (E1) — occupancy, stall reasons, local-memory
  traffic, integer pipe utilization
- Registers-per-thread on the production build (`make ptxas-info` on the
  target machine)
