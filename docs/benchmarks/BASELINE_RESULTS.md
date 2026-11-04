# Baseline Results — 2^71 project

**Status: PENDING — no measurements exist yet.** This project has not been
built or run on target hardware (RTX 5090 / RTX PRO 6000 Blackwell) under
the new validation framework. Nothing below is a measurement.

## Measured results

| Commit | GPU | CUDA | Config | Correctness | Peak Mkeys/s | Record |
|---|---|---|---|---|---|---|
| — | — | — | — | — | **pending hardware run** | — |

Records will be appended by `benchmarks/baseline/run_baseline.sh` as
`docs/benchmarks/data/*.json` and summarized here by machine, never
hand-typed.

## Historical context (NOT a baseline of this project)

The original CUDACyclone project reported (via README, unverified in this
project, attribution in ATTRIBUTION.md):

| GPU | Grid | Speed | Source |
|---|---|---|---|
| RTX 4090 | 128,1024 | 6214 Mkeys/s | community report, original README |
| RTX 4060 | 512,512 | 1238 Mkeys/s | original author, original README |
| RTX 5090 | 128,256 | 8408 Mkeys/s (7 s sample) | original README example |

These numbers are retained for context only. They were produced by
CUDACyclone builds with unknown compiler/toolkit/driver state and no
recorded methodology, and must not be used as comparison points for
optimization claims in this project.

## Required fields for the first real baseline

Per the Phase-3 spec, the first hardware run must record: GPU model and
compute capability, driver and CUDA versions, compiler options
(`make info` output), GPU clock and power conditions (dmon samples),
kernel execution time, throughput, registers per thread
(`make ptxas-info`), and — where Nsight Compute is available — achieved
occupancy, warp stall reasons, memory throughput, and integer pipeline
utilization.
