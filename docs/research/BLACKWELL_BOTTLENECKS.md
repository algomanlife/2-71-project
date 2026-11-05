# Blackwell Bottleneck Analysis — Status

Date: 2026-11-05 · **Status: counter-based profiling is currently
BLOCKED** on the RTX 5090 rig (`ERR_NVGPUCTRPERM` — GPU performance
counters not permitted for the user; see
`benchmarks/profile/ncu_profile.sh` which detects and quarantines this).
This document therefore contains **no profiler measurements** — only
compile-time evidence, benchmark observations, and lab experiment
definitions. Nothing here claims a confirmed bottleneck without profiler
or controlled-experiment evidence.

## Evidence inventory

| Evidence | Type | Source |
|---|---|---|
| Main kernel: 16,688 B stack frame, 251 static LDL/STL (sm_120 SASS), spill 8/16 B at B≤512 | compile-time | CI kernel-resource-matrix (run 41906226978) + artifacts |
| Frame = 16·MAX_BATCH_SIZE + 304 (measured at 4 sizes) | compile-time | same |
| Hash pipeline uses zero local memory (SASS census) | compile-time | same |
| Grid sweep: `512,128` dips ~12% on both GPUs; `1024,64` outlier on 5090 | single-run best-observed | grid-sweep-2026-11-05.json |
| Grid `1024,*` (16.7 KB frame) ≈ `128,*` (2.3 KB frame) throughput | single-run best-observed | same |
| ncu profiling produces no valid metrics (counter permission) | environment limitation | observed on rig; script detects |

## Bottleneck candidates (for the standalone lab workloads)

| ID | Candidate bottleneck | Evidence so far | Impact | Possible causes | Confidence | Candidate experiment |
|---|---|---|---|---|---|---|
| BN-1 | Local-memory latency (subp[] class) | frame + LDL/STL compile evidence; 1024-vs-128 sweep suggests NOT dominant | unknown | L1-resident, ILP-hidden | M (either way) | lab `local-rw` vs `modmul256`; E1 when counters unlocked |
| BN-2 | Integer multiply throughput (IMAD chains in 256-bit mult) | structural (16×64-bit mults per modmul) | likely the compute floor for modmul-heavy work | pipeline saturation | M | lab `int64-mul` vs `modmul256` ratio |
| BN-3 | Launch-geometry sensitivity (`512,128` dip) | sweep pattern (single runs) | up to ~12% in that config | threadsTotal divisibility / occupancy edge | M | E5 repeated sweep; lab launch-latency bench |
| BN-4 | Field-inversion latency (setup-phase class) | one divsteps inversion per affine op | setup only (not search hot path) | divstep loop dependency chain | H (measured as setup-only cost by construction) | lab `ec-scalar-mul` throughput |
| BN-5 | Warp divergence in hash prefix check | uniform exit paths by construction | likely negligible | — | M | lab `divergent` vs `int32-mix` |
| BN-6 | Memory bandwidth (global working set) | 64 B/thread × threadsTotal | unknown | streaming pattern | L | lab `global-rw` GB/s vs spec bandwidth |

## What would confirm/refute each

- BN-1/BN-2/BN-5/BN-6: run `python3 research/cuda-lab/scripts/run_experiments.py --reps 5`
  on the rig (medians + variance), then compare cross-bench ratios.
- BN-3: E5 (repeated `512,64` vs `512,128` sweep with the hardened
  grid_sweep.sh).
- Anything counter-level (occupancy, stall reasons): unlock counters
  (`NVreg_RestrictProfilingToAdminUsers=0`, see ncu_profile.sh header)
  and run `bash benchmarks/profile/ncu_profile.sh`.

Rules: no bottleneck is "confirmed" without either (a) profiler metrics
or (b) a controlled repeated experiment exceeding the noise guard.
