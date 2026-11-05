# Blackwell Performance Research — RTX 5090 (sm_120)

Research baseline: 2026-11-05 · CUDA 13.4 / driver 615.71.09 · RTX 5090
(170 SM) and RTX PRO 6000 Blackwell WE (188 SM) · sm_120.

Every finding below is tagged: **[DOC-FACT]** (authoritative
documentation), **[MEASURED]** (our hardware evidence), or
**[HYPOTHESIS]** (needs measurement). Confidence: H/M/L.

---

## 1. Hardware capabilities

| Finding | Tag | Confidence | Source |
|---|---|---|---|
| RTX 5090: Blackwell, compute capability 12.0, 170 SMs, 32 GB | DOC-FACT | H | NVIDIA product documentation + `nvidia-smi` on the rig |
| RTX PRO 6000 Blackwell WE: compute 12.0, 188 SMs | DOC-FACT | H | `nvidia-smi` on the rig |
| Same `sm_120` does not imply identical runtime behavior: different SM counts, clocks, memory configs → per-GPU launch tuning required | DOC-FACT | H | device properties; observed per-GPU optimal grids (MEASURED: `512,64` vs `128,64`) |
| Local memory is backed by L1/TEX caches; frame size alone does not determine throughput | DOC-FACT | H | CUDA C++ Programming Guide (local memory & thread execution); corroborated MEASURED: grid `1024,*` (16.7 KB frame) ≈ `128,*` (2.3 KB frame) throughput in the sweep |
| Warp size 32; warp-collective ops require participating-lane masks (entry ballot pattern in the search kernel since PR #27) | DOC-FACT | H | CUDA Programming Guide (warp collective functions) |

## 2. Compiler / CUDA 13 behavior

| Finding | Tag | Confidence | Source |
|---|---|---|---|
| CUDA 13 nvcc minimum supported arch is `sm_75` (Turing); pre-Turing dropped; `sm_75` is the documented default target | DOC-FACT | H | NVIDIA CUDA Compiler Driver NVCC documentation (docs.nvidia.com/cuda/cuda-compiler-driver-nvcc) — verified 2026-11-04 |
| `sm_120` SASS compiles with CUDA ≥ 12.8; CUDA 13.4 compiles our kernels for sm_120 (SASS + PTX artifacts in CI) | MEASURED | H | CI kernel-resource-matrix runs |
| `-rdc=true` (relocatable device code) requires `-lcudadevrt` and separates device compilation/linked device code; whole-program compilation (`RDC=0`) can differ in codegen | DOC-FACT | H | NVCC docs; A/B experiment EXP-01 prepared (runtime pending) |
| `-use_fast_math` affects floating-point only; our workloads are integer → no-op | DOC-FACT | H | NVCC docs |
| ptxas resource report (`-Xptxas -v`): main search kernel = 16.7 KB stack frame + negligible register spill at B≤512 (CI-measured) | MEASURED | H | CI ptxas reports |
| ~128 registers/thread reported for some architectures (maintainer observation) — resource numbers vary by arch/flags; treat as hypothesis until the sm_120 production report is attached | HYPOTHESIS | M | maintainer ptxas runs; script `benchmarks/profile/ncu_profile.sh` now captures sm_120 ptxas reports automatically |

## 3. Known limitations & blockers

| Limitation | Tag | Impact | Source |
|---|---|---|---|
| `ERR_NVGPUCTRPERM`: GPU performance counters not permitted for the user on the 5090 rig → Nsight Compute metrics unavailable until driver option changed | MEASURED | E1 profiling blocked; timing-only fallback exists | observed session; `ncu_profile.sh` now detects, explains, quarantines invalid output |
| Early-termination known-answer benchmark records best-observed speed, not sustained throughput | MEASURED | benchmark label discipline | GRID_RESULTS.md |
| Benchmark values so far are single best-observed runs (pre-hardening sweeps) | MEASURED | no variance/median yet | grid-sweep-2026-11-05.json; repetition workflow now available |

## 4. Performance hypotheses (to be tested in the lab / on hardware)

| ID | Hypothesis | Confidence | Required measurement |
|---|---|---|---|
| BH-1 | The search kernel's local-memory traffic is latency-hidden by field-multiply ILP (frame ≠ bottleneck) | M | E1 ncu local-mem metrics + lab `local-rw` vs `modmul256` comparison |
| BH-2 | `512,128` grid dip (~12% both GPUs) is real (launch-geometry sensitivity, e.g. threadsTotal divisibility pattern) | M | E5: repeated sweep at `512,64`/`512,128` with medians |
| BH-3 | Integer multiply throughput (IMAD chains) is the compute floor; 256-bit modmul ≈ 16×64-bit multiplies + reduction | M | lab `int64-mul` vs `modmul256` Gops |
| BH-4 | Warp divergence cost in the hash prefix check is negligible (uniform found-flag exits) | M | lab `divergent` vs `int32-mix` |
| BH-5 | RDC=0 improves codegen (0–5%) | L | EXP-01 A/B with medians |

## 5. Measurement requirements (binding)

1. Medians of ≥ 5 repetitions with min/max/stdev — never single runs.
2. Environment captured (GPU, driver, CUDA, commit, compiler flags).
3. Counter-based profiling (ncu) only when counters are permitted;
   otherwise CUDA-event timing with explicit "cannot classify
   bottlenecks" labeling.
4. Synthetic ops/s, EC ops/s, and key-search Mkeys/s never mixed.
5. Correctness gate before any benchmark run (lab tests / project suite).

Sources:
- NVCC documentation: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- CUDA C++ Programming Guide (memory model, warp functions):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/
