# CUDA 13 & Blackwell Compatibility — 2^71 project

Status: **implemented, awaiting hardware verification** (see Verification
Status below). This document defines the supported toolchain matrix and the
reproducible build procedure.

---

## Supported targets

| Target | Architecture | Requirement |
|---|---|---|
| NVIDIA RTX 5090 | `sm_120` (Blackwell GeForce) | CUDA Toolkit ≥ 12.8 to compile SASS; CUDA 13.x recommended |
| NVIDIA RTX PRO 6000 Blackwell Workstation | `sm_120` | same toolchain requirements as RTX 5090 |
| Legacy targets | `sm_75`, `sm_86`, `sm_89` | CUDA 13 minimum arch is sm_75 — pre-Turing (Maxwell/Pascal/Volta) is **not supported** by CUDA 13 |

**Both primary GPUs are `sm_120`, but they are NOT identical runtime
configurations.** The RTX 5090 has 170 SMs / 32 GB GDDR7; the RTX PRO 6000
Blackwell has 240 SMs / 96 GB. The application auto-sizes its grid from
device properties (`project_2_71.cu:540` uses `prop.multiProcessorCount` and
`prop.totalGlobalMem`), so no code change is needed per GPU — but optimal
`--grid`/`--slices` values must be re-measured per model. The historical
README benchmark table shows how strongly speed depends on grid tuning;
those numbers were measured on CUDACyclone, not on this project (see
ATTRIBUTION.md), and must not be reused as baseline claims.

---

## Toolchain requirements

| Component | Minimum | Notes |
|---|---|---|
| CUDA Toolkit | 12.8 (sm_120 SASS); **13.x recommended** | `nvcc --version` to check; CUDA 13 default target arch is `sm_75` |
| NVIDIA driver | R570+ for Blackwell hardware; R580+ for CUDA 13.x runtime | verify with `nvidia-smi` — the "CUDA Version" field shows the driver's maximum supported CUDA |
| OS | Linux x86_64 (primary); Windows via WSL2 | under WSL2 the *Windows* driver version matters |
| Compiler mode | C++17 (`-std=c++17`) | unchanged from original project |

## Reproducible build

```bash
git clone https://github.com/algomanlife/2-71-project.git
cd 2-71-project
make                # auto-detects toolkit + GPU, prints full flag set
make ptxas-info     # register/spill report per translation unit
make clean
```

The `info` target prints the exact toolkit version, detected GPU compute
capability, architecture list, and full nvcc command line — capture this
output in every benchmark report.

Explicit overrides (reproducibility on GPU-less build hosts):

```bash
make GPU_ARCH=120 SM_ARCHS="75 86 89 120"   # pin to a fixed set
make GPU_ARCH=                              # build without GPU detection
```

### What the Makefile now does (vs. original)

1. Detects the CUDA toolkit version and includes `sm_120` only when the
   toolkit supports it (≥ 12.8), keeping CUDA 12 compatibility.
2. Deduplicates and sorts the architecture list; appends the runtime-detected
   GPU arch.
3. Embeds PTX for the newest arch (`compute_<PTX>`), so future GPUs can JIT
   even without matching SASS.
4. GPU detection no longer prints errors on GPU-less machines.
5. Adds `ptxas-info` target (`-Xptxas -v`) for register/spill evidence
   (audit hypothesis H2).
6. Runtime diagnostics: the binary already prints device name, compute
   capability, SM count, and memory utilization at startup
   (`project_2_71.cu:703-715`).

### Unchanged (deliberately)

- All kernel and host source code — computational behavior is preserved.
- `-rdc=true -use_fast_math --ptxas-options=-O3` flags. `-use_fast_math` is
  a no-op for this integer-only workload (audit §4); `-rdc=true` is a
  performance *hypothesis* (H4) that will be A/B tested in an isolated
  `opt/*` PR, not changed as part of compatibility.

---

## Verification status

| Check | Status |
|---|---|
| Compiles with CUDA 13 for sm_75/86/89/120 | **Pending** — no CUDA toolkit on authoring machine; CI job (CPU-only) will compile in the `nvidia/cuda:13.0.0-devel-ubuntu24.04` container |
| Runs on RTX 5090 | **Pending** — requires hardware |
| Runs on RTX PRO 6000 Blackwell | **Pending** — requires hardware |
| GPU benchmark numbers | **Pending** — none claimed |

Per project policy: no successful GPU testing is claimed without access to
the specified hardware. Update this table (and
`docs/benchmarks/BASELINE_RESULTS.md`) when hardware runs complete.
