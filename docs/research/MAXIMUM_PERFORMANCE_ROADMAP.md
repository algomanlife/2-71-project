# Maximum Performance Roadmap — 2^71 project

Research-phase document (Phase 5). Every proposal below is an **estimate**
grounded in the static audit (`docs/audits/INITIAL_TECHNICAL_AUDIT.md`,
hypotheses H1–H9). Nothing here is a measured result. No proposal may be
merged without: (1) profiler/benchmark evidence on target hardware, and
(2) passing the full correctness suite
(`tests/` + `proof.py`).

Legend — Priority: P1 (do first once hardware is available) … P4 (backlog).
Risk: correctness risk if implemented carelessly.

---

## Stage A — Evidence gathering (before any optimization)

| ID | Action | Output |
|---|---|---|
| A1 | `make ptxas-info` on RTX 5090 build | registers/thread + spill bytes per kernel (H2 evidence) |
| A2 | Nsight Compute profile of the canonical run | occupancy, warp stall breakdown, local-memory LD/ST volume (H1/H3 evidence) |
| A3 | `nvdisasm` SASS of `kernel_point_add_and_check_oneinv` for sm_120 | instruction mix, LDL/STL, IMAD/MUL chains |
| A4 | Baseline records via `benchmarks/baseline/run_baseline.sh` on both GPUs | first rows of `PERFORMANCE_HISTORY.md` |
| A5 | Grid/slices sweep on both GPUs (config-only, no code change) | per-GPU optimal runtime configuration table |

Stage A costs no code risk and gates everything else.

---

## Stage B — Foundational, low-risk candidates

### O-01 — Remove verified dead code
- **What:** delete `USE_SYMMETRY` block, `addBigEndian256`, `compare20`,
  `MatchResult`, `mul64x64_128`, `addWithCarry`, unused device globals
  (audit §6, exact lines listed there); fix triple declaration of
  `scalarMulKernelBase`.
- **Expected benefit:** none claimed (code is never executed) — compile
  time and review surface only.
- **Complexity:** trivial. **Risk:** minimal (build + tests confirm).
- **Validation:** full test suite; diff confirms no live-path change.
- **Priority:** P3 (hygiene).

### O-02 — Build-flag A/B: relocatable device code (H4)
- **What:** build variant without `-rdc=true`/`-lcudadevrt` (requires
  resolving `__noinline__` usage legality first).
- **Expected benefit (estimate):** 0–5%; whole-program codegen may
  schedule better and removes module load overhead.
- **Complexity:** low. **Risk:** low.
- **Validation:** identical correctness output on known-answer + proof.py;
  benchmark vs baseline on same GPU.
- **Priority:** P1 (cheap experiment, immediate signal).

### O-03 — Host-side found-flag polling (H5)
- **What:** replace the 10 ms blocking `cudaMemcpy` D2H of `found_flag`
  (`project_2_71.cu:771-773`) with a zero-copy mapped host-visible flag or
  `cudaEventQuery`-based completion detection.
- **Expected benefit (estimate):** near-zero to a few % (copy is small);
  removes potential stream serialization during launches.
- **Complexity:** low-medium. **Risk:** low (host-side only).
- **Validation:** benchmark A/B; Ctrl-C interrupt path still tested.
- **Priority:** P2.

### O-04 — Setup-phase speedup (H7)
- **What:** batch inversion for the initial `scalarMulKernelBase` phase
  (affine double-and-add currently does one field inversion per op).
- **Expected benefit (estimate):** reduces one-time startup latency only
  (seconds); no steady-state throughput change.
- **Complexity:** medium. **Risk:** medium (touches point arithmetic).
- **Validation:** scalar-mul correctness tests + reference cross-check.
- **Priority:** P4 (startup cost is amortized over long runs).

---

## Stage C — Hot-path restructuring (requires A1–A3 evidence)

### O-05 — Local-memory elimination for `subp[]` (H1)
- **What:** the `(B/2)·32 B` per-thread array `subp`
  (`project_2_71.cu:140`) lives in local memory. Restructure into
  segment-wise processing (e.g. segments of 16–32 points computed in
  registers with per-segment prefix products), or a two-pass
  forward/backward product scheme that stores only O(1) values per thread.
- **Expected benefit (estimate):** potentially large if H1 is confirmed by
  A2/A3 (local LD/ST currently on the critical path of every batch);
  bounded by the EC+hash arithmetic floor (H3).
- **Complexity:** high. **Risk:** high — exact same keys must be produced
  (proof.py mod-B residue coverage is the safety net).
- **Validation:** proof.py full run + new GPU tests + per-config sweep;
  benchmark vs baseline.
- **Priority:** P1 **if** A2/A3 confirm local-memory dominance, else P3.

### O-06 — Occupancy/launch-bounds tuning (H2)
- **What:** sweep `__launch_bounds__(256, N)` for N ∈ {1,2,3,4} and
  `threadsPerBlock` ∈ {128, 256, 512} (currently fixed 256
  `project_2_71.cu:520-522`).
- **Expected benefit (estimate):** 0–15% depending on A1 register report.
- **Complexity:** low per experiment. **Risk:** low.
- **Validation:** each variant through the full suite (grid divisibility
  must hold — `pick_threads_total` adapts).
- **Priority:** P1 (after A1).

### O-07 — Hash pipeline instruction efficiency (H3)
- **What:** investigate (a) merging SHA-256 schedule loads, (b) RIPEMD-160
  dual-line round pairing, (c) `__noinline__` boundary of
  `getHash160_33_from_limbs` (H6), on sm_120 SASS.
- **Expected benefit (estimate):** unknown until A3; if EC stalls dominate,
  hash work is free ILP and gains are small.
- **Complexity:** medium-high. **Risk:** medium (hash must remain exact —
  reference tests catch deviations).
- **Validation:** CPU reference vectors + GPU hash160 cross-check.
- **Priority:** P2 (after A3 evidence).

---

## Stage D — Architecture-specific research (Blackwell)

### O-08 — sm_120 codegen study
- **What:** inspect Blackwell SASS for the field-mul PTX carry chains
  (`CUDAMath.h:5-26`): 64-bit `mad` lowering, IMAD.HI usage, register bank
  conflicts. Compare 64-bit limb vs 32-bit limb implementations.
- **Expected benefit (estimate):** unknown; hypothesis only.
- **Complexity:** high. **Risk:** high (rewrite of core arithmetic).
- **Priority:** P3 (only after Stages B/C measured).

### O-09 — Per-GPU default tuning
- **What:** auto-select `--grid/--slices` defaults per device
  (SM count / VRAM already drive sizing; add a device-profile table from
  A5 sweep data for 5090 vs PRO 6000).
- **Complexity:** low. **Risk:** none to correctness.
- **Priority:** P2 after A5.

---

## Rejected / deferred ideas (documented to avoid re-litigating)

| Idea | Reason |
|---|---|
| Change secp256k1 algorithms / EC formulas | Out of scope by spec (behavior preservation); math changes only via dedicated, fully-validated rewrites |
| Skip RIPEMD-160 for most candidates | Not sound: hash160 output is the search filter; no cheaper sound prefilter exists |
| `-use_fast_math` removal | no-op for integer workload (audit §4) — cosmetic only |
| Persistent kernel + CUDA graphs | launches are already amortized (one per multi-second slice); expected gain negligible |

---

## Process

1. Each optimization = one branch `opt/<id>` + one PR with the 9-section
   template (objective, current, proposed, files, rationale, correctness
   results, benchmark comparison, limitations, rollback).
2. Merged only with Pass correctness + measured, non-regressed benchmark.
3. Regressions are reverted (rollback section) and recorded as Rejected in
   `PERFORMANCE_HISTORY.md`.
4. Never add percentages of unrelated optimizations (per-GPU cumulative
   baseline only).
