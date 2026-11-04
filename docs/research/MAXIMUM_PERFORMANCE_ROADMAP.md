# Maximum Performance Roadmap — 2^71 project (v2)

Evidence-gated optimization program for CUDA 13 / Blackwell `sm_120`.
Revision: 2026-11-05 (Phase 3). Supersedes v1; history is in git.

**Ground rules (unchanged, binding):**
1. Every optimization PR passes the full correctness suite (CPU reference
   tests + GPU suite + `proof.py` where hardware allows) before merge.
2. Every performance claim is a **measurement** recorded in
   `docs/benchmarks/PERFORMANCE_HISTORY.md`; estimates live only here,
   always with a confidence level.
3. Improvements are compared cumulatively per GPU+driver+CUDA
   configuration; percentages of unrelated optimizations are never added.
4. No arbitrary throughput target is promised. The objective is maximum
   practical performance with correctness preserved.

**New evidence since v1 (compile-time, CI-verified):**
- `docs/research/STACK_FRAME_INVESTIGATION.md`: kernel stack frame =
  `16·MAX_BATCH_SIZE + 304` B (subp[] is ~98%); 251 static LDL/STL in the
  hot kernel SASS; register spill negligible (8/16 B at B≤512) → **H1
  structurally confirmed, H2 weakened**. Runtime impact: unknown until
  experiments E1–E4 run on hardware.
- Hash pipeline uses zero local memory (SASS census) → hash-side
  restructuring will not relieve local-memory pressure.

---

## Stage A — evidence gathering (P0, gates everything)

| ID | Action | Output |
|---|---|---|
| A1 | ✅ done | `make ptxas-info` register reports (CI) |
| A2 | ✅ done | SASS census + stack-frame matrix (CI run 41906226978) |
| A3 | ⬜ Nsight Compute profile, canonical run, both GPUs | occupancy, stall reasons, local-memory volume (E1) |
| A4 | ⬜ First baseline records via `run_baseline.sh`, both GPUs | first measured rows of PERFORMANCE_HISTORY |
| A5 | ⬜ `--grid/--slices` sweep per GPU (config-only) | per-GPU optimal configuration table |
| A6 | ⬜ Batch-size sweep B=128/256/512 at fixed work (E2) | frame-vs-throughput correlation |

## Optimization proposals

Legend: Confidence = estimate confidence, not measured benefit.
Priority: P0 = gating/evidence, P1 = first implementation wave,
P2 = later, P3 = backlog. Status: pending / in-PR / done / rejected.

| Optimization | Current limitation | Expected benefit (confidence) | Cost | Correctness risk | Required hardware | Validation | Priority | Status |
|---|---|---|---|---|---|---|---|---|
| **O-01** remove verified dead code | none at runtime (compile/review surface only) | 0% runtime (high conf.) | Low | Low | none | CI compile + CPU tests + pure-deletion diff | P3 | **in PR #14** |
| **O-05** restructure `subp[]` out of local memory (segment-wise products or two-pass O(1) residency) | 16·B B/thread local frame; 251 static LDL/STL in hot loop; runtime impact unconfirmed | unknown until E1/E2; if local-memory-bound: potentially large (low conf.) | High | High | RTX 5090 + PRO 6000 | E1/E2 decision gate; full suite + proof.py; per-config sweep; benchmark records | P1 *after E1/E2* | pending |
| **O-06** occupancy / launch-bounds / threadsPerBlock sweep | `__launch_bounds__(256,2)` fixed; spill negligible → headroom may exist (low conf.) | 0–15% (low conf.) | Low per variant | Low | both GPUs | each variant: full suite + benchmark | P1 | pending |
| **O-02** build-flag A/B: rdc on/off | `-rdc=true` may limit whole-program codegen | 0–5% (low conf.) | Low | Low | one GPU enough | A/B build identical results + benchmark | P1 (cheap) | pending |
| **O-03** host found-flag polling → zero-copy mapped flag | 10 ms blocking D2H copy per poll (`project_2_71.cu:771-773`) | near-zero–few % (low conf.) | Low-Med | Low | one GPU | A/B benchmark; Ctrl-C path test | P2 | pending |
| **O-07** hash pipeline instruction efficiency | hash is register-only, zero local mem; EC dominates? unknown (A3) | small if EC-bound (low conf.) | Med-High | Med | both GPUs | A3 SASS + stall data first; CPU hash vectors catch deviations | P2 | pending |
| **O-09** per-GPU default grid/slices auto-tuning | defaults not per-device; SM count already dynamic | user-facing convenience + best-case perf (med conf.) | Low | none | both GPUs | A5 sweep data → device profile table | P2 | pending |
| **E3-knob** `NVCC_EXTRA=-DMAX_BATCH_SIZE=512` A/B | compile-time frame cap experiment | measures frame-only effect (med conf. of decision value) | Low | Low | one GPU | identical-results check + benchmark | P1 | knob **in PR (opt/nvcc-extra-knob)** |
| **O-04** setup-phase batch inversion | one field inversion per affine op (startup only) | startup seconds only (high conf.) | Med | Med | any GPU | setup timing + scalar-mul tests | P4 | pending |
| **O-08** sm_120 codegen study (32-bit limbs, IMAD chains) | unknown SASS quality vs hand-tuned | unknown (very low conf.) | High | High | both GPUs | A3 evidence first; reference cross-checks per rewrite | P3 | pending |

## Rejected / deferred (unchanged)

| Idea | Reason |
|---|---|
| Change secp256k1 algorithms / EC formulas | out of scope (behavior preservation); only via dedicated fully-validated rewrites |
| Skip RIPEMD-160 for most candidates | unsound: hash160 is the search filter |
| `-use_fast_math` removal | no-op for integer workload (audit §4) |
| Persistent kernel / CUDA graphs | launches amortized over multi-second slices; negligible expected gain |
| Reducing the stack frame for its own sake | frame size ≠ throughput; local memory is L1-cached; E1/E2 decide (investigation §4) |

## Process

Unchanged from v1: one branch `opt/<id>` per optimization, 9-section PR
template, merge only with correctness pass + non-regressed measurement,
regressions recorded as Rejected.
