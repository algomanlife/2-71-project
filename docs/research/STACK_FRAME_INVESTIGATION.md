# Stack Frame Investigation — main kernel (`kernel_point_add_and_check_oneinv`)

Investigation date: 2026-11-05
Code state: commit `da4b5a6` (main, post PR #12)
Evidence source: CI run `41906226978` — CUDA 13.0 container,
`nvidia/cuda:13.0.0-devel-ubuntu24.04`, CPU-only runner (compile-time
evidence only, **no GPU execution**). SASS/PTX artifacts retained 90 days
as `sm120-sass-ptx`.

All facts below are **compile-time measurements**. No runtime performance
claim is made anywhere in this document.

---

## 1. Executive summary

The 16,688-byte stack frame reported in the initial audit addendum is
**fully attributed**: it is dominated by the worst-case instantiation of
the per-thread batch-inversion product array `subp[]`
(`project_2_71.cu:140`), whose size is `16·B` bytes for batch size `B`
because the kernel compiles against the compile-time maximum
`MAX_BATCH_SIZE = 1024` while `B` is a runtime parameter. The frame scales
linearly with `MAX_BATCH_SIZE` exactly as predicted, local-memory
instructions (LDL/STL) are present in the hot kernel's SASS, and the
frame is *not* register-spill driven (spill is 8/16 bytes at B≤512).
Whether this local-memory traffic measurably limits throughput on RTX
5090 / RTX PRO 6000 **remains an open runtime question** (experiments in
§6).

## 2. Measured evidence

### 2.1 Stack frame vs `MAX_BATCH_SIZE` (ptxas -v, sm_120, CUDA 13.0)

| `MAX_BATCH_SIZE` | Kernel stack frame | Register spill (st/ld) | `subp[]` size `16·B` | Residual |
|---|---|---|---|---|
| 128  | 2,352 B  | 8 / 16 B | 2,048 B  | 304 B |
| 256  | 4,400 B  | 8 / 16 B | 4,096 B  | 304 B |
| 512  | 8,496 B  | 8 / 16 B | 8,192 B  | 304 B |
| 1024 | 16,688 B | 64 / 124 B | 16,384 B | 304 B |

Regression: `frame = 16·MAX_BATCH_SIZE + 304` (exact through 512;
at 1024 the extra spill accounts for the small deviation family
16,688/16,704/16,720 observed across rdc compilation passes).

**Attribution:**
- `subp[(MAX_BATCH_SIZE/2)][4]` × 8 B = `16·MAX_BATCH_SIZE` bytes —
  **~98% of the frame at B=1024** (`project_2_71.cu:140`).
- Residual 304 B: small arrays (`h20[20]`) and indexed locals; consistent
  with the 44 low-offset local accesses observed in SASS (§2.2).
- Spill stores/loads are negligible (8/16 B) for B ≤ 512 — the frame is
  **not** caused by register pressure (audit H2 is weakened; H1's
  structural premise is confirmed).

### 2.2 SASS instruction census (sm_120, `cuobjdump -sass`)

| Function | SASS lines | LDL | STL | LDG | STG |
|---|---|---|---|---|---|
| `kernel_point_add_and_check_oneinv` | 15,972 | **180** | **71** | 19 | 76 |
| `_ModInv` | 2,260 | 38 | 58 | 0 | 0 |
| `scalarMulKernelBase` | 7,092 | 12 | 15 | 5 | 24 |
| `getHash160_33_from_limbs` | 4,388 | 0 | 0 | 0 | 0 |

Observations:
- The hot kernel contains **251 static local-memory instructions**
  (LDL+STL). These sit inside the per-`i` loop
  (`project_2_71.cu:172-328`), so dynamic local-memory traffic scales
  with `(B/2) × iterations` per batch — static counts understate dynamic
  volume. Exact dynamic volume requires runtime profiling.
- Offset histogram of the kernel's local accesses: 44 accesses at offsets
  ≤ 0x30 (small-object traffic) and 136 accesses at offsets
  0x4000–0x4124 (the large-array region; max offset 0x4124 fits inside
  the 16,688 B = 0x4130 frame).
- `getHash160_33_from_limbs` uses **zero** local memory — the hash
  pipeline is register-only (relevant to roadmap O-07: hash work does not
  add local-memory pressure).

### 2.3 PTX note

The `cuobjdump -ptx` artifact (`ptx_sm120.txt`) contains only fatbin
headers — the linked binary embeds no reviewable PTX body for these
functions, so no PTX-level claim is made here. (A `nvcc -ptx` compile
step can be added to the CI matrix job if PTX-level inspection becomes
useful; SASS evidence in §2.2 is sufficient for this investigation.)

## 3. Answers to the Phase-3 questions

| # | Question | Answer | Status |
|---|---|---|---|
| 1 | Which variables contribute? | `subp[]` (~98%), residual 304 B small locals/spill | **measured** (frame scaling + offsets) |
| 2 | Is `subp[]` dominant? | Yes, exactly `16·B` of the frame | **measured** |
| 3 | Is local-memory traffic generated? | Yes: 251 static LDL/STL in the hot kernel, executed per loop iteration | **compile-verified; dynamic volume pending** |
| 4 | Can memory accesses be simplified? | Candidate: segment-wise subp or two-pass O(1) forward/backward product (roadmap O-05) | hypothesis |
| 5 | Can lifetimes be shortened? | `subp[i]` is consumed within the same batch (`project_2_71.cu:144-176`); lifetime is already one batch — restructure would shorten *residency*, not lifetime | analysis |
| 6 | Unnecessary instructions? | None identified statically; 8/16 B spill is benign. `--use_fast_math` is a no-op for integer code | analysis |
| 7 | Could changes improve resource utilization? | Potentially large for O-05 **if** runtime profiling confirms local-memory-bound stalls; frame reduction alone is not a goal | hypothesis |
| 8 | Measurable runtime overhead? | **Unknown — pending hardware.** Frame size alone does not imply throughput loss: local memory is cached in L1, and latency may be hidden by the EC/hash arithmetic ILP | pending |

## 4. Key caveat (explicit)

**Reducing the stack frame does not automatically increase throughput.**
Local memory is backed by L1/TEX caches on Blackwell; if the loop's
LDL/STL hit in L1 and latency is hidden by independent field-multiply
work, the local-memory traffic may be effectively free. Conversely, if
Nsight Compute shows `l1tex` local-op stalls or L1 miss traffic to L2,
the array is a first-class bottleneck. Only experiment E1 (§6) can
decide this. This document deliberately reports no runtime numbers.

## 5. Risks of the restructure (O-05)

- The ±G walk and the J-transition depend on the exact prefix-product
  recurrence; a two-pass rewrite must reproduce identical key sequences.
- `subp` elimination increases per-step arithmetic (recomputing products)
  — trading local-memory traffic for multiplications; net effect unknown
  a priori.
- Safety net: `proof.py` mod-B residue coverage + GPU reference tests.

## 6. Proposed experiments (hardware required)

| ID | Experiment | Decision it answers | Method |
|---|---|---|---|
| E1 | Nsight Compute local-memory metrics on the canonical baseline run | Is the kernel local-memory-bound? | `l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st`, warp stall reasons (`stalled_lg_throttle`, `stalled_long_scoreboard`), achieved occupancy |
| E2 | Runtime batch-size sweep (B=128/256/512) at fixed total work | Does frame size correlate with throughput? | `run_baseline.sh --grid ...` per config; records into `docs/benchmarks/data/` |
| E3 | A/B build with `MAX_BATCH_SIZE=512` compiled cap (behavior-identical for B≤512 runs) | Does a smaller compiled frame alone change throughput? | `make GPU_ARCH=120 SM_ARCHS=120 NVCC_EXTRA=-DMAX_BATCH_SIZE=512` — requires a Makefile knob (candidate small PR) |
| E4 | SASS dynamic hot-loop analysis with `ncu --set full` source correlation | Which LDL/STL dominate dynamic counts | profile + `ncu --import` analysis |

E1 and E2 are the decision gate for O-05 (roadmap priority flip).

## 7. Related items

- Audit H1/H2: `docs/audits/INITIAL_TECHNICAL_AUDIT.md` (addendum)
- Roadmap O-05, O-06: `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md`
- Evidence artifacts: CI run 41906226978, artifacts `sm120-sass-ptx`
  (sass_sm120.txt, ptx_sm120.txt, 90-day retention)
