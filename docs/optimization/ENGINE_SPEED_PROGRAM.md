# Engine Speed Program — RTX 5090 Target >10 Gkeys/s

Directive (maintainer, 2026-11-05): **engine search throughput is the top
priority** — from the observed 7.5–8.1 Gkeys/s toward >10 Gkeys/s
(+25–30%). The search engine may now be modified for speed; every change
must pass the correctness gates (27-test suite + `proof.py` on hardware)
and be measured with medians before any claim.

## Status (2026-11-05, rig measurements — Phase 1 CLOSED)

**Current measured: 9,672.7 Mkeys/s median (`512,64`, 5 reps, commit
886548d) — best observed 9,811.9.** Two accepted code gains so far:
+6.3% from #39+#40 (launch bounds + hash forceinline) and **+7.4% from
K-2a** (hash160 instruction cuts: SHA-256 K/IV immediates +
funnelshift rotations — median 9,672.7 vs 9,006.7 same-day baseline;
the worst K-2a run, 9,529.6, beats every prior median; power dropped
to 62.3 W). All Phase-1 zero-risk levers are measured NULL (E-S1
saturated, E-S2 +0.38% < noise, E-S4 +0.44% < noise). Remaining gap to
10 Gkeys/s: **+3.4% median — next slices: K-2b (RIPEMD round
scheduling), then K-1 (subp segmentation), then K-3 (point-add).**

Reproducibility anchors: same-binary drift 0.09% (T113559Z vs
T113659Z); same-day cross-record ~0.5% (noise floor); lab repeat
within 0.3% (T121316Z).

## Where the time goes (calibrated by the 2026-11-05 lab run)

Per hashed key (B=512 typical best config):

| Stage | Estimate | Basis |
|---|---|---|
| EC arithmetic | ~3.5 256-bit modmuls ≈ 180–250 instructions | kernel structure: 2 point adds per i-step (2 sqr + ~6 mul + 1 chain mul per 2 keys), 1 _ModInv per batch (amortized) |
| hash160 | ~250–400 instructions | SHA-256 (64 rounds) + RIPEMD-160 (2×80 steps), fully unrolled, register-only |
| Local memory | ~16 B/key L1 traffic (subp) | frame analysis; **not dominant** (1024,* ≈ 128,* throughput in the sweep) |

Lab calibration (record
[lab-20261105T113806Z](../../research/cuda-lab/reports/lab-20261105T113806Z.md),
RTX 5090, commit 5c78d872, medians of 5):

| Measurement | Value | What it calibrates |
|---|---:|---|
| `modmul256` | **95.42 Gops/s** | EC ceiling: engine at 9,040 Mkeys/s × 3.5 modmul/key ≈ 31.6 Gmodmul/s = **33% of ceiling** → EC arithmetic is NOT the wall |
| `int32-mix` | 12.53 Gops/s dependent 3–4-instr iterations ≈ ~44 T instr/s | issue-rate ceiling: with ~4,000 instr/key (EC + hash160 + overhead) the whole-kernel ceiling is **~11 Gkeys/s (estimate, M confidence)** → the engine is already near the instruction-issue wall |
| `local-rw` | 102.67 Gops/s | local-memory ceiling for subp[] traffic |
| power | 64–66 W of 575 W | NOT power/thermal bound — the wall is issue rate and latency, not energy |
| `ec-scalar-mul` | INVALID | zero-scalar early-return timing bug (fixed in this PR; number must be re-measured before any use) |

**Revised verdict (measured):** launch geometry is saturated (E-S1),
EC arithmetic has 3× ALU headroom, and the binding constraint is the
**per-key instruction count** — dominated by hash160. The path to >10
Gkeys/s is *reducing instructions per key* (K-2, K-3) plus removing
latency (K-1), not more tuning of launches.

## Phase 1 — zero-code-risk experiments (rig, hours)

All use the existing binary or build knobs; measure with
`benchmarks/sweep/grid_sweep.sh --grids "..." --repeat 5` (medians,
variance, telemetry) after unlocking steady clocks:

```bash
sudo nvidia-smi -pm 1                       # persistence
sudo nvidia-smi -lgc 2400,2400              # lock clocks (thermal variance out)
bash benchmarks/sweep/grid_sweep.sh --repeat 5 \
  --grids "512,64 512,32 512,16 256,64 256,32 128,16 1024,64 1024,32"
```

(Clock lock was DENIED on the 2026-11-05 rig — medians + same-day
back-to-back runs are the compensating protocol; ~0.5% same-day noise
floor measured, ~5% cross-day drift observed.)

| ID | Experiment | Rationale | Expected | Result |
|---|---|---|---|---|
| E-S1 | extended grid/slices sweep (incl. B=16/32 variants) + clock lock | current best 8.13 @ 512,64 came from geometry alone (+4% over default); 512,128 dips 12% → strong geometry sensitivity; slices untested beyond 16 | +0–8% | **DONE** — 6 grids × 5 reps ([record](../benchmarks/data/sweep-20261105T104604Z.json)): batch ≥ 512 plateau 8,545–8,551 median (±0.1%); only small geometries lose (128,16 → 7,849). Geometry is saturated; no meaningful gain left here. |
| E-S2 | `make RDC=0` (whole-program device code) | rdc limits codegen (roadmap EXP-01) | +0–5% | **INCONCLUSIVE** — the 11:36Z sweep pair ran on one binary (no rebuild between records) and the pre-fix records cannot prove the build. Needs a labeled back-to-back pair; `.build-flags` capture (this PR) makes future records self-proving. |
| E-S3 | `make NVCC_EXTRA="-DTPB_MAX=512"` (and `-DLAUNCH_MIN_BLOCKS=1/4` combos) | new compile knobs; occupancy/ILP balance | unknown, cheap | not yet run (fold into the E-S2 labeled pair) |
| E-S4 | `make NVCC_EXTRA="-DMAX_BATCH_SIZE=512"` | halves the 16.7 KB frame; L1 pressure | unknown, cheap | not yet run — **next cheapest rig experiment** |

## Phase 2 — measurement gates (rig)

1. **Lab calibration**: ✅ DONE 2026-11-05 (table above). modmul headroom
   ratio = 95.42 / 31.6 ≈ **3.0×** — EC has ample ALU headroom; the
   constraint is instruction issue.
2. **Profiling**: unlock counters
   (`sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`) →
   `bash benchmarks/profile/ncu_profile.sh` → stall reasons and pipe
   utilization decide Phase 3 priorities. If counters stay locked, the
   lab ratios + A/B builds remain the decision tool.

## Phase 3 — kernel work (this repository, gated on Phase 1/2 evidence)

Candidates in priority order (each: isolated PR, correctness gates,
5-rep medians, keep-or-revert) — **reordered 2026-11-05 by the measured
issue-rate verdict: hash is the instruction consumer, so K-2 leads:**

| ID | Change | Hypothesis | Risk |
|---|---|---|---|
| K-2a | hash160 instruction cuts, first slice: SHA-256 K/IV constants as literal macro arguments (immediates, not `__constant__` loads); RIPEMD `ROL` → explicit `__funnelshift_l` | removes up to 64 constant-memory loads (LDC) per key, each in the `T1` dependency chain; guarantees single-instruction rotations for all 166 RIPEMD rotations. Expected +0.5–2% | low (bit-identical output, host-verified vs hashlib + python RIPEMD oracle) |
| K-2b2 | message assembly via `__byte_perm` in `SHA256_33_from_limbs` (9 words/key: ~10 instr → 1 each) | removes ~70-80 instructions per key from the hot path — real instruction-count cut (slice 1 bswap was already compiler-optimized and reverted) | low (host-verified bit-exact; rig medians) |
| K-2 | further hash160 micro-optimizations (RIPEMD round scheduling, register allocation, shared subexpressions between the 2×80 steps) | hash ≈ dominant share of per-key instructions → single-digit % each, compounding | medium |
| K-1 | subp[] segmentation (`SUBP_SEG` compile knob, 0=legacy default) | **REJECTED 2026-11-06** (new rig, default-build baseline 9,453.4): SEG=16 → 6,911.8 (−27%); SEG=8 → 7,254.4 / 7,319.6 (−23%, reproduced). Coverage PERFECT on both (proof.py 848/848) — the register window q[SEG][4] (64/128 regs) spills and re-materialization muls add EC work. Knob stays (default 0 = release unaffected) as negative evidence. Consequence: local-memory indexing is NOT the wall — per-key instruction count remains the validated lever. Next: K-3. |
| K-3 | point-add restructure (share subexpressions between the ± pair) | fewer modmuls per i-step (EC has 3× headroom, so gains here are real but bounded) | medium-high |
| K-4 | __reduce/ballot-based hash counting, poll cadence | minor | low |

## External library evaluation (maintainer question, 2026-11-05)

"Can replacing the scalar-multiplication logic — e.g. with
[UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1) —
increase search speed?" — **No, not for the search hot path**, for a
structural reason: the engine does **not** perform scalar
multiplications per key. The hot loop is an incremental ±(i+1)·G walk
with batch inversion (~3.5 modmul/key); a full k·G ladder costs ~two
orders of magnitude more. Full scalar mults happen only in the one-time
setup phase (seconds), and the lab calibration shows the engine uses
just 33% of the modmul ceiling — EC arithmetic is not the bottleneck;
the hash pipeline is. UltrafastSecp256k1's self-reported ~1.5× wins are
CPU-side ECDSA/signing vs libsecp256k1, on a different workload; its
CUDA backend remains an evaluation candidate for the lab only
(see [SECP256K1_COMPARISON](../research/SECP256K1_COMPARISON.md)). No
integration into the search engine.

## Rules (binding)

1. No claim without medians ≥5 reps + correctness pass.
2. Synthetic lab Gops ≠ engine Mkeys/s — never compared.
3. Every accepted change lands in PERFORMANCE_HISTORY with its record.
4. The 10 Gkeys/s goal is a target, not a promise: the program's success
   criterion is **measured, reproducible improvement** and honest
   reporting of the ceiling we find.
