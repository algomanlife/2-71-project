# Engine Speed Program — RTX 5090 Target >10 Gkeys/s

Directive (maintainer, 2026-11-05): **engine search throughput is the top
priority** — from the observed 7.5–8.1 Gkeys/s toward >10 Gkeys/s
(+25–30%). The search engine may now be modified for speed; every change
must pass the correctness gates (27-test suite + `proof.py` on hardware)
and be measured with medians before any claim.

## Where the time goes (evidence-based estimate)

Per hashed key (B=512 typical best config):

| Stage | Estimate | Basis |
|---|---|---|
| EC arithmetic | ~3.5 256-bit modmuls ≈ 180–250 instructions | kernel structure: 2 point adds per i-step (2 sqr + ~6 mul + 1 chain mul per 2 keys), 1 _ModInv per batch (amortized) |
| hash160 | ~250–400 instructions | SHA-256 (64 rounds) + RIPEMD-160 (2×80 steps), fully unrolled, register-only |
| Local memory | ~16 B/key L1 traffic (subp) | frame analysis; **not dominant** (1024,* ≈ 128,* throughput in the sweep) |

Cross-checks from the lab (hardware): int32-mix 12.65 Gops of dependent
3–4-instr iterations (≈ near issue peak), int64-mul 4.49 Gops (64-bit
mul path is the expensive ingredient in every modmul). **The single
most valuable missing measurement is the lab's `modmul256` Gops on the
5090** — it directly calibrates the EC ceiling: at 8 Gkeys/s and ~3.5
modmul/key, the engine performs ~28 Gmodmul/s; the lab number tells us
how close that is to the modmul ceiling. Run it first.

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

| ID | Experiment | Rationale | Expected |
|---|---|---|---|
| E-S1 | extended grid/slices sweep (incl. B=16/32 variants) + clock lock | current best 8.13 @ 512,64 came from geometry alone (+4% over default); 512,128 dips 12% → strong geometry sensitivity; slices untested beyond 16 | +0–8% |
| E-S2 | `make RDC=0` (whole-program device code) | rdc limits codegen (roadmap EXP-01) | +0–5% |
| E-S3 | `make NVCC_EXTRA="-DTPB_MAX=512"` (and `-DLAUNCH_MIN_BLOCKS=1/4` combos) | new compile knobs; occupancy/ILP balance | unknown, cheap |
| E-S4 | `make NVCC_EXTRA="-DMAX_BATCH_SIZE=512"` | halves the 16.7 KB frame; L1 pressure | unknown, cheap |

Run E-S2/E-S3/E-S4 **before** E-S1's final numbers or re-run E-S1 per
build (identical workload requirement).

## Phase 2 — measurement gates (rig)

1. **Lab calibration**: `cd research/cuda-lab && python3
   scripts/run_experiments.py --reps 5` — get modmul256/local-rw/divergent
   medians. Compute the modmul headroom ratio (lab ceiling vs engine's
   28 Gmodmul/s).
2. **Profiling**: unlock counters
   (`sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`) →
   `bash benchmarks/profile/ncu_profile.sh` → stall reasons and pipe
   utilization decide Phase 3 priorities. If counters stay locked, the
   lab ratios + A/B builds remain the decision tool.

## Phase 3 — kernel work (this repository, gated on Phase 1/2 evidence)

Candidates in priority order (each: isolated PR, correctness gates,
5-rep medians, keep-or-revert):

| ID | Change | Hypothesis | Risk |
|---|---|---|---|
| K-1 | subp[] segmentation (S=8/16 per segment, registers only) | removes the 136 high-offset LDL/STL from the hot loop | high (must reproduce identical key coverage; proof.py is the gate) |
| K-2 | hash160 micro-optimizations (RIPEMD round scheduling, register allocation) | hash ≈ 40–50% of instructions → single-digit % each | medium |
| K-3 | point-add restructure (share subexpressions between the ± pair) | fewer modmuls per i-step | medium-high |
| K-4 | __reduce/ballot-based hash counting, poll cadence | minor | low |

## Rules (binding)

1. No claim without medians ≥5 reps + correctness pass.
2. Synthetic lab Gops ≠ engine Mkeys/s — never compared.
3. Every accepted change lands in PERFORMANCE_HISTORY with its record.
4. The 10 Gkeys/s goal is a target, not a promise: the program's success
   criterion is **measured, reproducible improvement** and honest
   reporting of the ceiling we find.
