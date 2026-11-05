# Experimental Benchmark Variants

Isolated experiments on standalone synthetic workloads. **The baseline
benchmarks and the application's search engine are untouched.** No
speedup is claimed for any variant until measured on hardware with the
correctness gates passing and the noise guard satisfied.

## EXP-V1: modmul256-foldmul (`cuda-lab-exp`)

- **Isolated change:** the 512→320-bit reduction multiplies the high
  half by C = 2^32+977 with plain 64-bit multiplies (UMULLO/UMULHI +
  carry chain) instead of the baseline's UMultSpecial shift-add
  decomposition. Product stage, op definition, and operands identical.
- **Hypothesis (unverified):** fewer integer instructions in the
  reduction may improve synthetic modmul256 throughput. Confidence low —
  SASS lowering decides; must be measured.
- **Correctness gates (mandatory, run before any timing):**
  G1 chain vs public vectors (A·B, A·B⁸, A·B⁶⁴ mod p); G2
  device-vs-device cross-check: foldmul and baseline `_ModMult` produce
  identical per-thread results for gid-mixed starts (semantics
  replicates the baseline's non-canonical mod-2^256 fold exactly).
- **Methodology:** identical workload (blocks/threads/iters) for both
  kernels; configurable warmup (default 1000 — steady-state intent);
  CUDA-event timing; ≥5 repetitions; medians + stdev; verdict only when
  |delta| > 2× pooled stdev, otherwise "inconclusive (within noise)".
- **Run:** `./build/cuda-lab-exp --reps 5 --warmup 1000 [--json]`
- **Status:** compile-verified (CI, sm_120); runtime measurement
  pending hardware. Output JSON carries `verified: true` (gates).

## Metric rules (binding)

Synthetic ops/s here are never EC ops/s of the application and never
key-search Mkeys/s. Improvement in a synthetic variant does NOT imply
faster Bitcoin private-key searches.
