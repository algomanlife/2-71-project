# Changelog — 2^71 project

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer once
releases exist.

## [Unreleased]

### Multi-GPU engine: one host thread per GPU over a split range (2026-11-06)

Added
- `--device N` (optional): pin the search to one explicit GPU. Default
  (no flag): **all detected GPUs share the range** — the validated
  power-of-two range is split into N equal power-of-two sub-ranges
  (largest N <= device count that keeps each sub-range >= one batch),
  one host thread per GPU runs the unchanged single-GPU engine on its
  sub-range. Two 5090-class cards => ~2x aggregate throughput.
- Cross-GPU coordination is host-side: whichever GPU finds the key
  sets a shared stop flag — the others stop within one kernel slice;
  progress/Speed/Count lines are aggregated (printed once); the FOUND
  banner reports the winning GPU and the fleet-wide keys-hashed total.
- Single-GPU behavior and console output are unchanged (same code
  path, one thread); `--device 0` reproduces the classic engine
  explicitly. Coverage is a partition by construction (each sub-range
  is exhaustively searched); proof.py + the test suite gate it.
- Exit codes unchanged: 0 found / 1 exhaustive / 130 interrupt / 2
  CUDA error (any device).

### K-3a verdict: REJECTED (2026-11-06)

Rig (proof.py 848/848 PASS, sweep T103324Z on 5f86f1d): `512,64`
median **9,263.1** vs this rig's baseline 9,453.4 = **-2.0%**. The
plain-C ModAdd256 carry chain costs more than the two PTX-chained
ModSubs it replaced. Reverted. With K-1 and K-3a both rejected on
measurements, the program's evidence is consistent: the engine sits at
~88% of its instruction-issue wall (~11 Gkeys/s lab estimate) and the
remaining levers are marginal. Accepted engine state: K-2a (+7.4%) +
K-2b2 (+1.0%); rig-dependent medians 9.4-9.7 Gkeys/s.

### K-1 verdict: REJECTED — register-window segmentation loses 23-27% (2026-11-06)

Measured on the fresh rig (default-build baseline 9,453.4 Mkeys/s,
records sweep-20261106T091743Z / T090153Z / T093824Z / T094028Z):
- `SUBP_SEG=16`: median 6,911.8 (**-27%**) — q[16][4] = 128 registers, spills.
- `SUBP_SEG=8`: median 7,254.4 then 7,319.6 (**-23%, reproduced**) — the
  64-register window still spills, plus re-materialization muls add EC work.
- **Coverage flawless on both: proof.py 848/848 PASS** — the correctness
  gate did its job; the loss is purely performance.
- Release untouched: `SUBP_SEG` defaults to 0 (legacy, binary-identical);
  the code stays as recorded negative evidence.
- Program consequence: local-memory indexing is not the wall — the
  validated lever stays per-key instruction count (K-2a +7.4%, K-2b2
  +1.0% accepted). Next: **K-3** (point-add restructure: shares
  subexpressions between the +/- pair, REDUCES modmuls, no extra
  registers).

### K-1: subp[] segmentation — compile-gated experiment (2026-11-05)

Added
- `project_2_71.cu`: new compile knob `SUBP_SEG` (default **0 = legacy
  path, binary identical to the baseline**). With `-DSUBP_SEG=8/16` the
  16 KB local-memory prefix-product array is replaced by per-segment
  checkpoints (~2 KB) plus a register-resident segment window rebuilt
  per segment — removing the runtime-indexed `subp[i]` local
  loads/stores from the hot loop (the ~130 LDL/STL identified in the
  SASS evidence) and shrinking the stack frame 8x.
- Math is bit-identical (same factors, same order — segments only
  re-materialize the same prefix products in registers); key coverage
  is unchanged — **proof.py is the acceptance gate**, then medians.
- CI: two compile gates (SUBP_SEG=8 and 16) added; default build
  untouched.

### K-2b slice 2 ACCEPTED: +1.0% median, 9,301 Mkeys/s sustained (2026-11-05)

- Rig verdict (record `sweep-20261105T155537Z.json`, self-proving
  build-flags): `512,64` median **9,720.6** (best 9,832.5 — project
  record) = **+1.0%** over the K-2a class, above the +0.5% accept line.
- **Real long-run: 9,301.3 Mkeys/s sustained over 324 s / 2.75 T keys**
  (third historical vector) — the highest sustained figure ever measured
  on this rig. Cumulative sustained vs the historical log: ~+30%.
- PERFORMANCE_HISTORY Accepted entry + records; README figures updated.

### Release 0.0.1 (2026-11-05)

- README: complete Quick Start (fresh server → verified search:
  setup.sh, test gates, search invocation, sweep workflow, research
  lab). Version 0.0.1 = today's measured state: RTX 5090 `512,64`
  median **9,720.6 Mkeys/s** (best observed 9,832.5), real-run
  sustained **9,301.3 Mkeys/s** (324 s / 2.75 T keys), two accepted
  kernel optimizations (K-2a +7.4%, K-2b slice 2 +1.0%), full
  correctness gates (33 tests + 528-test proof suite + 60/60 sweep
  runs finding the reference key).

### K-2b slice 2: message assembly via byte-permute (2026-11-05)

Changed
- `CUDAHash.cuh` `SHA256_33_from_limbs`: the 9 message words of the
  hot-path SHA-256 block are now built with single `__byte_perm`
  (PRMT-class) instructions instead of `pack_be4`'s 4-byte-extract +
  3-OR sequence per word (~10 instructions → 1 per word; ~70-80
  instructions removed per hashed key). Unlike the reverted slice 1
  (already-optimized bswap), this removes real instruction count.
- Verified bit-for-bit on host against hashlib SHA-256 + the
  pure-python RIPEMD-160 oracle on the limbs path (the host harness
  caught a limb-ordering bug in the TEST itself first — the CUDA code
  was correct). Rig medians decide accept/revert vs the K-2a class
  (~9,627–9,673 median; accept > +0.5%).
### Docs: honest sustained-speed status + address-paste troubleshooting (2026-11-05)

- README: sustained speed is **not stable on this rig** — 8.5-9
  Gkeys/s observed on real runs (cloud-GPU clock behavior, no clock
  lock); sweep best-observed medians (~9.6) remain a separate metric
  class. Cumulative sustained vs historical: ~+23%. Post-revert
  T152812Z median 9,627.6 confirms K-2a performance restored.
- README usage + TROUBLESHOOTING: `Error: invalid P2PKH address` =
  terminal paste corruption of the address string (verify 34 chars;
  extract from grid_sweep.sh).

### K-2a ACCEPTED: +7.4% measured — first kernel-level optimization (2026-11-05)

- Rig verdict (record `sweep-20261105T133135Z.json`, self-proving via
  `.build-flags`): `512,64` median **9,672.7 Mkeys/s** (runs
  9,529.6–9,811.9) vs 9,006.7 same-day baseline = **+7.4%**, >14× the
  0.5% noise floor. The worst K-2a run beats every prior median; power
  62.3 W (was ~64 W) — consistent with the removed instructions.
- Correctness: 33/33 tests, 5/5 sweep runs found the reference key,
  host bit-exactness vs hashlib + the pure-python RIPEMD oracle.
- PERFORMANCE_HISTORY: first Accepted kernel-level entry + record;
  README §7/§12/§14 and the speed-progression chart updated
  (9,672.7 bar added; goal gap now +3.4% median).

### K-2a: hash160 instruction cuts, first slice (2026-11-05)

Changed
- `CUDAHash.cuh`: the 64 SHA-256 round constants and 8 IV words are now
  LITERAL macro arguments (`SHA_ROUND_PLAIN`/`SHA_ROUND_SCHED`, the same
  pattern as the RIPEMD-160 round macros in this file) instead of
  `__constant__`-memory arrays — every constant and every `w[]` index
  is a compile-time constant, the values fold into instruction
  immediates, and up to 64 constant-memory loads (LDC) per hashed key
  disappear (each previously in the T1 dependency chain). A first
  attempt with `constexpr` arrays failed CI ("identifier K is
  undefined in device code" — runtime loop indices are not constant
  expressions at semantic-analysis time); the macro-literal form is the
  correct one. Values mechanically generated from the FIPS 180-4 table;
  verified bit-for-bit on host against hashlib + the project's
  pure-python RIPEMD reference (2 vectors), plus rig gates.
- `CUDAHash.cuh`: RIPEMD-160 `ROL` macro now uses `__funnelshift_l`
  explicitly — one SHF-class instruction per rotation instead of a
  shift-or sequence (compiler pattern-matching not guaranteed).
- First kernel-level experiment of the engine-speed program (Phase 1
  closed, all zero-risk levers null). Accept/revert decision by rig
  medians: `grid_sweep.sh --repeat 5 --grids "512,64"` vs the 9,006.7
  same-day default-build median.
### README results hub: new charts + refreshed data sections (2026-11-05)

Added
- Three new SVG assets (generated by `docs/assets/gen_engine_charts.py`,
  values verbatim from committed records):
  `speed-progression.svg` (baseline → config → real run → median →
  best, metric classes visually distinct, >10 Gkeys/s goal line),
  `phase1-verdicts.svg` (E-S1 geometry plateau + E-S2/E-S4 null
  verdicts with ±0.5% noise band), `lab-calibration.svg` (modmul
  ceiling 95.42 Gops vs engine 31.6 derived, same-unit only).
- README: §7 speed-progression chart, §8 Phase-1 verdict table + chart,
  §9 validation table extended (33 tests, 60/60 median-sweep runs,
  real-run logs, 9/9 lab checks), §10 methodology updated to the
  executed median workflow with measured noise floors and `.build-flags`
  provenance, §12 lab-calibration chart + engine-speed status, §13/§14
  limitation and roadmap refresh (Phase 4 ✅, K-2 active).

### Phase-1 closed: E-S2/E-S4 measured null; real-run result registered (2026-11-05)

Added
- Rig records committed verbatim: same-binary reproducibility pair
  (`sweep-20261105T113559Z.json`, 9,048.0 median — 0.09% drift),
  the labeled E-S2 pair (RDC=0 9,041.0 vs default 9,006.7) and E-S4
  (`-DMAX_BATCH_SIZE=512`, 9,046.0), plus the lab repeat
  (`lab-20261105T121316Z.md`, all benches within 0.3%).
- Real search run registered: `data/execution-log-2026-11-05c.txt`
  (manual, `512,64`, second historical test vector) — **8,534.1
  Mkeys/s sustained** over the full 22 s run vs 7,128.4 for the same
  vector in the historical log = **+19.8%**. README performance
  section updated with current medians (9,006–9,048) and the
  sustained-vs-best-observed distinction.

Measured verdicts (Phase 1 closed — all zero-risk levers null)
- E-S2 `RDC=0`: +0.38% vs default — below the 0.5% noise floor →
  no effect; build option only.
- E-S4 `MAX_BATCH_SIZE=512`: +0.44% — below noise → no effect;
  local-memory frame size is not the constraint.
- Provenance caveat ² resolved (E-S2 parity): the +6.3% of #39+#40
  stands regardless of which build the 11:36Z pair used.
- Next per ENGINE_SPEED_PROGRAM: kernel work K-2 (hash160 instruction
  count) → K-1 (subp segmentation) → K-3 (point-add restructure).

### First measured engine-speed results + record provenance (2026-11-05)

Added
- Rig records committed verbatim:
  `docs/benchmarks/data/sweep-20261105T104604Z.json` (E-S1 geometry,
  6 grids × 5 reps), `sweep-20261105T105257Z.json` (same-binary noise
  floor), `sweep-20261105T113659Z.json` (`512,64` median **9,040.3
  Mkeys/s** on 5c78d872, +6.3% vs same-day 07a8509 median) and the lab
  calibration report `research/cuda-lab/reports/lab-20261105T113806Z.md`
  (modmul256 ceiling **95.42 Gops/s** → engine at 33% of the EC
  ceiling; instruction-issue wall est. ~11 Gkeys/s).
- `PERFORMANCE_HISTORY.md`: first median-based entries with records,
  same-day noise floor (0.5%) and cross-day drift (~5%) documented;
  lab calibration section (synthetic, never mixed with Mkeys/s).
- `ENGINE_SPEED_PROGRAM.md`: measured status (9,040.3), E-S1 verdict
  (geometry saturated), E-S2 marked inconclusive (provenance), Phase-3
  candidates reprioritized (K-2 hash micro-opts lead), and the
  UltrafastSecp256k1 hot-path verdict (structural no — the engine does
  no per-key scalar mults; ~3.5 modmul/key batch-inversion walk).

Fixed
- Lab `ec-scalar-mul` bench timed a zero-scalar early-return path
  (buffers were memset to 0) — the published 2.64 Gops/s figure is
  INVALID. Scalars are now deterministic nonzero 256-bit values;
  re-measurement pending on the rig.
- Benchmark-record provenance: `make` now writes `.build-flags`
  (commit, rdc, full nvcc/ld flags of the actual binary) and
  `grid_sweep.sh` captures it into each record's environment — a sweep
  of an RDC=0 binary can no longer be mistaken for a default build
  (root cause of today's E-S2 ambiguity).

### Engine speed program (2026-11-05, maintainer directive)

Changed
- Search-engine optimization is now the project's top priority
  (target: >10 Gkeys/s on RTX 5090 from the observed 7.5–8.1);
  `docs/optimization/ENGINE_SPEED_PROGRAM.md` defines the measured,
  correctness-gated program (zero-risk build/geometry experiments →
  lab calibration + profiling → kernel candidates K-1..K-4).
- `project_2_71.cu`: launch-geometry compile knobs `TPB_MAX` (256,
  default) and `LAUNCH_MIN_BLOCKS` (2, default) — defaults reproduce
  the baseline binary exactly; variants via
  `make NVCC_EXTRA="-DTPB_MAX=512 ..."`.

### Documentation: usage, workflow, and proof reference (2026-11-05)

Added
- `docs/USAGE.md` — complete CLI reference verified against the argument
  parser (options, defaults, every validation constraint and error
  message, exit codes, build variants, CUDACyclone→project-2-71
  differences); user-facing examples use neutral placeholders.
- `docs/SEARCH_WORKFLOW.md` — implementation walkthrough of the existing
  search pipeline (parsing → validation → GPU init → work distribution
  incl. partial-block scheduling → hot-path batch inversion → result
  reporting/interruption/error behavior).
- `docs/PROOF_AND_VALIDATION.md` — what proof.py actually verifies
  (key-coverage categories, pass/fail semantics, limitations), the three
  test suites, and reproducible non-targeted public-vector commands.
- `docs/TROUBLESHOOTING.md` — argument/build/runtime errors mapped to
  causes and fixes.
- README: usage example neutralized to placeholders; navigation block.

No source changes; documentation only.

### Phase 3 — benchmark validity verification (2026-11-05)

Fixed
- Python `ResourceWarning` (unclosed file handles) in benchmark tooling:
  every read/write now uses with-statements; verified with
  `-W error::ResourceWarning` (no test weakened).

Added
- Lab validity gate: `cuda-lab-bench --verify/--verify-only` re-runs each
  bench kernel writing final per-thread values and cross-checks them
  bit-for-bit against exact host references (DCE/loop-elision/empty
  timing would diverge). Runner aborts before timing on failure; JSON
  records carry `verified` + theoretical reference points (attribute-based
  clocks — cudaDeviceProp clock fields were removed in CUDA 13).
- `docs/research/BENCHMARK_VALIDATION.md`: audit of the first RTX 5090
  lab measurements (plausible, pending verification on hardware).


### CUDA research lab & Phase-2 research docs (2026-11-05)

Added
- `research/cuda-lab/` — standalone synthetic benchmark laboratory
  (CMake, sm_120): 9 workloads with precise op definitions (integer
  mix/mul, 256-bit modmul with verbatim-extracted project arithmetic,
  global read/write, local-memory access, warp divergence, affine EC
  scalar-mul on public scalars, launch latency); public-vector
  correctness tests; automated experiment runner with medians/variance,
  JSON/CSV/Markdown records, baseline comparison with noise guard.
- Research docs: `docs/research/BLACKWELL_PERFORMANCE.md` (doc-fact vs
  measured vs hypothesis, cited), `SECP256K1_COMPARISON.md`
  (UltrafastSecp256k1 = MIT, evaluation-only), `BLACKWELL_BOTTLENECKS.md`
  (evidence table; counter profiling blocked by ERR_NVGPUCTRPERM),
  `PERFORMANCE_EXPERIMENTS.md` (EXP-L series),
  `OPTIMIZATION_RESULTS.md` (measured-only, currently empty).
- CI compiles the lab for sm_120.

### Benchmark publication & performance program (2026-11-05)

Added
- First measured GPU results published: RTX 5090 (170 SM, CUDA 13.4)
  best observed 8,133.0 Mkeys/s at grid `512,64`; RTX PRO 6000 Blackwell
  (188 SM, CUDA 13.4) best observed 7,452.4 Mkeys/s at grid `128,64`;
  full 12-configuration grid sweep on both GPUs with explicit
  early-termination/known-answer labeling
  (`docs/benchmarks/GRID_RESULTS.md`, machine-readable
  `docs/benchmarks/data/grid-sweep-2026-11-05.json`, updated
  `BASELINE_RESULTS.md` and `PERFORMANCE_HISTORY.md`).
- Verified validation status: 19/19 unit+GPU tests (RTX 5090), 528/528
  proof-suite tests (RTX 5090), 12/12 grid configs found reference key on
  both GPUs (`docs/testing/VALIDATION.md`, with scope honesty rules).
- `docs/architecture/CUDA_PIPELINE.md` (execution pipeline reference),
  `docs/optimization/PERFORMANCE_BASELINE.md` (baseline, bottleneck
  classification, E-series experiments),
  `docs/optimization/EXPERIMENTS.md` (experiment log; EXP-01 prepared).
- Repeatable benchmark infrastructure: `benchmarks/sweep/grid_sweep.sh`
  (per-grid repetitions, medians, variance, power telemetry, JSON
  records), `benchmarks/profile/ncu_profile.sh` (E1 Nsight Compute
  metric set).
- Build knob `RDC` (`make RDC=0`) for the whole-program-device-code A/B
  experiment; default unchanged (rdc on).

Changed
- README redesigned around the measured results with three new SVG
  assets (GPU performance cards, grid-sweep comparison chart, validation
  status).

### GPU validation round 1 fixes (2026-11-05)

Fixed
- Small-range scheduling: ranges with fewer batches than one full block
  (e.g. 4096 keys, batch 128) previously aborted with "failed to pick
  threadsTotal"; a fallback now launches a partially-filled final block
  with exact whole-batch per-thread work. Large-range scheduling path
  unchanged.
- Kernel entry now derives its warp mask from an entry ballot
  (`__ballot_sync` before any early return), making partial warps
  well-defined; hash-flush takes a partial-warp-safe branch. Full-warp
  hot path instruction-identical (one added ballot).
- CUDA error handling: all host/device allocations, transfers, symbol
  copies, and synchronizations checked; stream-query/kernel-launch
  failures reported with CUDA error names and no longer exit 0. Exit
  codes now distinguish FOUND (0) / NOT_FOUND (1) / INTERRUPTED (130) /
  ERROR (2).
- Test harness rewritten: bounded aligned ranges, fixed random seeds,
  return-code/stdout/stderr/command diagnostics, explicit timeouts,
  sub-block/full-block/multi-block regression matrix, parity coverage,
  exhaustive no-match, argument validation (runs GPU-less in CI).

Added
- `docs/testing/GPU_VALIDATION_ROUND1_FIXES.md` (root causes, changes,
  CPU validation, remaining hardware validation).
- CI: solver binary artifact passed to the python-tests job so
  argument-validation tests execute on CPU-only runners.

### Phase 3 (2026-11-05)

Added
- `docs/research/STACK_FRAME_INVESTIGATION.md` — full attribution of the
  16,688 B kernel stack frame (frame = 16·MAX_BATCH_SIZE + 304; `subp[]`
  ~98%), sm_120 SASS instruction census per function, local-access offset
  histogram; runtime impact explicitly pending experiments E1–E4.
- CI kernel-resource-matrix job: stack frames at MAX_BATCH_SIZE
  128/256/512/1024, SASS/PTX artifacts (90-day retention).
- `NVCC_EXTRA` Makefile knob for compile-time experiments (E3).
- README redesign (Phase 3 Workstream B) with SVG assets.

Changed
- `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md` — v2: proposal table with
  confidence/cost/risk/hardware/validation/priority/status per item;
  Stage A evidence updated (A1/A2 done); O-01 status in-PR.

Removed
- Verified dead code (audit §6 list; pure-deletion diff, PR #14).

### Phase 2 (2026-11-04) — merged via PRs #7–#12

Added
- `docs/audits/INITIAL_TECHNICAL_AUDIT.md` — full static audit with
  execution-flow diagram, verified dead-code list, and bottleneck
  hypotheses H1–H9 (no profiler claims).
- CUDA 13 / Blackwell `sm_120` build support: version-aware architecture
  selection, PTX fallback, `ptxas-info` register report, GPU-less-safe
  detection (`Makefile`), and
  `docs/compatibility/CUDA13_BLACKWELL.md`.
- Validation framework: CPU reference tests (RIPEMD-160/SHA-256/hash160/
  base58/secp256k1 vectors), GPU correctness suite (known-answer,
  reference cross-check, boundary, no-skip/duplicate, repeatability),
  pure-python RIPEMD-160 reference.
- Benchmark harness `benchmarks/baseline/run_baseline.sh` with full
  environment capture and JSON records;
  `docs/benchmarks/BASELINE_RESULTS.md` and
  `docs/benchmarks/PERFORMANCE_HISTORY.md`.
- `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md` — evidence-gated
  optimization plan (O-01 … O-09).
- CI quality gates (build + CPU tests + hygiene). CPU-only CI is never
  presented as GPU validation.

### Notes
- Computational algorithms are unchanged in all of the above; the hot
  kernel is byte-identical to the initial release except where a merged
  `opt/*` PR explicitly says otherwise.
