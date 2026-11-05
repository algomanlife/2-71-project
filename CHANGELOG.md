# Changelog — 2^71 project

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer once
releases exist.

## [Unreleased]

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
