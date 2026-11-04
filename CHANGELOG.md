# Changelog — 2^71 project

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer once
releases exist.

## [Unreleased]

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
