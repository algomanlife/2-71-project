# Changelog — 2^71 project

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer once
releases exist.

## [Unreleased]

### Added (pending PRs — see GitHub Issues / Pull Requests)
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
