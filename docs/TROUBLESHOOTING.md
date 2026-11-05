# Troubleshooting

Common errors and their causes, verified against the current
implementation. Argument-validation errors below exit `1` **before any
GPU work**; CUDA failures exit `2` with the error name on stderr.

## Argument errors

| Message | Cause → fix |
|---|---|
| `Usage: …` | missing `--range` or a target; provide `--range` and exactly one of `--address`/`--target-hash160` |
| `--grid expects "A,B" (positive integers).` | missing comma, empty part, non-numeric, zero, or > 2^20 |
| `--slices must be in 1..1048576` | out-of-range or non-numeric |
| `provide either --address or --target-hash160, not both.` | both target forms given |
| `range format must be start:end` | missing `:` |
| `invalid range hex` | non-hex characters or a bound longer than 64 hex chars |
| `invalid P2PKH address` | wrong base58, wrong version byte (must start with `1`), or failed double-SHA checksum |
| `invalid target hash160 hex` | not exactly 40 hex chars |
| `batch size must be even and a power of two.` | first `--grid` number invalid |
| `batch size must be <= 1024 (kernel limit).` | batch exceeds `MAX_BATCH_SIZE` |
| `range length (end - start + 1) must be a power of two.` | e.g. an odd-sized range |
| `start must be aligned to the range length.` | start is not a multiple of the range size |
| `range length must be divisible by batch size (A).` | e.g. a 4096-key range with `--grid 512,…` |

**Rule of thumb:** choose `start = k × length` and a batch `A` that
divides the length.

## Build problems

| Symptom | Cause → fix |
|---|---|
| `nvcc: command not found` | CUDA Toolkit not on PATH; install ≥ 12.8 (13.x recommended) or set `CUDA_HOME` |
| no GPU on the build machine | expected — `make` still compiles; pass `GPU_ARCH=` to silence detection |
| apt mirror transient failures in CI | handled by retry logic; locally re-run the install |
| `ecdsa` ModuleNotFoundError on a fresh server | run `bash scripts/setup.sh` (creates `.venv`, installs `requirements.txt` — verified end-to-end by the CI `clean-install` job) |

## Runtime problems

| Symptom | Cause → fix |
|---|---|
| `CUDA init error: …` (exit 2) | driver missing or too old — check `nvidia-smi`; Blackwell needs R570+ |
| `CUDA Error (…)` with name + context (exit 2) | real runtime failure; the message names the failing call — report it with the full stderr |
| Program exits `1` with `KEY NOT FOUND (exhaustive)` | **not an error** — the range was fully searched without a match |
| Program exits `130` | Ctrl+C interruption (by design; the current slice finishes first) |
| `ncu` profiling produces `ERR_NVGPUCTRPERM` | GPU performance counters not permitted for the user; see the remedies in `benchmarks/profile/ncu_profile.sh` (driver option or sudo); use `--timing` fallback meanwhile |

## Benchmark/test interpretation

- A **skipped** GPU test is never a pass (CPU-only environment).
- Benchmark speed lines are best-observed early-termination values
  unless produced by `benchmarks/sweep/grid_sweep.sh` (medians,
  repetitions, `verified` flags) — see
  `docs/research/BENCHMARK_VALIDATION.md`.
- `proof.py` failures list the exact category and key in
  `project_2_71_tests_results.txt`.


## `Error: invalid P2PKH address` when the address is correct

The string reaching the binary is not byte-identical to the address —
terminal paste corruption (invisible/zero-width characters, `l` vs `1`,
`O` vs `0`). The engine is fine (sweep runs with the same address
succeed). Diagnose: `echo -n '<address>' | wc -c` must print 34; fix:
`--address "$(grep -oP 'ADDRESS="\K[^"]+' benchmarks/sweep/grid_sweep.sh | tr -d '[:space:]')"`.
