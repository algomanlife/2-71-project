# GPU Validation Round 1 — Failure Analysis and Fixes

Date: 2026-11-05
Environment (reported by maintainer): NVIDIA RTX 5090 32 GB (sm_120),
Ubuntu 24.04.5, driver 615.71.09, CUDA 13.4, Python 3.12 + ecdsa 0.19.2.
Build: `make -j2` succeeded; failures below are runtime/test-harness.

Status of this document: fixes implemented and compile/CPU-validated in
CI. **GPU re-validation on the RTX 5090 server is still required** — the
fixes are not claimed as hardware-verified until the suite below passes
there (see "Remaining hardware validation").

---

## Root causes

### RC-1 — Scheduler rejects small ranges (execution failure)

`project_2_71.cu` (host, thread selection): `pick_threads_total()`
required `threadsTotal` to be a multiple of `threadsPerBlock` (256) and a
divisor of `total_batches`. For the regression range
`2000000000:2000000FFF` (4096 keys) with `--grid 128,8`: total batches =
32 < 256 → no valid multiple exists → `Error: failed to pick threadsTotal
satisfying divisibility` → non-zero exit before any search.

**Fix:** a fallback path — when no 256-multiple works, choose the largest
`threadsTotal ≤ upper` that divides `total_batches` (scan is bounded by
`upper`, which is small in exactly the cases where the pick fails). The
final launch block may then be **partially filled**:
`blocks = ceil(threadsTotal / threadsPerBlock)`. Invariants preserved:
per-thread work is always an exact whole number of batches (`per_thread`
is a multiple of B), no overlapping/missing scalar ranges, and large-range
scheduling (where the 256-multiple pick succeeds) is byte-for-byte
unchanged.

### RC-2 — Partial warps would make warp-collectives undefined (latent hazard)

Launching a partially-filled final block is only safe if every
warp-collective operation uses the set of *actually active* lanes: with
the previous unconditional `full_mask = 0xFFFFFFFF`, a partially-filled
warp (active lanes 0..k-1) would call `__shfl_sync/__any_sync/__syncwarp`
with lanes that have already returned — undefined behavior on Volta+
independent thread scheduling.

**Fix (kernel entry):** every lane executes
`full_mask = __ballot_sync(0xFFFFFFFFu, gid < threadsTotal)` **before**
any early return, so `full_mask` is exactly the active-lane set (a
contiguous prefix — gid ordering is linear). All existing
shfl/any/syncwarp calls now use this mask unchanged. Hash accounting
(`WARP_FLUSH_HASHES`) takes an explicit branch: full warps keep the
unchanged warp-reduced single atomicAdd (hot path identical); partial
warps (small ranges only) use direct per-thread atomicAdd — trivial
volume at those sizes.

### RC-3 — Test harness used unrunnable/oversized ranges

`tests/test_gpu_validation.py` (old version): config-invariance and
repeatability tests used the 4096-key range that RC-1 could not schedule
→ solver exited non-zero with no output → subtest failures and an ERROR
in repeatability (no private key parsed). The randomized test derived
2^37-sized ranges from the full search interval → each run searched
billions of keys ("excessively long execution").

**Fix:** complete rewrite (below).

### RC-4 — CUDA error handling gaps

- `cudaStreamQuery` error path cleared the error and stopped the loop,
  then the final section printed `TERMINATED` and returned
  **exit code 0** — a CUDA failure silently became a successful exit.
- Kernel-launch errors took the same silent path.
- `cudaHostAlloc` (×4), `cudaMemcpyToSymbol` (×2), `cudaMemGetInfo`,
  `cudaStreamSynchronize`, `cudaDeviceSynchronize` were unchecked.
- Exit codes did not distinguish outcomes.

**Fix:** all listed calls wrapped with the `ck()` helper (now prints the
CUDA error NAME and description, exit code 2); stream-query and
kernel-launch failures set a runtime-error flag, print details, and end
with a `CUDA ERROR` banner and exit 2. Outcome exit codes are now:
**0** FOUND, **1** NOT_FOUND (exhaustive), **130** INTERRUPTED (Ctrl+C),
**2** CUDA/runtime error; invalid arguments exit 1 with an `Error:`
message on stderr before any GPU work. Successful output formatting is
unchanged; cleanup (free/destroy) runs on the error paths that reach the
end of `main` (hard `ck` failures exit the process, and the OS reclaims
device memory at process teardown).

## Files changed

| File | Change |
|---|---|
| `project_2_71.cu` | kernel entry ballot mask; partial-warp-safe flush; scheduler fallback + ceil blocks; ck() coverage; runtime-error flag; distinct exit codes |
| `tests/test_gpu_validation.py` | full rewrite (below) |
| `.github/workflows/ci.yml` | build job uploads the solver binary; python-tests job downloads it and runs argument-validation tests on the CPU-only runner |
| `README.md` | exit-code documentation updated |
| `CHANGELOG.md` | entry added |
| this document | new |

No algorithmic change: the EC arithmetic, hash pipeline, and search logic
are untouched. The kernel's hot path for full warps is instruction-
identical apart from one entry ballot (compile evidence via CI
kernel-resource-matrix: register/spill deltas are recorded in the run
log).

## Test cases added (tests/test_gpu_validation.py)

- **TestArgumentValidation** (no GPU needed): non-power-of-two length,
  misaligned start, address+hash160 together, odd batch, batch > 1024,
  missing arguments — all must exit non-zero with the expected stderr.
- **TestSmallRanges**: 512/4096 (sub-block), 32768 (exactly one full
  block = 256 threads), 65536 (two blocks); first/last keys; keys at
  batch boundaries (B−1, B, 2B−1, last-batch start); grids 128,8 / 512,8
  / 128,64; slices 1/4/64/absent.
- **TestParityCoverage**: fixtures include both compressed-pubkey
  parities (02/03), each key validated priv+pub.
- **TestRandomizedSeeded**: fixed seed (20261105), bounded aligned
  ranges ≤ 8192 keys.
- **TestConfigInvariance**: same keys under three configurations.
- **TestRepeatability**: identical output across repeated runs.
- **TestExhaustiveNoMatch**: out-of-range target → exit 1 +
  `KEY NOT FOUND (exhaustive)`.
- **TestExitCodes / TestKnownAnswer**: exit 0 + banners; README
  known-answer vector with priv+pub checks.

All solver calls: explicit timeouts (120 s default, 60 s for arg tests),
return-code assertions, and failure diagnostics containing command line,
stdout, stderr.

## CPU validation results (CI, no GPU)

- CPU reference suite: all pass (RIPEMD-160/SHA-256/hash160/base58/
  secp256k1 vs `ecdsa`).
- CUDA 13 / sm_120 compile: green (build + kernel-resource-matrix jobs).
- Scheduler fallback: simulation of the failing round-1 case
  (4096 keys, B=128) → `threadsTotal=32, blocks=1, per_thread=128`
  (exact whole batch per thread).
- Argument-validation tests: execute on the GPU-less CI runner against
  the freshly built binary; GPU search tests report visible skips there.

## Remaining hardware validation (RTX 5090 server)

Not executed yet — **no GPU correctness or performance claim is made**:

```bash
git pull && git checkout main && make clean && make -j2
python3 -m unittest discover -s tests -v
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

The full `TestSmallRanges` matrix plus `proof.py` mod-B coverage exercises
the partial-block paths; expected outcome: all pass with the round-1
cases now executing. Any failure output now includes return codes and
full solver diagnostics.
