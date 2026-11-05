# Validation Status — 2^71 project

Last updated: 2026-11-05 (commit `b3d13be` baseline series).

## Verified test results (executed on hardware)

| Suite | Result | GPU | Notes |
|---|---|---|---|
| Python unit + GPU tests (`tests/`) | **19/19 PASS** | RTX 5090 | includes argument validation, known-answer, small-range matrix, parity, seeded randoms, invariance, repeatability, exhaustive no-match |
| Python proof suite (`proof.py`) | **528/528 PASS** | RTX 5090 | no-skip/no-duplicate key-coverage harness |
| Grid-sweep correctness | **12/12** configs found reference key | RTX 5090 | see GRID_RESULTS.md |
| Grid-sweep correctness | **12/12** configs found reference key | RTX PRO 6000 Blackwell | 256,128 produced no speed line (key still found) |

**Scope honesty rules (binding):**

- The 528-test proof suite is evidenced on the RTX 5090 only; it is NOT
  claimed to have run on the RTX PRO 6000.
- These suites are not "547 independent proofs of complete correctness":
  they are strong evidence for key coverage in tested configurations and
  reference agreement for tested vectors. Full correctness is never
  claimed from a finite test set.
- CI runs (CPU-only) additionally execute the CPU reference suite and the
  argument-validation tests on every push; GPU search tests skip visibly
  there.

## Proof suite structure (`proof.py`)

The harness verifies that the solver finds **every** generated key in a
range (no skipped, no duplicated work) by construction:

- **Range-start coverage** — keys at the start of the range, both
  parities (start+2k, start+1+2k blocks).
- **Range-end coverage** — keys at the end of the range, both parities.
- **Full mod-B residue coverage** — a complete set of residues modulo the
  batch size, exercising every position within every batch stride.
- **Random quartile coverage** — random keys from each quarter of the
  range, with mod-B diversity.

Per-category totals of the 528-run were not recorded; the reported totals
are `Successes=528 Failures=0`. (The suite is configurable via
`--start-count/--end-count/--quartile-count`; historical runs with
defaults reported 848/848 — see the README provenance.)

Command: `python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512`

## Unit/GPU suite (`tests/`)

See `tests/README.md` and `docs/testing/GPU_VALIDATION_ROUND1_FIXES.md`.
CPU-executable reference tests (RIPEMD-160/SHA-256 official vectors,
hash160, base58, secp256k1 vs `ecdsa`) run everywhere; GPU tests cover
known-answer, small/sub-block/full-block/multi-block ranges, boundary and
batch-boundary keys, pubkey parity (02/03), seeded randoms,
configuration invariance, repeatability, and exhaustive no-match with
exit-code checks.

## Verified historical fixes (Git evidence)

Each item below is traceable to a merged PR; items without Git evidence
are classified unverified and are not listed as fixed.

| Area | Problem → fix | Files | PR / commit | Regression test | Status |
|---|---|---|---|---|---|
| Small-range scheduling | 4096-key range aborted: threadsTotal forced to 256-multiple → fallback + partial final block | `project_2_71.cu` | #27 (2545ddd) | `TestSmallRanges` matrix | Verified on RTX 5090 (19/19 incl. matrix) |
| Warp/thread boundary handling | partial warps + full-mask collectives would be UB → entry `__ballot_sync` mask + partial-warp-safe flush | `project_2_71.cu` | #27 (2545ddd) | exercised by all GPU tests (partial warps in small ranges) | Verified on RTX 5090 |
| Range-boundary handling | first/last key, batch-boundary keys untested → regression matrix | `tests/` | #27 | `RANGE_CASES` (13 cases) | Verified on RTX 5090 |
| CUDA error handling | stream/launch errors exited 0 silently; unchecked calls → `ck()` everywhere, error names, CUDA ERROR banner | `project_2_71.cu` | #27 (2545ddd) | manual/code review; exit-code 2 path documented | Code-verified; runtime-error injection not automated (unverified at runtime) |
| Exit-code behavior | single success code → 0/1/130/2 = FOUND/NOT_FOUND/INTERRUPTED/ERROR | `project_2_71.cu`, README | #27 | `TestExitCodes`, `TestExhaustiveNoMatch` | Verified on RTX 5090 |
| Public-key parity coverage | fixtures may cover only one parity → explicit 02/03 coverage test | `tests/` | #27 | `TestParityCoverage` | Verified on RTX 5090 |
| Repeatability | nondeterminism undetected → repeated-run equality test | `tests/` | #27 | `TestRepeatability` | Verified on RTX 5090 |
| Slice configuration | slices variants untested → 1/4/64/absent matrix | `tests/` | #27 | `TestSmallRanges.test_slice_configurations` | Verified on RTX 5090 |
| Grid validation | odd/oversized batch untested → argument-validation cases | `tests/` | #27 (+fixture fix in #29) | `TestArgumentValidation` | Runs GPU-less in CI (every push) |
| Test-suite integrity | CI masked a real failure (`bash -e` w/o pipefail) → `set -o pipefail` + artifact ordering | `.github/workflows/ci.yml` | #29 (9964e1b) | the previously-masked failure now fails CI | Verified in CI |
| Build-system compatibility | transient NVIDIA apt mirror failure killed CI → retry-tolerant install | `.github/workflows/ci.yml` | #26 (e9be79c) | CI runs since | Verified in CI |
| Dead code removal | verified-dead symbols removed (compile/review surface) | 5 files | #14 (61e8089+a9412b3) | full suite + pure-deletion diff | Verified (no behavior change by construction) |
| CI artifact race | python-tests downloaded binary before build uploaded → `needs: build-cuda13` | `.github/workflows/ci.yml` | #29 (4d19ebb) | argument tests now execute in CI | Verified in CI |

**Unverified items (not claimed):** runtime CUDA-failure injection; RTX
PRO 6000 execution of the 19-test and proof suites; sustained-throughput
behavior; Nsight profiling metrics.
