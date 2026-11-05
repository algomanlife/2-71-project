# Proof and Validation — What Is Actually Verified

This document explains the verification tooling as implemented
(inspected at commit `ad3b5cb`): what `proof.py` does, what the unit /
GPU test suites cover, how to interpret results, and what the
verification does **not** guarantee.

## 1. `proof.py` — key-coverage harness

**What it verifies:** that the solver finds **every** generated key in a
range — i.e. no skipped keys and no missed keys across batch-stride
positions. It is a *coverage* test of the search workflow, not a
performance benchmark.

**How it works (as implemented):**

1. Generates private keys with the independent python `ecdsa` library
   at chosen positions of the given range (see categories below).
2. Computes each key's compressed public key and P2PKH address with the
   same independent reference.
3. Runs the solver per key over the range and watches its output for
   the `FOUND MATCH!` marker and the private-key line.
4. Counts successes/failures per category and writes a detailed log
   (`project_2_71_tests_results.txt`), ending with
   `Successes=N Failures=M`.

**Test categories (the generated key positions):**

| Category | Positions |
|---|---|
| Range start A/B | `start + 2k` and `start + 1 + 2k` (both parities near the start) |
| Range end A/B | `end − 2k` and `end − 1 − 2k` (both parities near the end) |
| Full mod-B residue coverage | a complete residue set modulo the batch size — every position within the batch stride |
| Random quartiles | random keys from each quarter of the range, mod-B diverse |

**CLI (verified from the argparse block):**

```text
python3 proof.py --range START:END [--project-path ./project-2-71]
                 [--grid 512,512] [--batch N] [--timeout S]
                 [--start-count 128] [--end-count 128] [--quartile-count 20]
```

Defaults: binary `./project-2-71`, grid `512,512`, 128 start/end keys
per parity, 20 random keys per quartile. Output file:
`project_2_71_tests_results.txt`.

**How it differs from the main executable:** `proof.py` never searches
for a wallet — it *manufactures* targets from known keys so the answer
is known in advance, then checks the solver's answer against it. The
main executable searches for one user-given target.

**Pass / fail:**

- **Pass:** final line reports `Failures=0` (every generated key was
  found with the correct private key). Historical runs: 848/848
  (defaults, original project) and 528/528 (RTX 5090 session, 2026-11-05,
  as recorded in `docs/testing/VALIDATION.md`).
- **Fail:** any `Failures > 0`, a timeout, or a solver error — the log
  identifies which category and key. A solver crash (non-zero exit)
  counts as a failure for that key.

**Interpretation and limitations:**

- A pass certifies coverage for the tested range, grid, and batch
  configuration — finite evidence, **not** a proof of total correctness.
- Categories exercise boundary and residue positions; they do not
  enumerate all possible GPU/driver/toolchain states.
- Requires `ecdsa` (installed automatically by `scripts/setup.sh` /
  `requirements.txt`).

## 2. Unit + GPU test suites (`tests/`)

| Suite | Runs | Verifies |
|---|---|---|
| `test_reference.py` | anywhere (CPU) | RIPEMD-160/SHA-256 official vectors, hash160, base58 round-trip, secp256k1 generator constant, `proof.py` helpers vs the independent `ecdsa` reference |
| `test_benchmark_parsing.py` | anywhere (CPU) | benchmark log parsing: valid-run definition, missing-speed-is-null, non-zero-exit → invalid, median/max separation, JSON schema (synthetic fixture logs) |
| `test_gpu_validation.py` | GPU + binary | known-answer vector, GPU math vs `ecdsa` reference (priv+pub), boundary/batch-boundary keys, sub-block/full-block/multi-block ranges, pubkey parity (02/03), seeded randoms, config invariance, repeatability, exhaustive no-match exit code, argument validation (runs GPU-less in CI) |

Skips are visible and are never counted as passes. On CPU-only CI the
GPU tests report `skipped`; the argument-validation tests execute
against the freshly built binary.

## 3. Reproducible, non-targeted correctness tests (public vectors)

```bash
# CPU-only reference checks (no GPU, no wallet targets):
python3 -m unittest tests.test_reference -v

# GPU correctness with public vectors (independent ecdsa oracle):
python3 -m unittest tests.test_gpu_validation -v

# GPU key-coverage harness (generates its own targets):
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

The GPU test cases are generated at runtime from random/private-for-test
scalars via the independent `ecdsa` reference — no wallet addresses are
targeted. The single canonical public vector used for end-to-end
regression (`TestKnownAnswer`) is the repository's documented test
vector (see `tests/test_gpu_validation.py`).

## 4. Claim discipline

- We state that a proof/test **passed** only when its recorded output
  was checked (test logs, CI runs, or the results files cited in
  `docs/testing/VALIDATION.md`).
- Verified behavior (executed), documented-but-untested behavior,
  performance measurements, and theoretical estimates are kept separate
  — see `docs/testing/VALIDATION.md` and
  `docs/research/BENCHMARK_VALIDATION.md`.
