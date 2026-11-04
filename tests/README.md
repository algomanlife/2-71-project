# Tests — 2^71 project

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Dependencies: `pip install ecdsa` (for reference cross-checks and GPU test
case generation; tests that need it skip cleanly when absent).

## Layout

| File | Runs | Covers |
|---|---|---|
| `reference_ripenmd160.py` | — (library) | pure-python RIPEMD-160 reference, validated against the official test vectors and hashlib |
| `test_reference.py` | **anywhere (CPU-only)** | RIPEMD-160/SHA-256 vectors, hash160, base58, secp256k1 generator constant, proof.py helpers vs independent `ecdsa` reference |
| `test_gpu_validation.py` | **GPU required** (skips visibly otherwise) | known-answer E2E, GPU scalar-mul + hash160 vs ecdsa reference (CPU↔GPU consistency), boundary keys (both parities, range start/end), randomized keys, no-skip/no-duplicate across `--grid`/`--slices` configurations, repeatability |

GPU test configuration via environment:

```bash
PROJECT_2_71_BIN=./project-2-71 PROJECT_2_71_GRID=512,256 \
    python3 -m unittest tests.test_gpu_validation -v
```

A skip is never treated as a pass. GPU results belong in
`docs/benchmarks/BASELINE_RESULTS.md` only after a real run.
