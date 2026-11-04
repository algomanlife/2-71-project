# Baseline & Validation — Benchmark Methodology

## Prerequisites

- NVIDIA GPU + driver (see `docs/compatibility/CUDA13_BLACKWELL.md`)
- CUDA toolkit (`nvcc` on PATH)
- Python 3.8+ with `ecdsa` (`pip install ecdsa`) for the test framework

## Build

```bash
make                 # prints toolkit, detected GPU, arch list, full flags
make ptxas-info      # register/spill report (audit hypothesis H2 evidence)
```

## Correctness (run before every benchmark)

```bash
python3 -m unittest discover -s tests -v
```

- CPU-only reference tests always run.
- GPU tests skip (visible, never silently) without hardware.
- `proof.py` remains the canonical key-skip regression harness:

```bash
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

## Benchmark

```bash
bash benchmarks/baseline/run_baseline.sh --grid 512,256 --slices 16
```

Each run appends a JSON record to `docs/benchmarks/data/` containing:

| Field | Content |
|---|---|
| commit / utc | exact code state and time |
| gpu / compute_capability / driver / cuda_toolkit | hardware & toolchain |
| grid / slices | runtime configuration |
| measured.wall_seconds / peak_mkeys_per_s / total_keys_hashed | machine-parsed from solver output |
| measured.power_w_samples_* | `nvidia-smi dmon` sampling during the run |
| clocks_before / clocks_after | SM/mem clock, power draw, temperature |
| incomplete | true when speed could not be parsed or the target was not found |

The canonical case is the README known-answer vector
(range `2000000000:3FFFFFFFFF`, address `1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2`,
expected key `0x22382FACD0`) — every successful benchmark doubles as a
correctness check.

## Rules

1. Never hand-edit measurement records.
2. A record with `"incomplete": true` is not a result.
3. Optimizations are compared only against records with identical GPU,
   driver, and configuration fields (see `docs/benchmarks/PERFORMANCE_HISTORY.md`).
4. Register/occupancy metrics beyond `make ptxas-info` (warp stalls,
   achieved occupancy, memory throughput) require Nsight Compute and are
   recorded in the roadmap evidence log when hardware is available.
