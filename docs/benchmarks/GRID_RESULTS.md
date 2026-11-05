# Grid Sweep Results — RTX 5090 & RTX PRO 6000 Blackwell

Date: 2026-11-05 · Commit under test: `b3d13be` · Machine-readable data:
[`data/grid-sweep-2026-11-05.json`](data/grid-sweep-2026-11-05.json)

**Read the label first: these are *best observed* results from
early-termination known-answer runs, not sustained throughput.**

---

## Test setup

| | GPU A | GPU B |
|---|---|---|
| Model | NVIDIA GeForce RTX 5090 | NVIDIA RTX PRO 6000 Blackwell WE |
| Architecture | Blackwell, compute 12.0 | Blackwell, compute 12.0 |
| SM count | 170 | 188 |
| CUDA Toolkit | 13.4 | 13.4 |
| Driver | 615.71.09 | not recorded |

- **Search:** known-answer vector — range `2000000000:3FFFFFFFFF`
  (2^37 keys), target `1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2`, expected key
  `0x22382FACD0`.
- **Termination:** each run stops when the reference key is found
  (early-termination known-answer test). The printed speed is the best
  observed value during that run, **not** a median, **not** sustained
  throughput.
- **Correctness gate:** every configuration below found the reference key.
- `--grid A,B` = points-per-batch × batches-per-SM (see README Usage).

## Summary

| | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|
| Initial baseline | 7,800.4 Mkeys/s | 7,074.8 Mkeys/s |
| **Best observed (grid sweep)** | **8,133.0 Mkeys/s** | **7,452.4 Mkeys/s** |
| Best grid | `512,64` | `128,64` |
| Grid configs finding reference key | 12/12 | 12/12 |

## Full grid sweep (Mkeys/s)

| Grid | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|
| 128,64   | 7843.9 | 7452.4 |
| 128,128  | 7970.3 | 7416.4 |
| 128,256  | 7826.7 | 7295.3 |
| 256,64   | 8015.2 | 7141.7 |
| 256,128  | N/A¹ | N/A¹ |
| 256,256  | 7813.8 | 7058.2 |
| 512,64   | **8133.0** | 7447.7 |
| 512,128  | 7116.5 | 6276.5 |
| 512,256  | 7833.1 | 7059.8 |
| 1024,64  | 5698.1 | 6267.5 |
| 1024,128 | 7118.7 | 6264.7 |
| 1024,256 | 7835.2 | 7066.5 |

¹ Reference key found on both GPUs but no speed line was produced in that
run — recorded as missing, not zero.

## Interpretation and limitations (binding)

1. All 12 grid configurations found the known reference key on both GPUs.
2. Values are **best observed from single early-termination runs**, not
   independently verified sustained throughput.
3. Do **not** infer that the RTX 5090 is universally faster from this
   single series (different SM counts and clocks; one data point per
   cell).
4. Provenance: values supplied by the maintainer from run consoles; raw
   logs held by the maintainer; the JSON record preserves them verbatim.
5. No repetitions, medians, variance, or power/clock telemetry were
   recorded in this series — the Phase-7 sweep workflow
   (`benchmarks/sweep/grid_sweep.sh`) exists precisely to add these.

## Observations (hypotheses, not conclusions)

- Per-GPU optimal grid differs (`512,64` vs `128,64`) — consistent with
  different SM counts (170 vs 188) changing `threadsTotal` for the same
  grid, but unverified without telemetry.
- The `512,128` dip (both GPUs) and the RTX 5090 `1024,64` outlier
  (5698.1) suggest launch-geometry sensitivity; single-run values, so no
  conclusion is drawn.
- Batch size 1024 compiles a 16.7 KB stack frame (see the stack-frame
  investigation) yet `1024,256` performs comparably to `128,*` —
  consistent with the documented position that frame size alone does not
  determine throughput; runtime profiling (E1) remains the arbiter.

## Related

- [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md) — baseline status table
- [`PERFORMANCE_HISTORY.md`](PERFORMANCE_HISTORY.md) — cumulative tracking
- [`../optimization/PERFORMANCE_BASELINE.md`](../optimization/PERFORMANCE_BASELINE.md) —
  engineering baseline and analysis
