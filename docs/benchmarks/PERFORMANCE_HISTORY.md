# Performance History — 2^71 project

Centralized cumulative tracking of measured performance changes.

## Rules

1. Every entry is a **measured** result from
   `benchmarks/baseline/run_baseline.sh` records
   (`docs/benchmarks/data/*.json`). Estimated or projected gains are NOT
   recorded here — they live in
   `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md`.
2. Each entry references the exact commit, GPU, CUDA toolkit, and
   configuration.
3. "Improvement" is computed **cumulatively against the first baseline
   measured on the same GPU + driver + CUDA combination**, never by adding
   percentages of unrelated optimizations.
4. Correctness must be **Pass** for an entry to be Accepted.
5. Regressions are kept (status Rejected) with their measured numbers —
   they are evidence.

## Entries

| Commit | GPU | CUDA | Configuration | Baseline (Mkeys/s) | Updated (Mkeys/s) | Improvement | Correctness | Status |
|---|---|---|---|---|---|---|---|---|
| b3d13be | RTX 5090 | 13.4 | initial baseline (default config) ([record](data/grid-sweep-2026-11-05.json)) | — | 7,800.4 (best observed¹) | — | Pass (19/19 + 528/528 proof + 12/12 grids) | Baseline |
| b3d13be | RTX 5090 | 13.4 | `512,64` (grid sweep best) ([record](data/grid-sweep-2026-11-05.json)) | 7,800.4 | 8,133.0 (best observed¹) | +4.3% vs initial baseline (config change only, no code change) | Pass | Accepted (configuration finding) |
| b3d13be | RTX PRO 6000 | 13.4 | initial baseline (default config) ([record](data/grid-sweep-2026-11-05.json)) | — | 7,074.8 (best observed¹) | — | Pass (12/12 grids) | Baseline |
| b3d13be | RTX PRO 6000 | 13.4 | `128,64` (grid sweep best) ([record](data/grid-sweep-2026-11-05.json)) | 7,074.8 | 7,452.4 (best observed¹) | +5.3% vs initial baseline (config change only, no code change) | Pass | Accepted (configuration finding) |
| 07a8509 | RTX 5090 | 13.4 | E-S1 geometry sweep, 6 grids × 5 reps, medians ([record](data/sweep-20261105T104604Z.json)) | — | best median **8,551.2** at `512,32`; batch ≥ 512 plateau 8,545–8,551 | geometry saturated (plateau ±0.1%); no path to >10 Gkeys/s from launch geometry alone | Pass (all 30 runs valid, all found reference key) | Accepted (configuration finding) |
| 07a8509 | RTX 5090 | 13.4 | `512,64` repeat (same binary, ~30 min later, 5 reps) ([record](data/sweep-20261105T105257Z.json)) | 8,546.0 | **8,501.1** (median) | −0.53% same-binary drift | Pass | Measurement (same-day noise floor: ~0.5%; no clock lock available on this rig) |
| 5c78d872 | RTX 5090 | 13.4 | `512,64` after PR #39 (launch bounds) + PR #40 (hash core forceinline), 5 reps ([record](data/sweep-20261105T113659Z.json)) | 8,501.1 (median, 07a8509 same day) | **9,040.3** (median); 9,681.5 best observed | **+6.3%** vs same-day same-grid median (>12× the 0.5% noise floor) | Pass (5/5 runs valid, all found reference key) | Accepted (measured) — with a provenance caveat² |

¹ Best observed in early-termination known-answer runs — **not sustained
throughput, not medians**; see GRID_RESULTS.md interpretation rules. The
+4.3%/+5.3% figures compare single best-observed values of configuration
changes only (no code change between them) and must not be presented as
optimization gains. The 2026-11-05 UTC entries are **median-of-5-reps**
records from the hardened sweep workflow; percentages are only ever
computed within the same metric class (median vs median, or best vs best).

² The 5c78d872 record's `environment` field cannot prove which build
produced the binary (the pre-fix capture ran a fresh `make info`, which
prints current-default flags). The two same-day records 60 s apart
(T113559Z, T113659Z) were produced by the **same binary** (no rebuild
time between them). The +6.3% therefore measures the combined effect of
PRs #39+#40 on the measured build; if that build was `RDC=0`, part of
the gain belongs to E-S2. A labeled back-to-back pair (default vs
`RDC=0`) is required to split the attribution — `.build-flags`
provenance capture added in this PR makes future records self-proving.

Cross-day drift note: the b3d13be-era `512,64` best-observed 8,133 vs
the 07a8509-era median 8,501–8,546 on the same GPU shows ~5% cross-day
variance on this rig (no clock lock); within-day comparisons are the
only ones used for accept/reject decisions.

## Lab calibration records (synthetic, never compared with Mkeys/s)

| UTC | Record | Key numbers (RTX 5090, CUDA 13.4, commit 5c78d872) |
|---|---|---|
| 20261105T113806Z | [research/cuda-lab/reports/lab-20261105T113806Z.md](../../research/cuda-lab/reports/lab-20261105T113806Z.md) | modmul256 **95.42 Gops/s** (median of 5); int32-mix 12.53 Tops; global-rw 93.48 Gops (16 B/op ≈ 1.50 TB/s); local-rw 102.67 Gops; launch latency 1.83 µs; power 64–66 W. `ec-scalar-mul` row **INVALID** (zero-scalar early-return timing bug, fixed in this PR; re-measure pending). |

Engine calibration derived in
[ENGINE_SPEED_PROGRAM.md](../optimization/ENGINE_SPEED_PROGRAM.md):
at the 9,040.3 Mkeys/s median the engine performs ≈ 31.6 Gmodmul/s
(3.5 modmul/key) — **33% of the 95.42 Gops/s lab ceiling** — and the
measured issue-rate budget puts the whole-kernel ceiling near ~11
Gkeys/s. See that document for the gating of Phase-3 kernel work.

## Measurement record index

| UTC | Record file | Optimization PR | Notes |
|---|---|---|---|
| 2026-11-05 | data/grid-sweep-2026-11-05.json | — (baseline + grid sweep) | maintainer-supplied runs; 12 grid configs x 2 GPUs; 256,128 no speed line |
| 20261105T104604Z | data/sweep-20261105T104604Z.json | — (E-S1) | geometry sweep, 6 grids × 5 reps; commit 07a8509; plateau batch ≥ 512 ~8,550 median |
| 20261105T105257Z | data/sweep-20261105T105257Z.json | — (E-S1 noise floor) | `512,64` repeat; same-binary drift −0.53% |
| 20261105T113559Z | (on rig, not yet committed) | — | second of the back-to-back pair; **contents not yet provided** — needed to complete the same-binary reproducibility check |
| 20261105T113659Z | data/sweep-20261105T113659Z.json | #39 + #40 (+ possibly RDC=0) | `512,64`; median 9,040.3; build provenance unresolved (see caveat ²) |
| 20261105T113806Z | research/cuda-lab/reports/lab-20261105T113806Z.md | — (lab calibration) | modmul256 95.42 Gops ceiling; ec-scalar-mul row invalid (bench bug, fixed) |
