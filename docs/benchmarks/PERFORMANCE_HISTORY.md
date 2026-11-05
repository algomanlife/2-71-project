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
| 5c78d872 | RTX 5090 | 13.4 | `512,64` same-binary reproducibility pair (60 s later, 5 reps) ([record](data/sweep-20261105T113559Z.json)) | 9,040.3 | **9,048.0** (median); 9,331.3 best | +0.09% same-binary | Pass (5/5 valid) | Measurement (same-binary reproducibility: **0.1%** — the +6.3% claim above rests on this) |
| 5c78d872 | RTX 5090 | 13.4 | E-S2 labeled pair, `make RDC=0` leg, 5 reps ([record](data/sweep-20261105T121018Z.json)) | 9,006.7 (default-build leg, [T121117Z](data/sweep-20261105T121117Z.json), 4 min earlier) | **9,041.0** (median) | **+0.38%** — below the 0.5% same-day noise floor | Pass (5/5 valid) | **Rejected (no measurable effect)** — whole-program device code does not speed this kernel; `RDC=0` remains a build option only |
| 5c78d872 | RTX 5090 | 13.4 | E-S2 labeled pair, default `rdc=true` leg, 5 reps ([record](data/sweep-20261105T121117Z.json)) | 8,501.1 (07a8509 same day) | **9,006.7** (median) | +5.9% (independent rebuild of #39+#40 confirms the +6.3% within noise) | Pass (5/5 valid) | Measurement (baseline leg of the E-S2 pair) |
| 5c78d872 | RTX 5090 | 13.4 | E-S4 `-DMAX_BATCH_SIZE=512` (stack frame 16.7→8.5 KB), 5 reps ([record](data/sweep-20261105T121215Z.json)) | 9,006.7 (default build, [T121117Z](data/sweep-20261105T121117Z.json), ~2 min earlier) | **9,046.0** (median) | **+0.44%** — below the noise floor | Pass (5/5 valid) | **Rejected (no measurable effect)** — local-memory frame size is not the constraint (consistent with E-S1: 1024,* ≈ 128,* throughput) |
| ≥5c78d872³ | RTX 5090 | 13.4 | **real search run** — manual, `512,64`, full 22.0 s run, 193.9 G keys, second historical test vector ([log](data/execution-log-2026-11-05c.txt)) | 7,128.4 (same vector, historical log, default config — sustained metric) | **8,534.1 sustained** | **+19.8%** (config + #39/#40 code; sustained vs sustained) | Pass (reference key found) | Measurement (sustained-average metric class — separate from sweep medians/best-observed; ~5–6% below best-observed peaks) |
| 886548d | RTX 5090 | 13.4 | **K-2a** — hash160 instruction cuts: SHA-256 K/IV as instruction immediates (literal macro rounds) + explicit `__funnelshift_l` RIPEMD rotations, 5 reps ([record](data/sweep-20261105T133135Z.json)) | 9,006.7 (same-day default-build leg, [T121117Z](data/sweep-20261105T121117Z.json)) | **9,672.7** (median); 9,811.9 best observed | **+7.4%** vs same-day baseline (>14× the 0.5% noise floor); the **worst** K-2a run (9,529.6) beats every prior median; power 62.3 W (was ~64 W) | Pass (33/33 tests, 5/5 runs valid, all found reference key; host bit-exactness vs hashlib + python RIPEMD oracle) | **Accepted (measured)** — first kernel-level optimization; provenance self-evidenced by `.build-flags` inside the record |
| 041be91 | RTX 5090 | 13.4 | **K-2b slice 2** — SHA-256 message words via single `__byte_perm` (PRMT), 5 reps ([record](data/sweep-20261105T155537Z.json)) | 9,627.6 (post-revert K-2a class, same day) | **9,720.6** (median); 9,832.5 best | **+1.0%** (above the +0.5% accept line); best-observed project record | Pass (33/33 tests, 5/5 valid, host bit-exact) | **Accepted (measured)** |
| 041be91 | RTX 5090 | 13.4 | **real search run, K-2b2 build** — third historical vector, **324 s / 2.75 T keys** (longest sustained run) ([log](data/execution-log-2026-11-05e.txt)) | 8,767.1 (second vector, K-2a — sustained) | **9,301.3 sustained** | highest sustained ever on this rig (+6.1% vs prior best sustained) | Pass (reference key found) | Measurement (sustained class) |

¹ Best observed in early-termination known-answer runs — **not sustained
throughput, not medians**; see GRID_RESULTS.md interpretation rules. The
+4.3%/+5.3% figures compare single best-observed values of configuration
changes only (no code change between them) and must not be presented as
optimization gains. The 2026-11-05 UTC entries are **median-of-5-reps**
records from the hardened sweep workflow; percentages are only ever
computed within the same metric class (median vs median, or best vs best).

² The 5c78d872 records' `environment` field cannot prove which build
produced the binary (the pre-fix capture ran a fresh `make info`, which
prints current-default flags). The two same-day records 60 s apart
(T113559Z, T113659Z) were produced by the **same binary** (no rebuild
time between them). **Resolution (same day, later):** the labeled E-S2
pair measured `RDC=0` at parity with the default build (9,041.0 vs
9,006.7 — both ~9.0 Gkeys/s), so the ambiguity no longer affects any
conclusion: whichever build T113559Z/T113659Z used, the measured speed
is the same. The +6.3% is attributed to the #39+#40 code changes.
Build provenance for T121018Z/T121117Z/T121215Z comes from the rig
terminal transcript (the nvcc command lines between the record
writes); these records predate the `.build-flags` capture because the
rig's `git pull` conflicted with untracked record files and was only
completed afterwards. Records after `19ffaf6` are self-proving.

Cross-day drift note: the b3d13be-era `512,64` best-observed 8,133 vs
the 07a8509-era median 8,501–8,546 on the same GPU shows ~5% cross-day
variance on this rig (no clock lock); within-day comparisons are the
only ones used for accept/reject decisions.

³ The real-run engine commit is ≥ 5c78d872 (the rig was at 5c78d872 or
19ffaf6 — engine sources identical between them; the manual invocation
produced no sweep record, and the exact build variant is at measured
parity within noise per E-S2/E-S4).

## Lab calibration records (synthetic, never compared with Mkeys/s)

| UTC | Record | Key numbers (RTX 5090, CUDA 13.4, commit 5c78d872) |
|---|---|---|
| 20261105T113806Z | [research/cuda-lab/reports/lab-20261105T113806Z.md](../../research/cuda-lab/reports/lab-20261105T113806Z.md) | modmul256 **95.42 Gops/s** (median of 5); int32-mix 12.53 Tops; global-rw 93.48 Gops (16 B/op ≈ 1.50 TB/s); local-rw 102.67 Gops; launch latency 1.83 µs; power 64–66 W. `ec-scalar-mul` row **INVALID** (zero-scalar early-return timing bug, fixed in this PR; re-measure pending). |
| 20261105T121316Z | [research/cuda-lab/reports/lab-20261105T121316Z.md](../../research/cuda-lab/reports/lab-20261105T121316Z.md) | lab reproducibility repeat (stale pre-fix binary): modmul256 95.46 Gops (+0.04%), int32-mix 12.51 Tops (−0.2%), all other benches within 0.3% — lab numbers are stable to ~0.3%. `ec-scalar-mul` still the invalid pre-fix figure (binary predates the fix; a fixed-binary run T121624Z exists on the rig). |

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
| 20261105T113559Z | data/sweep-20261105T113559Z.json | — (reproducibility) | `512,64`; median 9,048.0; same-binary pair with T113659Z (0.09% drift) |
| 20261105T113659Z | data/sweep-20261105T113659Z.json | #39 + #40 (attribution resolved: E-S2 parity) | `512,64`; median 9,040.3 |
| 20261105T113806Z | research/cuda-lab/reports/lab-20261105T113806Z.md | — (lab calibration) | modmul256 95.42 Gops ceiling; ec-scalar-mul row invalid (bench bug, fixed) |
| 20261105T121018Z | data/sweep-20261105T121018Z.json | — (E-S2 RDC=0 leg) | median 9,041.0; provenance: rig terminal build lines |
| 20261105T121117Z | data/sweep-20261105T121117Z.json | — (E-S2 default leg) | median 9,006.7; independent #39+#40 rebuild confirmation |
| 20261105T121215Z | data/sweep-20261105T121215Z.json | — (E-S4) | `-DMAX_BATCH_SIZE=512`; median 9,046.0; no effect |
| 20261105T121316Z | research/cuda-lab/reports/lab-20261105T121316Z.md | — (lab repeat) | all benches within 0.3% of T113806Z; ec-scalar-mul still pre-fix figure |
| 20261105T121624Z | (on rig, not yet committed) | — (lab, fixed ec-scalar-mul binary, commit 19ffaf6) | contents not yet provided — first valid ec-scalar-mul measurement |
| 2026-11-05c | data/execution-log-2026-11-05c.txt | #39 + #40 + config | **real search run** (manual, `512,64`, second historical vector): 8,534.1 Mkeys/s sustained over 22.0 s / 193.9 G keys; vs 7,128.4 for the same vector in the historical log = +19.8% |
| 20261105T155537Z | data/sweep-20261105T155537Z.json | #50 (K-2b slice 2) | **ACCEPTED +1.0%**: median 9,720.6, best 9,832.5 (project record); power 63.1 W |
| 2026-11-05e | data/execution-log-2026-11-05e.txt | #50 | **real long-run: 9,301.3 sustained over 324 s / 2.75 T keys** (third vector) |
| 20261105T133135Z | data/sweep-20261105T133135Z.json | #44 (K-2a) | **ACCEPTED +7.4%**: median 9,672.7, best 9,811.9; first self-proving record (`.build-flags` inside `environment`) |
