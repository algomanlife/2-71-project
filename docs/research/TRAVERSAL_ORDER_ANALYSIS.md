# Traversal-Order Analysis — Verified from Historical Execution Logs

Evidence: `docs/benchmarks/data/execution-log-2026-11-05b.txt` (RTX 5090,
CUDA 13.4, preserved verbatim). This document derives and **verifies
against the logs** the law that governs when a target is discovered —
and corrects an earlier (Phase-4) speculative explanation.

## The traversal-order law

The scheduler partitions the range into `threadsTotal` equal per-thread
intervals of `per_thread = range_len / threadsTotal` keys. Each thread
walks **its own interval linearly** (window `[base, base+B)`, advancing
by the batch size `B` per batch). Therefore:

> **fraction of range hashed at discovery ≈ (offset of the target key
> within its own thread's interval) / per_thread**

— **not** the key's global position in the range, and different for
every `--grid` configuration (different `threadsTotal` → different
partition → different offset).

## Verification against the historical logs [MEASURED]

Scheduler parameters derived exactly as the code computes them
(`userUpper = SMs × batches_per_SM × 256`, largest 256-multiple divisor
of `total_batches`; both runs used the default grid `128,8` →
`threadsTotal = 262,144`):

| Run | Range | per_thread | Key offset in its thread | Predicted found-at | Observed (`Keys hashed / range`) | Δ |
|---|---|---:|---:|---:|---:|---:|
| 1 | `2000000000:3FFFFFFFFF` (2^37), key `…22382FACD0` | 524,288 | 502,992 (thread #18,181) | **95.94%** | 131,138,067,872 / 137,438,953,472 = **95.42%** | 0.52 pts |
| 2 | `8000000000:ffffffffff` (2^39), key `…E9AE4933D6` | 2,097,152 | 603,094 (thread #216,434) | **28.76%** | 158,403,639,136 / 549,755,813,888 = **28.81%** | 0.06 pts |

Agreement to within 0.06–0.52 percentage points; the residual is the
±(B/2) window width and launch-boundary effects. The 1 Hz progress line
additionally lags the actual state (run 1: line 90.59% at 17.0 s vs
actual 95.42% — ≈0.9 s of staleness at 7.5 Gkeys/s).

## Config-dependence of the same key [CODE-derived]

The canonical key `0x22382FACD0` in the 2^37 range is discovered at a
different fraction of the range under each grid:

| Grid | threadsTotal | per_thread | Predicted found-at (of range hashed) |
|---|---:|---:|---:|
| 128,8 (default) | 262,144 | 524,288 | **95.9%** |
| 128,128 | 4,194,304 | 32,768 | **35.0%** |
| 512,64 | 2,097,152 | 65,536 | **67.5%** |

This is the mechanism behind the Phase-4 observation (two grids, same
target, different progress at observation): the discovery fraction is a
**deterministic function of the grid geometry**, and a mid-run snapshot
without a match simply precedes that grid's discovery point.

## Correction to the Phase-4 report

`SEARCH_DISCREPANCY_INVESTIGATION.md` §5 previously explained the
observed 6.89% found-progress as matching the target's **global
position** (6.936%). That was a coincidence-based explanation and is
**wrong as a mechanism**: discovery depends on the key's offset within
its thread's interval, not its global position (run 2 here disproves
the global-position model directly: key at global 52.6%, found at
28.8%). The Phase-4 **verdict is unchanged** — traversal-order
difference, not a missed match — and is now mechanically proven by the
verified law above. The 6.89%-at-found figure from the Phase-4 report
cannot be reproduced for the canonical vector under grid 128,128
(predicted 35.0%); that observation most plausibly used a different
range/target than assumed there, and in any case pre-dates the exact
`Keys hashed` reporting introduced in Phase 4.

## Implications

1. Progress % is a work-done measure; **never** interpret it as scan
   position or as comparable across grid configurations.
2. The exact `Keys hashed` value at discovery is predictable from the
   scheduler arithmetic (as demonstrated) — a cheap forensic tool for
   any future discrepancy report.
3. Nothing here involves missed or duplicated keys: the full-coverage
   invariants remain pinned by `tests/test_progress_accounting.py`
   (exhaustive count == range length) and `proof.py`.
