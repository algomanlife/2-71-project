# Search Discrepancy Investigation — Grid 128,128 vs 512,64

Phase-4 root-cause report (2026-11-05). Question: two RTX 5090 runs of
the **same range and target** with different grids — A (`128,128`,
slices 16) found the match at printed progress 6.89%; B (`512,64`,
slices 16) showed 7.83% with no match at the observation point. Is a
match being missed, or is this expected behavior?

**Verdict: the available evidence establishes a traversal-order and
reporting-timing difference — NOT a missed match.** Mechanisms below;
each claim is tagged [CODE] (verified in source), [TEST] (pinned by a
new regression test), or [MEASURED] (existing hardware evidence).

---

## 1. How Progress is calculated [CODE]

`project_2_71.cu:796-798`:

```
progress = hashes_accum / range_len × 100   (clamped to 100)
```

- `hashes_accum` is a device-side u64 atomic counter; every hashed
  candidate increments a per-thread counter that is warp-reduced and
  atomically added (all exit paths flush).
- `range_len` = number of keys in the range (for the canonical vector:
  2^37 = 137,438,953,472 keys).
- **Progress is cumulative work done, not a position in a linear scan.**
- The line is printed **once per second** (`dt >= 1.0` gate,
  `project_2_71.cu:785`): at ~8 Gkeys/s over 2^37 keys, one print
  interval = **~5.7 percentage points** — the printed value can lag the
  actual state by that much.

## 2. Do grid configurations change candidate order? [CODE] + [TEST]

**Yes — by construction, and deterministically per configuration.**

- `--grid A,B` sets the thread budget `SMs × B × 256` (capped by
  batches and memory). A (`128,128`): 5.57 M threads × 128-key batches;
  B (`512,64`): 2.79 M threads × 512-key batches. The scalar range is
  partitioned differently, so each thread walks a different interval.
- Each thread explores its interval with a ±(i+1)·G walk around a
  center that advances by `A·G` per batch — keys are visited in
  window-order within each thread, not in ascending scalar order, and
  the interleaving of 5.57 M vs 2.79 M threads (multiple occupancy
  "waves") differs.
- Consequence: the **keys-hashed count at which a given target is
  discovered differs per configuration**. Pinned by
  `tests/test_progress_accounting.py::test_grid_configs_find_same_key_at_different_counts`
  [TEST]: both grids find the same key with the correct private key at
  **different counts** (deterministic per grid).

## 3. Do reported counts represent actual completed work? [CODE] + [TEST]

**Yes — exactly.** Every exit path flushes the per-thread hash count;
each key in a batch is hashed exactly once (1 + 2·(A/2−1) + 1 = A hashes
per batch), and the batch-inversion stride covers each key exactly once
(pinned by `proof.py` mod-B residue coverage). The new reporting line
`Keys hashed : N` (final read of the same counter) makes this directly
checkable:

- `test_exhaustive_count_equals_range_length` [TEST]: on exhaustive
  completion, **Count == range length exactly** — no missed, duplicated,
  or miscounted positions.
- Previously this invariant was unverifiable from the output: the final
  count was only observable via the 1 Hz progress line, which is stale
  by up to one print interval. That reporting gap is now fixed (the
  `FOUND`/`KEY NOT FOUND` banners print the exact final count).

## 4. Consistency of result and progress reporting [CODE] + [TEST]

Both use the **same device counter** (`d_hashes_accum`). The progress
line prints Count and Progress computed from one read;
`test_progress_percent_consistent_with_count` [TEST] asserts the last
printed pair agrees to print precision. The found banner's
`Keys hashed` uses a final read of the same counter.

## 5. The observation, quantitatively [MEASURED] + analysis

For the canonical vector (the only range/target pair with published
multi-config evidence):

- Target position in range: key `0x22382FACD0` lies at global position
  9,532,583,120 of 137,438,953,472 keys = **6.936%**.
- Config A printed progress at found: **6.89%** — equal to the target's
  global position to within 0.05 points. This is exactly what the
  mechanics predict (uniform thread progress ⇒ discovery at ≈ the
  target's global position).
- Config B at **7.83%** with no match **at the observation point**: a
  mid-run snapshot of a run that had not yet reached its (different)
  discovery point. Given the ~5.7-points/s print cadence and the
  deterministic per-grid traversal offsets (thread count halves, batch
  size quadruples, window width ±64 → ±256, wave structure changes),
  a sub-second difference in discovery timing between A and B is fully
  expected. B's observed snapshot is ~0.9 points past A's discovery
  point — within the explained variation.

**What the 7.83% snapshot is NOT:** it is not a completed run. A
completed exhaustive run without a match exits `1` with
`KEY NOT FOUND (exhaustive)` — that, with the key provably inside the
range, would be the signature of a real bug. No such evidence exists.

## 6. Existing multi-config hardware evidence [MEASURED]

The 12-configuration grid sweep (same range and target, RTX 5090 and
RTX PRO 6000, `docs/benchmarks/GRID_RESULTS.md`) found the reference
key in **every configuration on both GPUs** — including `128,128`
(7970.3 Mkeys/s) and `512,64` (8133.0 Mkeys/s). A missed match for
`512,64` is contradicted by direct hardware evidence.

## 7. CPU-reference consistency and boundary audit [CODE] + [TEST]

- CPU-vs-GPU consistency on synthetic, non-wallet-targeted vectors is
  covered by `tests/test_gpu_validation.py` (independent `ecdsa`
  reference, private AND public key comparison) — unchanged, passing.
- Boundary audit of general-purpose range math:
  - 2^256 wrap: a full-range request (`start=0`, `end=2^256−1`) makes
    `range_len` overflow to 0 and is **rejected** by the power-of-two
    validation [CODE].
  - Curve-order boundary: ranges ending at or beyond the secp256k1
    group order `n` were previously accepted silently; scalars ≥ n wrap
    on the curve (k and k−n give the same public key), so such ranges
    cover no additional distinct private keys. **Fixed (reporting):**
    a stderr warning now states this when `end ≥ n`
    (`test_range_beyond_curve_order_warns` [TEST], runs GPU-less).
    Search behavior unchanged.

## 8. Confirmed bugs vs hypotheses

| Finding | Classification | Resolution |
|---|---|---|
| "Missed match in config B" | **Refuted** (traversal-order + snapshot timing; contradicted by 12/12 sweep evidence) | no bug |
| Final candidate count not observable exactly (stale 1 Hz line) | **Confirmed reporting gap** | fixed: `Keys hashed : N` in result banners + regression test |
| Ranges ≥ curve order accepted silently | **Confirmed reporting gap** | fixed: boundary warning + regression test |
| Progress/count use inconsistent definitions | **Refuted** [CODE]+[TEST] | same counter, tested |
| Counted work ≠ actual work | **Refuted** [TEST] | exhaustive count == range length, exactly |

## Reproducing the investigation

```bash
python3 -m unittest tests.test_progress_accounting -v      # GPU: invariants
# GPU-less (binary required): boundary-warning tests inside the same file
```

Every test uses the canonical public vector or `ecdsa`-generated
synthetic targets — no wallet addresses.
