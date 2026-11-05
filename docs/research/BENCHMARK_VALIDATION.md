# Benchmark Validation — Do Reported Numbers Represent Executed Work?

Phase-3 audit (2026-11-05). Question: do the lab's reported performance
numbers accurately represent executed GPU workloads?

## Answer

**Infrastructure answer (implemented):** every benchmark now carries a
validity gate — `cuda-lab-bench --verify` / `--verify-only` re-runs each
kernel with its final per-thread value written unconditionally and
cross-checks **every thread** against an exact host reference of the same
arithmetic. Agreement proves the timed iterations executed; dead-code
elimination, loop elision, or timing an empty kernel would diverge. The
experiment runner aborts before timing if any bench fails verification,
and every JSON record carries `verified: true/false`.

**Status: the reported first measurements (below) are analyzed as
plausible-but-pending — they were collected before the verification gate
existed. Re-run with `--verify` on the rig to validate them.**

## Audit of the first RTX 5090 measurements (pre-verification, 5 reps)

| Benchmark | Reported | Plausibility analysis | Verdict |
|---|---:|---|---|
| int32-mix | 12,654.7 Gops/s (op = 1 iteration ≈ 3–4 SASS instr) | ≈ 38–50 T instr/s vs ~52 T theoretical INT32 issue (170 SM × 128 lanes × ~2.4 GHz) → 73–97% of peak. High but feasible for a dependent add/rotate/xor chain; verification decides | plausible, pending verify |
| int64-mul | 4,492.7 Gops/s | ratio vs int32 ≈ 0.35 — consistent with reduced 64-bit integer multiply throughput | plausible |
| global-rw | 1,488.55 GB/s | ~83% of GDDR7 theoretical (~1.79 TB/s) for a read-modify-write pattern | plausible |
| launch-latency | 1.74 µs/launch | typical for null-kernel launches incl. driver overhead | plausible |

Methodological cautions (unchanged, binding):
- op = one defined iteration, not one instruction; the op definition is
  in the lab README and in every JSON record.
- theoretical reference points are now emitted in each record
  (`theoretical_int32_ginstr_per_s`, `theoretical_mem_bw_gb_s`, from
  device attributes, labeled estimates).
- verification is exact (bit-for-bit vs host reference), not statistical.

## How to validate on the rig

```bash
cd research/cuda-lab && cmake --build build -j
./build/cuda-lab-bench --bench all --verify-only          # validity gate
./build/cuda-lab-bench --bench all --reps 5               # timing + verified flags
ctest --test-dir build                                     # includes verification
python3 scripts/run_experiments.py --reps 5                # gated end-to-end
```

## Verification coverage

| Benchmark | Verification method |
|---|---|
| int32-mix / int32-mul / int64-mul / divergent | exact per-thread final value vs host reference (all 16,384 verification threads) |
| modmul256 | device chain vs host reference replicating `_ModMult`'s exact fold semantics (transitively anchored: device already matches independent python-ecdsa vectors) |
| local-rw | per-thread final array state vs host simulation |
| global-rw | exact small-config simulation (XOR hit parity per element) |
| ec-scalar-mul | public-vector tests (2G/3G/7G/k·G) |
| launch-latency | n/a (timing-only; no computed values) |

## Remaining open item

The `256,128` grid observation (no speed line) in the application sweep
and the modmul256/ec-scalar-mul/local-rw/divergent lab numbers were not
part of the reported first-measurement table — run them with the
verification gate on the rig.
