# Search Workflow — Implementation Overview

What `project-2-71` actually does, verified against the source
(`project_2_71.cu`, commit `ad3b5cb`). This document describes the
existing implementation only — nothing here modifies or optimizes the
key-discovery engine. Line references refer to the current file.

```
main()
 1  CLI parsing + argument validation            (no GPU work yet)
 2  device selection & properties                (device 0)
 3  thread/grid sizing
 4  pinned host buffers + target upload          (__constant__)
 5  setup kernels (one-time)
 6  search loop (sliced launches, stream)
 7  result reporting + cleanup
```

## 1. Command-line parsing

Five options (`--range`, `--address`, `--target-hash160`, `--grid A,B`,
`--slices N`; defaults `128,8` and `64`). All validation
(power-of-two range, alignment, batch evenness/limit, mutual exclusion
of the two target forms) happens **before** any CUDA call — invalid
input never touches the GPU. See [USAGE.md](USAGE.md) for the full
constraint table.

## 2. Range validation and alignment

- `end − start + 1` must be a power of two; `start` must be a multiple
  of that length (so the range covers complete aligned strides).
- The range length must be divisible by the batch size `A`.
- Target: a base58 P2PKH address is decoded and double-SHA checksum
  verified to a 20-byte hash160 (host, `sha256.h`); or a raw hash160 is
  given. Both forms are equivalent.

## 3. GPU initialization

Device 0 is selected; `cudaGetDeviceProperties` provides SM count,
memory, and limits. A 64 MB VRAM reserve is subtracted and the
per-thread working set (64 B/thread) bounds the thread count, together
with `SMs × B × 256` from `--grid A,B`.

**Grid/slice configuration → work distribution:** the scheduler picks
`threadsTotal` such that

- `threadsTotal` divides `total_batches = range_length / A` exactly —
  every thread processes a whole number of batches, so the scalar range
  is partitioned with **no overlap and no gap**;
- preferably a multiple of 256 (full blocks). When no 256-multiple
  works (small ranges), a fallback selects the largest valid divisor
  and launches a **partially-filled final block**; the kernel
  bounds-checks `gid` and derives its warp mask from an entry
  `__ballot_sync`, keeping warp-collective operations well-defined in
  partial warps.
- `--slices N` sets how many batches a thread consumes per kernel
  launch (bounded residency); state (current point, remaining count,
  scalar) is written back and reloaded on the next launch.

## 4. Host buffers and target upload

Write-combined mapped pinned buffers hold per-thread start scalars and
remaining counters. The 4-byte target prefix and 20-byte hash160 are
copied into `__constant__` memory for broadcast access.

## 5. Setup kernels (one-time, not the hot path)

Affine double-and-add scalar multiplication computes, for every thread,
its starting point; the multiples `1·G … (A/2)·G` and `J = A·G` are
precomputed into constant memory. Each point operation here performs a
full field inversion — setup cost only.

## 6. Search loop (hot path)

One kernel per launch on a non-blocking stream. Per thread, per batch of
`A` keys:

- prefix products of `(G_i − x)` are formed, **one** modular inversion
  (divsteps) per batch is performed (Montgomery batch-inversion), then
  each candidate point `P ± (i+1)·G` is derived by multiplication;
- each candidate's compressed public key is hashed (single-block
  SHA-256 → RIPEMD-160 = `hash160`) **on the fly** — nothing is stored;
- a 4-byte prefix compare short-circuits before the full 20-byte
  compare; a warp `__any_sync` vote avoids per-lane divergence before
  the full check;
- on a match: atomic CAS claims the result slot
  (`FOUND_NONE → FOUND_LOCK → FOUND_READY`) with a system thread fence;
  other warps observe the flag and exit early;
- hash counting is warp-reduced (one `atomicAdd` per full warp; direct
  per-thread atomics in partial warps).

The host polls every ~10 ms (found flag, stream completion), prints the
progress line once per second, and swaps input/output point buffers
between launches. The loop ends when the target is found, when all
threads exhaust their ranges, on Ctrl+C (after finishing the current
slice), or on error.

## 7. Result reporting, completion, interruption, errors

- **FOUND:** the winner's private key (hex) and compressed public key
  are printed; exit `0`.
- **Exhaustive completion without match:** `KEY NOT FOUND (exhaustive)`;
  exit `1`.
- **Ctrl+C:** the signal handler sets a flag; the current kernel slice
  completes, then `INTERRUPTED` prints with partial progress; exit
  `130`.
- **CUDA/runtime errors:** every CUDA call is checked; failures print
  `CUDA Error (<name>) in <context>: <description>` to stderr and a
  `CUDA ERROR`/`TERMINATED` banner, exiting `2`. Cleanup (frees, stream
  destroy) runs on paths that reach the end of `main`.

## Where correctness comes from

The invariants above (exact range partition, whole batches per thread,
batch inversion identities) are what the proof harness exercises — see
[PROOF_AND_VALIDATION.md](PROOF_AND_VALIDATION.md).
