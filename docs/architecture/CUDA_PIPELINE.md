# CUDA Execution Pipeline — 2^71 project

Reference: commit `b3d13be`. Line references: `project_2_71.cu` (post-#27).

```
main()
 ├─ CLI parse (--range/--address/--target-hash160/--grid/--slices)
 ├─ argument validation (power-of-two, aligned, batch even pow2 ≤1024)
 ├─ device init (cudaGetDeviceProperties; ck() error gate, exit 2)
 ├─ thread selection
 │    pick_threads_total: threadsTotal = 256-multiple | total_batches
 │    fallback (small ranges): largest divisor of total_batches ≤ cap
 │    blocks = ceil(threadsTotal / 256)   ← final block may be partial
 ├─ pinned host buffers (h_counts256, h_start_scalars; ck-checked)
 ├─ target upload → __constant__ c_target_prefix / c_target_hash160
 ├─ device allocations (d_Px/Py/Rx/Ry, d_start_scalars, d_counts256,
 │    d_found_flag, d_found_result, d_hashes_accum, d_any_left; ck-checked)
 ├─ SETUP KERNELS (once; affine double-and-add, per-op inversion)
 │    scalarMulKernelBase: start scalars → d_Px/d_Py
 │    scalarMulKernelBase: 1..B/2 multiples of G → c_Gx/c_Gy (constant)
 │    scalarMulKernelBase: J = B·G → c_Jx/c_Jy (constant)
 └─ SEARCH LOOP (per launch, non-blocking stream)
      kernel_point_add_and_check_oneinv<<<blocks,256,stream>>>
        entry: ballot mask (active lanes) → gid bounds check
        per thread: load P,S,rem  →  per batch of B keys:
          hash P; batch-invert ∏(G_i−x) (subp[] prefix products,
          1× _ModInv divsteps); walk ±(i+1)·G for i=0..B/2−1;
          each candidate: hash160 (SHA-256→RIPEMD-160), 4-byte
          prefix test, warp __any_sync, full 20-byte compare,
          atomic-CAS FOUND_LOCK→FOUND_READY + threadfence
          advance P += J (B·G); S += B; rem −= B
        hash accounting: warp-reduced atomicAdd (full warps) or
          per-thread atomicAdd (partial warps)
      host: 10 ms poll (found_flag D2H, cudaStreamQuery),
      progress line each second; swap d_Px↔d_Rx; repeat while any_left
 └─ result: FOUND (exit 0) / NOT_FOUND exhaustive (exit 1) /
    INTERRUPTED (exit 130) / CUDA ERROR (exit 2); cleanup
```

## Kernels

| Kernel | Role | Local memory |
|---|---|---|
| `kernel_point_add_and_check_oneinv` | search: batch-inversion stride walk + hash160 + match | 16·MAX_BATCH_SIZE + 304 B stack frame (see stack-frame investigation); 251 static LDL/STL (sm_120 SASS) |
| `scalarMulKernelBase` | setup scalar multiplication (affine, per-op `fieldInv`) | 192 B frame; setup-only cost |

## Memory map

- **Constant:** `c_Gx/c_Gy` (B/2 points), `c_Jx/c_Jy`, target prefix +
  hash160, SHA-256 K/IV (hash file).
- **Global:** per-thread 64 B × threadsTotal working set + result/state
  scalars; grid auto-sized from `prop.multiProcessorCount` and
  `totalGlobalMem`.
- **Pinned host:** write-combined mapped buffers for counts/start scalars.
- **Shared:** unused (by design — constant-memory broadcast for G table).

## Error handling

All CUDA API calls are wrapped by `ck()` (error name + description,
exit 2). Kernel-launch and stream-query failures print details, emit a
`CUDA ERROR` banner, and exit 2 — never exit 0. Cleanup (free/destroy)
runs on paths that reach the end of `main`; hard `ck` exits rely on
process teardown (documented limitation).

## Launch configuration semantics

`--grid A,B`: A = points per batch (even power of two ≤ 1024, must divide
range length), B = batches per SM (caps threadsTotal at
`SMs·B·256`). `--slices N`: batches per thread per kernel launch.
`threadsTotal` divides `total_batches` exactly, so each thread processes
an integer number of batches and the range is partitioned without
overlap or gaps.
