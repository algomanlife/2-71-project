# Initial Technical Audit — 2^71 project

Audit date: 2026-11-04
Commit audited: `b68bf64` ("2^71 project: initial release")
Auditor: QwenWork (agentic engineering session)
Hardware available during audit: **none** (macOS, no NVIDIA GPU, no CUDA toolkit).
Every statement below is either a **[VERIFIED FACT]** read directly from source
(with file:line references) or a **[HYPOTHESIS]** requiring profiler evidence.
No performance claims in this document are measurements.

---

## 1. Repository inventory

| File | Lines | Role |
|---|---|---|
| `project_2_71.cu` | 830 | Host application + main search kernel |
| `CUDAMath.h` | 1233 | secp256k1 field arithmetic, modular inverse, affine EC, setup kernel |
| `CUDAHash.cu` | 501 | GPU SHA-256 + RIPEMD-160 + hash160 pipeline |
| `CUDAHash.cuh` | 21 | Hash pipeline declarations |
| `CUDAStructures.h` | 32 | Constants, `FoundResult`, dead device globals |
| `CUDAUtils.h` | 185 | Host 256-bit helpers, prefix-match device helpers, warp reduce |
| `sha256.h` | 133 | Host SHA-256 (address decode), base58, P2PKH decode |
| `Makefile` | 25 | Build |
| `proof.py` | 377 | End-to-end key-skip verification harness |

Build: `nvcc -O3 -rdc=true -use_fast_math --ptxas-options=-O3 -std=c++17`
with `-gencode` for `sm_75, sm_86, sm_89` plus a runtime-detected arch
(`Makefile:6-10`). Link: `-lcudadevrt -cudart=static` (`Makefile:13`).

**[VERIFIED FACT]** No unit tests exist other than `proof.py` (end-to-end only).
No CI. No benchmark harness. No LICENSE file.

---

## 2. Execution flow (host)

Entry: `main()` at `project_2_71.cu:385`.

1. **CLI parsing** (`project_2_71.cu:415-437`): `--range`, `--address`,
   `--target-hash160`, `--grid A,B` (points-per-batch × batches-per-SM),
   `--slices`. Defaults: batch 128, batches/SM 8, slices 64
   (`project_2_71.cu:389-391`).
2. **Input validation** (`project_2_71.cu:440-511`): range power-of-two,
   start aligned to range length, batch even power-of-two ≤ 1024
   (`MAX_BATCH_SIZE`, `project_2_71.cu:41`).
3. **Device setup** (`project_2_71.cu:513-518`): device 0 only, cache config
   `cudaFuncCachePreferL1`. `threadsPerBlock` fixed at 256
   (`project_2_71.cu:520-522`).
4. **Thread-count selection** (`project_2_71.cu:524-563`): VRAM-limited
   (64 B/thread, 64 MB reserve, `project_2_71.cu:524-527`), capped by
   `SMs × batches_per_SM × 256` and by `total_batches`; then walked down in
   steps of 256 until `total_batches % threadsTotal == 0`
   (`pick_threads_total`, `project_2_71.cu:543-552`).
5. **Host buffers** (`project_2_71.cu:573-599`): write-combined mapped pinned
   memory for per-thread counters and start scalars (start = range_start +
   half·B offset per thread).
6. **Target upload** (`project_2_71.cu:601-608`): 4-byte prefix + 20-byte
   hash160 → `__constant__` memory.
7. **Setup kernels** (setup cost only, not in hot path):
   - `scalarMulKernelBase` for all thread start scalars
     (`project_2_71.cu:638-643`) — affine double-and-add **with a field
     inversion per point op** (`CUDAMath.h:1098-1167`).
   - 1..B/2 multiples of G → constant memory `c_Gx/c_Gy`
     (`project_2_71.cu:645-672`).
   - J = B·G → `c_Jx/c_Jy` (`project_2_71.cu:673-697`).
8. **Search loop** (`project_2_71.cu:729-793`): one kernel launch per
   iteration on a non-blocking stream; host polls every ~10 ms: progress
   print at 1 s cadence (`754-769`), found-flag D2H copy each poll
   (`771-773`), `cudaStreamQuery` (`775-777`), `sleep 10 ms` (`779`).
   On completion of a launch without "found", swaps `d_Px↔d_Rx`,
   `d_Py↔d_Ry` (`789-790`) and continues while `d_any_left != 0`.
9. **Result output** (`project_2_71.cu:803-819`) and cleanup
   (`822-828`).

### Execution-flow diagram

```
main()
 ├─ parse CLI ──► validate range/batch/grid
 ├─ cudaGetDeviceProperties ──► L1 cache preference
 ├─ pick threadsTotal (VRAM ∩ user ∩ batches, divisible)
 ├─ pinned alloc: h_counts256, h_start_scalars
 ├─ cudaMemcpyToSymbol: c_target_prefix, c_target_hash160
 ├─ SETUP (one-time, affine double-and-add w/ per-op inversion):
 │    scalarMulKernelBase(start scalars)      → d_Px, d_Py
 │    scalarMulKernelBase(1..B/2 · G)         → c_Gx, c_Gy (constant mem)
 │    scalarMulKernelBase(B · G)              → c_Jx, c_Jy (constant mem)
 └─ SEARCH LOOP (per kernel launch):
      kernel_point_add_and_check_oneinv <<<blocks,256,stream>>>
        per thread:
          load P(x1,y1), S, rem
          per batch (B keys):
            hash P; batch-invert Δx products (subp[] + 1×_ModInv)
            for i in 0..B/2-1:  hash(P+(i+1)G), hash(P−(i+1)G)   [prefix test]
            P += J (B·G);  S += B;  rem −= B
          write back P, S, rem
      host: poll found_flag / progress / stream done every 10 ms
      swap d_Px↔d_Rx; repeat while any thread has rem > 0
```

---

## 3. Kernel architecture (hot path)

`kernel_point_add_and_check_oneinv` (`project_2_71.cu:52-375`):

- `__launch_bounds__(256, 2)` (`project_2_71.cu:52`) — forces ≥2
  blocks/SM residency, i.e. caps registers at 32K/2/256 = 64 regs/thread.
- **One thread per key-stride**, not one thread per key: each thread owns a
  base point and walks `B` keys per batch using the
  "one inversion per batch" endomorphism-style stride.
- **Batch inversion** (`project_2_71.cu:140-166`): prefix products
  `subp[i] = ∏(G_j − x1)` (`144-156`), single `_ModInv`
  (`158-166`), then un-multiplied per step (`172-273`). This is the
  classic Montgomery trick; the single inversion is `_ModInv`
  (divsteps-based, `CUDAMath.h:464-568`).
- **Symmetric ± walk**: for each i, computes both `P+(i+1)G` and
  `P−(i+1)G` (`172-266`), plus a final third variant at `i = half−1`
  (`275-328`), then adds J = B·G to advance (`330-350`).
- **Hash pipeline per candidate**: `getHash160_33_from_limbs`
  (`CUDAHash.cu:494-501`, `__noinline__`): SHA-256 single 33-byte block
  (`CUDAHash.cu:445-468`) → RIPEMD-160 single 32-byte block
  (`CUDAHash.cu:470-492`).
- **Prefix short-circuit**: 4-byte little-endian prefix compare
  (`CUDAUtils.h:119-124`, `load_u32_le` `CUDAUtils.h:95-100`), then
  `__any_sync` across the warp before full 20-byte compare
  (`project_2_71.cu:120-137`, `CUDAUtils.h:102-113`).
- **Found protocol**: atomic CAS lock `FOUND_NONE→FOUND_LOCK→FOUND_READY`
  with `__threadfence_system` (`project_2_71.cu:122-134`).
- **Hash accounting**: per-thread counter, warp-reduced every 2^16 hashes,
  lane-0 `atomicAdd` (`project_2_71.cu:83-89`, `CUDAUtils.h:145-153`).
- **Early-exit discipline**: `warp_found_ready` checked per i-step
  (`project_2_71.cu:173`) and per batch (`112`).

---

## 4. Mathematical implementation

All arithmetic is 4×64-bit limb, little-endian, on secp256k1 prime
p = 2^256 − 2^32 − 977 (constants `CUDAMath.h:950-968`, `MulP`
`CUDAMath.h:211-225`).

| Operation | Location | Algorithm |
|---|---|---|
| 256×256 multiply | `CUDAMath.h:86-94` (`UMult` macro) | schoolbook via inline PTX `mul.lo/hi.u64`, `mad` |
| Modular reduction | `CUDAMath.h:570-582` (`UMultSpecial`), `585-633` (`_ModMult`) | fold high 256 bits with 2^32+1 special form (2-step: 512→320→256) |
| Squaring | `CUDAMath.h:685-788` (`_ModSqr`, `#if 1` branch) | 10 partial products + doubling chains |
| Modular inverse | `CUDAMath.h:464-568` (`_ModInv`) | Bernstein-Yang divsteps, 62-bit steps, 320-bit signed accumulator, `__noinline__` |
| Point addition (hot) | inlined in `project_2_71.cu:172-350` | affine formulas with shared batch inverse |
| Point add/double (setup) | `CUDAMath.h:1098-1167` | affine with full `fieldInv` each op |
| Scalar mult (setup) | `CUDAMath.h:1169-1231` | MSB double-and-add |

**[VERIFIED FACT]** The hot-path field ops (`_ModMult`, `_ModSqr`,
`ModSub256`, `ModNeg256`) use inline PTX carry chains
(`CUDAMath.h:5-26`) — no `__uint128_t` in the hot path (that appears only
in setup/EC code `CUDAMath.h:981-1056` and host code `CUDAUtils.h:1-53`).

**[VERIFIED FACT]** Hashing: SHA-256 64-round rolling 16-word schedule
(`CUDAHash.cu:46-79`); RIPEMD-160 fully unrolled dual-line
(`CUDAHash.cu:114-291`); both single-block, constants in
`__constant__` memory (`CUDAHash.cu:26-40`).

---

## 5. Memory usage

| Memory | Contents | Location |
|---|---|---|
| Constant | `c_Gx/c_Gy` (B/2 points × 2 × 32 B), `c_Jx/c_Jy`, `c_target_hash160/prefix`, SHA K/IV | `project_2_71.cu:47-50`, `CUDAStructures.h:12-13`, `CUDAHash.cu:26-40` |
| Global (device) | d_Px/d_Py/d_Rx/d_Ry/d_start_scalars/d_counts256 (64 B/thread), d_found_flag/result, d_hashes_accum, d_any_left | `project_2_71.cu:621-630` |
| Pinned host | h_counts256, h_start_scalars (write-combined, mapped) | `project_2_71.cu:573-576` |
| Shared | **none used in any kernel** | — |
| Local (per-thread spill) | **`subp[MAX_BATCH_SIZE/2][4]`** — up to 512×4×8 B = **16 KB/thread** at B=1024 (`project_2_71.cu:140`) | see bottleneck H1 |

**[VERIFIED FACT]** The dominant per-thread array `subp[512][4]` cannot fit
in registers; it necessarily lives in local memory (spill), size
`(B/2)·32` bytes per thread (8 KB at B=512, 2 KB at B=128).
This is a code-structure fact; its *performance impact* is a hypothesis (H1).

---

## 6. Dead / dormant code (verified by symbol search)

| Symbol | Location | Status |
|---|---|---|
| `USE_SYMMETRY` block (`ModPositive256`, `ModNeg256Order`) | `CUDAMath.h:294-323` | never defined → never compiled |
| `addBigEndian256` | `CUDAHash.cu:388-424` | defined, never called |
| `addBigEndian32` | `CUDAHash.cuh:18` | declared, never defined or called |
| `compare20` | `CUDAHash.cu:426-437` | defined, never called (superseded by `hash160_matches_prefix_then_full`) |
| `MatchResult` struct | `CUDAHash.cuh:8-14` | never used |
| `mul64x64_128`, `addWithCarry` | `CUDAMath.h:937-946` | defined, never called |
| `found_result`/`found_flag` device globals, `Gx_d`/`Gy_d` | `CUDAStructures.h:17-24` | defined, never used (search uses pointer-passed buffers) |
| `_ModSqr` `#if 0` alternative | `CUDAMath.h:790-919` | disabled variant |
| `scalarMulKernelBase` | declared 3× | `CUDAStructures.h:19`, `project_2_71.cu:383`, defined `CUDAMath.h:1222` |

Dead code is inert for correctness but inflates compile time and review
surface. Removal is a candidate low-risk cleanup (opt/O-01 in roadmap).

---

## 7. Build system

`Makefile` (25 lines): **[VERIFIED FACT]**

- Arch list hardcodes `75 86 89` + runtime GPU arch (`Makefile:6-8`).
  **No `sm_120` (Blackwell) and no `sm_100/110` support** — on an RTX 5090
  the binary relies on the auto-detected `GPU_ARCH` appended at build time;
  JIT from PTX of older archs is possible but suboptimal.
- `GPU_ARCH` detection runs `nvidia-smi` at *parse time* — build fails
  noise on GPU-less machines (cosmetic only, `Makefile:6`).
- `-rdc=true` requires `-lcudadevrt`; relocatable device code limits some
  compiler optimizations and increases launch overhead vs whole-program
  compilation **[HYPOTHESIS H4 — must be measured]**.
- `-use_fast_math` affects only floating-point; all kernel math is integer
  **[VERIFIED FACT — flag is a no-op for this workload]**.
- No `ptxas -v` register reporting, no debug/profile targets.

---

## 8. Correctness & testing status

- `proof.py` (377 lines): randomized + edge-case (both parities at range
  ends, full mod-B residue coverage, quartiles) end-to-end runs against the
  compiled binary; verifies no skipped/duplicated keys. Requires the
  `ecdsa` python package and a GPU. **[VERIFIED FACT]** No CPU-executable
  tests exist. No hash test vectors. No field-arithmetic unit tests.
- README documents historical results of this harness: 848/848 successes
  (RTX 4060) — **not re-verified in this audit (no GPU)**.

---

## 9. Suspected bottlenecks — HYPOTHESES ONLY

> None of the following is confirmed. All require profiler evidence
> (NVIDIA Nsight Compute / `nvcc --ptxas-options=-v`) on target hardware
> before being treated as facts.

| ID | Hypothesis | Rationale (code location) | Validation required |
|---|---|---|---|
| H1 | **Local-memory traffic from `subp[]`** dominates memory bandwidth | `project_2_71.cu:140` — `(B/2)·32 B` per thread in local memory, written (`144-156`) and read back (`176`, `278`) every batch | Nsight Compute `l1tex__t_bytes_pipe_lsu_mem_local_op_ld/st`, SASS `LDL/STL` counts |
| H2 | **Register pressure / occupancy ceiling** from large live set (x1,y1,S,rem,subp pointers, inverse accumulators) + `__launch_bounds__(256,2)` | `project_2_71.cu:52`, locals `91-101` | `nvcc -Xptxas -v` regs/thread; achieved occupancy in Nsight |
| H3 | **RIPEMD-160 + SHA-256 pipeline latency** is the arithmetic floor (160+64 rounds per hash, 2–3 hashes per i-step) | `CUDAHash.cu:46-291` | `sm__inst_executed_pipe_*` counts vs EC pipes |
| H4 | **`-rdc=true`** prevents whole-program codegen optimizations and adds module-load overhead | `Makefile:10,13` | A/B build with rdc off (must also drop `-lcudadevrt`), identical results |
| H5 | **Host-side blocking D2H `cudaMemcpy` of `found_flag` every 10 ms** perturbs the stream / adds sync overhead | `project_2_71.cu:771-773` | Measure with/without; alternative: mapped host-visible flag or events |
| H6 | **`__noinline__` on `getHash160_33_from_limbs` and `_ModInv`** trades register pressure for call overhead | `CUDAHash.cu:494`, `CUDAMath.h:464` | A/B after register report |
| H7 | **Setup kernels are slow** (affine double-and-add with one inversion per op) — visible on small ranges / restarts | `CUDAMath.h:1098-1167` | Time setup phase separately |
| H8 | **No `sm_120` SASS** in the default build → JIT/PTX fallback or suboptimal code on RTX 5090 / PRO 6000 | `Makefile:7` | Build with `-gencode arch=compute_120,code=sm_120`; compare |
| H9 | **Redundant writes**: `start_scalars` written back per launch even when unchanged mid-run is fine, but `counts256` full writeback (`360-366`) is 32 B/thread per launch | `project_2_71.cu:360-369` | Nsight memory workload analysis |

Priority for investigation on hardware: **H1 → H2 → H3** (these interact:
smaller B lowers H1 but raises per-key inversion amortization cost).

---

## 10. Build/reproducibility gaps (verified facts)

1. No pinned CUDA toolkit version, no `nvcc --version` capture in output.
2. No register/occupancy report target (`ptxas -v`).
3. No way to build without GPU present (`nvidia-smi` call at parse time).
4. Arch list is not CUDA-version-aware (fails on sm_120-only toolkits if
   old arches unsupported, and vice versa).
5. No LICENSE file — blocks distribution (see ATTRIBUTION.md).

---

## 11. What this audit does NOT cover

- SASS-level inspection (requires CUDA toolkit on target machine).
- Profiler metrics (requires GPU).
- The historical benchmark claims in README (not re-measured).

**Next steps:** Phase 2 (CUDA 13 / sm_120 compatibility), Phase 3
(baseline + validation harness), then hypothesis-driven optimization PRs
gated on profiler evidence per `docs/research/MAXIMUM_PERFORMANCE_ROADMAP.md`.

---

## Addendum (2026-11-04): compile evidence from CI (CUDA 13.0, sm_120)

CI run 41903696856 (CPU-only runner, `nvidia/cuda:13.0.0-devel-ubuntu24.04`,
`make GPU_ARCH=120 SM_ARCHS="120"`) compiled successfully and produced:

**[MEASURED FACT — compile-time only, not runtime performance]**

| Function | sm_120 ptxas report |
|---|---|
| `kernel_point_add_and_check_oneinv` | **16,688 bytes stack frame**, 8 bytes spill stores / 16 bytes spill loads |
| `_ModInv` | 192 bytes stack frame, 108 bytes spill stores / 108 bytes spill loads |

Interpretation:
- The 16.7 KB per-thread stack frame **confirms the existence** of the
  local-memory footprint predicted by H1 (the `subp[512][4]` worst case at
  `MAX_BATCH_SIZE=1024` compiles in, since `B` is a runtime parameter).
  Whether this dominates runtime memory traffic still requires Nsight
  Compute on hardware — H1 remains a *performance* hypothesis, but its
  structural premise is now compile-verified.
- H2 (register pressure) is **weakened** for the main kernel: only 8/16
  bytes of register spill; the `__launch_bounds__(256,2)` cap is not the
  binding constraint on this compilation.
- The sm_120 build (SASS + PTX) compiles cleanly on CUDA 13.0.
