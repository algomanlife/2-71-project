// lab/kernels.cuh — synthetic benchmark kernel declarations.
// Each benchmark defines its unit of work ("op") PRECISELY in both this
// header and research/cuda-lab/README.md. Deterministic inputs; sinks
// prevent dead-code elimination. No wallet targets, no key search.

#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Benchmark kernels
// ---------------------------------------------------------------------------

// op = one iteration of: v = (v + 0x9E3779B9u) ^ rotr(v,13)  (uint32)
__global__ void k_int32_mix(uint32_t* sink, int iters);

// op = one iteration of: v = v * 0x85EBCA6Bu ^ rotr(v,11)     (uint32 mul)
__global__ void k_int32_mul(uint32_t* sink, int iters);

// op = one iteration of: v = v * 0x9E3779B185EBCA87ULL ^ (v >> 23) (uint64 mul)
__global__ void k_int64_mul(uint64_t* sink, int iters);

// op = one 256-bit secp256k1 modular multiplication (_ModMult, project
// arithmetic via lab/secp_ref.cuh) on deterministic public operands from
// lab/vectors.h: r = r * B mod p, chained per thread (r0 = A XOR gid).
__global__ void k_modmul256(uint64_t* sink, int iters);

// op = one 64-bit coalesced global load + one 64-bit global store
// (strided by gridDim to touch memory repeatedly) -> bytes/op = 16.
__global__ void k_global_rw(uint64_t* buf, size_t n_elems, int iters);

// op = one 64-bit local (stack) load + one 64-bit local store via a
// dynamically-indexed per-thread array of 64 words (512 B — the subp[]
// access pattern class from the stack-frame investigation).
__global__ void k_local_rw(uint64_t* sink, int iters);

// op = one iteration of a deliberately warp-divergent branch
// (odd lanes multiply, even lanes add) — compare throughput with
// k_int32_mix to estimate divergence cost.
__global__ void k_divergent(uint32_t* sink, int iters);

// op = one full affine scalar multiplication k·G (public scalar) using
// the project's scalarMulBaseAffine (isolated in lab/secp_ref.cuh).
// One field inversion per point operation — the project's SETUP-phase
// cost class. NOT the search kernel.
__global__ void k_ec_scalar_mul(const uint64_t* scalars, uint64_t* outX, uint64_t* outY, int N);

// empty kernel for launch-latency measurement (op = one launch)
__global__ void k_empty();

// ---------------------------------------------------------------------------
// Host-side launch wrappers (used by main.cu AND tests/)
// ---------------------------------------------------------------------------

namespace lab {

struct KernelConfig {
    int blocks = 0;
    int threads = 0;
};

// Run one benchmark for `iters` per thread and return elapsed milliseconds
// (CUDA events, device synchronized). Single measurement — repetition and
// statistics are handled by scripts/run_experiments.py.
double run_bench(const char* name, int blocks, int threads, int iters,
                 uint64_t scratch_bytes);

// EC scalar-mul correctness helper: compute k·G for k in scalars[N]
// (4-limb LE each) and return true iff all results match expected[].
bool ec_scalar_mul_check(const uint64_t* scalars, const uint64_t* expected_xy,
                         int N, int blocks, int threads);

// modmul correctness helper: verify A·B, A·B^8, A·B^64 (mod p) against
// precomputed public vectors (see lab/vectors.h).
bool modmul_check(int blocks, int threads);

}  // namespace lab
