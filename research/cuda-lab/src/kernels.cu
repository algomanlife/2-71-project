// lab kernels — synthetic benchmark implementations.
// Op definitions are binding (see include/lab/kernels.cuh + README.md).

#include "lab/kernels.cuh"
#include "lab/secp_ref.cuh"
#include "lab/vectors.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__global__ void k_int32_mix(uint32_t* sink, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u);
    for (int i = 0; i < iters; ++i) {
        v = (v + 0x9E3779B9u) ^ rotr32(v, 13);
    }
    if (v == 0xDEADBEEFu) sink[0] = v;  // sink (never taken; defeats DCE)
}

__global__ void k_int32_mul(uint32_t* sink, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u) | 1u;
    for (int i = 0; i < iters; ++i) {
        v = v * 0x85EBCA6Bu ^ rotr32(v, 11);
    }
    if (v == 0xDEADBEEFu) sink[0] = v;
}

__global__ void k_int64_mul(uint64_t* sink, int iters) {
    uint64_t v = (uint64_t)(threadIdx.x + blockIdx.x * 97u) | 1ull;
    for (int i = 0; i < iters; ++i) {
        v = v * 0x9E3779B185EBCA87ULL ^ (v >> 23);
    }
    if (v == 0xDEADBEEFull) sink[0] = (uint32_t)v;
}

__global__ void k_modmul256(uint64_t* sink, int iters) {
    uint64_t r[4];
    const uint64_t gid_mix = (uint64_t)(threadIdx.x + blockIdx.x * 97u);
    // deterministic per-thread start: r = (A ^ gid) mod p  (public operands)
    r[0] = MODMUL_A[0] ^ gid_mix; r[1] = MODMUL_A[1];
    r[2] = MODMUL_A[2]; r[3] = MODMUL_A[3];
    for (int i = 0; i < iters; ++i) {
        _ModMult(r, r, const_cast<uint64_t*>(MODMUL_B));
    }
    if (r[0] == 0xDEADBEEFull) sink[0] = r[0];
}

__global__ void k_global_rw(uint64_t* buf, size_t n_elems, int iters) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int i = 0; i < iters; ++i) {
        size_t idx = (tid + (size_t)i * stride) % n_elems;
        buf[idx] = buf[idx] ^ 0x9E3779B185EBCA87ULL;  // 1 load + 1 store
    }
}

__global__ void k_local_rw(uint64_t* sink, int iters) {
    // 512 B per-thread local array, dynamically indexed (subp[] class)
    uint64_t arr[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) arr[i] = (uint64_t)(threadIdx.x + i);
    for (int i = 0; i < iters; ++i) {
        const int idx = (i * 7) & 63;
        arr[idx] ^= 0x9E3779B185EBCA87ULL ^ (uint64_t)i;  // 1 LDL + 1 STL
    }
    if (arr[0] == 0xDEADBEEFull) sink[0] = arr[0];
}

__global__ void k_divergent(uint32_t* sink, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u) | 1u;
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x & 1) v = v * 0x85EBCA6Bu;   // odd lanes
        else                 v = v + 0x9E3779B9u;   // even lanes
        v ^= rotr32(v, 11);
    }
    if (v == 0xDEADBEEFu) sink[0] = v;
}

__global__ void k_ec_scalar_mul(const uint64_t* scalars, uint64_t* outX, uint64_t* outY, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    scalarMulBaseAffine(scalars + (size_t)idx * 4,
                        outX + (size_t)idx * 4, outY + (size_t)idx * 4);
}

__global__ void k_empty() {}

// ---------------------------------------------------------------------------
// host wrappers
// ---------------------------------------------------------------------------

namespace lab {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error in %s: %s\n", #x, cudaGetErrorString(e_)); \
    return -1.0; } } while (0)

double run_bench(const char* name, int blocks, int threads, int iters,
                 uint64_t scratch_bytes) {
    const size_t total_threads = (size_t)blocks * threads;

    uint64_t* d_sink = nullptr;
    if (scratch_bytes) {
        CK(cudaMalloc(&d_sink, scratch_bytes));
        CK(cudaMemset(d_sink, 0, scratch_bytes));
    }
    // ec-scalar-mul buffers (allocated before any timing; scalar 0 = fast
    // identity path, deterministic)
    uint64_t *d_ec_s = nullptr, *d_ec_x = nullptr, *d_ec_y = nullptr;
    const bool is_ec = !strcmp(name, "ec-scalar-mul");
    if (is_ec) {
        CK(cudaMalloc(&d_ec_s, total_threads * 32));
        CK(cudaMalloc(&d_ec_x, total_threads * 32));
        CK(cudaMalloc(&d_ec_y, total_threads * 32));
        CK(cudaMemset(d_ec_s, 0, total_threads * 32));
    }

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    // warmup (excluded from measurement)
    if (!strcmp(name, "int32-mix"))      k_int32_mix<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "int32-mul"))      k_int32_mul<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "int64-mul"))      k_int64_mul<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "modmul256"))      k_modmul256<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "local-rw"))       k_local_rw<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "divergent"))      k_divergent<<<blocks, threads>>>(d_sink, 16);
    if (!strcmp(name, "global-rw"))      k_global_rw<<<blocks, threads>>>(d_sink, scratch_bytes / 8, 16);
    if (is_ec) {
        k_ec_scalar_mul<<<blocks, threads>>>(d_ec_s, d_ec_x, d_ec_y, (int)total_threads);
        CK(cudaDeviceSynchronize());
    }
    if (!strcmp(name, "launch-latency")) { k_empty<<<1, 32>>>(); }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(t0));
    if (!strcmp(name, "int32-mix"))      k_int32_mix<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "int32-mul"))      k_int32_mul<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "int64-mul"))      k_int64_mul<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "modmul256"))      k_modmul256<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "local-rw"))       k_local_rw<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "divergent"))      k_divergent<<<blocks, threads>>>(d_sink, iters);
    if (!strcmp(name, "global-rw"))      k_global_rw<<<blocks, threads>>>(d_sink, scratch_bytes / 8, iters);
    if (is_ec) {
        // op = one full affine scalar multiplication per thread
        k_ec_scalar_mul<<<blocks, threads>>>(d_ec_s, d_ec_x, d_ec_y, (int)total_threads);
    }
    if (!strcmp(name, "launch-latency")) {
        for (int i = 0; i < iters; ++i) k_empty<<<1, 32>>>();
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, t0, t1));

    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    if (d_sink) CK(cudaFree(d_sink));
    if (d_ec_s) { CK(cudaFree(d_ec_s)); CK(cudaFree(d_ec_x)); CK(cudaFree(d_ec_y)); }
    return (double)ms;
}

bool ec_scalar_mul_check(const uint64_t* scalars, const uint64_t* expected_xy,
                         int N, int blocks, int threads) {
    uint64_t *d_s, *d_x, *d_y, *h_x, *h_y;
    cudaMalloc(&d_s, (size_t)N * 32);
    cudaMalloc(&d_x, (size_t)N * 32);
    cudaMalloc(&d_y, (size_t)N * 32);
    cudaMemcpy(d_s, scalars, (size_t)N * 32, cudaMemcpyHostToDevice);
    k_ec_scalar_mul<<<blocks, threads>>>(d_s, d_x, d_y, N);
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return false; }
    h_x = new uint64_t[(size_t)N * 4];
    h_y = new uint64_t[(size_t)N * 4];
    cudaMemcpy(h_x, d_x, (size_t)N * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, (size_t)N * 32, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N && ok; ++i) {
        ok = !memcmp(h_x + i * 4, expected_xy + (size_t)i * 8, 32)
          && !memcmp(h_y + i * 4, expected_xy + (size_t)i * 8 + 4, 32);
    }
    delete[] h_x; delete[] h_y;
    cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
    return ok;
}

// verify chained modmul: run iters = 1, 8, 64 and compare r against R1/R8/R64
__global__ void modmul_check_kernel(uint64_t* out, int iters) {
    uint64_t r[4];
    r[0] = MODMUL_A[0]; r[1] = MODMUL_A[1]; r[2] = MODMUL_A[2]; r[3] = MODMUL_A[3];
    for (int i = 0; i < iters; ++i)
        _ModMult(r, r, const_cast<uint64_t*>(MODMUL_B));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const uint64_t* want = (iters == 1) ? MODMUL_R1 : (iters == 8) ? MODMUL_R8 : MODMUL_R64;
        const bool eq = r[0] == want[0] && r[1] == want[1]
                     && r[2] == want[2] && r[3] == want[3];
        out[0] = eq ? 1 : 0;
    }
}

bool modmul_check(int blocks, int threads) {
    uint64_t* d_out;
    cudaMalloc(&d_out, 4096);
    bool ok = true;
    for (int iters : {1, 8, 64}) {
        modmul_check_kernel<<<1, 32>>>(d_out, iters);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { ok = false; break; }
        uint64_t res = 0;
        cudaMemcpy(&res, d_out, 8, cudaMemcpyDeviceToHost);
        if (res != 1) { ok = false; break; }
    }
    cudaFree(d_out);
    return ok;
}

}  // namespace lab
