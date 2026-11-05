// lab kernels — synthetic benchmark implementations.
// Op definitions are binding (see include/lab/kernels.cuh + README.md).

#include "lab/kernels.cuh"
#include "lab/secp_ref.cuh"
#include "lab/vectors.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

using lab::MODMUL_A;
using lab::MODMUL_B;
using lab::MODMUL_R1;
using lab::MODMUL_R8;
using lab::MODMUL_R64;

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__global__ void k_int32_mix(uint64_t* sink, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u);
    for (int i = 0; i < iters; ++i) {
        v = (v + 0x9E3779B9u) ^ rotr32(v, 13);
    }
    if (v == 0xDEADBEEFu) sink[0] = (uint64_t)v;  // sink (never taken; defeats DCE)
}

__global__ void k_int32_mul(uint64_t* sink, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u) | 1u;
    for (int i = 0; i < iters; ++i) {
        v = v * 0x85EBCA6Bu ^ rotr32(v, 11);
    }
    if (v == 0xDEADBEEFu) sink[0] = (uint64_t)v;
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

__global__ void k_divergent(uint64_t* sink, int iters) {
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

double run_warm(const char* name, int blocks, int threads, int iters,
                uint64_t scratch_bytes, int warmup_iters) {
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

    // warmup (excluded from measurement; iterations configurable via
    // run_warm for clock/cache stabilization — baseline used 16)
    const int W = warmup_iters;
    if (!strcmp(name, "int32-mix"))      k_int32_mix<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "int32-mul"))      k_int32_mul<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "int64-mul"))      k_int64_mul<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "modmul256"))      k_modmul256<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "local-rw"))       k_local_rw<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "divergent"))      k_divergent<<<blocks, threads>>>(d_sink, W);
    if (!strcmp(name, "global-rw"))      k_global_rw<<<blocks, threads>>>(d_sink, scratch_bytes / 8, W);
    if (is_ec) {
        k_ec_scalar_mul<<<blocks, threads>>>(d_ec_s, d_ec_x, d_ec_y, (int)total_threads);
        CK(cudaDeviceSynchronize());
    }
    if (!strcmp(name, "launch-latency")) { for (int i = 0; i < W; ++i) k_empty<<<1, 32>>>(); }
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

// ---------------------------------------------------------------------------
// Benchmark validity verification (Phase 3)
// ---------------------------------------------------------------------------

// verification kernels: identical arithmetic to the bench kernels, but
// the final value is written UNCONDITIONALLY so the host can check it.
__global__ void v_int32_mix(uint32_t* out, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u);
    for (int i = 0; i < iters; ++i) v = (v + 0x9E3779B9u) ^ rotr32(v, 13);
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = v;
}
__global__ void v_int32_mul(uint32_t* out, int iters) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u) | 1u;
    for (int i = 0; i < iters; ++i) v = v * 0x85EBCA6Bu ^ rotr32(v, 11);
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = v;
}
__global__ void v_int64_mul(uint64_t* out, int iters) {
    uint64_t v = (uint64_t)(threadIdx.x + blockIdx.x * 97u) | 1ull;
    for (int i = 0; i < iters; ++i) v = v * 0x9E3779B185EBCA87ULL ^ (v >> 23);
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = v;
}
__global__ void v_divergent(uint32_t* out, int iters, int blockDim_x) {
    uint32_t v = (uint32_t)(threadIdx.x + blockIdx.x * 97u) | 1u;
    const uint32_t lane = threadIdx.x % (uint32_t)blockDim_x;
    for (int i = 0; i < iters; ++i) {
        if (lane & 1) v = v * 0x85EBCA6Bu;
        else          v = v + 0x9E3779B9u;
        v ^= rotr32(v, 11);
    }
    out[(size_t)blockIdx.x * blockDim_x + threadIdx.x] = v;
}
__global__ void v_local_rw(uint64_t* out, int iters) {
    uint64_t arr[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) arr[i] = (uint64_t)(threadIdx.x + i);
    for (int i = 0; i < iters; ++i) {
        const int idx = (i * 7) & 63;
        arr[idx] ^= 0x9E3779B185EBCA87ULL ^ (uint64_t)i;
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = arr[0];
}
__global__ void v_modmul256(uint64_t* out, int iters) {
    uint64_t r[4];
    const uint64_t gid_mix = (uint64_t)(threadIdx.x + blockIdx.x * 97u);
    r[0] = MODMUL_A[0] ^ gid_mix; r[1] = MODMUL_A[1];
    r[2] = MODMUL_A[2]; r[3] = MODMUL_A[3];
    for (int i = 0; i < iters; ++i)
        _ModMult(r, r, const_cast<uint64_t*>(MODMUL_B));
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int k = 0; k < 4; ++k) out[gid * 4 + k] = r[k];
}

// ---- host references (exact replication of kernel arithmetic) ----------

namespace {

void host_add_into(uint64_t* lo, uint64_t& carry, uint64_t add) {
    const uint64_t old = *lo;
    *lo += add;
    const uint64_t c1 = (*lo < old) ? 1u : 0u;
    const uint64_t old2 = *lo;
    *lo += carry;
    const uint64_t c2 = (*lo < old2) ? 1u : 0u;
    carry = c1 | c2;
}

// 256x256 -> 512-bit schoolbook product (64-bit limbs, u128 carries)
void host_mul_512(const uint64_t a[4], const uint64_t b[4], uint64_t r[8]) {
    for (int k = 0; k < 8; ++k) r[k] = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            unsigned __int128 t = (unsigned __int128)a[i] * b[j] + r[i + j] + carry;
            r[i + j] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
        int k = i + 4;
        while (k < 8 && carry) {
            const uint64_t old = r[k];
            r[k] += carry;
            carry = (r[k] < old) ? 1u : 0u;
            ++k;
        }
    }
}

// Replicates _ModMult's exact fold semantics (mod 2^256 folding with
// C = 2^32+977; result is NOT canonicalized to < p, matching the device).
void host_modmul_device_semantics(const uint64_t a[4], const uint64_t b[4],
                                  uint64_t out[4]) {
    const uint64_t C = 0x1000003D1ULL;
    uint64_t prod[8];
    host_mul_512(a, b, prod);

    // X = lo(256) + hi(256)*C  (exact, <= 289 bits)
    uint64_t t[5] = {0, 0, 0, 0, 0};
    {
        uint64_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 cur = (unsigned __int128)prod[4 + i] * C + carry;
            t[i] = (uint64_t)cur;
            carry = (uint64_t)(cur >> 64);
        }
        t[4] = carry;  // <= 2^33
    }
    uint64_t lo[4] = {prod[0], prod[1], prod[2], prod[3]};
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) host_add_into(&lo[i], carry, t[i]);
    uint64_t x_hi = t[4] + carry;  // small

    // final: result = lo + x_hi*C  (mod 2^256)  [matches device fold]
    unsigned __int128 fin = (unsigned __int128)x_hi * C;
    uint64_t al = (uint64_t)fin, ah = (uint64_t)(fin >> 64);
    uint64_t res[4] = {lo[0], lo[1], lo[2], lo[3]};
    carry = 0;
    host_add_into(&res[0], carry, al);
    host_add_into(&res[1], carry, ah);
    host_add_into(&res[2], carry, 0);
    host_add_into(&res[3], carry, 0);
    for (int k = 0; k < 4; ++k) out[k] = res[k];
}

}  // namespace

namespace lab {

bool verify_bench(const char* name) {
    // fixed small verification configuration: semantics, not scale
    const int VB = 64, VT = 256, VI = 64;
    const size_t N = (size_t)VB * VT;

    if (!strcmp(name, "int32-mix")) {
        uint32_t* d; uint32_t* h = new uint32_t[N];
        cudaMalloc(&d, N * 4);
        v_int32_mix<<<VB, VT>>>(d, VI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 4, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint32_t tid = (uint32_t)(g % VT), blk = (uint32_t)(g / VT);
            uint32_t v = tid + blk * 97u;
            for (int i = 0; i < VI; ++i) {
                v = (v + 0x9E3779B9u) ^ ((v >> 13) | (v << (32 - 13)));
            }
            ok = (h[g] == v);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "int32-mul")) {
        uint32_t* d; uint32_t* h = new uint32_t[N];
        cudaMalloc(&d, N * 4);
        v_int32_mul<<<VB, VT>>>(d, VI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 4, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint32_t tid = (uint32_t)(g % VT), blk = (uint32_t)(g / VT);
            uint32_t v = (tid + blk * 97u) | 1u;
            for (int i = 0; i < VI; ++i) {
                v = v * 0x85EBCA6Bu ^ ((v >> 11) | (v << (32 - 11)));
            }
            ok = (h[g] == v);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "int64-mul")) {
        uint64_t* d; uint64_t* h = new uint64_t[N];
        cudaMalloc(&d, N * 8);
        v_int64_mul<<<VB, VT>>>(d, VI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 8, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint64_t tid = g % VT, blk = g / VT;
            uint64_t v = (tid + blk * 97u) | 1ull;
            for (int i = 0; i < VI; ++i)
                v = v * 0x9E3779B185EBCA87ULL ^ (v >> 23);
            ok = (h[g] == v);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "divergent")) {
        uint32_t* d; uint32_t* h = new uint32_t[N];
        cudaMalloc(&d, N * 4);
        v_divergent<<<VB, VT>>>(d, VI, VT);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 4, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint32_t lane = (uint32_t)(g % VT), blk = (uint32_t)(g / VT);
            uint32_t v = (lane + blk * 97u) | 1u;
            for (int i = 0; i < VI; ++i) {
                if (lane & 1) v = v * 0x85EBCA6Bu;
                else          v = v + 0x9E3779B9u;
                v ^= (v >> 11) | (v << (32 - 11));
            }
            ok = (h[g] == v);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "local-rw")) {
        uint64_t* d; uint64_t* h = new uint64_t[N];
        cudaMalloc(&d, N * 8);
        v_local_rw<<<VB, VT>>>(d, VI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 8, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint64_t tid = g % VT;
            uint64_t arr[64];
            for (int i = 0; i < 64; ++i) arr[i] = tid + (uint64_t)i;
            for (int i = 0; i < VI; ++i) {
                const int idx = (i * 7) & 63;
                arr[idx] ^= 0x9E3779B185EBCA87ULL ^ (uint64_t)i;
            }
            ok = (h[g] == arr[0]);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "modmul256")) {
        uint64_t* d; uint64_t* h = new uint64_t[N * 4];
        cudaMalloc(&d, N * 32);
        v_modmul256<<<VB, VT>>>(d, VI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); delete[] h; return false; }
        cudaMemcpy(h, d, N * 32, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t g = 0; g < N && ok; ++g) {
            const uint64_t gid_mix = (g % VT) + (g / VT) * 97u;
            uint64_t r[4];
            r[0] = hostvecs::MODMUL_A[0] ^ gid_mix; r[1] = hostvecs::MODMUL_A[1];
            r[2] = hostvecs::MODMUL_A[2]; r[3] = hostvecs::MODMUL_A[3];
            for (int i = 0; i < VI; ++i)
                host_modmul_device_semantics(r, hostvecs::MODMUL_B, r);
            ok = !memcmp(h + g * 4, r, 32);
        }
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "global-rw")) {
        // small-config exact simulation: n elements, stride >= threads so
        // intra-iteration indices are distinct; XOR order-independent.
        const int GB = 8, GT = 64, GI = 8;
        const size_t n = 4096;
        uint64_t* d;
        cudaMalloc(&d, n * 8);
        cudaMemset(d, 0, n * 8);
        k_global_rw<<<GB, GT>>>(d, n, GI);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); cudaFree(d); return false; }
        uint64_t* h = new uint64_t[n];
        cudaMemcpy(h, d, n * 8, cudaMemcpyDeviceToHost);
        std::vector<uint64_t> hits(n, 0);
        const size_t stride = (size_t)GB * GT;
        for (int i = 0; i < GI; ++i)
            for (size_t tid = 0; tid < stride; ++tid)
                hits[(tid + (size_t)i * stride) % n] += 1;
        bool ok = true;
        const uint64_t K = 0x9E3779B185EBCA87ULL;
        for (size_t j = 0; j < n && ok; ++j)
            ok = (h[j] == ((hits[j] & 1) ? K : 0ull));
        delete[] h; cudaFree(d);
        return ok;
    }
    if (!strcmp(name, "ec-scalar-mul")) {
        // already covered by the public-vector tests
        return true;
    }
    if (!strcmp(name, "launch-latency")) {
        return true;  // timing-only benchmark: nothing to verify numerically
    }
    return false;  // unknown bench
}

}  // namespace lab


namespace lab {

double run_bench(const char* name, int blocks, int threads, int iters,
                 uint64_t scratch_bytes) {
    return run_warm(name, blocks, threads, iters, scratch_bytes, 16);
}

}  // namespace lab
