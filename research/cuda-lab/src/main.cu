// cuda-lab — standalone synthetic benchmark runner.
//
// Usage:
//   ./cuda-lab --list
//   ./cuda-lab --bench modmul256 --blocks 1024 --threads 256 --iters 1000
//   ./cuda-lab --bench all --reps 5            (default configs)
//   ./cuda-lab --bench modmul256 --json        (machine-readable line)
//
// Throughput semantics per benchmark (op definitions in README.md):
//   ops = blocks*threads*iters  (except ec-scalar-mul: blocks*threads;
//   launch-latency: iters)
// Output: "name median_ms ops_per_s" human line and, with --json, one
// JSON object per bench with per-rep times. A single measurement is
// NEVER a result — use scripts/run_experiments.py for medians/variance.

#include "lab/kernels.cuh"
#include "lab/vectors.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct BenchDef {
    const char* name;
    int blocks;
    int threads;
    int iters;
    uint64_t scratch_bytes;
    const char* op_unit;      // precise definition (short form)
    bool per_thread_ops;      // ops = threads*iters
};

// default configurations (tuned for visibility, not peak)
const BenchDef kBenches[] = {
    {"int32-mix",   1024, 256, 20000, 0,       "1 iteration of add+xor+rotate (u32)", true},
    {"int32-mul",   1024, 256, 20000, 0,       "1 iteration of mul+xor+rotate (u32)", true},
    {"int64-mul",   1024, 256, 20000, 0,       "1 iteration of mul+xor+shift (u64)", true},
    {"modmul256",   1024, 256, 2000,  0,       "1 secp256k1 256-bit modular multiplication", true},
    {"global-rw",   1024, 256, 2000,  1ull<<28, "1 coalesced 64-bit load + 64-bit store (16 B/op)", true},
    {"local-rw",    1024, 256, 20000, 0,       "1 local-mem 64-bit load + 64-bit store (512 B/thread array)", true},
    {"divergent",   1024, 256, 20000, 0,       "1 divergent-branch iteration (odd: mul, even: add)", true},
    {"ec-scalar-mul", 64, 256, 1,     0,       "1 full affine scalar multiplication k*G (public scalar)", false},
    {"launch-latency", 1, 32, 10000,  0,       "1 empty kernel launch", false},
};
const int kNumBench = sizeof(kBenches) / sizeof(kBenches[0]);

}  // namespace

int main(int argc, char** argv) {
    const char* bench = "all";
    int blocks = -1, threads = -1, iters = -1, reps = 1;
    bool json = false, list = false, verify = false, verify_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench = argv[++i];
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) json = true;
        else if (!strcmp(argv[i], "--list")) list = true;
        else if (!strcmp(argv[i], "--verify")) verify = true;
        else if (!strcmp(argv[i], "--verify-only")) verify_only = true;
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    if (list) {
        for (int i = 0; i < kNumBench; ++i)
            std::printf("%-14s blocks=%-5d threads=%-4d iters=%-6d op: %s\n",
                        kBenches[i].name, kBenches[i].blocks, kBenches[i].threads,
                        kBenches[i].iters, kBenches[i].op_unit);
        return 0;
    }

    // environment report
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "CUDA init error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }

    // theoretical reference points (for utilization context; labeled
    // estimates — see docs/research/BENCHMARK_VALIDATION.md).
    // cudaDeviceProp::clockRate/memoryClockRate were removed in CUDA 13;
    // query device attributes instead (0 = unavailable).
    int clock_khz = 0, mem_clock_khz = 0, mem_bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    const double clock_ghz = (double)clock_khz / 1000000.0;
    const double int32_ginstr_per_s =
        (clock_khz > 0) ? (double)prop.multiProcessorCount * 128.0 * clock_ghz * 1e9 : 0.0;
    const double mem_bw_gbs =
        (mem_clock_khz > 0)
            ? ((double)mem_clock_khz * 2.0 * (double)mem_bus_bits / 8.0) / 1e9
            : 0.0;

    if (!json) {
        std::printf("# cuda-lab — synthetic benchmarks (research only)\n");
        std::printf("# device: %s (compute %d.%d, %d SM, %.2f GHz)\n", prop.name,
                    prop.major, prop.minor, prop.multiProcessorCount, clock_ghz);
        std::printf("# theoretical: int32 ~%.1f Ginstr/s, mem ~%.0f GB/s (estimates)\n",
                    int32_ginstr_per_s / 1e9, mem_bw_gbs);
    }

    int rc = 0;
    int verify_failures = 0;
    for (int i = 0; i < kNumBench; ++i) {
        const BenchDef& b = kBenches[i];
        if (strcmp(bench, "all") && strcmp(bench, b.name)) continue;

        // Phase-3 validity gate: cross-check executed work against the host
        // reference before reporting any timing (or standalone with
        // --verify-only). Divergence means the numbers are NOT valid.
        bool verified = lab::verify_bench(b.name);
        if (verify || verify_only) {
            if (json) {
                std::printf("{\"bench\":\"%s\",\"verified\":%s}\n", b.name,
                            verified ? "true" : "false");
            } else {
                std::printf("%-14s verification ......... %s\n", b.name,
                            verified ? "PASS (matches host reference)"
                                     : "FAIL (diverges from host reference)");
            }
            if (!verified) ++verify_failures;
        }
        if (verify_only) continue;

        int B = blocks > 0 ? blocks : b.blocks;
        int T = threads > 0 ? threads : b.threads;
        int I = iters > 0 ? iters : b.iters;

        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            double t = lab::run_bench(b.name, B, T, I, b.scratch_bytes);
            if (t < 0) { rc = 2; break; }
            ms.push_back(t);
        }
        if (rc) { std::fprintf(stderr, "benchmark %s failed\n", b.name); break; }

        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        double median = sorted[sorted.size() / 2];

        double ops;
        if (!strcmp(b.name, "ec-scalar-mul"))      ops = (double)B * T;
        else if (!strcmp(b.name, "launch-latency")) ops = (double)I;
        else                                        ops = (double)B * T * I;
        double ops_per_s = ops / (median / 1000.0);

        if (json) {
            std::printf("{\"bench\":\"%s\",\"blocks\":%d,\"threads\":%d,\"iters\":%d,"
                        "\"reps\":%d,\"ms\":%.4f,\"median_ms\":%.4f,\"ops\":%.0f,"
                        "\"ops_unit\":\"%s\",\"verified\":%s,"
                        "\"theoretical_int32_ginstr_per_s\":%.1f,"
                        "\"theoretical_mem_bw_gb_s\":%.0f}\n",
                        b.name, B, T, I, reps, ms[0], median, ops, b.op_unit,
                        verified ? "true" : "false",
                        int32_ginstr_per_s / 1e9, mem_bw_gbs);
        } else {
            const char* rate = strstr(b.op_unit, "B/op") ? "GB/s" : "Gops/s";
            if (!strcmp(b.name, "launch-latency"))
                std::printf("%-14s median %10.3f ms  (%.2f us/launch)\n",
                            b.name, median, median * 1000.0 / I);
            else if (!strcmp(rate, "GB/s"))
                std::printf("%-14s median %10.3f ms  %8.2f GB/s  (16 B/op)\n",
                            b.name, median, ops * 16.0 / (median / 1000.0) / 1e9);
            else
                std::printf("%-14s median %10.3f ms  %8.3f Gops/s  op: %s\n",
                            b.name, median, ops_per_s / 1e9, b.op_unit);
        }
    }
    if (verify_failures) {
        std::fprintf(stderr, "%d benchmark(s) FAILED verification — timings are NOT valid\n",
                     verify_failures);
        return 3;
    }
    return rc;
}
