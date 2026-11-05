// cuda-lab correctness tests — public vectors only.
// Exit 0 = all pass; non-zero with a named failure otherwise.
// Run via ctest or directly: ./cuda-lab-tests

#include "lab/kernels.cuh"
#include "lab/vectors.h"

#include <cstdio>

int main() {
    int failures = 0;

    // 1. 256-bit modular multiplication vs public vectors (1, 8, 64 chain)
    if (lab::modmul_check(1, 32)) std::printf("modmul256 vectors ......... PASS\n");
    else { std::printf("modmul256 vectors ......... FAIL\n"); ++failures; }

    // 2. affine scalar multiplication vs public vectors (2G, 3G, 7G, RAND*G)
    {
        const uint64_t RAND_K[4] = {  // public constant (see lab/vectors.h)
            0x1234567890ABCDEFULL, 0x1234567890ABCDEFULL,
            0x1234567890ABCDEFULL, 0x1234567890ABCDEFULL };
        const uint64_t scalars[4][4] = {
            {2, 0, 0, 0}, {3, 0, 0, 0}, {7, 0, 0, 0}, {RAND_K[0], RAND_K[1], RAND_K[2], RAND_K[3]}};
        if (lab::ec_scalar_mul_check(&scalars[0][0], &lab::EC_EXPECTED[0][0], 4, 1, 128)) {
            std::printf("ec scalar-mul vectors .... PASS\n");
        } else {
            std::printf("ec scalar-mul vectors .... FAIL\n"); ++failures;
        }
    }

    // 3. ptxas-level check is environment-dependent; not asserted here.
    if (failures) { std::printf("%d FAILURES\n", failures); return 1; }
    std::printf("all correctness tests PASS\n");
    return 0;
}
