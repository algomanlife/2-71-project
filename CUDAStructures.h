#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2

struct FoundResult {
    int      threadId;
    int      iter;
    uint64_t scalar[4];
    uint64_t Rx[4];
    uint64_t Ry[4];
};

__device__ __constant__ uint8_t  c_target_hash160[20];
__device__ __constant__ uint32_t c_target_prefix;


#define CUDA_CHECK(ans) do { cudaError_t err = ans; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)






