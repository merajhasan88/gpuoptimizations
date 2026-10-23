#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int AUX_THREADS = 1024;
constexpr int MAX_N = 100000000;
constexpr int MAX_BLOCKS = (MAX_N + THREADS - 1) / THREADS;
constexpr int MAX_GROUPS = (MAX_BLOCKS + AUX_THREADS - 1) / AUX_THREADS;

// Scratch space for every supported N, allocated with the CUDA module.
// No cudaMalloc/cudaFree is needed inside solve().
__device__ double block_sums[MAX_BLOCKS];
__device__ double group_sums[MAX_GROUPS];

__device__ __forceinline__
double warp_scan(double value)
{
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double earlier =
            __shfl_up_sync(0xffffffffu, value, offset);

        if (lane >= offset) {
            value += earlier;
        }
    }

    return value;
}

template<int BLOCK_THREADS>
__device__ __forceinline__
double block_scan(double value, double* warp_totals)
{
    constexpr int WARPS = BLOCK_THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // First compute an inclusive scan inside each warp.
    value = warp_scan(value);

    if (lane == 31) {
        warp_totals[warp] = value;
    }
    __syncthreads();

    // The first warp scans the totals of all warps in this block.
    if (warp == 0) {
        double total = (lane < WARPS) ? warp_totals[lane] : 0.0;
        total = warp_scan(total);

        if (lane < WARPS) {
            warp_totals[lane] = total;
        }
    }
    __syncthreads();

    // Add the contribution from preceding warps.
    if (warp > 0) {
        value += warp_totals[warp - 1];
    }

    return value;
}

__device__ __forceinline__
double warp_sum(double value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }

    return value;
}

// Pass 1: compute one total for each 256-element input chunk.
__global__ void reduce_chunks(const float* input, int N)
{
    __shared__ double warp_totals[THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int i = blockIdx.x * THREADS + tid;

    double value = (i < N) ? static_cast<double>(input[i]) : 0.0;
    value = warp_sum(value);

    if (lane == 0) {
        warp_totals[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = (lane < THREADS / 32) ? warp_totals[lane] : 0.0;
        value = warp_sum(value);

        if (lane == 0) {
            block_sums[blockIdx.x] = value;
        }
    }
}

// Pass 2: scan chunk totals in groups of 1024.
// For very large N, reuse this kernel to scan the group totals too.
__global__ void scan_totals(int count, bool upper_level)
{
    __shared__ double warp_totals[AUX_THREADS / 32];

    double* values = upper_level ? group_sums : block_sums;
    const int i = blockIdx.x * AUX_THREADS + threadIdx.x;

    double value = (i < count) ? values[i] : 0.0;
    value = block_scan<AUX_THREADS>(value, warp_totals);

    if (i < count) {
        values[i] = value;
    }

    // Zero padding means the last thread holds the whole group total.
    if (!upper_level && threadIdx.x == AUX_THREADS - 1) {
        group_sums[blockIdx.x] = value;
    }
}

// Final pass: scan each input chunk, add its offset, and write float32.
__global__ void write_prefix(
    const float* input, float* output, int N)
{
    __shared__ double warp_totals[THREADS / 32];

    const int block = blockIdx.x;
    const int i = block * THREADS + threadIdx.x;

    double value = (i < N) ? static_cast<double>(input[i]) : 0.0;
    value = block_scan<THREADS>(value, warp_totals);

    // All chunks before this one, within its group.
    double offset = 0.0;

    if (block % AUX_THREADS != 0) {
        offset = block_sums[block - 1];
    }

    // All complete groups before this chunk's group.
    const int group = block / AUX_THREADS;

    if (group > 0) {
        offset += group_sums[group - 1];
    }

    if (i < N) {
        output[i] = static_cast<float>(offset + value);
    }
}

} // namespace

// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N)
{
    const int blocks = (N + THREADS - 1) / THREADS;

    if (blocks > 1) {
        reduce_chunks<<<blocks, THREADS>>>(input, N);

        const int groups =
            (blocks + AUX_THREADS - 1) / AUX_THREADS;

        scan_totals<<<groups, AUX_THREADS>>>(blocks, false);

        if (groups > 1) {
            scan_totals<<<1, AUX_THREADS>>>(groups, true);
        }
    }

    // Launches on the same stream execute in order.
    // For N <= 256, this is the only kernel needed.
    write_prefix<<<blocks, THREADS>>>(input, output, N);
}
