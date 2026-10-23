#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int ITEMS = 4;
constexpr int TILE = THREADS * ITEMS;
constexpr int AUX_THREADS = 1024;
constexpr int DIRECT_LIMIT = 256;
constexpr int MAX_N = 100000000;
constexpr int MAX_BLOCKS = (MAX_N + TILE - 1) / TILE;
constexpr int MAX_GROUPS =
    (MAX_BLOCKS + AUX_THREADS - 1) / AUX_THREADS;

__device__ double block_sums[MAX_BLOCKS];
__device__ double group_sums[MAX_GROUPS];

// Full, aligned groups use one float4 load.
// Handle tails and nonaligned pointers without out-of-bounds reads.
__device__ __forceinline__
float4 load_four(const float* input, int i, int N)
{
    if (i + 3 < N &&
        (reinterpret_cast<uintptr_t>(input) & 15) == 0) {
        return reinterpret_cast<const float4*>(input)[i / 4];
    }

    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (i < N)     v.x = input[i];
    if (i + 1 < N) v.y = input[i + 1];
    if (i + 2 < N) v.z = input[i + 2];
    if (i + 3 < N) v.w = input[i + 3];

    return v;
}

__device__ __forceinline__
double warp_scan(double value)
{
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const double previous =
            __shfl_up_sync(0xffffffffu, value, offset);

        if (lane >= offset) value += previous;
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

// First launch: one total for each 1024-element chunk.
__global__ void reduce_chunks4(const float* input, int N)
{
    __shared__ double warp_totals[THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int i = blockIdx.x * TILE + tid * ITEMS;
    const float4 v = load_four(input, i, N);

    double value =
        ((static_cast<double>(v.x) + v.y) + v.z) + v.w;

    value = warp_sum(value);

    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = (lane < THREADS / 32) ? warp_totals[lane] : 0.0;
        value = warp_sum(value);

        if (lane == 0) block_sums[blockIdx.x] = value;
    }
}

// Used only when there are more than DIRECT_LIMIT chunks.
// Scan chunk totals in groups, then optionally scan the group totals.
__global__ void scan_totals4(int count, bool upper_level)
{
    __shared__ double warp_totals[AUX_THREADS / 32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int i = blockIdx.x * AUX_THREADS + threadIdx.x;

    double* data = upper_level ? group_sums : block_sums;

    double value = (i < count) ? data[i] : 0.0;
    value = warp_scan(value);

    if (lane == 31) warp_totals[warp] = value;
    __syncthreads();

    if (warp == 0) {
        double total = warp_totals[lane];
        total = warp_scan(total);
        warp_totals[lane] = total;
    }
    __syncthreads();

    if (warp > 0) value += warp_totals[warp - 1];
    if (i < count) data[i] = value;

    if (!upper_level && threadIdx.x == AUX_THREADS - 1) {
        group_sums[blockIdx.x] = value;
    }
}

// Final launch: each thread writes four consecutive prefix sums.
__global__ void write_prefix4(
    const float* input, float* output, int N)
{
    __shared__ double warp_totals[THREADS / 32];
    __shared__ double chunk_offset;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int block = blockIdx.x;
    const int i = block * TILE + tid * ITEMS;

    const float4 v = load_four(input, i, N);

    // Keep local prefixes in double until the final float32 stores.
    const double p0 = static_cast<double>(v.x);
    const double p1 = p0 + v.y;
    const double p2 = p1 + v.z;
    const double p3 = p2 + v.w;

    // Scan one total per thread.
    const double inclusive = warp_scan(p3);

    // Prefix belonging to the preceding thread in this warp.
    const double previous =
        __shfl_up_sync(0xffffffffu, inclusive, 1);

    if (lane == 31) warp_totals[warp] = inclusive;

    // The first warp determines how much precedes this entire chunk.
    if (warp == 0) {
        double offset = 0.0;

        if (gridDim.x <= DIRECT_LIMIT) {
            // Small grids: directly sum earlier chunk totals.
            // This avoids a separate totals-scan launch.
            for (int j = lane; j < block; j += 32) {
                offset += block_sums[j];
            }

            offset = warp_sum(offset);
        } else if (lane == 0) {
            // Large grids: use the hierarchical scan.
            if (block % AUX_THREADS != 0) {
                offset = block_sums[block - 1];
            }

            const int group = block / AUX_THREADS;
            if (group > 0) {
                offset += group_sums[group - 1];
            }
        }

        if (lane == 0) chunk_offset = offset;
    }
    __syncthreads();

    // Scan the eight warp totals.
    if (warp == 0) {
        double total =
            (lane < THREADS / 32) ? warp_totals[lane] : 0.0;

        total = warp_scan(total);

        if (lane < THREADS / 32) {
            warp_totals[lane] = total;
        }
    }
    __syncthreads();

    double offset = chunk_offset;

    if (warp > 0) offset += warp_totals[warp - 1];
    if (lane > 0) offset += previous;

    const float4 result = make_float4(
        static_cast<float>(offset + p0),
        static_cast<float>(offset + p1),
        static_cast<float>(offset + p2),
        static_cast<float>(offset + p3));

    if (i + 3 < N &&
        (reinterpret_cast<uintptr_t>(output) & 15) == 0) {
        reinterpret_cast<float4*>(output)[i / 4] = result;
    } else {
        if (i < N)     output[i] = result.x;
        if (i + 1 < N) output[i + 1] = result.y;
        if (i + 2 < N) output[i + 2] = result.z;
        if (i + 3 < N) output[i + 3] = result.w;
    }
}

} // namespace

// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N)
{
    const int blocks = (N + TILE - 1) / TILE;

    if (blocks > 1) {
        reduce_chunks4<<<blocks, THREADS>>>(input, N);

        if (blocks > DIRECT_LIMIT) {
            const int groups =
                (blocks + AUX_THREADS - 1) / AUX_THREADS;

            scan_totals4<<<groups, AUX_THREADS>>>(blocks, false);

            if (groups > 1) {
                scan_totals4<<<1, AUX_THREADS>>>(groups, true);
            }
        }
    }

    // N = 250,000: 245 chunks and two launches in total.
    // N <= 1024: this is the only launch.
    write_prefix4<<<blocks, THREADS>>>(input, output, N);
}
