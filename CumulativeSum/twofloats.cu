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

// Vectorized load, with safe fallbacks for tails and alignment.
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
        value += __shfl_down_sync(
            0xffffffffu, value, offset);
    }

    return value;
}

// A leading float plus a small rounding correction.
struct SumPair {
    float hi;
    float lo;
};

// Recover the rounding error of a float addition.
// Explicit rounding intrinsics preserve the required operations.
__device__ __forceinline__
SumPair two_sum(float a, float b)
{
    const float s = __fadd_rn(a, b);
    const float z = __fsub_rn(s, a);

    const float e = __fadd_rn(
        __fsub_rn(a, __fsub_rn(s, z)),
        __fsub_rn(b, z));

    return {s, e};
}

__device__ __forceinline__
SumPair pair_add(SumPair a, SumPair b)
{
    const SumPair s = two_sum(a.hi, b.hi);
    const float e =
        __fadd_rn(s.lo, __fadd_rn(a.lo, b.lo));

    return two_sum(s.hi, e);
}

__device__ __forceinline__
SumPair warp_scan_pair(SumPair value)
{
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const SumPair previous = {
            __shfl_up_sync(
                0xffffffffu, value.hi, offset),
            __shfl_up_sync(
                0xffffffffu, value.lo, offset)
        };

        if (lane >= offset) {
            value = pair_add(previous, value);
        }
    }

    return value;
}

// First launch: calculate one double-precision total per chunk.
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
        value = (lane < THREADS / 32)
            ? warp_totals[lane] : 0.0;

        value = warp_sum(value);

        if (lane == 0) {
            block_sums[blockIdx.x] = value;
        }
    }
}

// Larger arrays use a hierarchical scan of chunk totals.
__global__ void scan_totals4(int count, bool upper_level)
{
    __shared__ double warp_totals[AUX_THREADS / 32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int i =
        blockIdx.x * AUX_THREADS + threadIdx.x;

    double* data =
        upper_level ? group_sums : block_sums;

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

    if (!upper_level &&
        threadIdx.x == AUX_THREADS - 1) {
        group_sums[blockIdx.x] = value;
    }
}

// Final launch: each thread writes four consecutive prefixes.
__global__ void write_prefix4_compensated(
    const float* input,
    float* output,
    int N)
{
    __shared__ SumPair warp_totals[THREADS / 32];
    __shared__ SumPair chunk_offset;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int block = blockIdx.x;
    const int i = block * TILE + tid * ITEMS;

    const float4 v = load_four(input, i, N);

    // Local prefixes use FP32 arithmetic with corrections.
    const SumPair p0 = {v.x, 0.0f};
    const SumPair p1 = two_sum(v.x, v.y);
    const SumPair p2 = pair_add(p1, {v.z, 0.0f});
    const SumPair p3 = pair_add(p2, {v.w, 0.0f});

    // Scan one total per thread.
    const SumPair inclusive = warp_scan_pair(p3);

    const SumPair previous = {
        __shfl_up_sync(
            0xffffffffu, inclusive.hi, 1),
        __shfl_up_sync(
            0xffffffffu, inclusive.lo, 1)
    };

    if (lane == 31) {
        warp_totals[warp] = inclusive;
    }

    // Determine the sum preceding this entire chunk.
    if (warp == 0) {
        double offset = 0.0;

        if (gridDim.x <= DIRECT_LIMIT) {
            // Small grids: directly sum earlier chunk totals,
            // avoiding another kernel launch.
            for (int j = lane; j < block; j += 32) {
                offset += block_sums[j];
            }

            offset = warp_sum(offset);
        } else if (lane == 0) {
            // Large grids: consume the hierarchical scan.
            if (block % AUX_THREADS != 0) {
                offset = block_sums[block - 1];
            }

            const int group = block / AUX_THREADS;

            if (group > 0) {
                offset += group_sums[group - 1];
            }
        }

        if (lane == 0) {
            const float hi = __double2float_rn(offset);

            chunk_offset = {
                hi,
                __double2float_rn(offset - double(hi))
            };
        }
    }

    __syncthreads();

    // Scan the eight warp totals.
    if (warp == 0) {
        SumPair total = (lane < THREADS / 32)
            ? warp_totals[lane]
            : SumPair{0.0f, 0.0f};

        total = warp_scan_pair(total);

        if (lane < THREADS / 32) {
            warp_totals[lane] = total;
        }
    }

    __syncthreads();

    SumPair offset = chunk_offset;

    if (warp > 0) {
        offset = pair_add(offset, warp_totals[warp - 1]);
    }

    if (lane > 0) {
        offset = pair_add(offset, previous);
    }

    // Round each completed prefix to float32.
    const float4 result = make_float4(
        pair_add(offset, p0).hi,
        pair_add(offset, p1).hi,
        pair_add(offset, p2).hi,
        pair_add(offset, p3).hi);

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

extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    const int blocks = (N + TILE - 1) / TILE;

    if (blocks > 1) {
        reduce_chunks4<<<blocks, THREADS>>>(input, N);

        if (blocks > DIRECT_LIMIT) {
            const int groups =
                (blocks + AUX_THREADS - 1) / AUX_THREADS;

            scan_totals4<<<groups, AUX_THREADS>>>(
                blocks, false);

            if (groups > 1) {
                scan_totals4<<<1, AUX_THREADS>>>(
                    groups, true);
            }
        }
    }

    // N = 250,000: 245 chunks and two launches.
    // N <= 1024: only this launch is needed.
    write_prefix4_compensated<<<blocks, THREADS>>>(
        input, output, N);
}
