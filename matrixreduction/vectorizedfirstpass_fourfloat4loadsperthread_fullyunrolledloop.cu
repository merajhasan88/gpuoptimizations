#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;
constexpr int TILE = THREADS * 4;

constexpr int FAST_N = 4194304;
constexpr int FAST_BLOCKS = 1024;
constexpr int FAST_STRIDE = FAST_BLOCKS * THREADS;
constexpr int FAST_ITERS = FAST_N / (FAST_STRIDE * 4);

__device__ float partial_sums[MAX_BLOCKS];

__device__ __forceinline__
float warp_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__
float block_sum(float value)
{
    __shared__ float warp_sums[THREADS / 32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_sum(value);

    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = (lane < THREADS / 32) ? warp_sums[lane] : 0.0f;
        value = warp_sum(value);
    }

    return value;
}

// Specialized first pass for N = 4,194,304.
// Each thread reads four float4 vectors: 16 floats total.
__global__ void reduce_fast(
    const float4* __restrict__ input)
{
    const int i = blockIdx.x * THREADS + threadIdx.x;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    // Exactly four iterations, with no bounds checks.
    #pragma unroll
    for (int k = 0; k < FAST_ITERS; ++k) {
        const float4 v = input[i + k * FAST_STRIDE];

        s0 += v.x;
        s1 += v.y;
        s2 += v.z;
        s3 += v.w;
    }

    const float total = block_sum((s0 + s1) + (s2 + s3));

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = total;
    }
}

// Original scalar path for other sizes or unaligned input.
template <bool WRITE_OUTPUT>
__global__ void reduce_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int i = blockIdx.x * TILE + threadIdx.x;
    const int stride = gridDim.x * TILE;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    for (; i + 3 * THREADS < N; i += stride) {
        s0 += input[i];
        s1 += input[i + THREADS];
        s2 += input[i + 2 * THREADS];
        s3 += input[i + 3 * THREADS];
    }

    if (i < N)
        s0 += input[i];
    if (i + THREADS < N)
        s1 += input[i + THREADS];
    if (i + 2 * THREADS < N)
        s2 += input[i + 2 * THREADS];
    if (i + 3 * THREADS < N)
        s3 += input[i + 3 * THREADS];

    const float total = block_sum((s0 + s1) + (s2 + s3));

    if (threadIdx.x == 0) {
        if (WRITE_OUTPUT) {
            output[0] = total;
        } else {
            partial_sums[blockIdx.x] = total;
        }
    }
}

__global__ void reduce_partials(
    float* __restrict__ output,
    int count)
{
    float sum = 0.0f;

    for (int i = threadIdx.x; i < count; i += THREADS) {
        sum += partial_sums[i];
    }

    sum = block_sum(sum);

    if (threadIdx.x == 0) {
        output[0] = sum;
    }
}

} // namespace

extern "C" void solve(const float* input, float* output, int N)
{
    // float4 loads require 16-byte alignment.
    if (N == FAST_N &&
        (reinterpret_cast<uintptr_t>(input) & 15u) == 0) {

        reduce_fast<<<FAST_BLOCKS, THREADS>>>(
            reinterpret_cast<const float4*>(input));

        reduce_partials<<<1, THREADS>>>(output, FAST_BLOCKS);
        return;
    }

    int blocks = (N + TILE - 1) / TILE;
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    if (blocks == 1) {
        reduce_generic<true><<<1, THREADS>>>(input, output, N);
    } else {
        reduce_generic<false><<<blocks, THREADS>>>(input, output, N);
        reduce_partials<<<1, THREADS>>>(output, blocks);
    }
}
