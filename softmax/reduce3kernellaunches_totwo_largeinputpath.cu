#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int ITEMS = 8;
constexpr int TILE = THREADS * ITEMS;
constexpr int MAX_N = 500000;
constexpr int MAX_TILES = (MAX_N + TILE - 1) / TILE;

static_assert(MAX_TILES <= THREADS,
              "Final reduction needs one thread per tile");

// Scratch storage reused by sequential solve calls.
__device__ float tile_maxima[MAX_TILES];
__device__ float tile_sums[MAX_TILES];
__device__ unsigned int blocks_finished = 0;

// x = global maximum; y = reciprocal of the exponential sum.
__device__ float2 normalization;

template <bool MAXIMUM>
__device__ __forceinline__
float warp_reduce(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other =
            __shfl_down_sync(0xffffffffu, value, offset);

        if (MAXIMUM) value = fmaxf(value, other);
        else value += other;
    }
    return value;
}

// Every thread participates. Only thread 0 gets the complete result.
// Callers synchronize before reusing scratch.
template <bool MAXIMUM>
__device__ __forceinline__
float block_reduce(float value, float* scratch)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce<MAXIMUM>(value);

    if (lane == 0) scratch[warp] = value;
    __syncthreads();

    if (warp == 0) {
        const float identity = MAXIMUM ? -CUDART_INF_F : 0.0f;
        value = (lane < THREADS / 32) ? scratch[lane] : identity;
        value = warp_reduce<MAXIMUM>(value);
    }

    return value;
}

template <bool SINGLE_TILE>
__global__ void compute_normalization(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    __shared__ float scratch[THREADS / 32];
    __shared__ float shared_max;
    __shared__ float shared_inverse;
    __shared__ int last_block;

    const int t = threadIdx.x;
    const int base = blockIdx.x * TILE + t;

    float values[ITEMS];
    float local_max = -CUDART_INF_F;

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = base + k * THREADS;
        const float x =
            (i < N) ? __ldcg(input + i) : -CUDART_INF_F;

        values[k] = x;
        local_max = fmaxf(local_max, x);
    }

    const float maximum = block_reduce<true>(local_max, scratch);

    if (t == 0) shared_max = maximum;
    __syncthreads();

    const float tile_max = shared_max;
    float local_sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        values[k] = __expf(values[k] - tile_max);
        local_sum += values[k];
    }

    const float tile_sum = block_reduce<false>(local_sum, scratch);

    // Small inputs still finish in a single kernel.
    if (SINGLE_TILE) {
        if (t == 0) shared_inverse = 1.0f / tile_sum;
        __syncthreads();

        const float inverse = shared_inverse;

        #pragma unroll
        for (int k = 0; k < ITEMS; ++k) {
            const int i = base + k * THREADS;
            if (i < N) output[i] = values[k] * inverse;
        }

        return;
    }

    volatile float* maxima = tile_maxima;
    volatile float* sums = tile_sums;

    if (t == 0) {
        maxima[blockIdx.x] = tile_max;
        sums[blockIdx.x] = tile_sum;

        // Publish both statistics before announcing completion.
        __threadfence();

        const unsigned int ticket =
            atomicInc(&blocks_finished, gridDim.x);

        last_block = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // Other blocks exit; none spins waiting for unfinished blocks.
    if (!last_block) return;

    // The last block combines the statistics from every tile.
    const int count = gridDim.x;
    const float m = (t < count) ? maxima[t] : -CUDART_INF_F;
    const float global_max = block_reduce<true>(m, scratch);

    if (t == 0) shared_max = global_max;
    __syncthreads();

    float contribution = 0.0f;

    if (t < count) {
        contribution = sums[t] * __expf(m - shared_max);
    }

    const float total = block_reduce<false>(contribution, scratch);

    if (t == 0) {
        normalization = make_float2(shared_max, 1.0f / total);
        blocks_finished = 0;  // Ready for the next solve call.
    }
}

__global__ void write_softmax_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const float2 params = normalization;

    if (i + 3 < N) {
        const float4 x =
            __ldcg(reinterpret_cast<const float4*>(input + i));

        const float4 y = make_float4(
            __expf(x.x - params.x) * params.y,
            __expf(x.y - params.x) * params.y,
            __expf(x.z - params.x) * params.y,
            __expf(x.w - params.x) * params.y
        );

        *reinterpret_cast<float4*>(output + i) = y;
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (i + k < N) {
                output[i + k] =
                    __expf(input[i + k] - params.x) * params.y;
            }
        }
    }
}

__global__ void write_softmax_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        const float2 params = normalization;
        output[i] = __expf(__ldcg(input + i) - params.x) * params.y;
    }
}

} // namespace

extern "C" void solve(const float* input, float* output, int N)
{
    const int tiles = (N + TILE - 1) / TILE;

    if (tiles == 1) {
        compute_normalization<true><<<1, THREADS>>>(input, output, N);
        return;
    }

    // Launch 1: tile statistics and global normalization.
    compute_normalization<false>
        <<<tiles, THREADS>>>(input, output, N);

    // Launch 2: output. Default-stream ordering makes normalization ready.
    const uintptr_t addresses =
        reinterpret_cast<uintptr_t>(input) |
        reinterpret_cast<uintptr_t>(output);

    if ((addresses & 15u) == 0) {
        const int blocks =
            (N + THREADS * 4 - 1) / (THREADS * 4);

        write_softmax_vec4<<<blocks, THREADS>>>(input, output, N);
    } else {
        const int blocks = (N + THREADS - 1) / THREADS;

        write_softmax_scalar<<<blocks, THREADS>>>(input, output, N);
    }
}
