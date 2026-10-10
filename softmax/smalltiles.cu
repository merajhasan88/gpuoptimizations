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

__device__ float tile_maxima[MAX_TILES];
__device__ float tile_sums[MAX_TILES];

// x = global maximum, y = reciprocal of the exponential sum.
__device__ float2 normalization;

template <bool MAXIMUM>
__device__ __forceinline__
float warp_reduce(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other =
            __shfl_down_sync(0xffffffffu, value, offset);

        if (MAXIMUM) {
            value = fmaxf(value, other);
        } else {
            value += other;
        }
    }

    return value;
}

// Every thread must participate.
// Only thread 0 receives the complete block result.
template <bool MAXIMUM>
__device__ __forceinline__
float block_reduce(float value, float* scratch)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce<MAXIMUM>(value);

    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        const float identity = MAXIMUM ? -CUDART_INF_F : 0.0f;

        value = (lane < THREADS / 32)
                    ? scratch[lane]
                    : identity;

        value = warp_reduce<MAXIMUM>(value);
    }

    return value;
}

template <bool SINGLE_TILE>
__global__ void compute_tiles(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    __shared__ float scratch[THREADS / 32];
    __shared__ float shared_max;
    __shared__ float shared_inverse;

    const int base = blockIdx.x * TILE + threadIdx.x;

    // Keep this thread's input values in registers.
    float values[ITEMS];
    float local_max = -CUDART_INF_F;

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        const int i = base + k * THREADS;

        const float value =
            (i < N) ? __ldcg(input + i) : -CUDART_INF_F;

        values[k] = value;
        local_max = fmaxf(local_max, value);
    }

    const float maximum = block_reduce<true>(local_max, scratch);

    if (threadIdx.x == 0) {
        shared_max = maximum;
    }
    __syncthreads();

    const float m = shared_max;
    float local_sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
        values[k] = __expf(values[k] - m);
        local_sum += values[k];
    }

    const float sum = block_reduce<false>(local_sum, scratch);

    if (SINGLE_TILE) {
        // This block contains the entire input.
        if (threadIdx.x == 0) {
            shared_inverse = 1.0f / sum;
        }
        __syncthreads();

        const float scale = shared_inverse;

        #pragma unroll
        for (int k = 0; k < ITEMS; ++k) {
            const int i = base + k * THREADS;

            if (i < N) {
                output[i] = values[k] * scale;
            }
        }
    } else {
        if (threadIdx.x == 0) {
            tile_maxima[blockIdx.x] = m;
            tile_sums[blockIdx.x] = sum;
        }
    }
}

// At most 245 tiles, so one 256-thread block handles all statistics.
__global__ void combine_tiles(int count)
{
    __shared__ float scratch[THREADS / 32];
    __shared__ float shared_max;

    const int t = threadIdx.x;

    const float local_max =
        (t < count) ? tile_maxima[t] : -CUDART_INF_F;

    const float maximum = block_reduce<true>(local_max, scratch);

    if (t == 0) {
        shared_max = maximum;
    }
    __syncthreads();

    // Convert every tile's sum to the same global maximum:
    // sum(exp(x - global_max))
    //   = tile_sum * exp(tile_max - global_max).
    float contribution = 0.0f;

    if (t < count) {
        contribution =
            tile_sums[t] * __expf(local_max - shared_max);
    }

    const float sum = block_reduce<false>(contribution, scratch);

    if (t == 0) {
        normalization = make_float2(shared_max, 1.0f / sum);
    }
}

__global__ void write_softmax_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i =
        (blockIdx.x * blockDim.x + threadIdx.x) * 4;

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
        // Handle the final one to three elements.
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

        output[i] =
            __expf(__ldcg(input + i) - params.x) * params.y;
    }
}

} // namespace

extern "C" void solve(const float* input, float* output, int N)
{
    const int tiles = (N + TILE - 1) / TILE;

    if (tiles == 1) {
        compute_tiles<true><<<1, THREADS>>>(input, output, N);
        return;
    }

    compute_tiles<false><<<tiles, THREADS>>>(input, output, N);
    combine_tiles<<<1, THREADS>>>(tiles);

    const uintptr_t addresses =
        reinterpret_cast<uintptr_t>(input) |
        reinterpret_cast<uintptr_t>(output);

    if ((addresses & 15u) == 0) {
        const int blocks = (N + THREADS * 4 - 1) / (THREADS * 4);

        write_softmax_vec4<<<blocks, THREADS>>>(input, output, N);
    } else {
        const int blocks = (N + THREADS - 1) / THREADS;

        write_softmax_scalar<<<blocks, THREADS>>>(input, output, N);
    }
}
