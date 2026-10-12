#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

namespace {

constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int KEYS_PER_TILE = 256;
constexpr unsigned FULL_MASK = 0xffffffffu;

template <bool MAXIMUM>
__device__ __forceinline__
float warp_reduce(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other =
            __shfl_down_sync(FULL_MASK, value, offset);

        if (MAXIMUM) value = fmaxf(value, other);
        else value += other;
    }

    return value;
}

// Every thread participates; thread 0 receives the complete result.
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
        value = (lane < WARPS) ? scratch[lane] : identity;
        value = warp_reduce<MAXIMUM>(value);
    }

    return value;
}

// One block computes one complete output row.
template <int D>
__global__ void attention_row_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int N,
    int d)
{
    constexpr int PARTS = D / 32;

    __shared__ float weights[KEYS_PER_TILE];
    __shared__ float scratch[WARPS];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float correction;

    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    // Each warp holds a query copy distributed across its lanes.
    float query[PARTS];

    #pragma unroll
    for (int p = 0; p < PARTS; ++p) {
        const int c = lane + p * 32;
        query[p] = (c < d) ? Q[row * d + c] : 0.0f;
    }

    if (t == 0) {
        running_max = -CUDART_INF_F;
        running_sum = 0.0f;
    }

    const float scale = rsqrtf(static_cast<float>(d));
    float numerator = 0.0f;

    for (int first = 0; first < N; first += KEYS_PER_TILE) {
        const int remaining = N - first;
        const int count =
            (remaining < KEYS_PER_TILE) ? remaining : KEYS_PER_TILE;

        // Each warp computes four key scores together.
        #pragma unroll 1
        for (int j = warp * 4; j < count; j += WARPS * 4) {
            const int key = first + j;

            const bool valid1 = j + 1 < count;
            const bool valid2 = j + 2 < count;
            const bool valid3 = j + 3 < count;

            float dot0 = 0.0f;
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            float dot3 = 0.0f;

            #pragma unroll
            for (int p = 0; p < PARTS; ++p) {
                const int c = lane + p * 32;

                if (c < d) {
                    const float q = query[p];

                    dot0 = fmaf(q, K[key * d + c], dot0);

                    if (valid1) {
                        dot1 = fmaf(
                            q, K[(key + 1) * d + c], dot1);
                    }

                    if (valid2) {
                        dot2 = fmaf(
                            q, K[(key + 2) * d + c], dot2);
                    }

                    if (valid3) {
                        dot3 = fmaf(
                            q, K[(key + 3) * d + c], dot3);
                    }
                }
            }

            // Interleave four independent warp reductions.
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot0 += __shfl_down_sync(FULL_MASK, dot0, offset);
                dot1 += __shfl_down_sync(FULL_MASK, dot1, offset);
                dot2 += __shfl_down_sync(FULL_MASK, dot2, offset);
                dot3 += __shfl_down_sync(FULL_MASK, dot3, offset);
            }

            if (lane == 0) {
                weights[j] = dot0 * scale;

                if (valid1) weights[j + 1] = dot1 * scale;
                if (valid2) weights[j + 2] = dot2 * scale;
                if (valid3) weights[j + 3] = dot3 * scale;
            }
        }
        __syncthreads();

        // Each thread processes up to two scores.
        const float s0 = (t < count)
            ? weights[t] : -CUDART_INF_F;

        const float s1 = (t + THREADS < count)
            ? weights[t + THREADS] : -CUDART_INF_F;

        const float tile_max =
            block_reduce<true>(fmaxf(s0, s1), scratch);

        if (t == 0) {
            const float next_max = fmaxf(running_max, tile_max);

            correction = __expf(running_max - next_max);
            running_max = next_max;
        }
        __syncthreads();

        const float w0 = __expf(s0 - running_max);
        const float w1 = __expf(s1 - running_max);

        if (t < count) weights[t] = w0;
        if (t + THREADS < count) weights[t + THREADS] = w1;

        const float tile_sum =
            block_reduce<false>(w0 + w1, scratch);

        if (t == 0) {
            running_sum = fmaf(running_sum, correction, tile_sum);
        }
        __syncthreads();

        // Thread t computes output column t.
        if (t < d) {
            float a0 = 0.0f;
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;

            int j = 0;

            #pragma unroll 2
            for (; j + 3 < count; j += 4) {
                a0 = fmaf(
                    weights[j],
                    V[(first + j) * d + t],
                    a0);

                a1 = fmaf(
                    weights[j + 1],
                    V[(first + j + 1) * d + t],
                    a1);

                a2 = fmaf(
                    weights[j + 2],
                    V[(first + j + 2) * d + t],
                    a2);

                a3 = fmaf(
                    weights[j + 3],
                    V[(first + j + 3) * d + t],
                    a3);
            }

            // Handle an incomplete final group.
            for (; j < count; ++j) {
                a0 = fmaf(
                    weights[j],
                    V[(first + j) * d + t],
                    a0);
            }

            const float tile_numerator = (a0 + a1) + (a2 + a3);

            numerator =
                fmaf(numerator, correction, tile_numerator);
        }

        // Finish reading weights before the next tile overwrites them.
        __syncthreads();
    }

    if (t < d) {
        output[row * d + t] = numerator / running_sum;
    }
}

} // namespace

extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M,
    int N,
    int d)
{
    if (d <= 32) {
        attention_row_kernel<32><<<M, THREADS>>>(
            Q, K, V, output, N, d);
    } else if (d <= 64) {
        attention_row_kernel<64><<<M, THREADS>>>(
            Q, K, V, output, N, d);
    } else if (d <= 96) {
        attention_row_kernel<96><<<M, THREADS>>>(
            Q, K, V, output, N, d);
    } else {
        attention_row_kernel<128><<<M, THREADS>>>(
            Q, K, V, output, N, d);
    }
}
