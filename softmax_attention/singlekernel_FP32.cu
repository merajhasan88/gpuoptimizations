#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

namespace {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__
float warp_max(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(FULL_MASK, x, offset));
    }
    return x;  // Result is available in every lane.
}

__device__ __forceinline__
float warp_sum(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(FULL_MASK, x, offset);
    }
    return x;
}

// D is d rounded up to a multiple of 32.
// One warp handles one query row; four query rows share each K/V tile.
template <int D>
__global__ void attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int d)
{
    constexpr int PARTS = D / 32;

    // Reused for K and V. At D=128 this occupies 16 KiB.
    // XOR permutation supports both key-wise and feature-wise accesses.
    __shared__ float tile[D * 32];

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int row = blockIdx.x * WARPS + warp;

    float query[PARTS];
    float acc[PARTS];

    #pragma unroll
    for (int p = 0; p < PARTS; ++p) {
        const int c = p * 32 + lane;

        query[p] =
            (row < M && c < d) ? Q[row * d + c] : 0.0f;

        acc[p] = 0.0f;
    }

    const float scale = rsqrtf(static_cast<float>(d));

    float running_max = -CUDART_INF_F;
    float denominator = 0.0f;

    for (int first_key = 0; first_key < N; first_key += 32) {
        // Cooperatively load 32 key vectors.
        #pragma unroll
        for (int i = t; i < 32 * D; i += THREADS) {
            const int key = i / D;
            const int c = i % D;
            const int j = first_key + key;

            const float x =
                (j < N && c < d) ? K[j * d + c] : 0.0f;

            tile[c * 32 + (key ^ (c & 31))] = x;
        }
        __syncthreads();

        // Each lane computes the score for one key.
        float score = 0.0f;

        #pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            #pragma unroll
            for (int c = 0; c < 32; ++c) {
                const float q =
                    __shfl_sync(FULL_MASK, query[p], c);

                const float k =
                    tile[(p * 32 + c) * 32 + (lane ^ c)];

                score = fmaf(q, k, score);
            }
        }

        score = (first_key + lane < N)
            ? score * scale
            : -CUDART_INF_F;

        // Update the stable softmax maximum and denominator.
        const float tile_max = warp_max(score);
        const float next_max = fmaxf(running_max, tile_max);

        // Rescale previous contributions if the maximum increases.
        const float correction = __expf(running_max - next_max);
        const float weight = __expf(score - next_max);

        denominator =
            fmaf(denominator, correction, warp_sum(weight));

        #pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            acc[p] *= correction;
        }

        // All warps must finish reading K before tile is overwritten.
        __syncthreads();

        // Cooperatively load the corresponding 32 value vectors.
        #pragma unroll
        for (int i = t; i < 32 * D; i += THREADS) {
            const int key = i / D;
            const int c = i % D;
            const int j = first_key + key;

            const float x =
                (j < N && c < d) ? V[j * d + c] : 0.0f;

            tile[c * 32 + (key ^ (c & 31))] = x;
        }
        __syncthreads();

        // Accumulate the weighted values.
        // Each lane owns output columns lane, lane+32, ...
        #pragma unroll
        for (int key = 0; key < 32; ++key) {
            const float w =
                __shfl_sync(FULL_MASK, weight, key);

            #pragma unroll
            for (int p = 0; p < PARTS; ++p) {
                const int c = p * 32 + lane;
                const float v = tile[c * 32 + (key ^ lane)];

                acc[p] = fmaf(w, v, acc[p]);
            }
        }

        running_max = next_max;

        // Finish reading V before loading the next K tile.
        __syncthreads();
    }

    const float inverse = 1.0f / denominator;

    // Invalid query rows stayed active for all block barriers.
    if (row < M) {
        #pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            const int c = p * 32 + lane;

            if (c < d) {
                output[row * d + c] = acc[p] * inverse;
            }
        }
    }
}

} // namespace

extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M, int N, int d)
{
    const int blocks = (M + WARPS - 1) / WARPS;

    if (d <= 32) {
        attention_kernel<32><<<blocks, THREADS>>>(
            Q, K, V, output, M, N, d);
    } else if (d <= 64) {
        attention_kernel<64><<<blocks, THREADS>>>(
            Q, K, V, output, M, N, d);
    } else if (d <= 96) {
        attention_kernel<96><<<blocks, THREADS>>>(
            Q, K, V, output, M, N, d);
    } else {
        attention_kernel<128><<<blocks, THREADS>>>(
            Q, K, V, output, M, N, d);
    }
}
