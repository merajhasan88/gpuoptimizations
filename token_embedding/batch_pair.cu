#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

// Lane 0 receives the warp's sum.
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);

    return value;
}

// Used by the generic fallback.
__device__ __forceinline__ float block_sum(
    float value, float* warp_totals, float* result)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_sum(value);
    if (lane == 0)
        warp_totals[warp] = value;

    __syncthreads();

    if (warp == 0) {
        float total = lane < WARPS ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);

        if (lane == 0)
            *result = total;
    }

    __syncthreads();
    return *result;
}

// Reduce two independent tokens together.
// Lane 0 receives both totals.
__device__ __forceinline__ float2 warp_sum_pair(float a, float b) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }

    return make_float2(a, b);
}

// One warp handles two tokens at the same position in adjacent batches.
// Position embeddings, gamma and beta are loaded once for the pair.
// Four independent warps per block: no shared memory or block barriers.
__global__ void token_embedding_768_batch_pair(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int B, int T, float eps)
{
    constexpr int D = 768;
    constexpr int VECTORS = D / (32 * 4);

    const int lane = threadIdx.x & 31;
    const int batch =
        2 * (blockIdx.x * 4 + (threadIdx.x >> 5));

    if (batch >= B)
        return; // Uniform across the entire warp.

    const int t = blockIdx.y;
    const int row0 = batch * T + t;
    const int row1 = row0 + T;

    const float* token0 =
        token_embeddings + token_ids[row0] * D;
    const float* token1 =
        token_embeddings + token_ids[row1] * D;
    const float* position =
        position_embeddings + position_ids[t] * D;

    float4 values0[VECTORS];
    float4 values1[VECTORS];

    float4 sums0 = make_float4(0, 0, 0, 0);
    float4 sums1 = make_float4(0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        const int d = (i * 32 + lane) * 4;

        // Independent token-table loads.
        const float4 a0 =
            __ldcg(reinterpret_cast<const float4*>(token0 + d));
        const float4 a1 =
            __ldcg(reinterpret_cast<const float4*>(token1 + d));

        // Reuse this positional vector for both tokens.
        const float4 p =
            *reinterpret_cast<const float4*>(position + d);

        const float4 v0 = make_float4(
            a0.x + p.x, a0.y + p.y,
            a0.z + p.z, a0.w + p.w);

        const float4 v1 = make_float4(
            a1.x + p.x, a1.y + p.y,
            a1.z + p.z, a1.w + p.w);

        values0[i] = v0;
        values1[i] = v1;

        sums0.x += v0.x;
        sums0.y += v0.y;
        sums0.z += v0.z;
        sums0.w += v0.w;

        sums1.x += v1.x;
        sums1.y += v1.y;
        sums1.z += v1.z;
        sums1.w += v1.w;
    }

    float2 totals = warp_sum_pair(
        (sums0.x + sums0.y) + (sums0.z + sums0.w),
        (sums1.x + sums1.y) + (sums1.z + sums1.w));

    const float mean0 =
        __shfl_sync(0xffffffff, totals.x, 0) / float(D);
    const float mean1 =
        __shfl_sync(0xffffffff, totals.y, 0) / float(D);

    // Separate centered variance for each token.
    float4 squares0 = make_float4(0, 0, 0, 0);
    float4 squares1 = make_float4(0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        float4 v0 = values0[i];
        float4 v1 = values1[i];

        v0.x -= mean0;
        v0.y -= mean0;
        v0.z -= mean0;
        v0.w -= mean0;

        v1.x -= mean1;
        v1.y -= mean1;
        v1.z -= mean1;
        v1.w -= mean1;

        values0[i] = v0;
        values1[i] = v1;

        squares0.x = fmaf(v0.x, v0.x, squares0.x);
        squares0.y = fmaf(v0.y, v0.y, squares0.y);
        squares0.z = fmaf(v0.z, v0.z, squares0.z);
        squares0.w = fmaf(v0.w, v0.w, squares0.w);

        squares1.x = fmaf(v1.x, v1.x, squares1.x);
        squares1.y = fmaf(v1.y, v1.y, squares1.y);
        squares1.z = fmaf(v1.z, v1.z, squares1.z);
        squares1.w = fmaf(v1.w, v1.w, squares1.w);
    }

    totals = warp_sum_pair(
        (squares0.x + squares0.y) + (squares0.z + squares0.w),
        (squares1.x + squares1.y) + (squares1.z + squares1.w));

    const float variance0 =
        __shfl_sync(0xffffffff, totals.x, 0) / float(D);
    const float variance1 =
        __shfl_sync(0xffffffff, totals.y, 0) / float(D);

    const float inverse_std0 = rsqrtf(variance0 + eps);
    const float inverse_std1 = rsqrtf(variance1 + eps);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        const int d = (i * 32 + lane) * 4;

        // Load affine parameters once and apply to both tokens.
        const float4 g =
            *reinterpret_cast<const float4*>(gamma + d);
        const float4 b =
            *reinterpret_cast<const float4*>(beta + d);

        const float4 v0 = values0[i];
        const float4 v1 = values1[i];

        const float4 y0 = make_float4(
            g.x * (v0.x * inverse_std0) + b.x,
            g.y * (v0.y * inverse_std0) + b.y,
            g.z * (v0.z * inverse_std0) + b.z,
            g.w * (v0.w * inverse_std0) + b.w);

        const float4 y1 = make_float4(
            g.x * (v1.x * inverse_std1) + b.x,
            g.y * (v1.y * inverse_std1) + b.y,
            g.z * (v1.z * inverse_std1) + b.z,
            g.w * (v1.w * inverse_std1) + b.w);

        *reinterpret_cast<float4*>(output + row0 * D + d) = y0;
        *reinterpret_cast<float4*>(output + row1 * D + d) = y1;
    }
}

// Generic fallback for all supported dimensions, odd batch counts,
// and pointers that are not aligned for float4 accesses.
template<int DIM>
__global__ void token_embedding_kernel(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int T, int runtime_D, float eps)
{
    constexpr int ITEMS = DIM == 0 ? 4 : DIM / THREADS;

    const int D = DIM == 0 ? runtime_D : DIM;
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int token = token_ids[row];
    const int position = position_ids[row % T];

    __shared__ float warp_totals[WARPS];
    __shared__ float results[2];

    float values[ITEMS];
    float local_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int d = tid + i * THREADS;
        float value = 0.0f;

        if (DIM != 0 || d < D)
            value = token_embeddings[token * D + d]
                  + position_embeddings[position * D + d];

        values[i] = value;
        local_sum += value;
    }

    const float mean =
        block_sum(local_sum, warp_totals, results) / float(D);

    float local_variance = 0.0f;

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int d = tid + i * THREADS;
        float centered = 0.0f;

        if (DIM != 0 || d < D)
            centered = values[i] - mean;

        values[i] = centered;
        local_variance = fmaf(centered, centered, local_variance);
    }

    const float variance =
        block_sum(local_variance, warp_totals, results + 1)
        / float(D);

    const float inverse_std = rsqrtf(variance + eps);

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int d = tid + i * THREADS;

        if (DIM != 0 || d < D) {
            const float normalized = values[i] * inverse_std;
            output[row * D + d] = gamma[d] * normalized + beta[d];
        }
    }
}
} // namespace

extern "C" void solve(
    const int* token_ids, const int* position_ids,
    const float* token_embeddings, const float* position_embeddings,
    const float* gamma, const float* beta, float* output,
    int B, int T, int V, int P, int D, float eps)
{
    (void)V;
    (void)P;

    const uintptr_t addresses =
          reinterpret_cast<uintptr_t>(token_embeddings)
        | reinterpret_cast<uintptr_t>(position_embeddings)
        | reinterpret_cast<uintptr_t>(gamma)
        | reinterpret_cast<uintptr_t>(beta)
        | reinterpret_cast<uintptr_t>(output);

    // Pair adjacent batches only when B is even.
    // D=768 preserves float4 alignment for every embedding row.
    if (D == 768 && (B & 1) == 0 && (addresses & 15) == 0) {
        const dim3 grid((B + 7) / 8, T);

        token_embedding_768_batch_pair<<<grid, 128>>>(
            token_ids, position_ids,
            token_embeddings, position_embeddings,
            gamma, beta, output,
            B, T, eps);
    } else {
        token_embedding_kernel<0><<<B * T, THREADS>>>(
            token_ids, position_ids,
            token_embeddings, position_embeddings,
            gamma, beta, output,
            T, D, eps);
    }
}
