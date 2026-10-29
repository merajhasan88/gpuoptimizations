#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

// All 32 lanes participate. Lane 0 receives the warp's sum.
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// Used by the generic fallback.
// Separate result slots prevent the next reduction from overwriting
// a result another warp is still reading.
__device__ __forceinline__ float block_sum(
    float value, float* warp_totals, float* result)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_sum(value);
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float total = lane < WARPS ? warp_totals[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) *result = total;
    }
    __syncthreads();
    return *result;
}

// One warp handles one 768-component token.
// Four independent warps share a block.
// No shared memory or block-wide barriers.
__global__ void token_embedding_768_vector(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows, int T, float eps)
{
    constexpr int D = 768;
    constexpr int VECTORS = D / (32 * 4);

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);

    // The entire warp takes the same branch.
    if (row >= rows) return;

    const int token = token_ids[row];
    const int position = position_ids[row % T];

    const float* token_row = token_embeddings + token * D;
    const float* position_row = position_embeddings + position * D;

    // Six float4 values = 24 components per lane.
    float4 values[VECTORS];
    float4 sums = make_float4(0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        const int d = (i * 32 + lane) * 4;

        const float4 a =
            *reinterpret_cast<const float4*>(token_row + d);
        const float4 b =
            *reinterpret_cast<const float4*>(position_row + d);

        const float4 v = make_float4(
            a.x + b.x, a.y + b.y,
            a.z + b.z, a.w + b.w);

        values[i] = v;

        // Four independent accumulation chains.
        sums.x += v.x;
        sums.y += v.y;
        sums.z += v.z;
        sums.w += v.w;
    }

    float total = warp_sum(
        (sums.x + sums.y) + (sums.z + sums.w));

    const float mean =
        __shfl_sync(0xffffffff, total, 0) / float(D);

    // Centered variance; values remain in registers.
    float4 squares = make_float4(0, 0, 0, 0);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        float4 v = values[i];

        v.x -= mean;
        v.y -= mean;
        v.z -= mean;
        v.w -= mean;

        values[i] = v;

        squares.x = fmaf(v.x, v.x, squares.x);
        squares.y = fmaf(v.y, v.y, squares.y);
        squares.z = fmaf(v.z, v.z, squares.z);
        squares.w = fmaf(v.w, v.w, squares.w);
    }

    total = warp_sum(
        (squares.x + squares.y) + (squares.z + squares.w));

    const float variance =
        __shfl_sync(0xffffffff, total, 0) / float(D);

    const float inverse_std = rsqrtf(variance + eps);

    #pragma unroll
    for (int i = 0; i < VECTORS; ++i) {
        const int d = (i * 32 + lane) * 4;

        const float4 g =
            *reinterpret_cast<const float4*>(gamma + d);
        const float4 b =
            *reinterpret_cast<const float4*>(beta + d);
        const float4 v = values[i];

        const float4 y = make_float4(
            g.x * (v.x * inverse_std) + b.x,
            g.y * (v.y * inverse_std) + b.y,
            g.z * (v.z * inverse_std) + b.z,
            g.w * (v.w * inverse_std) + b.w);

        *reinterpret_cast<float4*>(output + row * D + d) = y;
    }
}

// Previous working implementation retained for other dimensions
// and pointers that do not meet float4 alignment requirements.
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
        block_sum(local_sum, warp_totals, results)
        / static_cast<float>(D);

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
        / static_cast<float>(D);

    const float inverse_std = rsqrtf(variance + eps);

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int d = tid + i * THREADS;

        if (DIM != 0 || d < D) {
            const float normalized = values[i] * inverse_std;
            output[row * D + d] =
                gamma[d] * normalized + beta[d];
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

    // D=768 preserves 16-byte alignment for every row.
    if (D == 768 && (addresses & 15) == 0) {
        token_embedding_768_vector<<<(B * T + 3) / 4, 128>>>(
            token_ids, position_ids,
            token_embeddings, position_embeddings,
            gamma, beta, output, B * T, T, eps);
    } else {
        token_embedding_kernel<0><<<B * T, THREADS>>>(
            token_ids, position_ids,
            token_embeddings, position_embeddings,
            gamma, beta, output, T, D, eps);
    }
}
