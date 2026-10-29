#include <cuda_runtime.h>

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

// Only the eight warp totals pass through shared memory.
// The caller supplies a different result slot for each reduction, so the
// next reduction cannot overwrite a result another warp is still reading.
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

// DIM=768 specializes the benchmark shape: three values per thread.
// DIM=0 handles every supported D using at most four values per thread.
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

    // Consecutive threads access consecutive components in each iteration.
    // Values stay in registers through both reductions and the final store.
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

    const float mean = block_sum(local_sum, warp_totals, results)
                     / static_cast<float>(D);

    // Keep the centered variance formula to avoid subtracting two
    // potentially similar numbers in E[x*x] - E[x]*E[x].
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

    if (D == 768) {
        token_embedding_kernel<768><<<B * T, THREADS>>>(
            token_ids, position_ids, token_embeddings, position_embeddings,
            gamma, beta, output, T, D, eps);
    } else {
        token_embedding_kernel<0><<<B * T, THREADS>>>(
            token_ids, position_ids, token_embeddings, position_embeddings,
            gamma, beta, output, T, D, eps);
    }
}
