#include <cuda_runtime.h>

// One block processes one token's D-dimensional embedding.
// The block size is the next power of two >= D.
__global__ void token_embedding_kernel(
    const int* token_ids,
    const int* position_ids,
    const float* token_embeddings,
    const float* position_embeddings,
    const float* gamma,
    const float* beta,
    float* output,
    int T, int D, float eps)
{
    // The constraints guarantee D <= 1024.
    __shared__ float sums[1024];

    const int row = blockIdx.x;  // Flattened (batch, time) index.
    const int d = threadIdx.x;  // Embedding component.
    const int t = row % T;

    // 1. Gather and add the token and positional embeddings.
    // Extra threads beyond D contribute zero to both reductions.
    float value = 0.0f;
    if (d < D) {
        const int token = token_ids[row];
        const int position = position_ids[t]; // Shared across batches.

        value = token_embeddings[token * D + d]
              + position_embeddings[position * D + d];
    }

    // 2. Sum all D components using a shared-memory tree reduction.
    sums[d] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (d < stride)
            sums[d] += sums[d + stride];

        __syncthreads();
    }

    const float mean = sums[0] / static_cast<float>(D);

    // Every thread must read the mean before sums[] is reused.
    __syncthreads();

    // 3. Compute variance as mean((value - mean)^2).
    // Divide by D, not D-1.
    const float centered = value - mean;
    sums[d] = d < D ? centered * centered : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (d < stride)
            sums[d] += sums[d + stride];

        __syncthreads();
    }

    // 4. Normalize, then apply the learned scale and shift.
    // rsqrtf(x) computes 1/sqrt(x).
    if (d < D) {
        const float variance = sums[0] / static_cast<float>(D);
        const float normalized = centered * rsqrtf(variance + eps);

        output[row * D + d] = gamma[d] * normalized + beta[d];
    }
}

// All pointer arguments refer to device memory.
extern "C" void solve(
    const int* token_ids,
    const int* position_ids,
    const float* token_embeddings,
    const float* position_embeddings,
    const float* gamma,
    const float* beta,
    float* output,
    int B, int T, int V, int P, int D, float eps)
{
    // IDs are guaranteed valid, so V and P need no runtime checks.
    (void)V;
    (void)P;

    int threads = 1;
    while (threads < D)
        threads *= 2;

    token_embedding_kernel<<<B * T, threads>>>(
        token_ids, position_ids, token_embeddings, position_embeddings,
        gamma, beta, output, T, D, eps);
}
