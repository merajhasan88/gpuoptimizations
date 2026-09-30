#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

// GPU storage for one sum per block.
// Double precision preserves more accuracy during accumulation.
__device__ double partial_sums[MAX_BLOCKS];

__global__ void sum_blocks(const float* input, int N)
{
    // Each thread gets one slot in this block's shared memory.
    __shared__ double sums[THREADS];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    // Distance between successive elements assigned to this thread.
    int step = blockDim.x * gridDim.x;

    // Each thread sums its own portion of the input.
    double local_sum = 0.0;
    for (; i < N; i += step) {
        local_sum += (double)input[i];
    }

    // Publish each thread's sum for the other threads in this block.
    sums[t] = local_sum;
    __syncthreads();

    // Combine 256 sums into 128, then 64, then 32, ... then 1.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            sums[t] += sums[t + stride];
        }

        // Finish this round before starting the next round.
        // Every thread must reach this barrier.
        __syncthreads();
    }

    // Thread 0 now holds the sum of this block's entire portion.
    if (t == 0) {
        partial_sums[blockIdx.x] = sums[0];
    }
}

__global__ void finish_sum(float* output, int count)
{
    // This kernel launches with exactly ONE thread.
    // It combines at most 1,024 block sums.
    double total = 0.0;

    for (int i = 0; i < count; ++i) {
        total += partial_sums[i];
    }

    // Convert to float32 only when writing the final answer.
    output[0] = (float)total;
}

extern "C" void solve(const float* input, float* output, int N)
{
    int blocks = (N + THREADS - 1) / THREADS;

    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    // First, compute the block sums in parallel.
    sum_blocks<<<blocks, THREADS>>>(input, N);

    // The same stream ensures all block sums are ready before this runs.
    finish_sum<<<1, 1>>>(output, blocks);
}
