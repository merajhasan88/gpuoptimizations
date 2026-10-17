#include <cuda_runtime.h>

__global__ void histogram_shared(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int N,
    int num_bins)
{
    // Each block has its own histogram.
    // The constraints allow at most 1024 bins.
    __shared__ int local_hist[1024];

    const int tid = threadIdx.x;

    // Clear every active bin before counting.
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        local_hist[bin] = 0;
    }
    __syncthreads();

    const int first = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // Neighboring threads read neighboring input elements.
    // Each thread continues through the input using a grid-stride loop.
    for (int i = first; i < N; i += stride) {
        const int value = input[i];

        // Threads can hit the same bin simultaneously.
        // Atomic addition ensures that every increment is counted.
        atomicAdd(&local_hist[value], 1);
    }

    // Wait until this block has finished counting.
    __syncthreads();

    // Merge the block's totals into the final histogram.
    // Different blocks can update the same bin, so use atomics here too.
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        const int count = local_hist[bin];
        if (count != 0) {
            atomicAdd(&histogram[bin], count);
        }
    }
}

// input and histogram are device pointers.
extern "C" void solve(
    const int* input,
    int* histogram,
    int N,
    int num_bins)
{
    // Reset the output on every call.
    // The clear and kernel execute in order on the default stream.
    cudaMemsetAsync(histogram, 0, num_bins * sizeof(int), 0);

    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    // Limit how many block histograms must be merged.
    // The grid-stride loop still processes all N elements.
    if (blocks > 1024) {
        blocks = 1024;
    }

    histogram_shared<<<blocks, THREADS>>>(
        input, histogram, N, num_bins);
}
