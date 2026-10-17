#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

// Padding shifts each replica's starting shared-memory bank.
constexpr int SMALL_STRIDE = 256 + 1;
constexpr int LARGE_STRIDE = 1024 + 1;

// Fits eight small histograms or two large histograms.
constexpr int SHARED_INTS = WARPS * SMALL_STRIDE;

__global__ void histogram_warp_private(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int N,
    int num_bins)
{
    __shared__ int local_hist[SHARED_INTS];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool small_bins = num_bins <= 256;

    // Clear all private histograms, including padding.
    for (int i = tid; i < SHARED_INTS; i += THREADS) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Up to 256 bins: each warp gets its own histogram.
    // More bins: two groups of warps share two larger histograms.
    const int replica_offset = small_bins
        ? warp_id * SMALL_STRIDE
        : (warp_id & 1) * LARGE_STRIDE;

    int* my_hist = local_hist + replica_offset;

    const int first = blockIdx.x * THREADS + tid;
    const int stride = gridDim.x * THREADS;

    // Keep the same input-reading pattern as the baseline.
    for (int i = first; i < N; i += stride) {
        atomicAdd(&my_hist[input[i]], 1);
    }

    // All counting must finish before the replicas are combined.
    __syncthreads();

    for (int bin = tid; bin < num_bins; bin += THREADS) {
        int count = 0;

        if (small_bins) {
            // Combine eight private copies of this bin.
            #pragma unroll
            for (int warp = 0; warp < WARPS; ++warp) {
                count += local_hist[warp * SMALL_STRIDE + bin];
            }
        } else {
            // Combine the two larger histograms.
            count = local_hist[bin]
                  + local_hist[LARGE_STRIDE + bin];
        }

        // One global update per nonempty bin per block.
        if (count != 0) {
            atomicAdd(&histogram[bin], count);
        }
    }
}

} // namespace

// input and histogram are device pointers.
extern "C" void solve(
    const int* input,
    int* histogram,
    int N,
    int num_bins)
{
    // Clear on every call; the kernel uses the same default stream.
    cudaMemsetAsync(histogram, 0, num_bins * sizeof(int), 0);

    int blocks = (N + THREADS - 1) / THREADS;
    if (blocks > 1024) {
        blocks = 1024;
    }

    histogram_warp_private<<<blocks, THREADS>>>(
        input, histogram, N, num_bins);
}
