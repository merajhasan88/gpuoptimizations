#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int SMALL_STRIDE = 257;
constexpr int LARGE_STRIDE = 1025;
constexpr int SHARED_INTS = WARPS * SMALL_STRIDE;

__global__ void histogram_vec8_cg(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int N,
    int num_bins)
{
    __shared__ int local_hist[SHARED_INTS];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool small_bins = num_bins <= 256;

    for (int i = tid; i < SHARED_INTS; i += THREADS) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Eight private histograms for <=256 bins.
    // Two larger histograms for 257–1024 bins.
    const int replica_offset = small_bins
        ? warp_id * SMALL_STRIDE
        : (warp_id & 1) * LARGE_STRIDE;
    int* my_hist = local_hist + replica_offset;

    const int first = blockIdx.x * THREADS + tid;
    const int stride = gridDim.x * THREADS;

    // Align the vectorized portion to 16 bytes.
    // Handle up to three leading integers separately.
    const int misalignment =
        static_cast<int>(reinterpret_cast<uintptr_t>(input) & 15);
    const int prefix = min(N, ((16 - misalignment) & 15) / 4);

    if (first < prefix) {
        atomicAdd(&my_hist[input[first]], 1);
    }

    const int vector_count = (N - prefix) / 4;
    const int4* input4 = reinterpret_cast<const int4*>(input + prefix);

    // Load two vectors through L2, bypassing L1.
    // Both loads remain coalesced across the warp.
    int i = first;
    for (; i + stride < vector_count; i += 2 * stride) {
        const int4 a = __ldcg(input4 + i);
        const int4 b = __ldcg(input4 + i + stride);

        atomicAdd(&my_hist[a.x], 1);
        atomicAdd(&my_hist[a.y], 1);
        atomicAdd(&my_hist[a.z], 1);
        atomicAdd(&my_hist[a.w], 1);

        atomicAdd(&my_hist[b.x], 1);
        atomicAdd(&my_hist[b.y], 1);
        atomicAdd(&my_hist[b.z], 1);
        atomicAdd(&my_hist[b.w], 1);
    }

    // A thread can have one vector left after processing pairs.
    if (i < vector_count) {
        const int4 a = __ldcg(input4 + i);
        atomicAdd(&my_hist[a.x], 1);
        atomicAdd(&my_hist[a.y], 1);
        atomicAdd(&my_hist[a.z], 1);
        atomicAdd(&my_hist[a.w], 1);
    }

    // Handle the final zero to three integers.
    const int tail = prefix + 4 * vector_count + first;
    if (tail < N) {
        atomicAdd(&my_hist[input[tail]], 1);
    }

    __syncthreads();

    // Combine replicas, then merge this block's counts into the output.
    for (int bin = tid; bin < num_bins; bin += THREADS) {
        int count = 0;

        if (small_bins) {
            #pragma unroll
            for (int warp = 0; warp < WARPS; ++warp) {
                count += local_hist[warp * SMALL_STRIDE + bin];
            }
        } else {
            count = local_hist[bin] + local_hist[LARGE_STRIDE + bin];
        }

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
    // Reset the output on every call, on the same stream as the kernel.
    cudaMemsetAsync(histogram, 0, num_bins * sizeof(int), 0);

    // Use the winning grid cap: 1024 blocks.
    int blocks = (N + THREADS - 1) / THREADS;
    if (blocks > 1024) {
        blocks = 1024;
    }

    histogram_vec8_cg<<<blocks, THREADS>>>(
        input, histogram, N, num_bins);
}
