#include <cuda_runtime.h>

namespace {

// Each block contains 256 threads = 8 warps.
constexpr int THREADS = 256;

// Limit the first pass to 1,024 blocks.
// For larger inputs, each block processes more data through a loop.
constexpr int MAX_BLOCKS = 1024;

// Each thread processes four elements per loop iteration.
// Therefore, a whole block processes 256 × 4 = 1,024 elements.
constexpr int TILE = THREADS * 4;

// Temporary storage in GPU GLOBAL memory, shared between kernel launches.
// Each first-pass block writes one partial sum.
//
// Size: 1,024 floats × 4 bytes = 4 KB.
// This is separate from the reported 32 bytes of shared memory.
__device__ float partial_sums[MAX_BLOCKS];


// Sum the 32 values held by the threads of one warp.
//
// __device__: this function runs on the GPU.
// __forceinline__: ask the compiler to insert its instructions at the call site.
__device__ __forceinline__
float warp_sum(float value)
{
    // A warp contains 32 threads, numbered by "lane" from 0 to 31.
    //
    // __shfl_down_sync transfers a value directly between lanes.
    // With offset 16, lane 0 receives lane 16's value,
    // lane 1 receives lane 17's value, and so on.
    //
    // Offsets 16, 8, 4, 2, 1 form a reduction tree.
    // After all five additions, lane 0 holds the entire warp's sum.
    //
    // 0xffffffffu indicates that all 32 lanes participate.
    // All callers below ensure that a complete warp executes this code.
    //
    // Unrolling lets the compiler emit the five steps directly.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }

    // Only lane 0 is guaranteed to contain the complete warp sum.
    return value;
}


// Combine the values held by all 256 threads of a block.
//
// Every thread in the block must call this function because it contains
// a block-wide synchronization barrier.
//
// Only thread 0 receives the complete block sum.
__device__ __forceinline__
float block_sum(float value)
{
    // Shared memory is visible to every thread within this block.
    // Reserve one float for each of the block's eight warps.
    //
    // 8 floats × 4 bytes = 32 bytes of shared memory per block.
    __shared__ float warp_sums[THREADS / 32];

    // Position within the warp: threadIdx.x % 32.
    const int lane = threadIdx.x & 31;

    // Which warp this thread belongs to: threadIdx.x / 32.
    const int warp = threadIdx.x >> 5;

    // First, independently reduce the 32 values in each warp.
    value = warp_sum(value);

    // Each warp's lane 0 publishes its total into shared memory.
    // Eight threads write eight different array entries.
    if (lane == 0) {
        warp_sums[warp] = value;
    }

    // Wait until every warp has published its total.
    // This also makes those shared-memory writes visible to the block.
    __syncthreads();

    // The first warp combines the eight warp totals.
    if (warp == 0) {
        // Lanes 0–7 load the eight totals.
        // Lanes 8–31 contribute zero, allowing another full-warp reduction.
        value = (lane < THREADS / 32) ? warp_sums[lane] : 0.0f;
        value = warp_sum(value);
    }

    // Thread 0 now holds the complete block sum.
    // Other threads' returned values are not used.
    return value;
}


// Compile two versions of this kernel:
//
// WRITE_OUTPUT = true:
//     A single block reduces the input and writes the final answer.
//
// WRITE_OUTPUT = false:
//     Multiple blocks reduce different portions of the input.
//     Each block writes its total into partial_sums.
template <bool WRITE_OUTPUT>
__global__ void reduce_input(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    // __restrict__ tells the compiler that these input/output regions
    // do not overlap, allowing better optimization.
    //
    // Within a tile, thread t processes these four positions:
    //     tile_start + t
    //     tile_start + t + 256
    //     tile_start + t + 512
    //     tile_start + t + 768
    //
    // Consecutive threads therefore read consecutive floats at each load.
    // This allows the GPU to combine their accesses efficiently.
    int i = blockIdx.x * TILE + threadIdx.x;

    // Once a block finishes one tile, skip past all the other blocks'
    // tiles to reach its next tile.
    //
    // This distributes the entire input across the grid without overlap.
    const int stride = gridDim.x * TILE;

    // Four independent running sums, normally kept in registers.
    //
    // Using four sums shortens the dependency chain compared with adding
    // every loaded value into a single accumulator.
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    // The fourth position is the largest index read in this iteration.
    // If it is valid, all four loads are valid.
    //
    // At N = 4,194,304 with 1,024 blocks:
    //     stride = 1,048,576 elements
    //     each thread executes this loop four times
    //     each thread reads 16 elements in total
    for (; i + 3 * THREADS < N; i += stride) {
        s0 += input[i];
        s1 += input[i + THREADS];
        s2 += input[i + 2 * THREADS];
        s3 += input[i + 3 * THREADS];
    }

    // If fewer than four valid positions remain, handle them separately.
    // These checks also make small inputs and arbitrary N safe.
    //
    // At the benchmark size, the main loop handles every element,
    // so these conditions are false.
    if (i < N)
        s0 += input[i];
    if (i + THREADS < N)
        s1 += input[i + THREADS];
    if (i + 2 * THREADS < N)
        s2 += input[i + 2 * THREADS];
    if (i + 3 * THREADS < N)
        s3 += input[i + 3 * THREADS];

    // First combine this thread's four sums.
    // Then combine the resulting values from all threads in this block.
    const float total = block_sum((s0 + s1) + (s2 + s3));

    // Only thread 0 has the complete block total.
    if (threadIdx.x == 0) {
        // WRITE_OUTPUT is a compile-time constant.
        // The compiler can eliminate the unused branch in each version.
        if (WRITE_OUTPUT) {
            output[0] = total;
        } else {
            // Each block owns a different entry, so no atomic is needed.
            partial_sums[blockIdx.x] = total;
        }
    }
}


// Second pass: one block sums the first pass's partial results.
__global__ void reduce_partials(
    float* __restrict__ output,
    int count)
{
    float sum = 0.0f;

    // Distribute the partial sums across 256 threads.
    //
    // With 1,024 partial sums, thread t adds:
    //     partial_sums[t]
    //     partial_sums[t + 256]
    //     partial_sums[t + 512]
    //     partial_sums[t + 768]
    for (int i = threadIdx.x; i < count; i += THREADS) {
        sum += partial_sums[i];
    }

    // Combine all 256 thread-local sums.
    sum = block_sum(sum);

    // Assign the final answer directly.
    // The previous contents of output do not matter.
    if (threadIdx.x == 0) {
        output[0] = sum;
    }
}

} // namespace


// Host entry point called by LeetGPU.
// input and output are pointers to GPU memory.
//
// extern "C" gives the function a C-linkage name.
extern "C" void solve(const float* input, float* output, int N)
{
    // Ceiling division: enough blocks for one 1,024-element tile each.
    int blocks = (N + TILE - 1) / TILE;

    // Cap the grid size. The kernel's loop handles additional tiles.
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    if (blocks == 1) {
        // N <= 1,024: one block can produce the final answer directly.
        // Avoid launching a second kernel for this small input.
        reduce_input<true><<<1, THREADS>>>(input, output, N);
    } else {
        // Pass 1: produce one partial sum per block.
        reduce_input<false><<<blocks, THREADS>>>(input, output, N);

        // Pass 2: combine those partial sums into output[0].
        //
        // Both launches use the same default stream, so this kernel
        // executes after the first kernel finishes.
        //
        // Every partial entry read here was overwritten by the first pass,
        // so the temporary array does not require initialization.
        reduce_partials<<<1, THREADS>>>(output, blocks);
    }
}
