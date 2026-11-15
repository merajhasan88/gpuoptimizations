#include <cuda_runtime.h>

namespace {
constexpr int THREADS = 128;
constexpr int COARSE = 4;
constexpr int SHARDS = 4096;
constexpr int FINISH_THREADS = 256;

__device__ float shard_sums[SHARDS];

__global__ void mse_clear_v8()
{
    for (int i = threadIdx.x; i < SHARDS; i += blockDim.x)
        shard_sums[i] = 0.0f;
}

template<bool DIRECT>
__global__ void mse_sharded_v8(
    const float* predictions,
    const float* targets,
    float* mse,
    int N)
{
    __shared__ float sdata[THREADS / 32];

    const int idx =
        blockIdx.x * blockDim.x * COARSE + threadIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Preserve the top kernel's four scalar loads per thread.
    // Adjacent lanes read adjacent elements.
    float val = 0.0f;

    for (int j = 0; j < COARSE; ++j) {
        const int i = idx + j * blockDim.x;

        if (i < N) {
            const float d = predictions[i] - targets[i];
            val += d * d;
        }
    }

    // Reduce within each warp.
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    if (lane == 0)
        sdata[warp] = val;

    __syncthreads();

    // First warp combines the four warp sums.
    if (warp == 0) {
        val = threadIdx.x < THREADS / 32
            ? sdata[threadIdx.x]
            : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffffu, val, offset);

        if (lane == 0) {
            if constexpr (DIRECT) {
                // Small inputs need only this kernel.
                *mse = val / N;
            } else {
                // Distribute additions across independent accumulators.
                // SHARDS is a power of two, so modulo becomes a bit mask.
                atomicAdd(
                    shard_sums + (blockIdx.x & (SHARDS - 1)),
                    val);
            }
        }
    }
}

__global__ void mse_finish_v8(float* mse, float inv_N)
{
    __shared__ float sdata[FINISH_THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = 0.0f;

    // Four independent sums reduce the accumulation dependency chain.
    #pragma unroll
    for (int i = tid; i < SHARDS; i += 4 * FINISH_THREADS) {
        a += shard_sums[i];
        b += shard_sums[i + FINISH_THREADS];
        c += shard_sums[i + 2 * FINISH_THREADS];
        d += shard_sums[i + 3 * FINISH_THREADS];
    }

    float val = (a + b) + (c + d);

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffffu, val, offset);

    if (lane == 0)
        sdata[warp] = val;

    __syncthreads();

    if (warp == 0) {
        val = lane < FINISH_THREADS / 32
            ? sdata[lane]
            : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffffu, val, offset);

        // Normalize once and overwrite the output.
        if (lane == 0)
            *mse = val * inv_N;
    }
}
} // namespace

extern "C" void solve(
    const float* predictions,
    const float* targets,
    float* mse,
    int N)
{
    const int blocks =
        (N + THREADS * COARSE - 1) / (THREADS * COARSE);

    if (blocks == 1) {
        mse_sharded_v8<true><<<1, THREADS>>>(
            predictions, targets, mse, N);
    } else {
        // Same-stream ordering guarantees clear -> accumulate -> finish.
        mse_clear_v8<<<1, FINISH_THREADS>>>();

        mse_sharded_v8<false><<<blocks, THREADS>>>(
            predictions, targets, mse, N);

        mse_finish_v8<<<1, FINISH_THREADS>>>(
            mse, 1.0f / static_cast<float>(N));
    }
}
