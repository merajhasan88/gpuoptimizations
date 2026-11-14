// MSE candidate v4: register-prefetched input pairs.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int THREADS = 128;
constexpr int MAX_BLOCKS = 640;

// Every active entry is overwritten on each invocation.
__device__ volatile float partials[MAX_BLOCKS];
__device__ unsigned int completed = 0;

__device__ __forceinline__ void accumulate4(
    float4 p, float4 t,
    float& s0, float& s1, float& s2, float& s3)
{
    const float d0 = p.x - t.x, d1 = p.y - t.y;
    const float d2 = p.z - t.z, d3 = p.w - t.w;

    s0 = fmaf(d0, d0, s0);
    s1 = fmaf(d1, d1, s1);
    s2 = fmaf(d2, d2, s2);
    s3 = fmaf(d3, d3, s3);
}

__device__ __forceinline__ float warp_sum(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

template<bool VECTOR, bool MULTI_BLOCK>
__global__ __launch_bounds__(THREADS, 8)
void mse_pipelined_v4(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ mse,
    int N, float inv_N)
{
    __shared__ float warp_sums[THREADS / 32];
    __shared__ int last_block;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int first = blockIdx.x * THREADS + tid;
    const int stride = gridDim.x * THREADS;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

    if constexpr (VECTOR) {
        const float4* p4 =
            reinterpret_cast<const float4*>(predictions);
        const float4* t4 =
            reinterpret_cast<const float4*>(targets);

        const int count4 = N / 4;
        int i = first;

        // Prime the pipeline with eight input pairs.
        if (i + stride < count4) {
            float4 p0 = __ldg(p4 + i);
            float4 t0 = __ldg(t4 + i);
            float4 p1 = __ldg(p4 + i + stride);
            float4 t1 = __ldg(t4 + i + stride);

            // Load NEXT values before computing with CURRENT values.
            #pragma unroll 1
            for (; i + 3 * stride < count4; i += 2 * stride) {
                const float4 next_p0 =
                    __ldg(p4 + i + 2 * stride);
                const float4 next_t0 =
                    __ldg(t4 + i + 2 * stride);
                const float4 next_p1 =
                    __ldg(p4 + i + 3 * stride);
                const float4 next_t1 =
                    __ldg(t4 + i + 3 * stride);

                accumulate4(p0, t0, s0, s1, s2, s3);
                accumulate4(p1, t1, s4, s5, s6, s7);

                p0 = next_p0;
                t0 = next_t0;
                p1 = next_p1;
                t1 = next_t1;
            }

            // Consume the final buffered group without rereading it.
            accumulate4(p0, t0, s0, s1, s2, s3);
            accumulate4(p1, t1, s4, s5, s6, s7);
            i += 2 * stride;
        }

        // A thread can have one remaining float4 pair.
        if (i < count4) {
            const float4 p = __ldg(p4 + i);
            const float4 t = __ldg(t4 + i);
            accumulate4(p, t, s0, s1, s2, s3);
        }

        // Handle the final 0–3 scalar elements exactly once.
        const int tail = 4 * count4 + first;
        if (tail < N) {
            const float d = predictions[tail] - targets[tail];
            s0 = fmaf(d, d, s0);
        }
    } else {
        // Fallback for inputs without 16-byte alignment.
        int i = first;

        for (; i + 3 * stride < N; i += 4 * stride) {
            const float d0 = predictions[i] - targets[i];
            const float d1 =
                predictions[i + stride] - targets[i + stride];
            const float d2 =
                predictions[i + 2 * stride] - targets[i + 2 * stride];
            const float d3 =
                predictions[i + 3 * stride] - targets[i + 3 * stride];

            s0 = fmaf(d0, d0, s0);
            s1 = fmaf(d1, d1, s1);
            s2 = fmaf(d2, d2, s2);
            s3 = fmaf(d3, d3, s3);
        }

        for (; i < N; i += stride) {
            const float d = predictions[i] - targets[i];
            s0 = fmaf(d, d, s0);
        }
    }

    float sum =
        ((s0 + s1) + (s2 + s3)) +
        ((s4 + s5) + (s6 + s7));

    // Reduce each warp, then publish its sum in shared memory.
    sum = warp_sum(sum);
    if (lane == 0)
        warp_sums[warp] = sum;
    __syncthreads();

    // Warp 0 reduces the four warp sums.
    if (warp == 0) {
        float block_sum =
            lane < THREADS / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);

        if (lane == 0) {
            if constexpr (MULTI_BLOCK) {
                partials[blockIdx.x] = block_sum;

                // Publish the partial before announcing completion.
                __threadfence();

                const unsigned int ticket =
                    atomicInc(&completed, gridDim.x - 1);

                // The final increment also resets the counter.
                last_block = (ticket == gridDim.x - 1);
            } else {
                *mse = block_sum * inv_N;
            }
        }
    }

    if constexpr (MULTI_BLOCK) {
        __syncthreads();

        // Only the last finishing block combines all block totals.
        if (last_block && warp == 0) {
            float total = 0.0f;

            for (int b = lane; b < gridDim.x; b += 32)
                total += partials[b];

            total = warp_sum(total);

            if (lane == 0)
                *mse = total * inv_N;
        }
    }
}
} // namespace

extern "C" void solve(
    const float* predictions,
    const float* targets,
    float* mse, int N)
{
    int blocks = (N + 2047) / 2048;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;

    const float inv_N = 1.0f / static_cast<float>(N);
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(predictions) |
          reinterpret_cast<uintptr_t>(targets)) & 15u) == 0;

    if (blocks == 1) {
        if (aligned)
            mse_pipelined_v4<true, false><<<1, THREADS>>>(
                predictions, targets, mse, N, inv_N);
        else
            mse_pipelined_v4<false, false><<<1, THREADS>>>(
                predictions, targets, mse, N, inv_N);
    } else {
        if (aligned)
            mse_pipelined_v4<true, true><<<blocks, THREADS>>>(
                predictions, targets, mse, N, inv_N);
        else
            mse_pipelined_v4<false, true><<<blocks, THREADS>>>(
                predictions, targets, mse, N, inv_N);
    }
}
