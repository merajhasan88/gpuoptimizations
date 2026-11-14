
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 320;

// Reused scratch. Every active entry is overwritten on each invocation.
__device__ volatile float partials[MAX_BLOCKS];
__device__ unsigned int completed = 0;

__device__ __forceinline__ float warp_sum(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, offset);
    return x;
}

template<bool VECTOR, bool MULTI_BLOCK>
__global__ __launch_bounds__(THREADS, 4)
void mse_kernel(const float* __restrict__ predictions,
                const float* __restrict__ targets,
                float* __restrict__ mse, int N, float inv_N)
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
        const float4* p4 = reinterpret_cast<const float4*>(predictions);
        const float4* t4 = reinterpret_cast<const float4*>(targets);
        const int count4 = N / 4;
        int i = first;

        // Two vector pairs per iteration, eight independent sum chains.
        for (; i + stride < count4; i += 2 * stride) {
            const float4 a = __ldg(p4 + i);
            const float4 b = __ldg(t4 + i);
            const float4 c = __ldg(p4 + i + stride);
            const float4 d = __ldg(t4 + i + stride);
            const float d0 = a.x - b.x, d1 = a.y - b.y;
            const float d2 = a.z - b.z, d3 = a.w - b.w;
            const float d4 = c.x - d.x, d5 = c.y - d.y;
            const float d6 = c.z - d.z, d7 = c.w - d.w;
            s0 = fmaf(d0, d0, s0); s1 = fmaf(d1, d1, s1);
            s2 = fmaf(d2, d2, s2); s3 = fmaf(d3, d3, s3);
            s4 = fmaf(d4, d4, s4); s5 = fmaf(d5, d5, s5);
            s6 = fmaf(d6, d6, s6); s7 = fmaf(d7, d7, s7);
        }
        if (i < count4) {
            const float4 a = __ldg(p4 + i);
            const float4 b = __ldg(t4 + i);
            const float d0 = a.x - b.x, d1 = a.y - b.y;
            const float d2 = a.z - b.z, d3 = a.w - b.w;
            s0 = fmaf(d0, d0, s0); s1 = fmaf(d1, d1, s1);
            s2 = fmaf(d2, d2, s2); s3 = fmaf(d3, d3, s3);
        }

        // At most three remaining elements, handled once across the grid.
        const int tail = 4 * count4 + first;
        if (tail < N) {
            const float d = predictions[tail] - targets[tail];
            s0 = fmaf(d, d, s0);
        }
    } else {
        // Scalar fallback when either input is not 16-byte aligned.
        int i = first;
        for (; i + 3 * stride < N; i += 4 * stride) {
            const float d0 = predictions[i] - targets[i];
            const float d1 = predictions[i + stride] - targets[i + stride];
            const float d2 = predictions[i + 2 * stride] - targets[i + 2 * stride];
            const float d3 = predictions[i + 3 * stride] - targets[i + 3 * stride];
            s0 = fmaf(d0, d0, s0); s1 = fmaf(d1, d1, s1);
            s2 = fmaf(d2, d2, s2); s3 = fmaf(d3, d3, s3);
        }
        for (; i < N; i += stride) {
            const float d = predictions[i] - targets[i];
            s0 = fmaf(d, d, s0);
        }
    }

    float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    sum = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < THREADS / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            if constexpr (MULTI_BLOCK) {
                partials[blockIdx.x] = block_sum;

                // Publish the partial BEFORE announcing block completion.
                __threadfence();
                const unsigned int ticket =
                    atomicInc(&completed, gridDim.x - 1);

                // atomicInc wraps the counter to zero for the next call.
                last_block = (ticket == gridDim.x - 1);
            } else {
                *mse = block_sum * inv_N;
            }
        }
    }

    if constexpr (MULTI_BLOCK) {
        __syncthreads();

        // Earlier blocks exit; nobody spins or waits for an unscheduled block.
        if (last_block && warp == 0) {
            float total = 0.0f;
            for (int b = lane; b < gridDim.x; b += 32)
                total += partials[b];

            total = warp_sum(total);
            if (lane == 0) *mse = total * inv_N;
        }
    }
}
} // namespace

extern "C" void solve(const float* predictions, const float* targets,
                      float* mse, int N)
{
    // Approximately 4096 input elements per block until the grid cap.
    int blocks = (N + 4095) / 4096;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    const float inv_N = 1.0f / static_cast<float>(N);
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(predictions) |
          reinterpret_cast<uintptr_t>(targets)) & 15u) == 0;

    if (blocks == 1) {
        if (aligned)
            mse_kernel<true, false><<<1, THREADS>>>(
                predictions, targets, mse, N, inv_N);
        else
            mse_kernel<false, false><<<1, THREADS>>>(
                predictions, targets, mse, N, inv_N);
    } else {
        if (aligned)
            mse_kernel<true, true><<<blocks, THREADS>>>(
                predictions, targets, mse, N, inv_N);
        else
            mse_kernel<false, true><<<blocks, THREADS>>>(
                predictions, targets, mse, N, inv_N);
    }
}
