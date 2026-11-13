#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>

namespace {
namespace cg = cooperative_groups;

constexpr int BLOCKS = 80;
__device__ float block_sums[BLOCKS];

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template<bool VECTOR>
__device__ __forceinline__ float4 load4(
    const float* p, int i, int N)
{
    if constexpr (VECTOR) {
        if (i + 3 < N)
            return *reinterpret_cast<const float4*>(p + i);
    }

    float4 v = make_float4(0, 0, 0, 0);
    if (i     < N) v.x = p[i];
    if (i + 1 < N) v.y = p[i + 1];
    if (i + 2 < N) v.z = p[i + 2];
    if (i + 3 < N) v.w = p[i + 3];
    return v;
}

template<bool VECTOR>
__device__ __forceinline__ void store4(
    float* p, int i, int N, float4 v)
{
    if constexpr (VECTOR) {
        if (i + 3 < N) {
            *reinterpret_cast<float4*>(p + i) = v;
            return;
        }
    }

    if (i     < N) p[i]     = v.x;
    if (i + 1 < N) p[i + 1] = v.y;
    if (i + 2 < N) p[i + 2] = v.z;
    if (i + 3 < N) p[i + 3] = v.w;
}

// Large path: 80 blocks, 320 threads, one float4 per thread.
// Small path: one block, 256 threads, up to four float4 values per thread.
template<int THREADS, int GRID, int ITEMS, bool VECTOR>
__global__ __launch_bounds__(THREADS, 1024 / THREADS)
void rms_kernel(
    const float* input, float* output, int N,
    float inv_N, float gamma, float beta, float eps)
{
    __shared__ float scratch[THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int vectors = (N + 3) / 4;
    const int begin = (vectors * blockIdx.x) / GRID;
    const int end = (vectors * (blockIdx.x + 1)) / GRID;

    // Retain input values in registers until the scale is available.
    float4 values[ITEMS];
    float4 acc = make_float4(0, 0, 0, 0);

    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int vi = begin + tid + j * THREADS;

        const float4 v =
            vi < end ? load4<VECTOR>(input, 4 * vi, N)
                     : make_float4(0, 0, 0, 0);

        values[j] = v;

        acc.x = fmaf(v.x, v.x, acc.x);
        acc.y = fmaf(v.y, v.y, acc.y);
        acc.z = fmaf(v.z, v.z, acc.z);
        acc.w = fmaf(v.w, v.w, acc.w);
    }

    float sum = (acc.x + acc.y) + (acc.z + acc.w);
    sum = warp_sum(sum);

    if (lane == 0)
        scratch[warp] = sum;

    __syncthreads();

    float total = 0.0f;

    if constexpr (GRID > 1) {
        // One warp combines the block's 10 warp sums.
        if (warp == 0) {
            float local =
                lane < THREADS / 32 ? scratch[lane] : 0.0f;

            local = warp_sum(local);

            if (lane == 0)
                block_sums[blockIdx.x] = local;
        }

        // This requires a cooperative kernel launch.
        cg::this_grid().sync();

        // Every warp reduces the same 80 block sums independently.
        // The input array itself is not read again.
        #pragma unroll
        for (int b = lane; b < GRID; b += 32)
            total += block_sums[b];
    } else {
        // Shared warp sums remain read-only after the first barrier.
        total = lane < THREADS / 32 ? scratch[lane] : 0.0f;
    }

    total = warp_sum(total);

    float scale = 0.0f;
    if (lane == 0)
        scale = gamma * rsqrtf(fmaf(total, inv_N, eps));

    scale = __shfl_sync(0xffffffffu, scale, 0);

    // Each warp now has its scale; no final block barrier is needed.
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int vi = begin + tid + j * THREADS;
        const float4 v = values[j];

        const float4 y = make_float4(
            fmaf(v.x, scale, beta),
            fmaf(v.y, scale, beta),
            fmaf(v.z, scale, beta),
            fmaf(v.w, scale, beta));

        if (vi < end)
            store4<VECTOR>(output, 4 * vi, N, y);
    }
}

template<bool VECTOR>
void launch(
    const float* input, float* output, int N,
    float inv_N, float gamma, float beta, float eps)
{
    if (N <= 4096) {
        rms_kernel<256, 1, 4, VECTOR><<<1, 256>>>(
            input, output, N, inv_N, gamma, beta, eps);
    } else {
        void* args[] = {
            &input, &output, &N, &inv_N, &gamma, &beta, &eps
        };

        cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(
                rms_kernel<320, BLOCKS, 1, VECTOR>),
            dim3(BLOCKS),
            dim3(320),
            args,
            0,
            nullptr);
    }
}

} // namespace

extern "C" void solve(
    const float* input, float gamma, float beta,
    float* output, int N, float eps)
{
    const float inv_N = 1.0f / static_cast<float>(N);

    const bool aligned =
        ((reinterpret_cast<uintptr_t>(input) |
          reinterpret_cast<uintptr_t>(output)) & 15u) == 0;

    if (aligned)
        launch<true>(input, output, N, inv_N, gamma, beta, eps);
    else
        launch<false>(input, output, N, inv_N, gamma, beta, eps);
}
