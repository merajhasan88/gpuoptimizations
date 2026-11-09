#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

// 128 threads produce 32x32 outputs: eight vertical pixels per thread.
// VEC selects aligned float4 tile loading when the input layout permits it.
template<bool VEC>
__global__ __launch_bounds__(128, 4)
void blur7x7(const float* __restrict__ input,
             const float* __restrict__ kernel,
             float* __restrict__ output, int H, int W)
{
    constexpr int PITCH = VEC ? 40 : 38;
    constexpr int SHIFT = VEC ? 1 : 0;
    __shared__ __align__(16) float tile[38][PITCH];
    __shared__ float weights[49];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const int x0 = blockIdx.x * 32;
    const int y0 = blockIdx.y * 32;

    if constexpr (VEC) {
        // Ten float4 vectors per row; extra padding keeps loads aligned.
        for (int i = tid; i < 38 * 10; i += 128) {
            const int sy = i / 10;
            const int sx = 4 * (i % 10);
            const int y = y0 + sy - 3;
            const int x = x0 + sx - 4;
            float4 v = make_float4(0, 0, 0, 0);
            if (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
                x >= 0 && x + 3 < W)
                v = *reinterpret_cast<const float4*>(input + y * W + x);
            *reinterpret_cast<float4*>(&tile[sy][sx]) = v;
        }
    } else {
        // Supports arbitrary widths and input alignment.
        for (int i = tid; i < 38 * 38; i += 128) {
            const int sy = i / 38;
            const int sx = i % 38;
            const int y = y0 + sy - 3;
            const int x = x0 + sx - 3;
            tile[sy][sx] =
                (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
                 static_cast<unsigned>(x) < static_cast<unsigned>(W))
                ? input[y * W + x] : 0.0f;
        }
    }
    if (tid < 49) weights[tid] = kernel[tid];
    __syncthreads();

    const int x = x0 + tx;
    const int y = y0 + 8 * ty;
    if (x >= W || y >= H) return;

    // Seven input rows remain in registers. Each filter row loads one
    // additional row, reusing the window across eight output accumulators.
    float rows[7][7];
    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        #pragma unroll
        for (int kx = 0; kx < 7; ++kx)
            rows[r][kx] = tile[8 * ty + r][tx + SHIFT + kx];
    }

    float sums[8] = {};
    #pragma unroll
    for (int ky = 0; ky < 7; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
            const float v = tile[8 * ty + ky + 7][tx + SHIFT + kx];
            const float w = weights[ky * 7 + kx];

            #pragma unroll
            for (int r = 0; r < 7; ++r)
                sums[r] = fmaf(rows[r][kx], w, sums[r]);
            sums[7] = fmaf(v, w, sums[7]);

            // Fully unrolled: these arrays become scalar registers.
            #pragma unroll
            for (int r = 0; r < 6; ++r)
                rows[r][kx] = rows[r + 1][kx];
            rows[6][kx] = v;
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r)
        if (y + r < H)
            output[(y + r) * W + x] = sums[r];
}

// General path for the other supported odd, possibly rectangular filters.
__global__ void blur_generic(const float* __restrict__ input,
                             const float* __restrict__ kernel,
                             float* __restrict__ output,
                             int H, int W, int KH, int KW)
{
    extern __shared__ float shared[];
    const int tile_w = 16 + KW - 1;
    const int tile_h = 16 + KH - 1;
    const int tile_size = tile_w * tile_h;
    float* tile = shared;
    float* weights = shared + tile_size;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int x0 = blockIdx.x * 16;
    const int y0 = blockIdx.y * 16;

    for (int i = tid; i < tile_size; i += 256) {
        const int y = y0 + i / tile_w - KH / 2;
        const int x = x0 + i % tile_w - KW / 2;
        tile[i] = (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
                   static_cast<unsigned>(x) < static_cast<unsigned>(W))
                ? input[y * W + x] : 0.0f;
    }
    for (int i = tid; i < KH * KW; i += 256)
        weights[i] = kernel[i];
    __syncthreads();

    const int x = x0 + tx;
    const int y = y0 + ty;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = 0; ky < KH; ++ky)
        for (int kx = 0; kx < KW; ++kx)
            sum = fmaf(tile[(ty + ky) * tile_w + tx + kx],
                       weights[ky * KW + kx], sum);
    output[y * W + x] = sum;
}
} // namespace

// input, kernel and output are device pointers.
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols)
{
    if (kernel_rows == 7 && kernel_cols == 7) {
        const dim3 block(32, 4);
        const dim3 grid((input_cols + 31) / 32,
                        (input_rows + 31) / 32);

        const bool aligned =
            (reinterpret_cast<uintptr_t>(input) & 15u) == 0 &&
            (input_cols & 3) == 0;

        if (aligned)
            blur7x7<true><<<grid, block>>>(
                input, kernel, output, input_rows, input_cols);
        else
            blur7x7<false><<<grid, block>>>(
                input, kernel, output, input_rows, input_cols);
    } else {
        const dim3 block(16, 16);
        const dim3 grid((input_cols + 15) / 16,
                        (input_rows + 15) / 16);

        const int shared_floats =
            (16 + kernel_rows - 1) * (16 + kernel_cols - 1)
            + kernel_rows * kernel_cols;

        blur_generic<<<grid, block, shared_floats * sizeof(float)>>>(
            input, kernel, output,
            input_rows, input_cols, kernel_rows, kernel_cols);
    }
}
