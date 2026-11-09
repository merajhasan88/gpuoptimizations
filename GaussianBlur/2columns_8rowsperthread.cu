#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

// A 32x2 block produces a 64x16 output tile.
// Each thread computes two adjacent columns across eight rows.
template<bool VECTOR_STORE>
__global__ __launch_bounds__(64, 8)
void blur7x7_paired(const float* __restrict__ input,
                    const float* __restrict__ kernel,
                    float* __restrict__ output, int H, int W)
{
    // The halo makes the input tile 70x22. Pairing values gives aligned
    // shared-memory float2 loads without requiring aligned global input.
    __shared__ float2 tile[22][35];
    __shared__ float weights[49];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const int x0 = blockIdx.x * 64;
    const int y0 = blockIdx.y * 16;

    for (int i = tid; i < 22 * 35; i += 64) {
        const int sy = i / 35;
        const int sx = i % 35;
        const int y = y0 + sy - 3;
        const int x = x0 + 2 * sx - 3;

        float2 v = make_float2(0.0f, 0.0f);
        if (static_cast<unsigned>(y) < static_cast<unsigned>(H)) {
            if (static_cast<unsigned>(x) < static_cast<unsigned>(W))
                v.x = input[y * W + x];
            if (static_cast<unsigned>(x + 1) < static_cast<unsigned>(W))
                v.y = input[y * W + x + 1];
        }
        tile[sy][sx] = v;
    }

    if (tid < 49) weights[tid] = kernel[tid];
    __syncthreads();

    const int x = x0 + 2 * tx;
    const int y = y0 + 8 * ty;
    if (x >= W || y >= H) return;

    // Eight input columns cover both seven-column filter windows.
    // Seven rows persist; the eighth row is replaced at each ky step.
    float rows[8][8];

    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 v = tile[8 * ty + r][tx + p];
            rows[r][2 * p] = v.x;
            rows[r][2 * p + 1] = v.y;
        }
    }

    float left[8] = {};
    float right[8] = {};

    #pragma unroll
    for (int ky = 0; ky < 7; ++ky) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 v = tile[8 * ty + ky + 7][tx + p];
            rows[7][2 * p] = v.x;
            rows[7][2 * p + 1] = v.y;
        }

        // Preserve ky-then-kx FMA order for every output pixel.
        #pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
            const float w = weights[ky * 7 + kx];

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                left[r]  = fmaf(rows[r][kx],     w, left[r]);
                right[r] = fmaf(rows[r][kx + 1], w, right[r]);
            }
        }

        // Fully unrolled register-window rotation.
        #pragma unroll
        for (int r = 0; r < 7; ++r) {
            #pragma unroll
            for (int c = 0; c < 8; ++c)
                rows[r][c] = rows[r + 1][c];
        }
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (y + r < H) {
            const int i = (y + r) * W + x;

            if constexpr (VECTOR_STORE) {
                // Even width and aligned output are guaranteed by solve.
                *reinterpret_cast<float2*>(output + i) =
                    make_float2(left[r], right[r]);
            } else {
                output[i] = left[r];
                if (x + 1 < W) output[i + 1] = right[r];
            }
        }
    }
}

// Fallback for every other supported odd, possibly rectangular filter.
__global__ void blur_generic(const float* __restrict__ input,
                             const float* __restrict__ kernel,
                             float* __restrict__ output,
                             int H, int W, int KH, int KW)
{
    extern __shared__ float shared[];
    const int tw = 16 + KW - 1;
    const int th = 16 + KH - 1;
    float* tile = shared;
    float* weights = shared + tw * th;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int x0 = blockIdx.x * 16, y0 = blockIdx.y * 16;

    for (int i = tid; i < tw * th; i += 256) {
        const int y = y0 + i / tw - KH / 2;
        const int x = x0 + i % tw - KW / 2;

        tile[i] =
            (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
             static_cast<unsigned>(x) < static_cast<unsigned>(W))
            ? input[y * W + x] : 0.0f;
    }

    for (int i = tid; i < KH * KW; i += 256)
        weights[i] = kernel[i];
    __syncthreads();

    const int x = x0 + tx, y = y0 + ty;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = 0; ky < KH; ++ky)
        for (int kx = 0; kx < KW; ++kx)
            sum = fmaf(tile[(ty + ky) * tw + tx + kx],
                       weights[ky * KW + kx], sum);

    output[y * W + x] = sum;
}

} // namespace

// input, kernel and output are device pointers.
extern "C" void solve(const float* input,
                      const float* kernel,
                      float* output,
                      int input_rows,
                      int input_cols,
                      int kernel_rows,
                      int kernel_cols)
{
    if (kernel_rows == 7 && kernel_cols == 7) {
        const dim3 block(32, 2);
        const dim3 grid((input_cols + 63) / 64,
                        (input_rows + 15) / 16);

        const bool vector_store =
            (input_cols & 1) == 0 &&
            (reinterpret_cast<uintptr_t>(output) & 7u) == 0;

        if (vector_store)
            blur7x7_paired<true><<<grid, block>>>(
                input, kernel, output, input_rows, input_cols);
        else
            blur7x7_paired<false><<<grid, block>>>(
                input, kernel, output, input_rows, input_cols);
    } else {
        const dim3 block(16, 16);
        const dim3 grid((input_cols + 15) / 16,
                        (input_rows + 15) / 16);

        const int bytes =
            ((16 + kernel_rows - 1) * (16 + kernel_cols - 1)
             + kernel_rows * kernel_cols) * sizeof(float);

        blur_generic<<<grid, block, bytes>>>(
            input, kernel, output,
            input_rows, input_cols, kernel_rows, kernel_cols);
    }
}
