#include <cuda_runtime.h>
#include <math.h>

namespace {

// 256 threads produce a 32x32 output tile.
// A 3-pixel halo makes the shared input tile 38x38.
__global__ __launch_bounds__(256, 4)
void blur7x7(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H,
    int W)
{
    __shared__ float tile[38][38];
    __shared__ float weights[49];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 32 + tx;

    const int x0 = blockIdx.x * 32;
    const int y0 = blockIdx.y * 32;

    // Load the input tile cooperatively.
    // Positions outside the image become zero.
    for (int i = tid; i < 38 * 38; i += 256) {
        const int sy = i / 38;
        const int sx = i % 38;

        const int y = y0 + sy - 3;
        const int x = x0 + sx - 3;

        tile[sy][sx] =
            (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
             static_cast<unsigned>(x) < static_cast<unsigned>(W))
            ? input[y * W + x]
            : 0.0f;
    }

    if (tid < 49)
        weights[tid] = kernel[tid];

    __syncthreads();

    const int x = x0 + tx;
    const int y = y0 + 4 * ty;

    // This return is safe because the block barrier is already complete.
    if (x >= W || y >= H)
        return;

    // Keep three input rows in registers.
    // Loading one additional row supplies all four output accumulators.
    float r0[7], r1[7], r2[7];

    #pragma unroll
    for (int kx = 0; kx < 7; ++kx) {
        r0[kx] = tile[4 * ty    ][tx + kx];
        r1[kx] = tile[4 * ty + 1][tx + kx];
        r2[kx] = tile[4 * ty + 2][tx + kx];
    }

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    #pragma unroll
    for (int ky = 0; ky < 7; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
            const float v = tile[4 * ty + ky + 3][tx + kx];
            const float w = weights[ky * 7 + kx];

            s0 = fmaf(r0[kx], w, s0);
            s1 = fmaf(r1[kx], w, s1);
            s2 = fmaf(r2[kx], w, s2);
            s3 = fmaf(v,      w, s3);

            // Slide the register window down by one row.
            r0[kx] = r1[kx];
            r1[kx] = r2[kx];
            r2[kx] = v;
        }
    }

    output[y * W + x] = s0;

    if (y + 1 < H)
        output[(y + 1) * W + x] = s1;

    if (y + 2 < H)
        output[(y + 2) * W + x] = s2;

    if (y + 3 < H)
        output[(y + 3) * W + x] = s3;
}

// General path for other supported odd, possibly rectangular filters.
__global__ void blur_generic(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int H,
    int W,
    int KH,
    int KW)
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

        tile[i] =
            (static_cast<unsigned>(y) < static_cast<unsigned>(H) &&
             static_cast<unsigned>(x) < static_cast<unsigned>(W))
            ? input[y * W + x]
            : 0.0f;
    }

    for (int i = tid; i < KH * KW; i += 256)
        weights[i] = kernel[i];

    __syncthreads();

    const int x = x0 + tx;
    const int y = y0 + ty;

    if (x >= W || y >= H)
        return;

    float sum = 0.0f;

    for (int ky = 0; ky < KH; ++ky) {
        for (int kx = 0; kx < KW; ++kx) {
            sum = fmaf(
                tile[(ty + ky) * tile_w + tx + kx],
                weights[ky * KW + kx],
                sum);
        }
    }

    output[y * W + x] = sum;
}

} // namespace

// input, kernel and output are device pointers.
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols)
{
    if (kernel_rows == 7 && kernel_cols == 7) {
        const dim3 block(32, 8);
        const dim3 grid(
            (input_cols + 31) / 32,
            (input_rows + 31) / 32);

        blur7x7<<<grid, block>>>(
            input, kernel, output, input_rows, input_cols);
    } else {
        const dim3 block(16, 16);
        const dim3 grid(
            (input_cols + 15) / 16,
            (input_rows + 15) / 16);

        const int shared_floats =
            (16 + kernel_rows - 1) * (16 + kernel_cols - 1)
            + kernel_rows * kernel_cols;

        blur_generic<<<grid, block, shared_floats * sizeof(float)>>>(
            input, kernel, output,
            input_rows, input_cols,
            kernel_rows, kernel_cols);
    }
}
