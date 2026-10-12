#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BX = 32;
constexpr int BY = 8;
constexpr int ROWS_PER_THREAD = 4;
constexpr int K = 15;
constexpr int TILE_W = BX + K - 1;                     // 46
constexpr int TILE_H = BY * ROWS_PER_THREAD + K - 1;   // 46

// Process one filter row for four adjacent output rows.
// a, b, c hold three input rows in registers.
// Load one new row and recycle the oldest register buffer.
__device__ __forceinline__
void filter_row(
    const float tile[TILE_H][TILE_W],
    const float filter[K][K],
    int y, int x, int kr,
    float (&a)[K], float (&b)[K], float (&c)[K],
    float& s0, float& s1, float& s2, float& s3)
{
    #pragma unroll
    for (int kc = 0; kc < K; ++kc) {
        const float w = filter[kr][kc];
        const float next = tile[y + kr + 3][x + kc];

        s0 = fmaf(a[kc], w, s0);
        s1 = fmaf(b[kc], w, s1);
        s2 = fmaf(c[kc], w, s2);
        s3 = fmaf(next,  w, s3);

        // This row will be reused by the next three filter rows.
        a[kc] = next;
    }
}

// The second launch-bounds argument guides register allocation.
// It does not guarantee three resident blocks.
__global__ __launch_bounds__(BX * BY, 3)
void conv2d_15x15_registers(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows, int input_cols,
    int output_rows, int output_cols)
{
    __shared__ float tile[TILE_H][TILE_W];
    __shared__ float filter[K][K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    const int base_col = blockIdx.x * BX;
    const int base_row = blockIdx.y * (BY * ROWS_PER_THREAD);

    // Each block stages a 46x46 patch for a 32x32 output tile.
    for (int y = ty; y < TILE_H; y += BY) {
        for (int x = tx; x < TILE_W; x += BX) {
            const int r = base_row + y;
            const int c = base_col + x;
            tile[y][x] = (r < input_rows && c < input_cols)
                      ? input[r * input_cols + c] : 0.0f;
        }
    }

    if (tid < K * K) {
        filter[tid / K][tid % K] = kernel[tid];
    }

    __syncthreads();

    // Unlike the previous kernel, each thread owns adjacent rows.
    const int local_row = ty * ROWS_PER_THREAD;
    const int row = base_row + local_row;
    const int col = base_col + tx;

    // Returning here is safe: there are no later block barriers.
    if (row >= output_rows || col >= output_cols) return;

    float r0[K], r1[K], r2[K];

    // Constant indices from unrolling let the compiler use registers.
    #pragma unroll
    for (int kc = 0; kc < K; ++kc) {
        r0[kc] = tile[local_row    ][tx + kc];
        r1[kc] = tile[local_row + 1][tx + kc];
        r2[kc] = tile[local_row + 2][tx + kc];
    }

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    // Rotate buffer roles instead of copying three register arrays.
    // Every output still accumulates in filter-row, filter-column order.
    #pragma unroll 1
    for (int kr = 0; kr < K; kr += 3) {
        filter_row(tile, filter, local_row, tx, kr,
                   r0, r1, r2, s0, s1, s2, s3);
        filter_row(tile, filter, local_row, tx, kr + 1,
                   r1, r2, r0, s0, s1, s2, s3);
        filter_row(tile, filter, local_row, tx, kr + 2,
                   r2, r0, r1, s0, s1, s2, s3);
    }

    output[row * output_cols + col] = s0;
    if (row + 1 < output_rows)
        output[(row + 1) * output_cols + col] = s1;
    if (row + 2 < output_rows)
        output[(row + 2) * output_cols + col] = s2;
    if (row + 3 < output_rows)
        output[(row + 3) * output_cols + col] = s3;
}

// General fallback for all other filter sizes.
__global__ void conv2d_generic(
    const float* input, const float* kernel, float* output,
    int input_cols, int output_rows, int output_cols,
    int kernel_rows, int kernel_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= output_rows || col >= output_cols) return;

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        for (int kc = 0; kc < kernel_cols; ++kc) {
            sum += input[(row + kr) * input_cols + col + kc] *
                   kernel[kr * kernel_cols + kc];
        }
    }
    output[row * output_cols + col] = sum;
}

} // namespace

extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols)
{
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;

    if (kernel_rows == K && kernel_cols == K) {
        const dim3 threads(BX, BY);
        const dim3 blocks(
            (output_cols + BX - 1) / BX,
            (output_rows + BY * ROWS_PER_THREAD - 1) /
                (BY * ROWS_PER_THREAD));

        conv2d_15x15_registers<<<blocks, threads>>>(
            input, kernel, output, input_rows, input_cols,
            output_rows, output_cols);
    } else {
        const dim3 threads(16, 16);
        const dim3 blocks(
            (output_cols + 15) / 16,
            (output_rows + 15) / 16);

        conv2d_generic<<<blocks, threads>>>(
            input, kernel, output, input_cols,
            output_rows, output_cols, kernel_rows, kernel_cols);
    }
}
