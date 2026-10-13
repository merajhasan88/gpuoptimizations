#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int K = 15;
constexpr int ROWS = 8;
constexpr int CACHED_ROWS = ROWS - 1;

constexpr int OUT_W = BX;
constexpr int OUT_H = BY * ROWS;       // 32
constexpr int TILE_W = OUT_W + K - 1;  // 46
constexpr int TILE_H = OUT_H + K - 1;  // 46

// PHASE identifies the oldest cached input row.
// Unrolling makes all cache indices compile-time constants.
template<int PHASE>
__device__ __forceinline__
void filter_row8(
    const float tile[TILE_H][TILE_W],
    const float filter[K][K],
    int y,
    int x,
    int kr,
    float (&cache)[CACHED_ROWS][K],
    float (&sum)[ROWS])
{
    #pragma unroll
    for (int kc = 0; kc < K; ++kc) {
        const float w = filter[kr][kc];
        const float next =
            tile[y + kr + CACHED_ROWS][x + kc];

        // Seven outputs use input values already in registers.
        #pragma unroll
        for (int r = 0; r < CACHED_ROWS; ++r) {
            sum[r] = fmaf(
                cache[(PHASE + r) % CACHED_ROWS][kc],
                w,
                sum[r]);
        }

        // The eighth output uses the newly loaded input.
        sum[ROWS - 1] = fmaf(next, w, sum[ROWS - 1]);

        // Replace the oldest row.
        cache[PHASE][kc] = next;
    }
}

// Allow more registers per thread for seven cached input rows.
// This target guides register allocation, not actual residency.
__global__ __launch_bounds__(BX * BY, 3)
void conv2d_15x15_eight(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int output_rows,
    int output_cols)
{
    __shared__ float tile[TILE_H][TILE_W];
    __shared__ float filter[K][K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    const int base_col = blockIdx.x * OUT_W;
    const int base_row = blockIdx.y * OUT_H;

    // Same input/output tile dimensions as the 3.37 ms version.
    for (int y = ty; y < TILE_H; y += BY) {
        for (int x = tx; x < TILE_W; x += BX) {
            const int r = base_row + y;
            const int c = base_col + x;

            tile[y][x] = (r < input_rows && c < input_cols)
                      ? input[r * input_cols + c]
                      : 0.0f;
        }
    }

    // 128 threads load all 225 filter coefficients.
    for (int i = tid; i < K * K; i += BX * BY) {
        filter[i / K][i % K] = kernel[i];
    }

    __syncthreads();

    const int local_row = ty * ROWS;
    const int row = base_row + local_row;
    const int col = base_col + tx;

    // No barriers follow this return.
    if (row >= output_rows || col >= output_cols) return;

    // 105 cached input values and eight independent accumulators.
    float cache[CACHED_ROWS][K];
    float sum[ROWS];

    #pragma unroll
    for (int r = 0; r < CACHED_ROWS; ++r) {
        #pragma unroll
        for (int kc = 0; kc < K; ++kc) {
            cache[r][kc] = tile[local_row + r][tx + kc];
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sum[r] = 0.0f;
    }

    // Process filter rows 0..13 as two cycles of seven.
    // Rotate buffer roles instead of copying register arrays.
    #pragma unroll 1
    for (int kr = 0; kr < 14; kr += 7) {
        filter_row8<0>(
            tile, filter, local_row, tx, kr,
            cache, sum);

        filter_row8<1>(
            tile, filter, local_row, tx, kr + 1,
            cache, sum);

        filter_row8<2>(
            tile, filter, local_row, tx, kr + 2,
            cache, sum);

        filter_row8<3>(
            tile, filter, local_row, tx, kr + 3,
            cache, sum);

        filter_row8<4>(
            tile, filter, local_row, tx, kr + 4,
            cache, sum);

        filter_row8<5>(
            tile, filter, local_row, tx, kr + 5,
            cache, sum);

        filter_row8<6>(
            tile, filter, local_row, tx, kr + 6,
            cache, sum);
    }

    // Final filter row.
    // Every output retains the original row-major FMA order.
    filter_row8<0>(
        tile, filter, local_row, tx, 14,
        cache, sum);

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (row + r < output_rows) {
            output[(row + r) * output_cols + col] = sum[r];
        }
    }
}

// General fallback for every other filter size.
__global__ void conv2d_generic(
    const float* input,
    const float* kernel,
    float* output,
    int input_cols,
    int output_rows,
    int output_cols,
    int kernel_rows,
    int kernel_cols)
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
            (output_cols + OUT_W - 1) / OUT_W,
            (output_rows + OUT_H - 1) / OUT_H);

        conv2d_15x15_eight<<<blocks, threads>>>(
            input, kernel, output,
            input_rows, input_cols,
            output_rows, output_cols);
    } else {
        const dim3 threads(16, 16);

        const dim3 blocks(
            (output_cols + 15) / 16,
            (output_rows + 15) / 16);

        conv2d_generic<<<blocks, threads>>>(
            input, kernel, output,
            input_cols, output_rows, output_cols,
            kernel_rows, kernel_cols);
    }
}
