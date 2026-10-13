#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int K = 15;
constexpr int ROWS = 4;
constexpr int COLS = 2;

constexpr int OUT_W = BX * COLS;       // 64
constexpr int OUT_H = BY * ROWS;       // 16
constexpr int TILE_W = OUT_W + K - 1;  // 78
constexpr int TILE_H = OUT_H + K - 1;  // 30
constexpr int HALF_W = TILE_W / 2;    // 39
constexpr int CACHE_W = K + COLS - 1; // 16

// Each thread computes four rows by two columns.
// PHASE selects the oldest cached input row.
template<int PHASE>
__device__ __forceinline__
void filter_row4x2(
    const float tile[TILE_H][2][HALF_W],
    const float filter[K][K],
    int y,
    int tx,
    int kr,
    float (&cache)[ROWS - 1][CACHE_W],
    float (&sum)[ROWS][COLS])
{
    const int next_row = y + kr + ROWS - 1;

    // Reuse the previous right-hand input as the next left-hand input.
    float next_left = tile[next_row][0][tx];

    #pragma unroll
    for (int kc = 0; kc < K; ++kc) {
        const float w = filter[kr][kc];

        const float next_right =
            tile[next_row][(kc + 1) & 1][tx + ((kc + 1) >> 1)];

        #pragma unroll
        for (int r = 0; r < ROWS - 1; ++r) {
            constexpr int PERIOD = ROWS - 1;

            sum[r][0] = fmaf(
                cache[(PHASE + r) % PERIOD][kc],
                w,
                sum[r][0]);

            sum[r][1] = fmaf(
                cache[(PHASE + r) % PERIOD][kc + 1],
                w,
                sum[r][1]);
        }

        sum[ROWS - 1][0] =
            fmaf(next_left, w, sum[ROWS - 1][0]);

        sum[ROWS - 1][1] =
            fmaf(next_right, w, sum[ROWS - 1][1]);

        // The old cached value has had its final use this row.
        cache[PHASE][kc] = next_left;
        next_left = next_right;
    }

    cache[PHASE][CACHE_W - 1] = next_left;
}

__global__ __launch_bounds__(BX * BY, 3)
void conv2d_15x15_rect(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int output_rows,
    int output_cols)
{
    // Plane 0 stores even columns; plane 1 stores odd columns.
    // Scalar reads across a warp access consecutive shared words.
    __shared__ float tile[TILE_H][2][HALF_W];
    __shared__ float filter[K][K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    const int base_col = blockIdx.x * OUT_W;
    const int base_row = blockIdx.y * OUT_H;

    // Load and rearrange the input patch in shared memory.
    for (int y = ty; y < TILE_H; y += BY) {
        for (int h = tx; h < HALF_W; h += BX) {
            const int r = base_row + y;
            const int c = base_col + 2 * h;

            float even = 0.0f;
            float odd = 0.0f;

            if (r < input_rows) {
                if (c < input_cols)
                    even = input[r * input_cols + c];

                if (c + 1 < input_cols)
                    odd = input[r * input_cols + c + 1];
            }

            tile[y][0][h] = even;
            tile[y][1][h] = odd;
        }
    }

    for (int i = tid; i < K * K; i += BX * BY) {
        filter[i / K][i % K] = kernel[i];
    }

    __syncthreads();

    const int local_row = ty * ROWS;
    const int row = base_row + local_row;
    const int col = base_col + tx * COLS;

    // No barriers follow this return.
    if (row >= output_rows || col >= output_cols) return;

    // Three rows x sixteen values = 48 retained input values.
    float cache[ROWS - 1][CACHE_W];
    float sum[ROWS][COLS];

    #pragma unroll
    for (int r = 0; r < ROWS - 1; ++r) {
        #pragma unroll
        for (int c = 0; c < CACHE_W; ++c) {
            cache[r][c] =
                tile[local_row + r][c & 1][tx + (c >> 1)];
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sum[r][0] = 0.0f;
        sum[r][1] = 0.0f;
    }

    // Rotate cached rows while preserving each output's FMA order.
    #pragma unroll 1
    for (int kr = 0; kr < K; kr += 3) {
        filter_row4x2<0>(
            tile, filter, local_row, tx, kr,
            cache, sum);

        filter_row4x2<1>(
            tile, filter, local_row, tx, kr + 1,
            cache, sum);

        filter_row4x2<2>(
            tile, filter, local_row, tx, kr + 2,
            cache, sum);
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (row + r < output_rows) {
            const int dst = (row + r) * output_cols + col;

            output[dst] = sum[r][0];

            if (col + 1 < output_cols)
                output[dst + 1] = sum[r][1];
        }
    }
}

// General fallback for other filter sizes.
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

        conv2d_15x15_rect<<<blocks, threads>>>(
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
