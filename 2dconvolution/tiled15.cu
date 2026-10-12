#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int OUTPUTS_PER_THREAD = 4;

constexpr int FILTER_SIZE = 15;
constexpr int OUTPUT_TILE_W = BLOCK_X;
constexpr int OUTPUT_TILE_H = BLOCK_Y * OUTPUTS_PER_THREAD;

constexpr int INPUT_TILE_W =
    OUTPUT_TILE_W + FILTER_SIZE - 1;

constexpr int INPUT_TILE_H =
    OUTPUT_TILE_H + FILTER_SIZE - 1;

// Fast path for a 15x15 filter, on any valid input size.
__global__ void conv2d_15x15_tiled(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int output_rows,
    int output_cols)
{
    __shared__ float tile[INPUT_TILE_H][INPUT_TILE_W];
    __shared__ float filter[FILTER_SIZE][FILTER_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_X + tx;

    const int base_col = blockIdx.x * OUTPUT_TILE_W;
    const int base_row = blockIdx.y * OUTPUT_TILE_H;

    // Load the input patch, including the extra rows and columns
    // needed by outputs near the edge of this tile.
    for (int y = ty; y < INPUT_TILE_H; y += BLOCK_Y) {
        for (int x = tx; x < INPUT_TILE_W; x += BLOCK_X) {
            const int input_row = base_row + y;
            const int input_col = base_col + x;

            tile[y][x] =
                (input_row < input_rows && input_col < input_cols)
                ? input[input_row * input_cols + input_col]
                : 0.0f;
        }
    }

    // Load all 225 filter coefficients once per block.
    if (tid < FILTER_SIZE * FILTER_SIZE) {
        filter[tid / FILTER_SIZE][tid % FILTER_SIZE] = kernel[tid];
    }

    // All threads must finish loading before computation starts.
    // No thread returns before this barrier.
    __syncthreads();

    const int col = base_col + tx;
    const int row = base_row + ty;

    if (col >= output_cols || row >= output_rows) return;

    // Each thread computes four rows, spaced eight rows apart.
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    // Keep the row loop compact; unroll the 15 filter columns.
    #pragma unroll 1
    for (int kr = 0; kr < FILTER_SIZE; ++kr) {
        #pragma unroll
        for (int kc = 0; kc < FILTER_SIZE; ++kc) {
            const float w = filter[kr][kc];

            s0 = fmaf(
                tile[ty + kr][tx + kc],
                w, s0);

            s1 = fmaf(
                tile[ty + BLOCK_Y + kr][tx + kc],
                w, s1);

            s2 = fmaf(
                tile[ty + 2 * BLOCK_Y + kr][tx + kc],
                w, s2);

            s3 = fmaf(
                tile[ty + 3 * BLOCK_Y + kr][tx + kc],
                w, s3);
        }
    }

    // Store only valid output rows.
    output[row * output_cols + col] = s0;

    if (row + BLOCK_Y < output_rows) {
        output[(row + BLOCK_Y) * output_cols + col] = s1;
    }

    if (row + 2 * BLOCK_Y < output_rows) {
        output[(row + 2 * BLOCK_Y) * output_cols + col] = s2;
    }

    if (row + 3 * BLOCK_Y < output_rows) {
        output[(row + 3 * BLOCK_Y) * output_cols + col] = s3;
    }
}

// Original simple implementation for other filter dimensions.
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

    if (kernel_rows == FILTER_SIZE && kernel_cols == FILTER_SIZE) {
        const dim3 threads(BLOCK_X, BLOCK_Y);

        const dim3 blocks(
            (output_cols + OUTPUT_TILE_W - 1) / OUTPUT_TILE_W,
            (output_rows + OUTPUT_TILE_H - 1) / OUTPUT_TILE_H
        );

        conv2d_15x15_tiled<<<blocks, threads>>>(
            input,
            kernel,
            output,
            input_rows,
            input_cols,
            output_rows,
            output_cols
        );
    } else {
        const dim3 threads(16, 16);

        const dim3 blocks(
            (output_cols + threads.x - 1) / threads.x,
            (output_rows + threads.y - 1) / threads.y
        );

        conv2d_generic<<<blocks, threads>>>(
            input,
            kernel,
            output,
            input_cols,
            output_rows,
            output_cols,
            kernel_rows,
            kernel_cols
        );
    }
}
