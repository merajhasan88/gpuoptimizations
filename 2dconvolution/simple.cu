#include <cuda_runtime.h>

// One GPU thread computes one output element.
__global__ void conv2d_simple_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int input_cols,
    int output_rows,
    int output_cols,
    int kernel_rows,
    int kernel_cols)
{
    // x selects the output column; y selects the output row.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ignore extra threads outside the output matrix.
    if (row >= output_rows || col >= output_cols) return;

    float sum = 0.0f;

    // Place the filter's top-left corner at input[row][col].
    // Multiply each covered input element by its filter coefficient.
    for (int kr = 0; kr < kernel_rows; ++kr) {
        for (int kc = 0; kc < kernel_cols; ++kc) {
            // Convert 2D coordinates to row-major array indices.
            const int input_index =
                (row + kr) * input_cols + (col + kc);

            const int kernel_index =
                kr * kernel_cols + kc;

            // Use the filter coefficients in their original order.
            sum += input[input_index] * kernel[kernel_index];
        }
    }

    // This thread is the only writer of this output element.
    output[row * output_cols + col] = sum;
}

// input, kernel, and output already point to GPU memory.
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols)
{
    // "Valid" keeps only positions where the filter fully fits.
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;

    // Each block contains 16 x 16 = 256 threads.
    const dim3 threads(16, 16);

    // Round up so every output element has a thread.
    const dim3 blocks(
        (output_cols + threads.x - 1) / threads.x,
        (output_rows + threads.y - 1) / threads.y
    );

    conv2d_simple_kernel<<<blocks, threads>>>(
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
