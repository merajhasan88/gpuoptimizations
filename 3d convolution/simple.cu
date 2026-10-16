#include <cuda_runtime.h>
#include <math.h>

// One GPU thread computes one output voxel.
__global__ void conv3d_simple(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int output_depth,
    int output_rows,
    int output_cols,
    int kernel_depth,
    int kernel_rows,
    int kernel_cols)
{
    // x = column, y = row, z = depth slice.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Edge blocks may contain threads outside the output volume.
    if (x >= output_cols || y >= output_rows || z >= output_depth) {
        return;
    }

    float sum = 0.0f;

    // Move through every element of the 3D filter.
    // The filter is used as stored: do not reverse its axes.
    for (int kz = 0; kz < kernel_depth; ++kz) {
        for (int ky = 0; ky < kernel_rows; ++ky) {
            for (int kx = 0; kx < kernel_cols; ++kx) {

                // Flat index for input[z + kz][y + ky][x + kx].
                // Columns are contiguous, followed by rows, then slices.
                const int input_index =
                    ((z + kz) * input_rows + (y + ky)) * input_cols
                    + (x + kx);

                // Flat index for kernel[kz][ky][kx].
                const int kernel_index =
                    (kz * kernel_rows + ky) * kernel_cols + kx;

                // Multiply and add with a single rounding step.
                sum = fmaf(input[input_index], kernel[kernel_index], sum);
            }
        }
    }

    // Each thread writes its own output voxel.
    const int output_index = (z * output_rows + y) * output_cols + x;
    output[output_index] = sum;
}

// input, kernel, and output are device pointers.
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_depth,
    int input_rows,
    int input_cols,
    int kernel_depth,
    int kernel_rows,
    int kernel_cols)
{
    // Valid convolution: keep only positions where the whole filter fits.
    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows  = input_rows  - kernel_rows  + 1;
    const int output_cols  = input_cols  - kernel_cols  + 1;

    // 32 columns x 4 rows x 2 slices = 256 threads per block.
    // Adjacent threads along x read adjacent input columns.
    const dim3 threads(32, 4, 2);

    // Round up so the grid covers the entire output volume.
    const dim3 blocks(
        (output_cols  + threads.x - 1) / threads.x,
        (output_rows  + threads.y - 1) / threads.y,
        (output_depth + threads.z - 1) / threads.z);

    conv3d_simple<<<blocks, threads>>>(
        input, kernel, output,
        input_rows, input_cols,
        output_depth, output_rows, output_cols,
        kernel_depth, kernel_rows, kernel_cols);
}
