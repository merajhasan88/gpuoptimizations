#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
 // Find which output element this thread is responsible for.
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;
// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)

int output_size = input_size - kernel_size + 1;
 if (i < output_size) {

        float sum = 0.0f;

        // Slide the kernel across the input starting at input[i].
        //
        // For output[i]:
        //
        // input[i]     * kernel[0]
        // input[i + 1] * kernel[1]
        // input[i + 2] * kernel[2]
        // ...
        //
        // Then add all products together.
        for (int j = 0; j < kernel_size; ++j) {

            sum +=
                input[i + j] *
                kernel[j];
        }

        // Store this thread's final convolution result.
        output[i] = sum;
    }
    }
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
