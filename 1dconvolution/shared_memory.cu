#include <cuda_runtime.h>

#define THREADS 256
#define MAX_KERNEL_SIZE 2047

// ------------------------------------------------------------
// Constant memory for the convolution filter.
//
// Maximum required:
// 2047 floats * 4 bytes = 8188 bytes
//
// This comfortably fits in CUDA constant memory.
//
// Every warp evaluates the same j at the same time:
//
// thread 0 -> c_kernel[j]
// thread 1 -> c_kernel[j]
// ...
// thread 31 -> c_kernel[j]
//
// so the value can be broadcast efficiently.
// ------------------------------------------------------------

__constant__ float c_kernel[MAX_KERNEL_SIZE];


// ------------------------------------------------------------
// Optimized 1D convolution
//
// One thread still computes one output element.
//
// Main difference:
// Instead of every thread repeatedly loading its complete
// input window from global memory, the block cooperatively
// loads one larger input tile into shared memory.
//
// For 256 output threads and a kernel of 2047:
//
// shared tile size
// = 256 + 2047 - 1
// = 2302 floats
//
// = only 9208 bytes of shared memory.
// ------------------------------------------------------------

__global__ void convolution_1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size,
    int output_size)
{
    // Dynamic shared memory.
    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;

    // First output position handled by this block.
    const int block_start =
        blockIdx.x * blockDim.x;

    // This thread's output position.
    const int i =
        block_start + tx;


    // --------------------------------------------------------
    // STEP 1:
    // Cooperatively load input into shared memory.
    //
    // To calculate 256 neighboring outputs:
    //
    // output[block_start]
    // ...
    // output[block_start + 255]
    //
    // we need:
    //
    // input[block_start]
    // through
    // input[block_start + 255 + kernel_size - 1]
    //
    // So:
    //
    // shared_size = blockDim.x + kernel_size - 1
    //
    // For kernel_size = 2047:
    //
    // 256 + 2047 - 1 = 2302 floats
    // --------------------------------------------------------

    const int shared_size =
        blockDim.x + kernel_size - 1;

    // There are more shared elements than threads,
    // so each thread may load several elements.
    //
    // Example:
    //
    // thread 0:
    //     shared[0]
    //     shared[256]
    //     shared[512]
    //     ...
    //
    // thread 1:
    //     shared[1]
    //     shared[257]
    //     shared[513]
    //     ...
    //
    // Neighboring threads therefore perform neighboring
    // global-memory accesses -> coalesced.

    for (int s = tx;
         s < shared_size;
         s += blockDim.x)
    {
        const int global_index =
            block_start + s;

        if (global_index < input_size) {
            shared_input[s] =
                input[global_index];
        }
    }


    // --------------------------------------------------------
    // Every thread must wait until the entire input tile has
    // been loaded.
    // --------------------------------------------------------

    __syncthreads();


    // --------------------------------------------------------
    // STEP 2:
    // Each thread computes one output.
    // --------------------------------------------------------

    if (i < output_size) {

        // Multiple independent accumulators.
        //
        // Instead of:
        //
        // sum = sum + ...
        // sum = sum + ...
        // sum = sum + ...
        //
        // which creates one long dependency chain,
        // use four independent sums.
        //
        // This gives the GPU more instruction-level
        // parallelism.

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        int j = 0;


        // ----------------------------------------------------
        // Process four kernel elements at a time.
        //
        // For this thread:
        //
        // shared_input[tx + j]
        //
        // corresponds exactly to:
        //
        // input[i + j]
        //
        // because:
        //
        // i = block_start + tx
        // ----------------------------------------------------

        for (;
             j + 3 < kernel_size;
             j += 4)
        {
            sum0 +=
                shared_input[tx + j] *
                c_kernel[j];

            sum1 +=
                shared_input[tx + j + 1] *
                c_kernel[j + 1];

            sum2 +=
                shared_input[tx + j + 2] *
                c_kernel[j + 2];

            sum3 +=
                shared_input[tx + j + 3] *
                c_kernel[j + 3];
        }


        // Combine the four independent accumulator chains.
        float sum =
            (sum0 + sum1) +
            (sum2 + sum3);


        // ----------------------------------------------------
        // Handle 0-3 leftover kernel elements.
        //
        // kernel_size = 2047 gives:
        //
        // 2044 processed by loop
        // 3 remaining
        // ----------------------------------------------------

        for (; j < kernel_size; ++j) {
            sum +=
                shared_input[tx + j] *
                c_kernel[j];
        }


        // Final result.
        output[i] = sum;
    }
}


// ------------------------------------------------------------
// solve()
// ------------------------------------------------------------

extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size)
{
    const int output_size =
        input_size - kernel_size + 1;


    // --------------------------------------------------------
    // Copy the filter from GPU global memory into CUDA
    // constant memory.
    //
    // "kernel" is already a device pointer, so this is a
    // device-to-device copy.
    //
    // Maximum transfer:
    //
    // 2047 * 4 = only 8188 bytes.
    // --------------------------------------------------------

    cudaMemcpyToSymbol(
        c_kernel,
        kernel,
        kernel_size * sizeof(float),
        0,
        cudaMemcpyDeviceToDevice
    );


    // --------------------------------------------------------
    // Grid
    // --------------------------------------------------------

    const int blocks =
        (output_size + THREADS - 1) /
        THREADS;


    // --------------------------------------------------------
    // Shared-memory size.
    //
    // For benchmark:
    //
    // (256 + 2047 - 1) * 4
    // = 2302 * 4
    // = 9208 bytes
    // --------------------------------------------------------

    const int shared_bytes =
        (THREADS + kernel_size - 1) *
        sizeof(float);


    convolution_1d_kernel
        <<<blocks, THREADS, shared_bytes>>>(
            input,
            output,
            input_size,
            kernel_size,
            output_size
        );


    cudaDeviceSynchronize();
}
