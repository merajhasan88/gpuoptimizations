#include <cuda_runtime.h>

#define THREADS 256

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    // Find the array element handled by this thread.
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    // The last block may contain more threads than elements.
    if (i < N)
    {
        const float x = input[i];

        // ReLU:
        // negative -> 0
        // zero/positive -> unchanged
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}


extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    // Number of blocks required to cover all N elements.
    const int blocks =
        (N + THREADS - 1) /
        THREADS;

    relu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N
    );

    cudaDeviceSynchronize();
}
