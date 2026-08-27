#include <cuda_runtime.h>

#define THREADS 256

__global__ void relu_kernel_2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int half)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < half)
    {
        // First contiguous half
        const float x0 = input[i];
        output[i] = (x0 > 0.0f) ? x0 : 0.0f;

        // Second contiguous half
        const int j = i + half;

        if (j < N)
        {
            const float x1 = input[j];
            output[j] = (x1 > 0.0f) ? x1 : 0.0f;
        }
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    // Ceiling(N / 2)
    const int half = (N + 1) >> 1;

    const int blocks =
        (half + THREADS - 1) /
        THREADS;

    relu_kernel_2<<<blocks, THREADS>>>(
        input,
        output,
        N,
        half
    );

    cudaDeviceSynchronize();
}
