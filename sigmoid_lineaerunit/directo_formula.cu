#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < N)
    {
        const float x = input[i];

        output[i] =
            x / (1.0f + expf(-x));
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    const int blocks =
        (N + THREADS - 1) /
        THREADS;

    silu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N
    );

    cudaDeviceSynchronize();
}
