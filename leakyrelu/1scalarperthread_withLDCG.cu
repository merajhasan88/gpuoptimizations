#include <cuda_runtime.h>

#define THREADS 256

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < N)
    {
        const float x = __ldcg(input + i);

        output[i] =
            (x > 0.0f)
                ? x
                : x * 0.01f;
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

    leaky_relu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N
    );

    cudaDeviceSynchronize();
}
