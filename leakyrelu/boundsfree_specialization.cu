#include <cuda_runtime.h>

#define FAST_N 50000000
#define FAST_THREADS 160
#define GENERIC_THREADS 256

__global__ void leaky_relu_50m_kernel(
    const float* __restrict__ input,
    float* __restrict__ output)
{
    const int i =
        blockIdx.x * FAST_THREADS +
        threadIdx.x;

    const float x = __ldcg(input + i);

    output[i] =
        (x > 0.0f)
            ? x
            : x * 0.01f;
}


__global__ void leaky_relu_generic_kernel(
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
    if (N == FAST_N)
    {
        constexpr int BLOCKS =
            FAST_N / FAST_THREADS;

        leaky_relu_50m_kernel<<<BLOCKS, FAST_THREADS>>>(
            input,
            output
        );
    }
    else
    {
        const int blocks =
            (N + GENERIC_THREADS - 1) /
            GENERIC_THREADS;

        leaky_relu_generic_kernel
            <<<blocks, GENERIC_THREADS>>>(
                input,
                output,
                N
            );
    }

    cudaDeviceSynchronize();
}
