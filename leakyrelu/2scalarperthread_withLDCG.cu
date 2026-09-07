#include <cuda_runtime.h>

#define FAST_N       50000000
#define HALF_N       25000000
#define FAST_THREADS 160

#define GENERIC_THREADS 256


__global__ void leaky_relu_50m_x2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output)
{
    const int i =
        blockIdx.x * FAST_THREADS +
        threadIdx.x;

    const int j = i + HALF_N;


    // Two independent global loads.
    const float x0 = __ldcg(input + i);
    const float x1 = __ldcg(input + j);


    // Two independent Leaky-ReLU calculations.
    const float y0 =
        (x0 > 0.0f)
            ? x0
            : x0 * 0.01f;

    const float y1 =
        (x1 > 0.0f)
            ? x1
            : x1 * 0.01f;


    // Normal stores.
    output[i] = y0;
    output[j] = y1;
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
        const float x =
            __ldcg(input + i);

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
            HALF_N / FAST_THREADS;

        leaky_relu_50m_x2_kernel
            <<<BLOCKS, FAST_THREADS>>>(
                input,
                output
            );
    }
    else
    {
        const int blocks =
            (N + GENERIC_THREADS - 1)
            / GENERIC_THREADS;

        leaky_relu_generic_kernel
            <<<blocks, GENERIC_THREADS>>>(
                input,
                output,
                N
            );
    }

    cudaDeviceSynchronize();
}
