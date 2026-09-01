#include <cuda_runtime.h>

#define THREADS 256

__global__ void leaky_relu_kernel(
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
        // L2/global cached load
        const float x0 = __ldcg(input + i);

        const float y0 =
            (x0 > 0.0f)
                ? x0
                : x0 * 0.01f;

        // Normal store
        output[i] = y0;


        const int j = i + half;

        if (j < N)
        {
            const float x1 = __ldcg(input + j);

            const float y1 =
                (x1 > 0.0f)
                    ? x1
                    : x1 * 0.01f;

            output[j] = y1;
        }
    }
}


extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    const int half =
        (N + 1) >> 1;

    const int blocks =
        (half + THREADS - 1) /
        THREADS;

    leaky_relu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N,
        half
    );

    cudaDeviceSynchronize();
}
