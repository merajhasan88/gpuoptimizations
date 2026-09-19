#include <cuda_runtime.h>

#define THREADS 256

__global__ void clip_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    float lo,
    float hi)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < N)
    {
        const float x = input[i];

        // Clamp x into [lo, hi].
        //
        // x < lo  -> lo
        // x > hi  -> hi
        // otherwise x
        output[i] =
            (x < lo)
                ? lo
                : ((x > hi) ? hi : x);
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N,
    float lo,
    float hi)
{
    const int blocks =
        (N + THREADS - 1) /
        THREADS;

    clip_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N,
        lo,
        hi
    );

    // No cudaDeviceSynchronize();
}
