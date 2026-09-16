#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

__global__ void swiglu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int half)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < half)
    {
        // First half: value going through SiLU.
        const float a = input[i];

        // Second half: gating value.
        const float b = input[i + half];

        // SiLU(a) = a / (1 + exp(-a))
        //
        // SWiGLU(a,b) = SiLU(a) * b
        //
        // Written as one expression to keep the hot path lean.
        output[i] =
            (a * b) /
            (1.0f + expf(-a));
    }
}


extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    const int half = N >> 1;

    const int blocks =
        (half + THREADS - 1) /
        THREADS;

    swiglu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        half
    );

    cudaDeviceSynchronize();
}
