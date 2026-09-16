#include <cuda_runtime.h>

#define THREADS 256

// Turing hardware approximate 2^x.
__device__ __forceinline__
float fast_exp2(float x)
{
    float y;

    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(y)
        : "f"(x)
    );

    return y;
}


// Turing hardware approximate 1/x.
__device__ __forceinline__
float fast_rcp(float x)
{
    float y;

    asm volatile(
        "rcp.approx.ftz.f32 %0, %1;"
        : "=f"(y)
        : "f"(x)
    );

    return y;
}


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
        const float a = input[i];
        const float b = input[i + half];

        // exp(-a) = 2^(-a * log2(e))
        constexpr float LOG2E =
            1.4426950408889634f;

        const float e =
            fast_exp2(-a * LOG2E);

        // sigmoid(a) = 1 / (1 + exp(-a))
        const float sigmoid =
            fast_rcp(1.0f + e);

        // SWiGLU = SiLU(a) * b
        //         = a * sigmoid(a) * b
        output[i] =
            (a * b) * sigmoid;
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
