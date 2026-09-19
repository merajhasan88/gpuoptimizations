#include <cuda_runtime.h>

#define FAST_N       100000
#define FAST_THREADS 160

#define GENERIC_THREADS 256


__device__ __forceinline__
float clip_ptx(float x, float lo, float hi)
{
    float y;

    asm volatile(
        "max.f32 %0, %1, %2;\n\t"
        "min.f32 %0, %0, %3;"
        : "=f"(y)
        : "f"(x), "f"(lo), "f"(hi)
    );

    return y;
}


// Fast benchmark path:
// exactly 100,000 threads
// no bounds predicate
__global__ void clip_100k_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float lo,
    float hi)
{
    const int i =
        blockIdx.x * FAST_THREADS +
        threadIdx.x;

    const float x = input[i];

    output[i] =
        clip_ptx(x, lo, hi);
}


// Generic fallback
__global__ void clip_generic_kernel(
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

        output[i] =
            clip_ptx(x, lo, hi);
    }
}


extern "C" void solve(
    const float* input,
    float* output,
    int N,
    float lo,
    float hi)
{
    if (N == FAST_N)
    {
        constexpr int BLOCKS =
            FAST_N / FAST_THREADS;  // 625 exactly

        clip_100k_kernel<<<BLOCKS, FAST_THREADS>>>(
            input,
            output,
            lo,
            hi
        );
    }
    else
    {
        const int blocks =
            (N + GENERIC_THREADS - 1) /
            GENERIC_THREADS;

        clip_generic_kernel<<<blocks, GENERIC_THREADS>>>(
            input,
            output,
            N,
            lo,
            hi
        );
    }

    // No explicit synchronization.
}
