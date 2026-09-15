#include <cuda_runtime.h>
#include <math.h>

// Start with the standard exponential to establish correctness.
//
// 0 = expf:   standard CUDA exponential
// 1 = __expf: faster approximate CUDA exponential
#define USE_FAST_EXP 0

constexpr int THREADS = 256;


// Compute SiLU for one input value.
__device__ __forceinline__
float silu_value(float x)
{
    // Numerically stable formulation:
    //
    // Let e = exp(-abs(x)).
    //
    // If x >= 0:
    //     SiLU(x) = x / (1 + e)
    //
    // If x < 0:
    //     SiLU(x) = (x * e) / (1 + e)
    //
    // The exponential's argument is never positive,
    // avoiding overflow for large negative inputs.

#if USE_FAST_EXP
    const float e = __expf(-fabsf(x));
#else
    const float e = expf(-fabsf(x));
#endif

    const float numerator =
        (x >= 0.0f) ? x : x * e;

    return numerator / (1.0f + e);
}


// One GPU thread processes one input element.
__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    // Protect against excess threads in the last block.
    if (i < N)
    {
        const float x = input[i];

        output[i] = silu_value(x);
    }
}


// input and output are device pointers.
extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    if (N <= 0)
        return;

    const int blocks =
        (N + THREADS - 1) / THREADS;

    silu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        N
    );

    cudaDeviceSynchronize();
}
