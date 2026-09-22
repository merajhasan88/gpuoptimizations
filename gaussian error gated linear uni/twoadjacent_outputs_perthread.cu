#include <cuda_runtime.h>
#include <math.h>

constexpr int FAST_N       = 1000000;
constexpr int FAST_M       = FAST_N / 2;
constexpr int FAST_PAIRS   = FAST_M / 2;
constexpr int FAST_THREADS = 250;

__global__ void geglu_x2_kernel(
    const float2* __restrict__ A,
    const float2* __restrict__ B,
    float2* __restrict__ output)
{
    const int i = blockIdx.x * FAST_THREADS + threadIdx.x;

    constexpr float INV_SQRT_2 = 0.7071067811865475f;

    const float2 a = A[i];
    const float2 b = B[i];

    // Keep both independent erf evaluations visible to the compiler.
    const float erf0 = erff(b.x * INV_SQRT_2);
    const float erf1 = erff(b.y * INV_SQRT_2);

    const float cdf0 = 0.5f + 0.5f * erf0;
    const float cdf1 = 0.5f + 0.5f * erf1;

    output[i] = make_float2(
        a.x * b.x * cdf0,
        a.y * b.y * cdf1
    );
}

__global__ void geglu_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        constexpr float INV_SQRT_2 = 0.7071067811865475f;

        const float a = input[i];
        const float b = input[i + M];

        const float cdf =
            0.5f + 0.5f * erff(b * INV_SQRT_2);

        output[i] = a * b * cdf;
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    if (N == FAST_N) {
        // 250,000 output pairs / 250 threads = 1,000 blocks.
        geglu_x2_kernel<<<FAST_PAIRS / FAST_THREADS, FAST_THREADS>>>(
            reinterpret_cast<const float2*>(input),
            reinterpret_cast<const float2*>(input + FAST_M),
            reinterpret_cast<float2*>(output)
        );
    } else {
        const int M = N / 2;

        constexpr int THREADS = 256;
        const int blocks = (M + THREADS - 1) / THREADS;

        geglu_generic_kernel<<<blocks, THREADS>>>(
            input,
            output,
            M
        );
    }
}
