#include <cuda_runtime.h>
#include <math.h>

constexpr int FAST_N       = 1000000;
constexpr int FAST_M       = FAST_N / 2;
constexpr int FAST_THREADS = 160;

__global__ void geglu_fast_kernel(
    const float* __restrict__ input,
    float* __restrict__ output)
{
    const int i = blockIdx.x * FAST_THREADS + threadIdx.x;

    constexpr float INV_SQRT_2 = 0.7071067811865475244f;

    const float a = input[i];
    const float b = input[i + FAST_M];

    const float normal_cdf =
        0.5f + 0.5f * erff(b * INV_SQRT_2);

    output[i] = a * b * normal_cdf;
}

__global__ void geglu_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        constexpr float INV_SQRT_2 = 0.7071067811865475244f;

        const float a = input[i];
        const float b = input[i + M];

        const float normal_cdf =
            0.5f + 0.5f * erff(b * INV_SQRT_2);

        output[i] = a * b * normal_cdf;
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int N)
{
    if (N == FAST_N) {
        geglu_fast_kernel
            <<<FAST_M / FAST_THREADS, FAST_THREADS>>>(
                input,
                output
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
