#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        constexpr float INV_SQRT_2 = 0.7071067811865475244f;

        const float a = input[i];
        const float b = input[i + M];

        // GELU(b) = 0.5 * b * (1 + erf(b / sqrt(2)))
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
    const int M = N / 2;

    constexpr int THREADS = 256;
    const int blocks = (M + THREADS - 1) / THREADS;

    geglu_kernel<<<blocks, THREADS>>>(
        input,
        output,
        M
    );
}
