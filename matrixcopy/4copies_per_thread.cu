#include <cuda_runtime.h>

constexpr int THREADS = 256;
constexpr int FAST_N = 4096;
constexpr int VALUES_PER_THREAD = 4;

// Every launched thread has four valid elements.
__global__ void matrix_copy_4096_kernel(
    const float* __restrict__ A,
    float* __restrict__ B)
{
    const int i =
        blockIdx.x * (VALUES_PER_THREAD * THREADS)
        + threadIdx.x;

    const float x0 = __ldcg(A + i);
    const float x1 = __ldcg(A + i + THREADS);
    const float x2 = __ldcg(A + i + 2 * THREADS);
    const float x3 = __ldcg(A + i + 3 * THREADS);

    B[i]               = x0;
    B[i + THREADS]     = x1;
    B[i + 2 * THREADS] = x2;
    B[i + 3 * THREADS] = x3;
}

__global__ void matrix_copy_generic_kernel(
    const float* __restrict__ A,
    float* __restrict__ B,
    int total)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total)
    {
        B[i] = __ldcg(A + i);
    }
}

extern "C" void solve(
    const float* A,
    float* B,
    int N)
{
    if (N == FAST_N)
    {
        constexpr int BLOCKS =
            (FAST_N * FAST_N)
            / (VALUES_PER_THREAD * THREADS);

        matrix_copy_4096_kernel<<<BLOCKS, THREADS>>>(
            A, B
        );
    }
    else
    {
        const int total = N * N;
        const int blocks =
            (total + THREADS - 1) / THREADS;

        matrix_copy_generic_kernel<<<blocks, THREADS>>>(
            A, B, total
        );
    }

    cudaDeviceSynchronize();
}
