#include <cuda_runtime.h>

__global__ void matrix_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int total)
{
    const int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < total) {
        const float a = __ldcs(A + i);
        const float b = __ldcs(B + i);

        C[i] = a + b;
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    const int total = N * N;

    constexpr int THREADS = 256;

    const int blocks =
        (total + THREADS - 1) / THREADS;

    matrix_add_kernel<<<blocks, THREADS>>>(
        A,
        B,
        C,
        total
    );

    cudaDeviceSynchronize();
}
