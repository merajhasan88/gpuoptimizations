#include <cuda_runtime.h>

__global__ void matrix_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int total = N * N;

    constexpr int THREADS = 256;

    int blocks =
        (total + THREADS - 1) / THREADS;

    matrix_add_kernel<<<blocks, THREADS>>>(
        A,
        B,
        C,
        total
    );

    cudaDeviceSynchronize();
}
