#include <cuda_runtime.h>

constexpr int FAST_N       = 25000000;
constexpr int FAST_THREADS = 160;

__global__ void interleave_x1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float2* __restrict__ output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const float a = A[i];
    const float b = B[i];

    output[i] = make_float2(a, b);
}

__global__ void interleave_generic_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float2* __restrict__ output,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = make_float2(A[i], B[i]);
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* output,
    int N)
{
    if (N == FAST_N) {
        // 25,000,000 / 160 = 156,250 exactly.
        interleave_x1_kernel<<<FAST_N / FAST_THREADS, FAST_THREADS>>>(
            A,
            B,
            reinterpret_cast<float2*>(output)
        );
    } else {
        constexpr int THREADS = 256;
        const int blocks = (N + THREADS - 1) / THREADS;

        interleave_generic_kernel<<<blocks, THREADS>>>(
            A,
            B,
            reinterpret_cast<float2*>(output),
            N
        );
    }
}
