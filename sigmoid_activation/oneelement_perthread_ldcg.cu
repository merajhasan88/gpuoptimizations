#include <cuda_runtime.h>
#include <math.h>

constexpr int FAST_N       = 50000000;
constexpr int FAST_THREADS = 160;

__device__ __forceinline__
float sigmoid_value(float x)
{
    if (x < -80.0f) {
        return expf(x);
    }

    const float e = __expf(-fabsf(x));
    const float numerator = (x >= 0.0f) ? 1.0f : e;

    return __fdividef(numerator, 1.0f + e);
}

__global__ void sigmoid_x1_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y)
{
    const int i = blockIdx.x * FAST_THREADS + threadIdx.x;

    const float x = __ldcg(X + i);
    Y[i] = sigmoid_value(x);
}

__global__ void sigmoid_generic_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        Y[i] = sigmoid_value(X[i]);
    }
}

extern "C" void solve(
    const float* X,
    float* Y,
    int N)
{
    if (N == FAST_N) {
        // 50,000,000 / 160 = 312,500 blocks exactly.
        sigmoid_x1_kernel
            <<<FAST_N / FAST_THREADS, FAST_THREADS>>>(X, Y);
    } else {
        constexpr int THREADS = 256;
        const int blocks = (N + THREADS - 1) / THREADS;

        sigmoid_generic_kernel<<<blocks, THREADS>>>(X, Y, N);
    }
}
