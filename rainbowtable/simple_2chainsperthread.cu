#include <cuda_runtime.h>

__device__ __forceinline__
unsigned int fnv1a_hash(unsigned int x)
{
    unsigned int h = 2166136261u;

    h = (h ^ ( x        & 255u)) * 16777619u;
    h = (h ^ ((x >>  8) & 255u)) * 16777619u;
    h = (h ^ ((x >> 16) & 255u)) * 16777619u;
    h = (h ^ ( x >> 24       )) * 16777619u;

    return h;
}

__global__ void rainbow_table_kernel(
    const int* __restrict__ input,
    unsigned int* __restrict__ output,
    int N,
    int R)
{
    // Two coalesced groups per block.
    const int i =
        blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    const int j = i + blockDim.x;

    if (i >= N)
        return;

    unsigned int x = static_cast<unsigned int>(input[i]);

    unsigned int y = (j < N)
        ? static_cast<unsigned int>(input[j])
        : 0u;

    // Independent chains; intermediate results stay in registers.
    #pragma unroll 2
    for (int r = 0; r < R; ++r)
    {
        x = fnv1a_hash(x);
        y = fnv1a_hash(y);
    }

    output[i] = x;

    if (j < N)
        output[j] = y;
}

extern "C" void solve(
    const int* input,
    unsigned int* output,
    int N,
    int R)
{
    constexpr int THREADS = 256;

    const int blocks =
        (N + 2 * THREADS - 1) / (2 * THREADS);

    rainbow_table_kernel<<<blocks, THREADS>>>(
        input, output, N, R
    );

    cudaDeviceSynchronize();
}
