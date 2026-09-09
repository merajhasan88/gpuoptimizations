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
    const int stride = blockDim.x;

    const int i =
        blockIdx.x * (4 * stride) + threadIdx.x;

    // Main path: all four elements exist.
    if (i + 3 * stride < N)
    {
        unsigned int x0 =
            static_cast<unsigned int>(input[i]);

        unsigned int x1 =
            static_cast<unsigned int>(input[i + stride]);

        unsigned int x2 =
            static_cast<unsigned int>(input[i + 2 * stride]);

        unsigned int x3 =
            static_cast<unsigned int>(input[i + 3 * stride]);

        #pragma unroll 2
        for (int r = 0; r < R; ++r)
        {
            x0 = fnv1a_hash(x0);
            x1 = fnv1a_hash(x1);
            x2 = fnv1a_hash(x2);
            x3 = fnv1a_hash(x3);
        }

        output[i]              = x0;
        output[i + stride]     = x1;
        output[i + 2 * stride] = x2;
        output[i + 3 * stride] = x3;
    }
    else
    {
        // Remaining elements in the final partial block.
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            const int j = i + k * stride;

            if (j < N)
            {
                unsigned int x =
                    static_cast<unsigned int>(input[j]);

                #pragma unroll 2
                for (int r = 0; r < R; ++r)
                    x = fnv1a_hash(x);

                output[j] = x;
            }
        }
    }
}

extern "C" void solve(
    const int* input,
    unsigned int* output,
    int N,
    int R)
{
    constexpr int THREADS = 256;

    const int blocks =
        (N + 4 * THREADS - 1) / (4 * THREADS);

    rainbow_table_kernel<<<blocks, THREADS>>>(
        input, output, N, R
    );

    cudaDeviceSynchronize();
}
