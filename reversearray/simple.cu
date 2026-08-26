#include <cuda_runtime.h>

__global__ void reverse_array_kernel(
    float* input,
    int N)
{
    // Each thread gets one index from the first half.
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    // Only N / 2 swaps are required.
    int half = N / 2;

    if (i < half) {

        // Matching element from the opposite end.
        int j = N - 1 - i;

        // Swap input[i] and input[j].
        float temp = input[i];

        input[i] = input[j];
        input[j] = temp;
    }
}


extern "C" void solve(
    float* input,
    int N)
{
    // Number of pairs that need swapping.
    int half = N / 2;

    constexpr int THREADS = 256;

    int blocks =
        (half + THREADS - 1) /
        THREADS;

    reverse_array_kernel<<<blocks, THREADS>>>(
        input,
        N
    );

    cudaDeviceSynchronize();
}
