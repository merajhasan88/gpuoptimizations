#include <cuda_runtime.h>

__global__ void invert_kernel(
    unsigned char* __restrict__ image,
    int pixels)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < pixels) {
        unsigned char* ptr = image + (p << 2);

        ptr[0] = ~ptr[0];   // R
        ptr[1] = ~ptr[1];   // G
        ptr[2] = ~ptr[2];   // B
        // ptr[3] = alpha, unchanged
    }
}

extern "C" void solve(
    unsigned char* image,
    int width,
    int height)
{
    const int pixels = width * height;

    constexpr int THREADS = 256;

    invert_kernel<<<
        (pixels + THREADS - 1) / THREADS,
        THREADS
    >>>(
        image,
        pixels
    );

    cudaDeviceSynchronize();
}
