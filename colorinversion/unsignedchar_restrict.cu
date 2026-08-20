#include <cuda_runtime.h>
//launch_bounds: “This kernel uses at most 256 threads per block, and please compile it so that resources ideally permit at least 4 such blocks to be resident simultaneously on one SM.”
__global__ void invert_kernel(
    unsigned char* __restrict__ image,
    int pixels)
//__restrict__ is a promise to the compiler not to do pointer aliasing.

//Aliasing means two different pointers might point at the same memory
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < pixels) {
        unsigned char* ptr = image + (p << 2);
//p << 2 means “shift p left by 2 bits,” which for nonnegative integers is equivalent to multiplying by 4: p * 4

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
