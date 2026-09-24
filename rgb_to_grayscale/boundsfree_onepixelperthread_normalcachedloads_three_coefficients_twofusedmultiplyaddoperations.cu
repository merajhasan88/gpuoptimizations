#include <cuda_runtime.h>

constexpr int FAST_WIDTH   = 2048;
constexpr int FAST_HEIGHT  = 2048;
constexpr int FAST_PIXELS  = FAST_WIDTH * FAST_HEIGHT;
constexpr int FAST_THREADS = 256;

__global__ void rgb_to_gray_fast_kernel(
    const float3* __restrict__ input,
    float* __restrict__ output)
{
    const int i = blockIdx.x * FAST_THREADS + threadIdx.x;

    const float3 rgb = input[i];

    output[i] = fmaf(
        0.299f,
        rgb.x,
        fmaf(0.587f, rgb.y, 0.114f * rgb.z)
    );
}

__global__ void rgb_to_gray_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int pixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < pixels) {
        const int base = 3 * i;

        const float r = input[base];
        const float g = input[base + 1];
        const float b = input[base + 2];

        output[i] = fmaf(
            0.299f,
            r,
            fmaf(0.587f, g, 0.114f * b)
        );
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int width,
    int height)
{
    const int pixels = width * height;

    if (pixels == FAST_PIXELS) {
        // 4,194,304 / 256 = 16,384 blocks exactly.
        rgb_to_gray_fast_kernel
            <<<FAST_PIXELS / FAST_THREADS, FAST_THREADS>>>(
                reinterpret_cast<const float3*>(input),
                output
            );
    } else {
        constexpr int THREADS = 256;
        const int blocks = (pixels + THREADS - 1) / THREADS;

        rgb_to_gray_generic_kernel<<<blocks, THREADS>>>(
            input,
            output,
            pixels
        );
    }
}
