#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;
constexpr int ELEMENTS_PER_BLOCK = THREADS * 16;

// Every used entry is overwritten on each invocation.
__device__ float partial_sums[MAX_BLOCKS];

template<typename T>
__device__ __forceinline__
T warp_sum(T value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffffu, value, offset);
    }
    return value;
}

// Only thread zero receives the complete block sum.
template<typename T>
__device__ __forceinline__
T block_sum(T value, T* shared)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_sum(value);

    if (lane == 0) shared[warp] = value;
    __syncthreads();

    if (warp == 0) {
        value = lane < THREADS / 32
            ? shared[lane] : T(0);

        value = warp_sum(value);
    }

    return value;
}

__global__ void dot_partials(
    const float* A,
    const float* B,
    float* output,
    int N)
{
    __shared__ float warp_totals[THREADS / 32];

    const int tid =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vectors = N / 4;

    // Four independent accumulation chains.
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    const bool aligned =
        ((reinterpret_cast<uintptr_t>(A) |
          reinterpret_cast<uintptr_t>(B)) & 15) == 0;

    if (aligned) {
        const float4* a4 =
            reinterpret_cast<const float4*>(A);
        const float4* b4 =
            reinterpret_cast<const float4*>(B);

        for (int i = tid; i < vectors; i += stride) {
            const float4 a = a4[i];
            const float4 b = b4[i];

            s0 = fmaf(a.x, b.x, s0);
            s1 = fmaf(a.y, b.y, s1);
            s2 = fmaf(a.z, b.z, s2);
            s3 = fmaf(a.w, b.w, s3);
        }
    } else {
        // Safe fallback for pointers without float4 alignment.
        for (int i = tid; i < vectors; i += stride) {
            const int j = 4 * i;

            s0 = fmaf(A[j],     B[j],     s0);
            s1 = fmaf(A[j + 1], B[j + 1], s1);
            s2 = fmaf(A[j + 2], B[j + 2], s2);
            s3 = fmaf(A[j + 3], B[j + 3], s3);
        }
    }

    // Handle the final 0–3 elements exactly once.
    const int tail = 4 * vectors + tid;
    if (tail < N) {
        s0 = fmaf(A[tail], B[tail], s0);
    }

    const float sum = block_sum(
        (s0 + s1) + (s2 + s3), warp_totals);

    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            // Small inputs need only this launch.
            output[0] = sum;
        } else {
            partial_sums[blockIdx.x] = sum;
        }
    }
}

__global__ void dot_finish(float* output, int count)
{
    __shared__ double warp_totals[THREADS / 32];

    double sum = 0.0;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        sum += double(partial_sums[i]);
    }

    // Only this tiny final reduction uses FP64.
    sum = block_sum(sum, warp_totals);

    if (threadIdx.x == 0) {
        output[0] = float(sum);
    }
}

} // namespace

// A, B and output are device pointers.
extern "C" void solve(
    const float* A,
    const float* B,
    float* output,
    int N)
{
    int blocks =
        (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;

    dot_partials<<<blocks, THREADS>>>(A, B, output, N);

    if (blocks > 1) {
        dot_finish<<<1, THREADS>>>(output, blocks);
    }
}
