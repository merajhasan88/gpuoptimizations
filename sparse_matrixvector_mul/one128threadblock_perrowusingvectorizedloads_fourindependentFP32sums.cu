#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);

    return value;
}

// One block computes one output row.
template<bool VECTORIZED>
__global__ void spmv_rows(
    const float* A, const float* x, float* y, int N)
{
    __shared__ float warp_totals[WARPS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* matrix_row = A + row * N;

    // Independent sums shorten the dependency chains.
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    if (VECTORIZED) {
        const float4* a4 =
            reinterpret_cast<const float4*>(matrix_row);
        const float4* x4 =
            reinterpret_cast<const float4*>(x);

        #pragma unroll 1
        for (int j = tid; j < N / 4; j += THREADS) {
            // Stream A through L2; allow the reused x vector in L1.
            const float4 a = __ldcg(a4 + j);
            const float4 v = __ldg(x4 + j);

            s0 = fmaf(a.x, v.x, s0);
            s1 = fmaf(a.y, v.y, s1);
            s2 = fmaf(a.z, v.z, s2);
            s3 = fmaf(a.w, v.w, s3);
        }
    } else {
        // Coalesced scalar loads for odd row lengths
        // or pointers without float4 alignment.
        #pragma unroll 1
        for (int j = tid; j < N; j += 4 * THREADS) {
            s0 = fmaf(
                __ldcg(matrix_row + j),
                __ldg(x + j), s0);

            if (j + THREADS < N)
                s1 = fmaf(
                    __ldcg(matrix_row + j + THREADS),
                    __ldg(x + j + THREADS), s1);

            if (j + 2 * THREADS < N)
                s2 = fmaf(
                    __ldcg(matrix_row + j + 2 * THREADS),
                    __ldg(x + j + 2 * THREADS), s2);

            if (j + 3 * THREADS < N)
                s3 = fmaf(
                    __ldcg(matrix_row + j + 3 * THREADS),
                    __ldg(x + j + 3 * THREADS), s3);
        }
    }

    float sum = warp_sum((s0 + s1) + (s2 + s3));

    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (lane == 0)
        warp_totals[warp] = sum;

    __syncthreads();

    // First warp combines the four warp totals.
    if (warp == 0) {
        sum = lane < WARPS ? warp_totals[lane] : 0.0f;
        sum = warp_sum(sum);

        if (lane == 0)
            y[row] = sum;
    }
}
} // namespace

// A, x and y are device pointers.
// A contains all M*N entries, including zeros.
extern "C" void solve(
    const float* A, const float* x, float* y,
    int M, int N, int nnz)
{
    // A count alone does not provide nonzero positions.
    (void)nnz;

    const bool vectorized =
        (N % 4 == 0) &&
        (((reinterpret_cast<uintptr_t>(A) |
           reinterpret_cast<uintptr_t>(x)) & 15) == 0);

    if (vectorized)
        spmv_rows<true><<<M, THREADS>>>(A, x, y, N);
    else
        spmv_rows<false><<<M, THREADS>>>(A, x, y, N);
}
