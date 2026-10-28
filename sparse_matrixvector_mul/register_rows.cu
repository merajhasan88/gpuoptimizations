#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;

constexpr int TILE_ROWS = 4;
constexpr int TILE_THREADS = 256;
constexpr int TILE_WARPS = TILE_THREADS / 32;

__device__ __forceinline__ float warp_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);

    return value;
}

// Each thread reuses its x vector across four output rows.
// Only the final warp totals use shared memory.
__global__ void spmv_register_rows(
    const float4* __restrict__ A,
    const float4* __restrict__ x,
    float* __restrict__ y,
    int M, int vectors)
{
    __shared__ float warp_totals[TILE_ROWS][TILE_WARPS];

    const int tid = threadIdx.x;
    const int first_row = blockIdx.x * TILE_ROWS;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Four rows, each with four independent accumulation chains.
    float acc[TILE_ROWS][4] = {};

    #pragma unroll 1
    for (int j = tid; j < vectors; j += TILE_THREADS) {
        const float4 v = __ldg(x + j);

        #pragma unroll
        for (int r = 0; r < TILE_ROWS; ++r) {
            if (first_row + r < M) {
                const float4 a =
                    __ldcg(A + (first_row + r) * vectors + j);

                acc[r][0] = fmaf(a.x, v.x, acc[r][0]);
                acc[r][1] = fmaf(a.y, v.y, acc[r][1]);
                acc[r][2] = fmaf(a.z, v.z, acc[r][2]);
                acc[r][3] = fmaf(a.w, v.w, acc[r][3]);
            }
        }
    }

    // Each warp produces one partial sum for each of the four rows.
    #pragma unroll
    for (int r = 0; r < TILE_ROWS; ++r) {
        float sum =
            (acc[r][0] + acc[r][1]) +
            (acc[r][2] + acc[r][3]);

        sum = warp_sum(sum);

        if (lane == 0)
            warp_totals[r][warp] = sum;
    }

    __syncthreads();

    // Four warps each finish one output row.
    if (warp < TILE_ROWS) {
        float sum =
            lane < TILE_WARPS ? warp_totals[warp][lane] : 0.0f;

        sum = warp_sum(sum);

        if (lane == 0 && first_row + warp < M)
            y[first_row + warp] = sum;
    }
}

// Original one-block-per-row fallback.
template<bool VECTORIZED>
__global__ void spmv_rows(
    const float* A, const float* x, float* y, int N)
{
    __shared__ float warp_totals[WARPS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* matrix_row = A + row * N;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    if (VECTORIZED) {
        const float4* a4 =
            reinterpret_cast<const float4*>(matrix_row);
        const float4* x4 =
            reinterpret_cast<const float4*>(x);

        #pragma unroll 1
        for (int j = tid; j < N / 4; j += THREADS) {
            const float4 a = __ldcg(a4 + j);
            const float4 v = __ldg(x4 + j);

            s0 = fmaf(a.x, v.x, s0);
            s1 = fmaf(a.y, v.y, s1);
            s2 = fmaf(a.z, v.z, s2);
            s3 = fmaf(a.w, v.w, s3);
        }
    } else {
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

    if (warp == 0) {
        sum = lane < WARPS ? warp_totals[lane] : 0.0f;
        sum = warp_sum(sum);

        if (lane == 0)
            y[row] = sum;
    }
}
} // namespace

// A, x and y are device pointers; A includes its zero entries.
extern "C" void solve(
    const float* A, const float* x, float* y,
    int M, int N, int nnz)
{
    (void)nnz;

    const bool vectorized =
        (N % 4 == 0) &&
        (((reinterpret_cast<uintptr_t>(A) |
           reinterpret_cast<uintptr_t>(x)) & 15) == 0);

    if (vectorized && N >= 2048 && M >= 128) {
        spmv_register_rows<<<
            (M + TILE_ROWS - 1) / TILE_ROWS,
            TILE_THREADS>>>(
                reinterpret_cast<const float4*>(A),
                reinterpret_cast<const float4*>(x),
                y, M, N / 4);
    } else if (vectorized) {
        spmv_rows<true><<<M, THREADS>>>(A, x, y, N);
    } else {
        spmv_rows<false><<<M, THREADS>>>(A, x, y, N);
    }
}
