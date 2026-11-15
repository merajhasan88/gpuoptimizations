// FP16 GEMM for T4:
// C = alpha * A * B + beta * C, with FP32 accumulation.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

namespace {
namespace wmma = nvcuda::wmma;

constexpr int BM = 128, BN = 64, BK = 32;
constexpr int THREADS = 256;
constexpr int AS = BK + 8, BS = BN + 8;

union Pack8 {
    int4 bits;
    half h[8];
};

// Input buffers and output staging share the same allocation.
// Total shared memory: 32768 bytes.
union Shared {
    struct {
        half a[2][BM][AS];
        half b[2][BK][BS];
    } tile;
    float out[BM][BN];
};

template<bool FAST>
__device__ __forceinline__ void fetch_tile(
    const half* A, const half* B,
    int M, int N, int K,
    int row, int col, int k0, int tid,
    Pack8& a0, Pack8& a1, Pack8& b)
{
    const int ar = tid / 4;
    const int ac = (tid % 4) * 8;
    const int br = tid / 8;
    const int bc = (tid % 8) * 8;

    if constexpr (FAST) {
        a0.bits = *reinterpret_cast<const int4*>(
            A + (row + ar) * K + k0 + ac);

        a1.bits = *reinterpret_cast<const int4*>(
            A + (row + ar + 64) * K + k0 + ac);

        b.bits = *reinterpret_cast<const int4*>(
            B + (k0 + br) * N + col + bc);
    } else {
        // Zero padding handles arbitrary dimensions and alignment.
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            a0.h[j] = row + ar < M && k0 + ac + j < K
                ? A[(row + ar) * K + k0 + ac + j]
                : __float2half_rn(0.0f);

            a1.h[j] = row + ar + 64 < M && k0 + ac + j < K
                ? A[(row + ar + 64) * K + k0 + ac + j]
                : __float2half_rn(0.0f);

            b.h[j] = k0 + br < K && col + bc + j < N
                ? B[(k0 + br) * N + col + bc + j]
                : __float2half_rn(0.0f);
        }
    }
}

__device__ __forceinline__ void publish_tile(
    Shared& s, int stage, int tid,
    const Pack8& a0, const Pack8& a1, const Pack8& b)
{
    const int ar = tid / 4;
    const int ac = (tid % 4) * 8;
    const int br = tid / 8;
    const int bc = (tid % 8) * 8;

    *reinterpret_cast<int4*>(&s.tile.a[stage][ar][ac]) =
        a0.bits;

    *reinterpret_cast<int4*>(&s.tile.a[stage][ar + 64][ac]) =
        a1.bits;

    *reinterpret_cast<int4*>(&s.tile.b[stage][br][bc]) =
        b.bits;
}

template<bool FAST, bool BETA_ZERO, int FIXED_DIM = 0>
__global__ __launch_bounds__(THREADS, 2)
void gemm_t4_wmma_v1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M_arg, int N_arg, int K_arg,
    float alpha, float beta)
{
    const int M = FIXED_DIM ? FIXED_DIM : M_arg;
    const int N = FIXED_DIM ? FIXED_DIM : N_arg;
    const int K = FIXED_DIM ? FIXED_DIM : K_arg;

    __shared__ __align__(32) Shared smem;

    const int tid = threadIdx.x;
    const int warp = tid / 32;

    // Visit two adjacent M tiles per N tile to encourage B reuse in L2.
    const int tiles_m = (M + BM - 1) / BM;
    const int tiles_n = (N + BN - 1) / BN;
    const int group = blockIdx.x / (2 * tiles_n);
    const int first_m = group * 2;
    const int group_m = first_m + 1 < tiles_m ? 2 : 1;
    const int within = blockIdx.x - group * 2 * tiles_n;

    const int row = (first_m + within % group_m) * BM;
    const int col = (within / group_m) * BN;

    // Eight warps form a 4 x 2 grid.
    // Each warp computes four 16 x 16 fragments: 32 x 32 outputs.
    const int wr = (warp / 2) * 32;
    const int wc = (warp % 2) * 32;

    wmma::fragment<
        wmma::accumulator, 16, 16, 16, float
    > acc[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);
    }

    Pack8 next_a0, next_a1, next_b;

    fetch_tile<FAST>(
        A, B, M, N, K, row, col, 0, tid,
        next_a0, next_a1, next_b);

    publish_tile(smem, 0, tid, next_a0, next_a1, next_b);
    __syncthreads();

    int stage = 0;

    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;

        // Prefetch the next global tile into registers.
        if (more) {
            fetch_tile<FAST>(
                A, B, M, N, K, row, col, k0 + BK, tid,
                next_a0, next_a1, next_b);
        }

        // Compute from the current shared-memory tile.
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<
                wmma::matrix_a, 16, 16, 16,
                half, wmma::row_major
            > af[2];

            wmma::fragment<
                wmma::matrix_b, 16, 16, 16,
                half, wmma::row_major
            > bf[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(
                    af[i],
                    &smem.tile.a[stage][wr + i * 16][kk],
                    AS);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(
                    bf[j],
                    &smem.tile.b[stage][kk][wc + j * 16],
                    BS);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(
                        acc[i][j], af[i], bf[j], acc[i][j]);
            }
        }

        if (more) {
            publish_tile(
                smem, stage ^ 1, tid,
                next_a0, next_a1, next_b);
        }

        // Publish the next tile and finish reads before buffer reuse.
        __syncthreads();
        stage ^= 1;
    }

    // Use WMMA's supported store rather than assuming a fragment layout.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(
                &smem.out[wr + i * 16][wc + j * 16],
                acc[i][j], BN, wmma::mem_row_major);
        }
    }

    __syncthreads();

    // Apply alpha and beta in FP32, then convert the final output to FP16.
    #pragma unroll
    for (int offset = tid * 8;
         offset < BM * BN;
         offset += THREADS * 8)
    {
        const int r = offset / BN;
        const int c = offset % BN;

        if constexpr (FAST) {
            Pack8 old_c, result;

            if constexpr (!BETA_ZERO) {
                old_c.bits = *reinterpret_cast<const int4*>(
                    C + (row + r) * N + col + c);
            }

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float ab = alpha * smem.out[r][c + j];

                const float out = BETA_ZERO
                    ? ab
                    : fmaf(beta, __half2float(old_c.h[j]), ab);

                result.h[j] = __float2half_rn(out);
            }

            *reinterpret_cast<int4*>(
                C + (row + r) * N + col + c) = result.bits;
        } else if (row + r < M) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (col + c + j < N) {
                    const int p = (row + r) * N + col + c + j;
                    const float ab = alpha * smem.out[r][c + j];

                    const float out = BETA_ZERO
                        ? ab
                        : fmaf(beta, __half2float(C[p]), ab);

                    C[p] = __float2half_rn(out);
                }
            }
        }
    }
}

template<bool FAST, int FIXED_DIM = 0>
void launch_gemm(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta)
{
    const int blocks =
        ((M + BM - 1) / BM) * ((N + BN - 1) / BN);

    if (beta == 0.0f) {
        gemm_t4_wmma_v1<FAST, true, FIXED_DIM>
            <<<blocks, THREADS>>>(
                A, B, C, M, N, K, alpha, beta);
    } else {
        gemm_t4_wmma_v1<FAST, false, FIXED_DIM>
            <<<blocks, THREADS>>>(
                A, B, C, M, N, K, alpha, beta);
    }
}
} // namespace

extern "C" void solve(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta)
{
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(A) |
          reinterpret_cast<uintptr_t>(B) |
          reinterpret_cast<uintptr_t>(C)) & 15u) == 0;

    if (aligned && M == 1024 && N == 1024 && K == 1024) {
        launch_gemm<true, 1024>(
            A, B, C, M, N, K, alpha, beta);
    } else if (
        aligned &&
        M % BM == 0 &&
        N % BN == 0 &&
        K % BK == 0)
    {
        launch_gemm<true>(
            A, B, C, M, N, K, alpha, beta);
    } else {
        launch_gemm<false>(
            A, B, C, M, N, K, alpha, beta);
    }
}
