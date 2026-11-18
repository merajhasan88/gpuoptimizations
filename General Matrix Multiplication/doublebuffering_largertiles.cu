// FP16 GEMM with FP32 accumulation.
// Turing MMA, XOR shared-memory layout, 64x64 warp tiles.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS = 128;

union Pack8 {
    int4 bits;
    half h[8];
};

struct Shared {
    half a[2][BM * BK];
    half b[2][BK * BN];
};

// Permute 16-byte chunks, preserving contiguous elements inside each.
__device__ __forceinline__
int a_index(int r, int c) {
    return r * BK +
           ((((c >> 3) ^ ((r >> 1) & 3)) << 3) | (c & 7));
}

__device__ __forceinline__
int b_index(int r, int c) {
    return r * BN +
           ((((c >> 3) ^ (r & 7)) << 3) | (c & 7));
}

__device__ __forceinline__
void load_a4(unsigned (&v)[4], const half* p) {
    const unsigned addr =
        static_cast<unsigned>(__cvta_generic_to_shared(p));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
        : "r"(addr)
        : "memory"
    );
}

__device__ __forceinline__
void load_b4(unsigned (&v)[4], const half* p) {
    const unsigned addr =
        static_cast<unsigned>(__cvta_generic_to_shared(p));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0,%1,%2,%3}, [%4];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
        : "r"(addr)
        : "memory"
    );
}

__device__ __forceinline__
void mma1688(
    float (&d)[4],
    unsigned a0,
    unsigned a1,
    unsigned b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b)
    );
}

template<bool FAST>
__device__ __forceinline__
void fetch(
    const half* A,
    const half* B,
    int M, int N, int K,
    int row, int col, int k,
    int tid,
    Pack8 (&a)[4],
    Pack8 (&b)[4])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = tid + i * THREADS;
        const int r = row + t / 4;
        const int c = k + (t & 3) * 8;

        if constexpr (FAST) {
            a[i].bits = __ldg(
                reinterpret_cast<const int4*>(
                    A + r * K + c));
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                a[i].h[j] =
                    r < M && c + j < K
                    ? A[r * K + c + j]
                    : __float2half_rn(0);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = tid + i * THREADS;
        const int r = k + t / 16;
        const int c = col + (t & 15) * 8;

        if constexpr (FAST) {
            b[i].bits = __ldg(
                reinterpret_cast<const int4*>(
                    B + r * N + c));
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                b[i].h[j] =
                    r < K && c + j < N
                    ? B[r * N + c + j]
                    : __float2half_rn(0);
            }
        }
    }
}

__device__ __forceinline__
void publish(
    Shared& s,
    int stage,
    int tid,
    const Pack8 (&a)[4],
    const Pack8 (&b)[4])
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = tid + i * THREADS;

        *reinterpret_cast<int4*>(
            s.a[stage] +
            a_index(t / 4, (t & 3) * 8)) = a[i].bits;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = tid + i * THREADS;

        *reinterpret_cast<int4*>(
            s.b[stage] +
            b_index(t / 16, (t & 15) * 8)) = b[i].bits;
    }
}

template<bool FAST, bool ZERO>
__device__ __forceinline__
void write_pair(
    half* C,
    int M, int N,
    int r, int c,
    float x, float y,
    float alpha, float beta)
{
    x *= alpha;
    y *= alpha;

    if constexpr (FAST) {
        half* p = C + r * N + c;

        if constexpr (!ZERO) {
            const float2 old = __half22float2(
                *reinterpret_cast<const half2*>(p));

            x = fmaf(beta, old.x, x);
            y = fmaf(beta, old.y, y);
        }

        *reinterpret_cast<half2*>(p) =
            __floats2half2_rn(x, y);
    } else if (r < M) {
        if (c < N) {
            const int i = r * N + c;

            if constexpr (!ZERO)
                x = fmaf(beta, __half2float(C[i]), x);

            C[i] = __float2half_rn(x);
        }

        if (c + 1 < N) {
            const int i = r * N + c + 1;

            if constexpr (!ZERO)
                y = fmaf(beta, __half2float(C[i]), y);

            C[i] = __float2half_rn(y);
        }
    }
}

template<bool FAST, bool ZERO, int FIXED = 0>
__global__ __launch_bounds__(THREADS, 2)
void gemm_square_v5(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int m, int n, int k,
    float alpha, float beta)
{
    const int M = FIXED ? FIXED : m;
    const int N = FIXED ? FIXED : n;
    const int K = FIXED ? FIXED : k;

    __shared__ __align__(32) Shared s;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Group neighboring M tiles to encourage cache reuse of B.
    const int tm = (M + BM - 1) / BM;
    const int tn = (N + BN - 1) / BN;
    const int group = blockIdx.x / (2 * tn);
    const int first_m = 2 * group;
    const int rows = first_m + 1 < tm ? 2 : 1;
    const int q = blockIdx.x - group * 2 * tn;

    const int row = (first_m + q % rows) * BM;
    const int col = (q / rows) * BN;

    // Four warps arranged as 2x2, each producing 64x64 outputs.
    const int wr = (warp / 2) * 64;
    const int wc = (warp & 1) * 64;

    // 32 MMA fragments, four FP32 accumulator values per lane each.
    float acc[4][8][4] = {};

    Pack8 next_a[4];
    Pack8 next_b[4];

    fetch<FAST>(
        A, B, M, N, K,
        row, col, 0, tid,
        next_a, next_b);

    publish(s, 0, tid, next_a, next_b);
    __syncthreads();

    int stage = 0;

    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;

        // Prefetch into registers while the current tile is computed.
        if (more) {
            fetch<FAST>(
                A, B, M, N, K,
                row, col, k0 + BK, tid,
                next_a, next_b);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            unsigned av0[4], av1[4];
            unsigned bv0[4], bv1[4];

            load_a4(
                av0,
                s.a[stage] + a_index(wr + lane, kk));

            load_a4(
                av1,
                s.a[stage] + a_index(wr + 32 + lane, kk));

            load_b4(
                bv0,
                s.b[stage] +
                b_index(kk + (lane & 7),
                        wc + (lane >> 3) * 8));

            load_b4(
                bv1,
                s.b[stage] +
                b_index(kk + (lane & 7),
                        wc + 32 + (lane >> 3) * 8));

            // Reuse A and B fragments across all four quadrants.
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma1688(
                        acc[i][j],
                        av0[2 * i], av0[2 * i + 1],
                        bv0[j]);

                    mma1688(
                        acc[i + 2][j],
                        av1[2 * i], av1[2 * i + 1],
                        bv0[j]);

                    mma1688(
                        acc[i][j + 4],
                        av0[2 * i], av0[2 * i + 1],
                        bv1[j]);

                    mma1688(
                        acc[i + 2][j + 4],
                        av1[2 * i], av1[2 * i + 1],
                        bv1[j]);
                }
            }
        }

        if (more)
            publish(s, stage ^ 1, tid, next_a, next_b);

        __syncthreads();
        stage ^= 1;
    }

    // Store directly from the documented MMA accumulator layout.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = row + wr + i * 16 + (lane >> 2);
            const int c = col + wc + j * 8 + (lane & 3) * 2;

            write_pair<FAST, ZERO>(
                C, M, N, r, c,
                acc[i][j][0], acc[i][j][1],
                alpha, beta);

            write_pair<FAST, ZERO>(
                C, M, N, r + 8, c,
                acc[i][j][2], acc[i][j][3],
                alpha, beta);
        }
    }
}

template<bool FAST, int FIXED = 0>
void launch(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int blocks =
        ((M + BM - 1) / BM) * ((N + BN - 1) / BN);

    if (beta == 0) {
        gemm_square_v5<FAST, true, FIXED>
            <<<blocks, THREADS>>>(
                A, B, C, M, N, K, alpha, beta);
    } else {
        gemm_square_v5<FAST, false, FIXED>
            <<<blocks, THREADS>>>(
                A, B, C, M, N, K, alpha, beta);
    }
}

} // namespace

extern "C" void solve(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha, float beta)
{
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(A) |
          reinterpret_cast<uintptr_t>(B) |
          reinterpret_cast<uintptr_t>(C)) & 15u) == 0;

    if (aligned && M == 1024 && N == 1024 && K == 1024) {
        launch<true, 1024>(
            A, B, C, M, N, K, alpha, beta);
    } else if (
        aligned &&
        M % BM == 0 &&
        N % BN == 0 &&
        K % BK == 0)
    {
        launch<true>(
            A, B, C, M, N, K, alpha, beta);
    } else {
        launch<false>(
            A, B, C, M, N, K, alpha, beta);
    }
}
