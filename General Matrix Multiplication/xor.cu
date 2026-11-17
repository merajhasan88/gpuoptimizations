// FP16 GEMM, FP32 accumulation.
// Native Turing MMA + XOR shared-memory layout.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int THREADS = 256;

union Pack8 {
    int4 bits;
    half h[8];
};

struct Shared {
    half a[2][BM * BK];
    half b[2][BK * BN];
};

// Permute 16-byte chunks while preserving eight contiguous halves.
__device__ __forceinline__ int a_index(int r, int c) {
    return r * BK +
           ((((c >> 3) ^ ((r >> 1) & 3)) << 3) | (c & 7));
}

__device__ __forceinline__ int b_index(int r, int c) {
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
    Pack8& a0,
    Pack8& a1,
    Pack8& b)
{
    const int ar = tid / 4;
    const int ac = (tid & 3) * 8;
    const int br = tid / 8;
    const int bc = (tid & 7) * 8;

    if constexpr (FAST) {
        a0.bits = __ldg(
            reinterpret_cast<const int4*>(
                A + (row + ar) * K + k + ac));

        a1.bits = __ldg(
            reinterpret_cast<const int4*>(
                A + (row + ar + 64) * K + k + ac));

        b.bits = __ldg(
            reinterpret_cast<const int4*>(
                B + (k + br) * N + col + bc));
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            a0.h[j] =
                row + ar < M && k + ac + j < K
                ? A[(row + ar) * K + k + ac + j]
                : __float2half_rn(0);

            a1.h[j] =
                row + ar + 64 < M && k + ac + j < K
                ? A[(row + ar + 64) * K + k + ac + j]
                : __float2half_rn(0);

            b.h[j] =
                k + br < K && col + bc + j < N
                ? B[(k + br) * N + col + bc + j]
                : __float2half_rn(0);
        }
    }
}

__device__ __forceinline__
void publish(
    Shared& s,
    int stage,
    int tid,
    const Pack8& a0,
    const Pack8& a1,
    const Pack8& b)
{
    const int ar = tid / 4;
    const int ac = (tid & 3) * 8;
    const int br = tid / 8;
    const int bc = (tid & 7) * 8;

    *reinterpret_cast<int4*>(
        s.a[stage] + a_index(ar, ac)) = a0.bits;

    *reinterpret_cast<int4*>(
        s.a[stage] + a_index(ar + 64, ac)) = a1.bits;

    *reinterpret_cast<int4*>(
        s.b[stage] + b_index(br, bc)) = b.bits;
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
void gemm_xor_v2(
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

    // Group neighboring M tiles to encourage reuse of B.
    const int tm = (M + BM - 1) / BM;
    const int tn = (N + BN - 1) / BN;
    const int group = blockIdx.x / (2 * tn);
    const int first_m = 2 * group;
    const int rows = first_m + 1 < tm ? 2 : 1;
    const int q = blockIdx.x - group * 2 * tn;

    const int row = (first_m + q % rows) * BM;
    const int col = (q / rows) * BN;

    const int wr = (warp / 2) * 32;
    const int wc = (warp & 1) * 32;

    // Each warp computes 32x32 outputs.
    // Eight 16x8 accumulators, four FP32 values per lane each.
    float acc[2][4][4] = {};

    Pack8 next_a0, next_a1, next_b;

    fetch<FAST>(
        A, B, M, N, K, row, col, 0, tid,
        next_a0, next_a1, next_b);

    publish(s, 0, tid, next_a0, next_a1, next_b);
    __syncthreads();

    int stage = 0;

    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;

        // Fetch the next tile before computing the current tile.
        if (more) {
            fetch<FAST>(
                A, B, M, N, K,
                row, col, k0 + BK, tid,
                next_a0, next_a1, next_b);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            unsigned av[4], bv[4];

            // All lanes provide valid 16-byte-aligned row addresses.
            load_a4(
                av,
                s.a[stage] + a_index(wr + lane, kk));

            load_b4(
                bv,
                s.b[stage] +
                b_index(kk + (lane & 7),
                        wc + (lane >> 3) * 8));

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma1688(
                        acc[i][j],
                        av[2 * i],
                        av[2 * i + 1],
                        bv[j]);
                }
            }
        }

        if (more)
            publish(
                s, stage ^ 1, tid,
                next_a0, next_a1, next_b);

        __syncthreads();
        stage ^= 1;
    }

    // Documented native MMA register mapping allows direct output.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
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
        gemm_xor_v2<FAST, true, FIXED>
            <<<blocks, THREADS>>>(
                A, B, C, M, N, K, alpha, beta);
    } else {
        gemm_xor_v2<FAST, false, FIXED>
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
