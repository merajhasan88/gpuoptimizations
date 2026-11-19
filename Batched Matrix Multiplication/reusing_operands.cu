#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int T = 128, BK = 16;

template<int E> struct __align__(16) Tiles {
    float a[2][BK][E], b[2][BK][E];
};

template<bool Full>
__device__ __forceinline__ float4 read4(const float* p, int r, int c,
                                         int rows, int cols) {
    if constexpr (Full)
        return __ldg(reinterpret_cast<const float4*>(p + r * cols + c));

    float4 v = {};
    if (r < rows && c < cols) {
        const float* q = p + r * cols + c;
        v.x = q[0];
        if (c + 1 < cols) v.y = q[1];
        if (c + 2 < cols) v.z = q[2];
        if (c + 3 < cols) v.w = q[3];
    }
    return v;
}

template<int E, bool Full>
__device__ __forceinline__ void fetch(const float* A, const float* B,
    int M, int N, int K, int by, int bx, int k0,
    float4 (&va)[E / 32], float4 (&vb)[E / 32]) {
    #pragma unroll
    for (int i = 0; i < E / 32; ++i) {
        int t = threadIdx.x + i * T;
        va[i] = read4<Full>(A, by + t / 4, k0 + (t % 4) * 4, M, K);
        vb[i] = read4<Full>(B, k0 + t / (E / 4),
                            bx + (t % (E / 4)) * 4, K, N);
    }
}

template<int E>
__device__ __forceinline__ void publish(Tiles<E>& s, int stage,
    const float4 (&va)[E / 32], const float4 (&vb)[E / 32]) {
    #pragma unroll
    for (int i = 0; i < E / 32; ++i) {
        int t = threadIdx.x + i * T, k = (t % 4) * 4;

        // Transpose A. The XOR spreads the four K groups across banks
        // while keeping groups of four output rows contiguous.
        int r = (t / 4) ^ ((k / 4) * 8);
        s.a[stage][k    ][r] = va[i].x;
        s.a[stage][k + 1][r] = va[i].y;
        s.a[stage][k + 2][r] = va[i].z;
        s.a[stage][k + 3][r] = va[i].w;

        *reinterpret_cast<float4*>(
            &s.b[stage][t / (E / 4)][(t % (E / 4)) * 4]) = vb[i];
    }
}

__device__ __forceinline__ void mac4(float4& s, float a, float4 b) {
    s.x = fmaf(a, b.x, s.x); s.y = fmaf(a, b.y, s.y);
    s.z = fmaf(a, b.z, s.z); s.w = fmaf(a, b.w, s.w);
}

template<int E, bool Full, int Fixed>
__global__ __launch_bounds__(T) void bmm(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int m, int n, int k) {
    const int M = Fixed ? Fixed : m, N = Fixed ? Fixed : n, K = Fixed ? Fixed : k;
    const int batch = blockIdx.z;
    A += batch * M * K;
    B += batch * K * N;
    C += batch * M * N;

    const int by = blockIdx.y * E, bx = blockIdx.x * E;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    constexpr int R = E == 64 ? 8 : 2;
    const int c = (warp % (E / 32)) * 32 + (lane % 8) * 4;
    const int r = E == 64 ? (warp / 2) * 32 + (lane / 8) * 4
                          : warp * 8 + (lane / 8) * 2;

    __shared__ Tiles<E> s;
    float4 sum[R] = {};
    float4 va[E / 32], vb[E / 32];

    fetch<E, Full>(A, B, M, N, K, by, bx, 0, va, vb);
    publish(s, 0, va, vb);
    __syncthreads();

    int stage = 0;
    #pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more)
            fetch<E, Full>(A, B, M, N, K, by, bx, k0 + BK, va, vb);

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const int x = r ^ ((kk / 4) * 8);
            const float4 b = *reinterpret_cast<const float4*>(&s.b[stage][kk][c]);

            if constexpr (E == 64) {
                const float4 lo = *reinterpret_cast<const float4*>(&s.a[stage][kk][x]);
                const float4 hi = *reinterpret_cast<const float4*>(&s.a[stage][kk][x ^ 16]);
                mac4(sum[0], lo.x, b); mac4(sum[1], lo.y, b);
                mac4(sum[2], lo.z, b); mac4(sum[3], lo.w, b);
                mac4(sum[4], hi.x, b); mac4(sum[5], hi.y, b);
                mac4(sum[6], hi.z, b); mac4(sum[7], hi.w, b);
            } else {
                const float2 a = *reinterpret_cast<const float2*>(&s.a[stage][kk][x]);
                mac4(sum[0], a.x, b); mac4(sum[1], a.y, b);
            }
        }

        if (more) {
            stage ^= 1;
            publish(s, stage, va, vb);
            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = by + r + (i < 4 ? i : i + 12), col = bx + c;
        if constexpr (Full) {
            *reinterpret_cast<float4*>(C + row * N + col) = sum[i];
        } else if (row < M && col < N) {
            float* q = C + row * N + col;
            q[0] = sum[i].x;
            if (col + 1 < N) q[1] = sum[i].y;
            if (col + 2 < N) q[2] = sum[i].z;
            if (col + 3 < N) q[3] = sum[i].w;
        }
    }
}

template<int E>
void launch(const float* A, const float* B, float* C,
            int batch, int M, int N, int K) {
    dim3 grid((N + E - 1) / E, (M + E - 1) / E, batch);
    const bool aligned = ((uintptr_t(A) | uintptr_t(B) | uintptr_t(C)) & 15) == 0;

    if (aligned && M == 256 && N == 256 && K == 256)
        bmm<E, true, 256><<<grid, T>>>(A, B, C, M, N, K);
    else if (aligned && M % E == 0 && N % E == 0 && K % BK == 0)
        bmm<E, true, 0><<<grid, T>>>(A, B, C, M, N, K);
    else
        bmm<E, false, 0><<<grid, T>>>(A, B, C, M, N, K);
}
} // namespace

extern "C" void solve(const float* A, const float* B, float* C,
                      int BATCH, int M, int N, int K) {
    const int tiles64 = ((M + 63) / 64) * ((N + 63) / 64) * BATCH;
    if (tiles64 < 128 || M <= 32 || N <= 32)
        launch<32>(A, B, C, BATCH, M, N, K);
    else
        launch<64>(A, B, C, BATCH, M, N, K);
}
