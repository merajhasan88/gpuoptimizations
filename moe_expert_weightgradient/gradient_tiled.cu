#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace {
constexpr int TILE = 64;
constexpr int REDUCE = 16;
constexpr int THREADS = 128;

// FAST is used when matrix widths are multiples of 64 and pointers
// are aligned. Each expert can still have any number of tokens.
template<bool FAST>
__device__ __forceinline__ float4 read4(
    const float* p, int row, int col, int width, int end)
{
    if (row >= end)
        return make_float4(0, 0, 0, 0);

    const float* r = p + static_cast<size_t>(row) * width;

    if constexpr (FAST) {
        return *reinterpret_cast<const float4*>(r + col);
    } else {
        return make_float4(
            col     < width ? r[col]     : 0.0f,
            col + 1 < width ? r[col + 1] : 0.0f,
            col + 2 < width ? r[col + 2] : 0.0f,
            col + 3 < width ? r[col + 3] : 0.0f);
    }
}

// Load operands for one thread's 8 x 4 outer product.
__device__ __forceinline__ void fragment(
    const float4* sx, const float4* sy,
    int r, int tx, int ty, float* a, float* b)
{
    const float4 x0 = sx[r * 16 + ty * 2];
    const float4 x1 = sx[r * 16 + ty * 2 + 1];
    const float4 y = sy[r * 16 + tx];

    a[0] = x0.x; a[1] = x0.y;
    a[2] = x0.z; a[3] = x0.w;
    a[4] = x1.x; a[5] = x1.y;
    a[6] = x1.z; a[7] = x1.w;

    b[0] = y.x; b[1] = y.y;
    b[2] = y.z; b[3] = y.w;
}

template<bool FAST>
__global__ void moe_gradient_tiled(
    const float* __restrict__ X,
    const float* __restrict__ dY,
    const int* __restrict__ group_offsets,
    float* __restrict__ dW,
    int K, int N)
{
    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    const int k0 = blockIdx.y * TILE;
    const int n0 = blockIdx.x * TILE;
    const int g = blockIdx.z;

    const int start = group_offsets[g];
    const int end = group_offsets[g + 1];

    // Both inputs are stored [token][feature].
    // Load contiguous features directly into shared memory.
    __shared__ float4 sx[REDUCE * 16];
    __shared__ float4 sy[REDUCE * 16];

    // Each thread owns 8 output rows and 4 output columns.
    float acc[8][4] = {};

    // Two sets of operands allow shared-memory load-ahead.
    float ar[2][8], br[2][4];

    #pragma unroll 1
    for (int m0 = start; m0 < end; m0 += REDUCE) {
        // Each thread loads two float4 vectors from each input.
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int v = tid + q * THREADS;
            const int r = v >> 4;
            const int c = (v & 15) * 4;

            sx[v] = read4<FAST>(X, m0 + r, k0 + c, K, end);
            sy[v] = read4<FAST>(dY, m0 + r, n0 + c, N, end);
        }
        __syncthreads();

        fragment(sx, sy, 0, tx, ty, ar[0], br[0]);

        #pragma unroll
        for (int r = 0; r < REDUCE; ++r) {
            const int cur = r & 1;
            const int next = cur ^ 1;

            // Fetch the next operands before using the current set.
            if (r + 1 < REDUCE)
                fragment(
                    sx, sy, r + 1, tx, ty, ar[next], br[next]);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(
                        ar[cur][i], br[cur][j], acc[i][j]);
                }
            }
        }

        // Finish all reads before replacing the shared-memory tiles.
        __syncthreads();
    }

    // Empty experts skip the loop and write zero accumulators.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = k0 + ty * 8 + i;
        const int n = n0 + tx * 4;

        if constexpr (FAST) {
            float* dst =
                dW + (static_cast<size_t>(g) * K + k) * N + n;

            *reinterpret_cast<float4*>(dst) = make_float4(
                acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        } else {
            if (k < K) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (n + j < N) {
                        dW[(static_cast<size_t>(g) * K + k)
                           * N + n + j] = acc[i][j];
                    }
                }
            }
        }
    }
}
} // namespace

extern "C" void solve(
    const float* X,
    const float* dY,
    const int* group_offsets,
    float* dW,
    int G,
    int M_total,
    int K,
    int N)
{
    (void)M_total;

    const dim3 grid(
        (N + TILE - 1) / TILE,
        (K + TILE - 1) / TILE,
        G);

    const uintptr_t addresses =
          reinterpret_cast<uintptr_t>(X)
        | reinterpret_cast<uintptr_t>(dY)
        | reinterpret_cast<uintptr_t>(dW);

    if (K % TILE == 0 &&
        N % TILE == 0 &&
        (addresses & 15) == 0)
    {
        moe_gradient_tiled<true><<<grid, THREADS>>>(
            X, dY, group_offsets, dW, K, N);
    } else {
        moe_gradient_tiled<false><<<grid, THREADS>>>(
            X, dY, group_offsets, dW, K, N);
    }
}
