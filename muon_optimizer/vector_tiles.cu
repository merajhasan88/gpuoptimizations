#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int REDUCE_THREADS = 256;
constexpr int BM = 64;
constexpr int BK = 16;
constexpr int AS = BM / 4;

// Swizzle float4 groups, keeping every vector 16-byte aligned.
// This permutes shared-memory banks during the transposed tile loads.
__device__ __forceinline__ int swizzle(int k) { return (k >> 2) * 2; }

union __align__(16) Shared4 { float4 v; float f[4]; };

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

// Called by all 256 threads. Only thread 0 receives the block sum.
__device__ __forceinline__ float block_sum(float v, float* shared) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    v = threadIdx.x < 8 ? shared[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    return v;
}

// Form V and reduce its squared norm in parallel.
// Tall matrices are transposed so the Gram matrix uses min(M,N).
__global__ void prepare(const float* G, const float* Buf, float* X,
                        float* partials, int M, int N, float momentum) {
    __shared__ float shared[8];
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < M * N; i += blockDim.x * gridDim.x) {
        float v = __fadd_rn(__fmul_rn(momentum, Buf[i]), G[i]);
        int dst = M > N ? (i % N) * M + i / N : i;
        X[dst] = v;
        sum = fmaf(v, v, sum);
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

__global__ void finish_norm(const float* partials, float* denominator,
                            int blocks) {
    __shared__ float shared[8];
    float v = threadIdx.x < blocks ? partials[threadIdx.x] : 0.0f;
    v = block_sum(v, shared);
    if (threadIdx.x == 0) denominator[0] = sqrtf(v) + 1.0e-7f;
}

__global__ void normalize(float* X, const float* denominator, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) X[i] /= denominator[0];
}

// FULL is used only when every tile fits and row strides are aligned.
// Other shapes use guarded vector loads or guarded scalar loads.
template<bool FULL>
__device__ __forceinline__ float4 load4(const float* p, int r, int c,
                                       int rows, int cols) {
    if constexpr (FULL) {
        return *reinterpret_cast<const float4*>(p + r * cols + c);
    } else {
        if (r < rows && c + 3 < cols && (cols & 3) == 0)
            return *reinterpret_cast<const float4*>(p + r * cols + c);
        float4 v = make_float4(0, 0, 0, 0);
        if (r < rows) {
            if (c     < cols) v.x = p[r * cols + c];
            if (c + 1 < cols) v.y = p[r * cols + c + 1];
            if (c + 2 < cols) v.z = p[r * cols + c + 2];
            if (c + 3 < cols) v.w = p[r * cols + c + 3];
        }
        return v;
    }
}

template<int BS>
__device__ __forceinline__ void fragment(
    const Shared4* sa, const Shared4* sb, int k, int tx, int ty,
    float* a, float* b) {
    int mask = swizzle(k);
    float4 lo = sa[k * AS + ((ty * 2) ^ mask)].v;
    float4 hi = sa[k * AS + ((ty * 2 + 1) ^ mask)].v;
    float4 bv = sb[k * BS + (tx ^ mask)].v;
    a[0] = lo.x; a[1] = lo.y; a[2] = lo.z; a[3] = lo.w;
    a[4] = hi.x; a[5] = hi.y; a[6] = hi.z; a[7] = hi.w;
    b[0] = bv.x; b[1] = bv.y; b[2] = bv.z; b[3] = bv.w;
}

// MODE 0: Gram = X*X^T.
// MODE 1: P = -4.7750*Gram + 2.0315*(Gram*Gram).
// MODE 2: next_X = 3.4445*X + P*X.
// MODE 3: same as MODE 2, with final scaling/output orientation.
//
// Modes 0/1 are symmetric: calculate upper tiles and mirror results.
// Each thread accumulates an 8x4 microtile in FP32 registers.
// Symmetric products: 128 threads. General products: 256 threads.
template<int MODE, bool FULL>
__global__ void tiled_product(
    const float* __restrict__ L, const float* __restrict__ R,
    const float* __restrict__ add, float* __restrict__ out,
    int rows, int cols, int inner, float scale, bool transpose_output) {
    constexpr bool SYMMETRIC = MODE < 2;
    constexpr int TX = SYMMETRIC ? 16 : 32;
    constexpr int BN = TX * 4;
    constexpr int NT = TX * 8;
    constexpr int BS = BN / 4;
    if constexpr (SYMMETRIC) {
        if (blockIdx.x < blockIdx.y) return; // Whole-block decision.
    }

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TX + tx;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    __shared__ Shared4 sa[BK][AS];
    __shared__ Shared4 sb[BK][BS];
    float acc[8][4] = {};
    float ar[2][8], br[2][4];

    #pragma unroll 1
    for (int k0 = 0; k0 < inner; k0 += BK) {
        // Coalesced float4 loads; transpose and swizzle the shared tile.
        #pragma unroll
        for (int v = tid; v < BM * BK / 4; v += NT) {
            int r = v / (BK / 4), k = (v % (BK / 4)) * 4;
            float4 x = load4<FULL>(L, row0 + r, k0 + k, rows, inner);
            sa[k    ][(r / 4) ^ swizzle(k    )].f[r & 3] = x.x;
            sa[k + 1][(r / 4) ^ swizzle(k + 1)].f[r & 3] = x.y;
            sa[k + 2][(r / 4) ^ swizzle(k + 2)].f[r & 3] = x.z;
            sa[k + 3][(r / 4) ^ swizzle(k + 3)].f[r & 3] = x.w;
        }

        if constexpr (SYMMETRIC) {
            // R is stored untransposed; transpose its tile on load.
            // For MODE 1, Gram is symmetric, so Gram*Gram^T=Gram*Gram.
            #pragma unroll
            for (int v = tid; v < BN * BK / 4; v += NT) {
                int r = v / (BK / 4), k = (v % (BK / 4)) * 4;
                float4 x = load4<FULL>(R, col0 + r, k0 + k, cols, inner);
                sb[k    ][(r / 4) ^ swizzle(k    )].f[r & 3] = x.x;
                sb[k + 1][(r / 4) ^ swizzle(k + 1)].f[r & 3] = x.y;
                sb[k + 2][(r / 4) ^ swizzle(k + 2)].f[r & 3] = x.z;
                sb[k + 3][(r / 4) ^ swizzle(k + 3)].f[r & 3] = x.w;
            }
        } else {
            #pragma unroll
            for (int v = tid; v < BK * BN / 4; v += NT) {
                int k = v / (BN / 4), c = (v % (BN / 4)) * 4;
                float4 x = load4<FULL>(R, k0 + k, col0 + c, inner, cols);
                sb[k][(c / 4) ^ swizzle(k)].v = x;
            }
        }
        __syncthreads();

        fragment<BS>(&sa[0][0], &sb[0][0], 0, tx, ty, ar[0], br[0]);
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int cur = k & 1, next = cur ^ 1;
            // Prefetch the next shared-memory operands into registers.
            if (k + 1 < BK)
                fragment<BS>(&sa[0][0], &sb[0][0], k + 1,
                                 tx, ty, ar[next], br[next]);
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = fmaf(ar[cur][i], br[cur][j], acc[i][j]);
        }
        __syncthreads(); // All readers finish before shared tiles change.
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row0 + ty * 8 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int c = col0 + tx * 4 + j;
            if (!FULL && (r >= rows || c >= cols)) continue;
            // Also avoid duplicate writes inside diagonal tiles.
            if (SYMMETRIC && r > c) continue;
            int index = r * cols + c;
            float v = acc[i][j];
            if constexpr (MODE == 1)
                v = __fadd_rn(__fmul_rn(-4.7750f, add[index]),
                              __fmul_rn(2.0315f, v));
            if constexpr (MODE >= 2)
                v = __fadd_rn(__fmul_rn(3.4445f, add[index]), v);
            if constexpr (MODE == 3) {
                v *= scale;
                out[transpose_output ? c * rows + r : index] = v;
            } else {
                out[index] = v;
                if (SYMMETRIC && r != c) out[c * cols + r] = v;
            }
        }
    }
}

template<int MODE>
void product(const float* L, const float* R, const float* add, float* out,
             int rows, int cols, int inner, float scale = 1.0f,
             bool transpose_output = false) {
    constexpr int TX = MODE < 2 ? 16 : 32;
    constexpr int BN = TX * 4;
    dim3 block(TX, 8), grid((cols + BN - 1) / BN, (rows + BM - 1) / BM);
    bool full = rows % BM == 0 && cols % BN == 0 && inner % BK == 0;
    if (full)
        tiled_product<MODE, true><<<grid, block>>>(
            L, R, add, out, rows, cols, inner, scale, transpose_output);
    else
        tiled_product<MODE, false><<<grid, block>>>(
            L, R, add, out, rows, cols, inner, scale, transpose_output);
}
} // namespace

extern "C" void solve(const float* G, const float* Buf, float* output,
                      int M, int N, float momentum) {
    int rows = M < N ? M : N;
    int cols = M > N ? M : N;
    int count = rows * cols;
    // Round offsets to preserve 16-byte alignment for every shape.
    int matrix_size = (count + 3) & ~3;
    int gram_size = (rows * rows + 3) & ~3;
    int blocks = (count + REDUCE_THREADS - 1) / REDUCE_THREADS;
    if (blocks > 256) blocks = 256;

    float* workspace = nullptr;
    cudaMalloc(&workspace, (2 * matrix_size + 2 * gram_size + 257)
                           * sizeof(float));
    float* X = workspace;
    float* next_X = X + matrix_size;
    float* A = next_X + matrix_size;
    float* P = A + gram_size;
    float* partials = P + gram_size;
    float* denominator = partials + 256;

    prepare<<<blocks, REDUCE_THREADS>>>(G, Buf, X, partials, M, N, momentum);
    finish_norm<<<1, REDUCE_THREADS>>>(partials, denominator, blocks);
    normalize<<<(count + 255) / 256, 256>>>(X, denominator, count);

    // Exactly five iterations; all launches use the same stream.
    for (int step = 0; step < 5; ++step) {
        product<0>(X, X, nullptr, A, rows, rows, cols);
        product<1>(A, A, A, P, rows, rows, rows);
        if (step == 4) {
            product<3>(P, X, X, output, rows, cols, rows,
                       sqrtf(static_cast<float>(cols)), M > N);
        } else {
            product<2>(P, X, X, next_X, rows, cols, rows);
            float* old = X; X = next_X; next_X = old;
        }
    }
    cudaFree(workspace);
}
