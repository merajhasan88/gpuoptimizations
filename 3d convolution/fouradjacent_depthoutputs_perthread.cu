#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int OUT_Z = 4;
constexpr int MAX_K = 5;
constexpr int THREADS = BX * BY;

constexpr int TX = BX + MAX_K - 1;     // 36
constexpr int TY = BY + MAX_K - 1;     // 8
constexpr int TZ = OUT_Z + MAX_K - 1;  // 8
constexpr int TILE_SIZE = TX * TY * TZ;

// This helper is inlined; it does not launch a separate GPU kernel.
// FIXED5 selects fully unrolled loops for the 5 x 5 x 5 filter.
template <bool FIXED5>
__device__ __forceinline__ void accumulate_z4(
    const float tile[TZ][TY][TX],
    const float* weights,
    int tx, int ty,
    int kernel_depth, int kernel_rows, int kernel_cols,
    float& s0, float& s1, float& s2, float& s3)
{
    const int kd = FIXED5 ? 5 : kernel_depth;
    const int kr = FIXED5 ? 5 : kernel_rows;
    const int kc = FIXED5 ? 5 : kernel_cols;

    #pragma unroll (FIXED5 ? 5 : 1)
    for (int z = 0; z < kd; ++z) {
        #pragma unroll (FIXED5 ? 5 : 1)
        for (int y = 0; y < kr; ++y) {
            #pragma unroll (FIXED5 ? 5 : 1)
            for (int x = 0; x < kc; ++x) {
                const float w = weights[(z * kr + y) * kc + x];

                // Reuse this weight across four depth-adjacent outputs.
                s0 = fmaf(tile[z    ][ty + y][tx + x], w, s0);
                s1 = fmaf(tile[z + 1][ty + y][tx + x], w, s1);
                s2 = fmaf(tile[z + 2][ty + y][tx + x], w, s2);
                s3 = fmaf(tile[z + 3][ty + y][tx + x], w, s3);
            }
        }
    }
}

// The only GPU kernel in this submission: all filter shapes use it.
__global__ void conv3d_z4(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_depth, int input_rows, int input_cols,
    int kernel_depth, int kernel_rows, int kernel_cols)
{
    __shared__ float tile[TZ][TY][TX];
    __shared__ float weights[MAX_K * MAX_K * MAX_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    const int base_x = blockIdx.x * BX;
    const int base_y = blockIdx.y * BY;
    const int base_z = blockIdx.z * OUT_Z;

    // Load a 36 x 8 x 8 input tile, including the filter halo.
    for (int i = tid; i < TILE_SIZE; i += THREADS) {
        const int sx = i % TX;
        const int sy = (i / TX) % TY;
        const int sz = i / (TX * TY);

        const int x = base_x + sx;
        const int y = base_y + sy;
        const int z = base_z + sz;

        float value = 0.0f;
        if (x < input_cols && y < input_rows && z < input_depth) {
            value = input[(z * input_rows + y) * input_cols + x];
        }

        tile[sz][sy][sx] = value;
    }

    const int filter_size = kernel_depth * kernel_rows * kernel_cols;
    if (tid < filter_size) {
        weights[tid] = kernel[tid];
    }

    // All threads finish loading before any thread reads shared memory.
    __syncthreads();

    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows  = input_rows  - kernel_rows  + 1;
    const int output_cols  = input_cols  - kernel_cols  + 1;

    const int x = base_x + tx;
    const int y = base_y + ty;

    if (x >= output_cols || y >= output_rows) {
        return;
    }

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    if (kernel_depth == 5 && kernel_rows == 5 && kernel_cols == 5) {
        accumulate_z4<true>(
            tile, weights, tx, ty,
            kernel_depth, kernel_rows, kernel_cols,
            s0, s1, s2, s3);
    } else {
        accumulate_z4<false>(
            tile, weights, tx, ty,
            kernel_depth, kernel_rows, kernel_cols,
            s0, s1, s2, s3);
    }

    const int plane = output_rows * output_cols;
    const int index = base_z * plane + y * output_cols + x;

    // The final depth tile may contain fewer than four valid outputs.
    if (base_z     < output_depth) output[index]             = s0;
    if (base_z + 1 < output_depth) output[index + plane]     = s1;
    if (base_z + 2 < output_depth) output[index + 2 * plane] = s2;
    if (base_z + 3 < output_depth) output[index + 3 * plane] = s3;
}

} // namespace

// input, kernel, and output are device pointers.
extern "C" void solve(
    const float* input, const float* kernel, float* output,
    int input_depth, int input_rows, int input_cols,
    int kernel_depth, int kernel_rows, int kernel_cols)
{
    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows  = input_rows  - kernel_rows  + 1;
    const int output_cols  = input_cols  - kernel_cols  + 1;

    // 128 threads, four outputs per thread.
    const dim3 threads(BX, BY);
    const dim3 blocks(
        (output_cols  + BX - 1) / BX,
        (output_rows  + BY - 1) / BY,
        (output_depth + OUT_Z - 1) / OUT_Z);

    conv3d_z4<<<blocks, threads>>>(
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols);
}
