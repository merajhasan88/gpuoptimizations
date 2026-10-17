#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int OUT_Z = 4;
constexpr int OUT_Y = 2;
constexpr int MAX_K = 5;
constexpr int THREADS = BX * BY;

constexpr int TILE_OUT_Y = BY * OUT_Y;       // 8 output rows
constexpr int TX = BX + MAX_K - 1;          // 36 input columns
constexpr int TY = TILE_OUT_Y + MAX_K - 1;   // 12 input rows
constexpr int TZ = OUT_Z + MAX_K - 1;        // 8 input slices
constexpr int TILE_SIZE = TX * TY * TZ;

// Inline helper: one weight contributes to eight output accumulators.
template <bool FIXED5>
__device__ __forceinline__ void accumulate_z4y2(
    const float tile[TZ][TY][TX],
    const float* weights,
    int tx, int local_y,
    int kernel_depth, int kernel_rows, int kernel_cols,
    float (&sum)[OUT_Z][OUT_Y])
{
    const int kd = FIXED5 ? 5 : kernel_depth;
    const int kr = FIXED5 ? 5 : kernel_rows;
    const int kc = FIXED5 ? 5 : kernel_cols;

    #pragma unroll (FIXED5 ? 5 : 1)
    for (int kz = 0; kz < kd; ++kz) {
        #pragma unroll (FIXED5 ? 5 : 1)
        for (int ky = 0; ky < kr; ++ky) {
            #pragma unroll (FIXED5 ? 5 : 1)
            for (int kx = 0; kx < kc; ++kx) {
                const float w = weights[(kz * kr + ky) * kc + kx];

                #pragma unroll
                for (int dz = 0; dz < OUT_Z; ++dz) {
                    #pragma unroll
                    for (int dy = 0; dy < OUT_Y; ++dy) {
                        sum[dz][dy] = fmaf(
                            tile[kz + dz][local_y + ky + dy][tx + kx],
                            w, sum[dz][dy]);
                    }
                }
            }
        }
    }
}

// One GPU kernel handles every supported filter shape.
__global__ void conv3d_z4y2(
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
    const int base_y = blockIdx.y * TILE_OUT_Y;
    const int base_z = blockIdx.z * OUT_Z;

    // The 128 threads cooperatively load the input tile and halo.
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

    // No thread may return before this barrier.
    __syncthreads();

    const int output_depth = input_depth - kernel_depth + 1;
    const int output_rows  = input_rows  - kernel_rows  + 1;
    const int output_cols  = input_cols  - kernel_cols  + 1;

    // Each thread owns two adjacent rows in four depth slices.
    const int local_y = ty * OUT_Y;
    const int x = base_x + tx;
    const int y = base_y + local_y;

    if (x >= output_cols || y >= output_rows) {
        return;
    }

    float sum[OUT_Z][OUT_Y] = {};

    if (kernel_depth == 5 && kernel_rows == 5 && kernel_cols == 5) {
        accumulate_z4y2<true>(
            tile, weights, tx, local_y,
            kernel_depth, kernel_rows, kernel_cols, sum);
    } else {
        accumulate_z4y2<false>(
            tile, weights, tx, local_y,
            kernel_depth, kernel_rows, kernel_cols, sum);
    }

    // Preserve valid convolution at partial row and depth tiles.
    #pragma unroll
    for (int dz = 0; dz < OUT_Z; ++dz) {
        #pragma unroll
        for (int dy = 0; dy < OUT_Y; ++dy) {
            if (base_z + dz < output_depth && y + dy < output_rows) {
                const int index =
                    ((base_z + dz) * output_rows + y + dy)
                    * output_cols + x;

                output[index] = sum[dz][dy];
            }
        }
    }
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

    // 128 threads, eight outputs per thread.
    const dim3 threads(BX, BY);
    const dim3 blocks(
        (output_cols  + BX - 1) / BX,
        (output_rows  + TILE_OUT_Y - 1) / TILE_OUT_Y,
        (output_depth + OUT_Z - 1) / OUT_Z);

    conv3d_z4y2<<<blocks, threads>>>(
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols);
}
