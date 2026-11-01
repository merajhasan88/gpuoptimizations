#include <cuda_runtime.h>
#include <stddef.h>

// Each thread computes ONE element of ONE expert's weight gradient.
__global__ void moe_expert_gradient_kernel(
    const float* X,
    const float* dY,
    const int* group_offsets,
    float* dW,
    int K,
    int N)
{
    // Output column, output row, and expert number.
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int g = blockIdx.z;

    // The last blocks may extend beyond the K x N output matrix.
    if (k >= K || n >= N)
        return;

    // This expert owns input rows [start, end).
    const int start = group_offsets[g];
    const int end = group_offsets[g + 1];

    // dW[g, k, n] = sum_m X[m, k] * dY[m, n].
    // There is no need to physically transpose X.
    float sum = 0.0f;

    for (int m = start; m < end; ++m) {
        const float x = X[static_cast<size_t>(m) * K + k];
        const float dy = dY[static_cast<size_t>(m) * N + n];

        // FP32 fused multiply-add: sum = x * dy + sum.
        sum = fmaf(x, dy, sum);
    }

    // Always overwrite the output. Empty experts have start == end,
    // so the loop is skipped and every output element is written as zero.
    // size_t keeps address arithmetic safe for large tensors.
    dW[(static_cast<size_t>(g) * K + k) * N + n] = sum;
}

// All pointers are device pointers. Keep this signature unchanged.
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
    // group_offsets already specifies each expert's row range.
    (void)M_total;

    // A block covers 8 output rows and 32 output columns.
    // A warp therefore reads consecutive columns of dY.
    const dim3 threads(32, 8);
    const dim3 blocks((N + 31) / 32, (K + 7) / 8, G);

    moe_expert_gradient_kernel<<<blocks, threads>>>(
        X, dY, group_offsets, dW, K, N);
}
