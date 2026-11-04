#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 256;
constexpr unsigned FULL = 0xffffffffu;

// W lanes own one row. P=0: the row fits in W lanes.
// P=4: aligned float4 loads. P=1: scalar loads for arbitrary strides.
template<int W, int P>
__global__ void cross_entropy(const float* __restrict__ logits,
                              const int* __restrict__ labels,
                              float* loss, int N, int C, float invN) {
    constexpr int ROWS = THREADS / W;
    const int tid = threadIdx.x, lane = tid % W;
    const unsigned mask = (FULL >> (32 - W)) << ((tid & 31) - lane);
    float subtotal = 0.0f;

    for (int row = blockIdx.x * ROWS + tid / W; row < N;
         row += gridDim.x * ROWS) {
        const int y = labels[row];

        // A C-column row has no class C. Never read beyond the row.
        if (static_cast<unsigned>(y) >= static_cast<unsigned>(C)) {
            if (lane == 0) subtotal = CUDART_NAN_F;
            continue;
        }

        const float* r = logits + row * C;
        float s = 0.0f, target = 0.0f;

        if constexpr (P == 0) {
            // The entire row is in registers. Broadcast the target logit.
            const float z = lane < C ? r[lane] : 0.0f;
            target = __shfl_sync(mask, z, y, W);
            if (lane < C && lane != y) s = __expf(z - target);

        } else if constexpr (P == 4) {
            const float4* v = reinterpret_cast<const float4*>(r);
            float a = 0, b = 0, c = 0, d = 0;

            for (int i = lane; i < C / 4; i += W) {
                float4 z = v[i];

                if (i == y / 4) {
                    // Keep the target; exp(-inf) excludes it from the sum.
                    switch (y & 3) {
                        case 0: target = z.x; z.x = -CUDART_INF_F; break;
                        case 1: target = z.y; z.y = -CUDART_INF_F; break;
                        case 2: target = z.z; z.z = -CUDART_INF_F; break;
                        default: target = z.w; z.w = -CUDART_INF_F; break;
                    }
                }

                a += __expf(z.x); b += __expf(z.y);
                c += __expf(z.z); d += __expf(z.w);
            }

            s = (a + b) + (c + d);
            target = __shfl_sync(mask, target, (y / 4) % W, W);

        } else {
            for (int k = lane; k < C; k += W) {
                const float z = r[k];
                if (k == y) target = z;
                else s += __expf(z);
            }

            target = __shfl_sync(mask, target, y % W, W);
        }

        #pragma unroll
        for (int step = W / 2; step; step /= 2)
            s += __shfl_down_sync(mask, s, step, W);

        if (lane == 0) {
            // Wide rows accumulated exp(z_k). Apply exp(-target) once.
            if constexpr (P != 0) s *= __expf(-target);
            subtotal += log1pf(s);
        }
    }

    // Only subgroup leaders carry losses. Combine them within each warp.
    #pragma unroll
    for (int step = 16; step >= W; step /= 2)
        subtotal += __shfl_down_sync(FULL, subtotal, step);

    __shared__ float warp_totals[THREADS / 32];
    if ((tid & 31) == 0) warp_totals[tid / 32] = subtotal;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; ++w)
            total += warp_totals[w];

        if (gridDim.x == 1) *loss = total * invN;
        else atomicAdd(loss, total * invN);
    }
}

template<int W, int P>
void launch(const float* x, const int* y, float* loss, int N, int C) {
    constexpr int ROWS = THREADS / W;
    const int needed = (N + ROWS - 1) / ROWS;
    const int blocks = needed < 160 ? needed : 160;

    if (blocks > 1) cudaMemsetAsync(loss, 0, sizeof(float), 0);
    cross_entropy<W,P><<<blocks, THREADS>>>(x, y, loss, N, C, 1.0f / N);
}
} // namespace

extern "C" void solve(const float* logits, const int* true_labels,
                      float* loss, int N, int C) {
    if      (C <=  2) launch< 2,0>(logits, true_labels, loss, N, C);
    else if (C <=  4) launch< 4,0>(logits, true_labels, loss, N, C);
    else if (C <=  8) launch< 8,0>(logits, true_labels, loss, N, C);
    else if (C <= 16) launch<16,0>(logits, true_labels, loss, N, C);
    else if (C <= 32) launch<32,0>(logits, true_labels, loss, N, C);
    else if (C % 4 == 0 &&
             (reinterpret_cast<uintptr_t>(logits) & 15u) == 0) {
        if (C <= 64) launch<16,4>(logits, true_labels, loss, N, C);
        else         launch<32,4>(logits, true_labels, loss, N, C);
    } else {
        launch<32,1>(logits, true_labels, loss, N, C);
    }
}
