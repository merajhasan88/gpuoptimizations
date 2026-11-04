#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 256;
constexpr unsigned FULL = 0xffffffffu;

// GROUP lanes per row. PACK=0: short row; 4: float4; 1: scalar.
template<int GROUP, int PACK, int ROUNDS>
__global__ __launch_bounds__(THREADS, 4)
void cross_entropy(const float* __restrict__ logits,
                   const int* __restrict__ labels,
                   float* loss, int N, int C, float invN) {
    constexpr int ROWS = THREADS / GROUP;
    const int tid = threadIdx.x, lane = tid % GROUP;
    const unsigned mask = (FULL >> (32 - GROUP)) << ((tid & 31) - lane);
    float subtotal = 0.0f;

    for (int row = blockIdx.x * ROWS + tid / GROUP; row < N;
         row += gridDim.x * ROWS) {
        const int y = labels[row];
        if (static_cast<unsigned>(y) >= static_cast<unsigned>(C)) {
            if (lane == 0) subtotal = CUDART_NAN_F;
            continue;
        }
        const float* r = logits + row * C;

        if constexpr (PACK == 0) {
            // The entire row fits in one group of lanes.
            const float z = lane < C ? r[lane] : 0.0f;
            const float t = __shfl_sync(mask, z, y, GROUP);
            float s = (lane < C && lane != y) ? __expf(z - t) : 0.0f;

            #pragma unroll
            for (int d = GROUP / 2; d; d /= 2)
                s += __shfl_down_sync(mask, s, d, GROUP);

            if (lane == 0)
                subtotal += (s > 0.5f) ? __logf(1.0f + s) : log1pf(s);
        } else {
            const float target = (lane == 0) ? r[y] : 0.0f;
            float local = 0.0f;

            // No label comparisons in the main loop.
            if constexpr (PACK == 4) {
                const float4* p = reinterpret_cast<const float4*>(r);
                float a = 0, b = 0, c = 0, d = 0;

                #pragma unroll
                for (int j = 0; j < ROUNDS; ++j) {
                    const int k = lane + j * GROUP;
                    if (k < C / 4) {
                        const float4 v = p[k];
                        a += __expf(v.x); b += __expf(v.y);
                        c += __expf(v.z); d += __expf(v.w);
                    }
                }
                local = (a + b) + (c + d);
            } else {
                for (int k = lane; k < C; k += GROUP)
                    local += __expf(r[k]);
            }

            // Keep each lane's original sum for the small-loss path.
            float sum = local;
            #pragma unroll
            for (int d = GROUP / 2; d; d /= 2)
                sum += __shfl_down_sync(mask, sum, d, GROUP);

            float value = 0.0f;
            if (lane == 0) value = __logf(sum) - target;

            const int repair =
                __shfl_sync(mask, int(value < 0.5f), 0, GROUP);

            if (repair) {
                // Only the owner's sum contains exp(target).
                const int owner = (y / PACK) % GROUP;
                if (lane == owner) {
                    local = 0.0f;
                    #pragma unroll 1
                    for (int k = lane * PACK; k < C; k += GROUP * PACK) {
                        #pragma unroll
                        for (int j = 0; j < PACK; ++j)
                            if (k + j != y) local += __expf(r[k + j]);
                    }
                }

                #pragma unroll
                for (int d = GROUP / 2; d; d /= 2)
                    local += __shfl_down_sync(mask, local, d, GROUP);

                if (lane == 0) value = log1pf(local * __expf(-target));
            }

            if (lane == 0) subtotal += value;
        }
    }

    // Combine row-group leaders, then combine the eight warps.
    #pragma unroll
    for (int d = 16; d >= GROUP; d /= 2)
        subtotal += __shfl_down_sync(FULL, subtotal, d);

    __shared__ float warp_totals[THREADS / 32];
    if ((tid & 31) == 0) warp_totals[tid / 32] = subtotal;
    __syncthreads();

    if (tid == 0) {
        float total = 0;
        #pragma unroll
        for (int w = 0; w < THREADS / 32; ++w) total += warp_totals[w];

        if (gridDim.x == 1) *loss = total * invN;
        else atomicAdd(loss, total * invN);
    }
}

template<int GROUP, int PACK, int ROUNDS = 0>
void launch(const float* x, const int* y, float* loss, int N, int C) {
    constexpr int ROWS = THREADS / GROUP;
    const int need = (N + ROWS - 1) / ROWS;
    const int blocks = need < 160 ? need : 160;

    if (blocks > 1) cudaMemsetAsync(loss, 0, sizeof(float), 0);
    cross_entropy<GROUP, PACK, ROUNDS><<<blocks, THREADS>>>(
        x, y, loss, N, C, 1.0f / N);
}
} // namespace

extern "C" void solve(const float* logits, const int* true_labels,
                      float* loss, int N, int C) {
    if      (C <=  2) launch< 2,0>(logits, true_labels, loss, N, C);
    else if (C <=  4) launch< 4,0>(logits, true_labels, loss, N, C);
    else if (C <=  8) launch< 8,0>(logits, true_labels, loss, N, C);
    else if (C <= 16) launch<16,0>(logits, true_labels, loss, N, C);
    else if (C <= 32) launch<32,0>(logits, true_labels, loss, N, C);
    else if (C % 4 != 0 || (reinterpret_cast<uintptr_t>(logits) & 15u))
        launch<32,1>(logits, true_labels, loss, N, C);
    else if (C <=  64) launch<16,4,1>(logits, true_labels, loss, N, C);
    else if (C <= 128) launch<32,4,1>(logits, true_labels, loss, N, C);
    else if (C <= 256) launch<32,4,2>(logits, true_labels, loss, N, C);
    else if (C <= 512) launch<32,4,4>(logits, true_labels, loss, N, C);
    else               launch<32,4,8>(logits, true_labels, loss, N, C);
}
