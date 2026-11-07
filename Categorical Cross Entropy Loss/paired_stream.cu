#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAX_BLOCKS = 160;
constexpr unsigned FULL = 0xffffffffu;

// Scratch for sequential default-stream calls.
// Every invocation recomputes all partial sums and the final loss.
__device__ volatile float partials[MAX_BLOCKS];
__device__ unsigned completed = 0;

// Each warp's lane 0 supplies a subtotal.
// All threads in the block must call this function.
__device__ __forceinline__
void finish_sum(float subtotal, float* loss, float invN)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    __shared__ float warp_totals[WARPS];
    __shared__ int last;

    if (lane == 0)
        warp_totals[tid / 32] = subtotal;

    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;

        #pragma unroll
        for (int w = 0; w < WARPS; ++w)
            total += warp_totals[w];

        if (gridDim.x == 1) {
            *loss = total * invN;
        } else {
            partials[blockIdx.x] = total;

            // Publish this partial before announcing completion.
            __threadfence();

            last = (atomicInc(&completed, gridDim.x)
                    == gridDim.x - 1);
        }
    }

    if (gridDim.x == 1)
        return;

    __syncthreads();

    // No spinning: only the last completed block reduces the partials.
    if (last && tid < 32) {
        float total = 0.0f;

        for (int i = lane; i < gridDim.x; i += 32)
            total += partials[i];

        #pragma unroll
        for (int d = 16; d; d /= 2)
            total += __shfl_down_sync(FULL, total, d);

        if (lane == 0) {
            *loss = total * invN;
            completed = 0;
        }
    }
}

template<int GROUP>
__global__ __launch_bounds__(THREADS, 4)
void small_rows(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* loss,
    int N,
    int C,
    float invN)
{
    constexpr int ROWS = THREADS / GROUP;

    const int tid = threadIdx.x;
    const int lane = tid % GROUP;
    const unsigned mask =
        (FULL >> (32 - GROUP)) << ((tid & 31) - lane);

    float subtotal = 0.0f;

    for (int row = blockIdx.x * ROWS + tid / GROUP;
         row < N;
         row += gridDim.x * ROWS)
    {
        const int y = labels[row];

        if (static_cast<unsigned>(y) >= static_cast<unsigned>(C)) {
            if (lane == 0)
                subtotal = CUDART_NAN_F;
            continue;
        }

        const float z =
            lane < C ? logits[row * C + lane] : 0.0f;

        const float target =
            __shfl_sync(mask, z, y, GROUP);

        float sum = lane < C && lane != y
                  ? __expf(z - target)
                  : 0.0f;

        #pragma unroll
        for (int d = GROUP / 2; d; d /= 2)
            sum += __shfl_down_sync(mask, sum, d, GROUP);

        if (lane == 0)
            subtotal += sum > 0.5f
                      ? __logf(1.0f + sum)
                      : log1pf(sum);
    }

    #pragma unroll
    for (int d = 16; d >= GROUP; d /= 2)
        subtotal += __shfl_down_sync(FULL, subtotal, d);

    finish_sum(subtotal, loss, invN);
}

// A full warp processes two rows with independent accumulators.
// PACK=4 requires aligned rows.
// PACK=1 covers arbitrary row widths and alignment.
template<int PACK, int ROUNDS>
__global__ __launch_bounds__(THREADS, 4)
void paired_rows(
    const float* __restrict__ logits,
    const int* __restrict__ labels,
    float* loss,
    int N,
    int C,
    float invN)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;

    float subtotal = 0.0f;

    for (int row = 2 * (blockIdx.x * WARPS + warp);
         row < N;
         row += 2 * gridDim.x * WARPS)
    {
        const bool has_second = row + 1 < N;
        const int y0 = labels[row];
        const int y1 = has_second ? labels[row + 1] : 0;

        if (static_cast<unsigned>(y0) >= static_cast<unsigned>(C) ||
            (has_second &&
             static_cast<unsigned>(y1) >= static_cast<unsigned>(C)))
        {
            if (lane == 0)
                subtotal = CUDART_NAN_F;
            continue;
        }

        const float* r0 = logits + row * C;
        const float* r1 = has_second ? r0 + C : r0;

        const float target0 =
            lane == 0 ? r0[y0] : 0.0f;

        const float target1 =
            lane == 0 && has_second ? r1[y1] : 0.0f;

        float local0 = 0.0f;
        float local1 = 0.0f;

        if constexpr (PACK == 4) {
            const float4* p0 =
                reinterpret_cast<const float4*>(r0);

            const float4* p1 =
                reinterpret_cast<const float4*>(r1);

            float a0 = 0, b0 = 0, c0 = 0, d0 = 0;
            float a1 = 0, b1 = 0, c1 = 0, d1 = 0;

            #pragma unroll
            for (int j = 0; j < ROUNDS; ++j) {
                const int k = lane + j * 32;

                if (k < C / 4) {
                    // Stream the large logit matrix through L2.
                    const float4 v0 = __ldcg(p0 + k);

                    const float4 v1 = has_second
                        ? __ldcg(p1 + k)
                        : make_float4(0, 0, 0, 0);

                    // Independent chains for both rows.
                    a0 += __expf(v0.x);
                    a1 += __expf(v1.x);
                    b0 += __expf(v0.y);
                    b1 += __expf(v1.y);
                    c0 += __expf(v0.z);
                    c1 += __expf(v1.z);
                    d0 += __expf(v0.w);
                    d1 += __expf(v1.w);
                }
            }

            local0 = (a0 + b0) + (c0 + d0);
            local1 = (a1 + b1) + (c1 + d1);
        } else {
            for (int k = lane; k < C; k += 32) {
                const float v0 = __ldcg(r0 + k);
                const float v1 =
                    has_second ? __ldcg(r1 + k) : 0.0f;

                local0 += __expf(v0);
                local1 += __expf(v1);
            }
        }

        float sum0 = local0;
        float sum1 = local1;

        #pragma unroll
        for (int d = 16; d; d /= 2) {
            sum0 += __shfl_down_sync(FULL, sum0, d);
            sum1 += __shfl_down_sync(FULL, sum1, d);
        }

        #pragma unroll
        for (int which = 0; which < 2; ++which) {
            if (which == 1 && !has_second)
                continue;

            const float* r = which == 0 ? r0 : r1;
            const int y = which == 0 ? y0 : y1;
            const float target = which == 0 ? target0 : target1;
            float local = which == 0 ? local0 : local1;
            const float sum = which == 0 ? sum0 : sum1;

            float value =
                lane == 0 ? __logf(sum) - target : 0.0f;

            const int repair =
                __shfl_sync(FULL, int(value < 0.5f), 0);

            if (repair) {
                // Avoid cancellation when the correct class dominates.
                // Only one lane needs to recompute its local sum.
                if (lane == (y / PACK) % 32) {
                    local = 0.0f;

                    #pragma unroll 1
                    for (int k = lane * PACK;
                         k < C;
                         k += 32 * PACK)
                    {
                        #pragma unroll
                        for (int j = 0; j < PACK; ++j) {
                            if (k + j != y)
                                local += __expf(r[k + j]);
                        }
                    }
                }

                #pragma unroll
                for (int d = 16; d; d /= 2)
                    local += __shfl_down_sync(FULL, local, d);

                if (lane == 0)
                    value = log1pf(local * __expf(-target));
            }

            if (lane == 0)
                subtotal += value;
        }
    }

    finish_sum(subtotal, loss, invN);
}

template<int GROUP>
void launch_small(
    const float* x,
    const int* y,
    float* loss,
    int N,
    int C)
{
    constexpr int ROWS = THREADS / GROUP;
    const int needed = (N + ROWS - 1) / ROWS;
    const int blocks = needed < MAX_BLOCKS ? needed : MAX_BLOCKS;

    small_rows<GROUP><<<blocks, THREADS>>>(
        x, y, loss, N, C, 1.0f / N);
}

template<int PACK, int ROUNDS = 0>
void launch_pair(
    const float* x,
    const int* y,
    float* loss,
    int N,
    int C)
{
    const int needed = (N + 2 * WARPS - 1) / (2 * WARPS);
    const int blocks = needed < MAX_BLOCKS ? needed : MAX_BLOCKS;

    paired_rows<PACK, ROUNDS><<<blocks, THREADS>>>(
        x, y, loss, N, C, 1.0f / N);
}

} // namespace

extern "C" void solve(
    const float* logits,
    const int* true_labels,
    float* loss,
    int N,
    int C)
{
    if (C <= 2)
        launch_small<2>(logits, true_labels, loss, N, C);
    else if (C <= 4)
        launch_small<4>(logits, true_labels, loss, N, C);
    else if (C <= 8)
        launch_small<8>(logits, true_labels, loss, N, C);
    else if (C <= 16)
        launch_small<16>(logits, true_labels, loss, N, C);
    else if (C <= 32)
        launch_small<32>(logits, true_labels, loss, N, C);
    else if ((C & 3) ||
             (reinterpret_cast<uintptr_t>(logits) & 15u))
        launch_pair<1>(logits, true_labels, loss, N, C);
    else if (C <= 128)
        launch_pair<4, 1>(logits, true_labels, loss, N, C);
    else if (C <= 256)
        launch_pair<4, 2>(logits, true_labels, loss, N, C);
    else if (C <= 512)
        launch_pair<4, 4>(logits, true_labels, loss, N, C);
    else
        launch_pair<4, 8>(logits, true_labels, loss, N, C);
}
