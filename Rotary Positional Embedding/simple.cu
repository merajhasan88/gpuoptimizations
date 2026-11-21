#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int THREADS = 256;
constexpr int ROWS = 32;  // Two waves of 16 rows per block.

// Each packet owns four columns and their four partners 64 columns later.
struct Packet { float4 lo, hi, c, s; };

__device__ __forceinline__ Packet fetch(const float4* q, const float4* c,
                                        const float4* s, int i) {
    return {__ldcg(q+i), __ldcg(q+i+16), __ldcg(c+i), __ldcg(s+i)};
}

__device__ __forceinline__ void rotate(float a, float b, float c, float s,
                                       float& lo, float& hi) {
    lo = fmaf(-b, s, a*c);
    hi = fmaf( a, s, b*c);
}

__device__ __forceinline__ void write_packet(float4* out, int i, Packet v) {
    float4 lo, hi;
    rotate(v.lo.x, v.hi.x, v.c.x, v.s.x, lo.x, hi.x);
    rotate(v.lo.y, v.hi.y, v.c.y, v.s.y, lo.y, hi.y);
    rotate(v.lo.z, v.hi.z, v.c.z, v.s.z, lo.z, hi.z);
    rotate(v.lo.w, v.hi.w, v.c.w, v.s.w, lo.w, hi.w);
    out[i] = lo;
    out[i+16] = hi;
}

template<bool Full>
__global__ void rope128(const float4* q, const float4* c, const float4* s,
                        float4* out, int M) {
    // Sixteen lanes span one 64-float half-row; one warp handles two rows.
    int t = threadIdx.x;
    int row = blockIdx.x*ROWS + (t >> 4);
    if (!Full && row >= M) return;

    int i = row*32 + (t & 15);  // Indices here are in float4 units.
    constexpr int next = 16*32;
    Packet a = fetch(q,c,s,i);

    if (Full || row+16 < M) {
        Packet b = fetch(q,c,s,i+next);
        write_packet(out,i,a);
        write_packet(out,i+next,b);
    } else {
        write_packet(out,i,a);
    }
}

// Any even D, including halves too short or unaligned for a float4 load.
__global__ void rope_pairs(const float* q, const float* c, const float* s,
                           float* out, int pairs, int half) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= pairs) return;

    int i = p + (p/half)*half;  // row*(2*half) + column
    float a=q[i], b=q[i+half], co=c[i], si=s[i];
    float lo, hi;
    rotate(a,b,co,si,lo,hi);
    out[i]=lo;
    out[i+half]=hi;
}
} // namespace

extern "C" void solve(float* Q, float* cos, float* sin, float* output,
                      int M, int D) {
    uintptr_t addr = uintptr_t(Q) | uintptr_t(cos) | uintptr_t(sin) | uintptr_t(output);
    if (D == 128 && (addr & 15u) == 0) {
        auto q = reinterpret_cast<const float4*>(Q);
        auto c = reinterpret_cast<const float4*>(cos);
        auto s = reinterpret_cast<const float4*>(sin);
        auto o = reinterpret_cast<float4*>(output);
        int blocks = (M+ROWS-1)/ROWS;

        if (M % ROWS == 0)
            rope128<true><<<blocks,THREADS>>>(q,c,s,o,M);
        else
            rope128<false><<<blocks,THREADS>>>(q,c,s,o,M);
    } else {
        int half = D/2, pairs = M*half;
        rope_pairs<<<(pairs+THREADS-1)/THREADS,THREADS>>>(Q,cos,sin,output,pairs,half);
    }
}
