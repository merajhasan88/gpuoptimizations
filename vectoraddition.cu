#include <cuda_runtime.h>

__global__ void vector_add(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
if(i<N){
    C[i]=A[i]+B[i];
    //C[i] = __ldg(&A[i]) + __ldg(&B[i]);
}

//  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//     if (i + 3 < N) {
//         float4 a = reinterpret_cast<const float4*>(A)[i / 4];
//         float4 b = reinterpret_cast<const float4*>(B)[i / 4];
//         float4 res;
//         res.x = a.x + b.x; res.y = a.y + b.y; res.z = a.z + b.z; res.w = a.w + b.w;
//         reinterpret_cast<float4*>(C)[i / 4] = res;
//     } else {
//       for (int j = i; j < N; j++) {
//         C[j] = A[j] + B[j];
//     }
//     }

//-----------------------------------

// int idx = threadIdx.x + blockIdx.x * blockDim.x;
// int stride = blockDim.x * gridDim.x;
    

//     for (int i = idx; i < N; i += stride)
//     {
//         C[i] = __ldg(&A[i]) + __ldg(&B[i]);
//     }

//-----------------------------------------
//   int vec_size = size / 4;
//     if(idx<vec_size){ 
//       const  float4* a = reinterpret_cast<const float4*>(A);
//     const float4* b = reinterpret_cast<const float4*>(B);
//    float4* c = reinterpret_cast<float4*>(C);
//     for (int i = idx; i < size/4; i += stride) {

//         float4 av = a[i];
//         float4 bv = b[i];
//     c[i] = make_float4(
//             av.x + bv.x,
//             av.y + bv.y,
//             av.z + bv.z,
//             av.w + bv.w
//         );
//         }
//         //C = c.x + c.y + c.z + c.w;
//     }
// int tail_idx = vec_size * 4 + idx;
// if (tail_idx < size) {
//     C[tail_idx] = A[tail_idx] + B[tail_idx];
// }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int threadsPerBlock = 256;
   // int blocksPerGrid = (N + (threadsPerBlock * 4) - 1) / (threadsPerBlock * 4);
    //int blocksPerGrid = (N/4 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (N+threadsPerBlock - 1) / threadsPerBlock;
//int blocksPerGrid = 160;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
