#include <cuda_runtime.h>
#define TILE_DIM 16
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) 
{
 __shared__ float ds_A[TILE_DIM][TILE_DIM];
    __shared__ float ds_B[TILE_DIM][TILE_DIM];
 int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Output target tracking coordinates
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    float sum = 0.0f;
// 2. Loop through all needed tiles to span the inner K dimension
    for (int phase = 0; phase < (K + TILE_DIM - 1) / TILE_DIM; ++phase) {

        // --- LOAD MATRIX A INTO SHARED TILES ---
        // Crucial Check: Ensure row is inside M AND global column index is inside K
        if (row < M && (phase * TILE_DIM + tx) < K) {
            ds_A[ty][tx] = A[row * K + phase * TILE_DIM + tx];
        } else {
            ds_A[ty][tx] = 0.0f; // Safe padding to prevent trash values in dot product
        }

        // --- LOAD MATRIX B INTO SHARED TILES ---
        // Crucial Check: Ensure global row index is inside K AND col is inside N
        if ((phase * TILE_DIM + ty) < K && col < N) {
            ds_B[ty][tx] = B[(phase * TILE_DIM + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0.0f; // Safe padding
        }

        // 3. Sync to guarantee tiles are fully populated before math begins
        __syncthreads();

        // 4. Dot product multiplication on local shared memory
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += ds_A[ty][i] * ds_B[i][tx];
        }

        // 5. Sync to ensure math finishes before loading the next phase's tile data
        __syncthreads();
    }

    // 6. Final Target Write Boundary Check
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
     dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
// dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
