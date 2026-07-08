#include <cuda_runtime.h>
#define TILE_DIM 16
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) 
{
 __shared__ float ds_A[TILE_DIM][TILE_DIM];
    __shared__ float ds_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // LeetGPU Mapping: row tracks M (Rows of A & C), col tracks K (Columns of B & C)
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;
    float sum = 0.0f;

    // LeetGPU Mapping: The inner shared dimension loop is bounded by N
    for (int phase = 0; phase < (N + TILE_DIM - 1) / TILE_DIM; ++phase) {

        // --- LOAD MATRIX A (M x N) ---
        // Stride is N. Row must be < M, Col index must be < N.
        if (row < M && (phase * TILE_DIM + tx) < N) {
            ds_A[ty][tx] = A[row * N + phase * TILE_DIM + tx];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // --- LOAD MATRIX B (N x K) ---
        // Stride is K. Row index must be < N, Col must be < K.
        if ((phase * TILE_DIM + ty) < N && col < K) {
            ds_B[ty][tx] = B[(phase * TILE_DIM + ty) * K + col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Dot product over the loaded tiles
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += ds_A[ty][i] * ds_B[i][tx];
        }

        __syncthreads();
    }

    // --- WRITE TO DESTINATION MATRIX C (M x K) ---
    // Stride is K.
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
     dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
 //dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
  //                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
