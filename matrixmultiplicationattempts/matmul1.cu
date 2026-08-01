#include <cuda_runtime.h>


#define BLOCK_THREADS 16
#define TILE_M 64
#define TILE_N 16
#define TILE_K 64

#define MICRO_TILE 4

__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
 const float* __restrict__ B, 
 float* __restrict__ C, 
 int M, 
 int N,
 int K) {
__shared__ float ds_A[TILE_M][TILE_N]; //64x16
__shared__ float ds_B[TILE_N][TILE_K]; //16x64

const int tx = threadIdx.x;
const int ty = threadIdx.y;

const int block_row = blockIdx.y * TILE_M;  
//Each block 
//handles 64 output rows. If blockIdx.y=0 these are 
//rows 0->63

const int block_col = blockIdx.x * TILE_K;
//Each block handles 64 output columns.
//If blockIdx.x=0 these are columns 0->63

const int local_row = ty * MICRO_TILE;
//Inside the 64-row output tile
//If ty=0 these are local rows 0->3

const int local_col = tx * MICRO_TILE;
//Inside the 64-column output tile.
//If tx=0 these are local columns 0->3 
const int global_row = block_row + local_row;

const int global_col = block_col + local_column;

float acc[MICRO_TILE][MICRO_TILE] = {};
//Register array holding this thread's 4 x 4 output values.


/*Divide the N dimension into phases of 16 elements.
     *
     * For example, when N = 35:
     *
     *     Phase 0 handles N indexes 0 through 15.
     *     Phase 1 handles N indexes 16 through 31.
     *     Phase 2 handles N indexes 32 through 34.
     *
     * The last phase is padded with zeros where necessary.*/
  for (
        int phase = 0;
        phase < (N + TILE_N - 1) / TILE_N;
        ++phase)
    {
        //Each thread loads four elements from matrix A.
        #pragma unroll
        for (int r = 0; r < MICRO_TILE; ++r)
        {


}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_THREADS, BLOCK_THREADS);
    //256 threads per block. 2 dim block with 16 threads in x
    //and 16 threads in y.
    dim3 blocksPerGrid((K + TILE_K - 1) / TILE_K,
                       (M + TILE_M - 1) / TILE_M);

    //Number of blocks in each direction.
//Each block computes: 64 rows, 64 columns

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
