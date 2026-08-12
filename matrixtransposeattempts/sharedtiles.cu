#include <cuda_runtime.h>

#define TILE_DIM   32
#define BLOCK_ROWS 8

__global__ void matrix_transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    // +1 avoids shared-memory bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // STEP 1: Read input row-wise.
    // Global reads are coalesced.
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] =
                input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Swap block coordinates for transpose
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // STEP 2: Read shared memory transposed,
    // then write output row-wise.
    // Global writes are now coalesced.
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            output[(y + j) * rows + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);  // 32 x 8 = 256 threads

    dim3 blocksPerGrid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, rows, cols
    );

    cudaDeviceSynchronize();
}
