#include <cuda_runtime.h>

#define TILE_DIM   32
#define BLOCK_ROWS 16

__global__ void transpose_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            tile[ty + j][tx] =
                input[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + tx;
    y = blockIdx.x * TILE_DIM + ty;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            output[(y + j) * rows + x] =
                tile[tx][ty + j];
        }
    }
}

extern "C" void solve(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    dim3 block(32, BLOCK_ROWS);

    dim3 grid(
        (cols + 31) / 32,
        (rows + 31) / 32
    );

    transpose_kernel<<<grid, block>>>(
        input, output, rows, cols
    );

    cudaDeviceSynchronize();
}
