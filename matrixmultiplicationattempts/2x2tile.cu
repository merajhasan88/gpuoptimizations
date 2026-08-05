#include <cuda_runtime.h>

#define TILE_SIZE 16
#define OUTPUT_TILE_SIZE 32

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    /*
     * Thread coordinates inside the 16 × 16 block.
     */
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /*
     * Each block calculates a 32 × 32 output tile.
     */
    int block_row = blockIdx.y * OUTPUT_TILE_SIZE;
    int block_col = blockIdx.x * OUTPUT_TILE_SIZE;

    /*
     * Each thread computes two rows.
     *
     * Thread ty=0 handles local rows 0 and 16.
     * Thread ty=1 handles local rows 1 and 17.
     * ...
     */
    int row0 = block_row + ty;
    int row1 = block_row + ty + TILE_SIZE;

    /*
     * Each thread computes two columns.
     *
     * Thread tx=0 handles local columns 0 and 16.
     * Thread tx=1 handles local columns 1 and 17.
     * ...
     */
    int col0 = block_col + tx;
    int col1 = block_col + tx + TILE_SIZE;

    /*
     * A must contain 32 output rows and 16 reduction elements.
     *
     * Shape:
     *
     * 32 rows × 16 values of N
     */
    __shared__ float shared_A[OUTPUT_TILE_SIZE][TILE_SIZE];

    /*
     * B must contain 16 reduction elements and 32 output columns.
     *
     * Shape:
     *
     * 16 values of N × 32 columns
     */
    __shared__ float shared_B[TILE_SIZE][OUTPUT_TILE_SIZE];

    /*
     * Four output accumulators.
     */
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    /*
     * Process the N dimension in chunks of 16.
     */
    int number_of_phases =
        (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int phase = 0; phase < number_of_phases; ++phase)
    {
        /*
         * Global column in A for the current phase.
         */
        int a_col = phase * TILE_SIZE + tx;

        /*
         * Each thread loads two A values:
         *
         * one for row0
         * one for row1
         */
        if (row0 < M && a_col < N)
        {
            shared_A[ty][tx] =
                A[row0 * N + a_col];
        }
        else
        {
            shared_A[ty][tx] = 0.0f;
        }

        if (row1 < M && a_col < N)
        {
            shared_A[ty + TILE_SIZE][tx] =
                A[row1 * N + a_col];
        }
        else
        {
            shared_A[ty + TILE_SIZE][tx] = 0.0f;
        }

        /*
         * Global row in B for the current phase.
         */
        int b_row = phase * TILE_SIZE + ty;

        /*
         * Each thread loads two B values:
         *
         * one for col0
         * one for col1
         */
        if (b_row < N && col0 < K)
        {
            shared_B[ty][tx] =
                B[b_row * K + col0];
        }
        else
        {
            shared_B[ty][tx] = 0.0f;
        }

        if (b_row < N && col1 < K)
        {
            shared_B[ty][tx + TILE_SIZE] =
                B[b_row * K + col1];
        }
        else
        {
            shared_B[ty][tx + TILE_SIZE] = 0.0f;
        }

        /*
         * Wait until all 32 × 16 A values and all 16 × 32 B values
         * have been loaded.
         */
        __syncthreads();

        /*
         * Calculate the contribution from the current 16-element
         * reduction tile.
         */
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            /*
             * Load two A values and two B values.
             */
            float a0 = shared_A[ty][i];
            float a1 = shared_A[ty + TILE_SIZE][i];

            float b0 = shared_B[i][tx];
            float b1 = shared_B[i][tx + TILE_SIZE];

            /*
             * Four multiplications from four loaded values.
             *
             * Each A value is reused twice.
             * Each B value is reused twice.
             */
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        /*
         * Ensure all threads have finished using the current tiles
         * before shared memory is overwritten for the next phase.
         */
        __syncthreads();
    }

    /*
     * Store the four output values.
     */
    if (row0 < M && col0 < K)
    {
        C[row0 * K + col0] = sum00;
    }

    if (row0 < M && col1 < K)
    {
        C[row0 * K + col1] = sum01;
    }

    if (row1 < M && col0 < K)
    {
        C[row1 * K + col0] = sum10;
    }

    if (row1 < M && col1 < K)
    {
        C[row1 * K + col1] = sum11;
    }
}


// A, B, and C are device pointers.
extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    /*
     * 16 × 16 = 256 threads.
     */
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    /*
     * Each block computes 32 × 32 output values.
     */
    dim3 blocksPerGrid(
        (K + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE,
        (M + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE
    );

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );

    cudaDeviceSynchronize();
}
