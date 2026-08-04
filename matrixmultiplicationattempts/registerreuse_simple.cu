#include <cuda_runtime.h>

#define TILE_SIZE 16

// Each block calculates 16 rows and 32 columns of C.
#define OUTPUT_TILE_COLUMNS (TILE_SIZE * 2)

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    /*
     * Thread coordinates inside the block.
     *
     * tx: 0 through 15
     * ty: 0 through 15
     */
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /*
     * Each block calculates 16 rows of C.
     *
     * Each thread calculates one of those rows.
     */
    int row = blockIdx.y * TILE_SIZE + ty;

    /*
     * Each block calculates 32 columns of C.
     */
    int block_col = blockIdx.x * OUTPUT_TILE_COLUMNS;

    /*
     * Each thread calculates two columns.
     *
     * Thread tx=0 calculates block columns 0 and 16.
     * Thread tx=1 calculates block columns 1 and 17.
     * Thread tx=2 calculates block columns 2 and 18.
     * ...
     * Thread tx=15 calculates block columns 15 and 31.
     */
    int col0 = block_col + tx;
    int col1 = block_col + tx + TILE_SIZE;

    /*
     * A shared-memory tile still contains:
     *
     * 16 rows × 16 reduction elements.
     */
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];

    /*
     * B must now contain 32 output columns:
     *
     * 16 reduction elements × 32 columns.
     */
    __shared__ float shared_B[TILE_SIZE][OUTPUT_TILE_COLUMNS];

    /*
     * Two output accumulators.
     *
     * These are normally kept in registers.
     */
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    /*
     * Process N in chunks of 16.
     */
    int number_of_phases =
        (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int phase = 0; phase < number_of_phases; ++phase)
    {
        /*
         * ---------------------------------------------------------
         * Load the A tile
         * ---------------------------------------------------------
         *
         * Every thread loads one A value.
         */
        int a_col = phase * TILE_SIZE + tx;

        if (row < M && a_col < N)
        {
            shared_A[ty][tx] =
                A[row * N + a_col];
        }
        else
        {
            shared_A[ty][tx] = 0.0f;
        }

        /*
         * ---------------------------------------------------------
         * Load the B tile
         * ---------------------------------------------------------
         *
         * Every thread loads two B values because the block now
         * calculates 32 output columns.
         */
        int b_row = phase * TILE_SIZE + ty;

        /*
         * Load the first half of the B tile:
         *
         * shared columns 0 through 15.
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

        /*
         * Load the second half of the B tile:
         *
         * shared columns 16 through 31.
         */
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
         * Wait until the entire A and B tiles have been loaded.
         */
        __syncthreads();

        /*
         * Multiply the current shared-memory tiles.
         */
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            /*
             * Load this A value once.
             *
             * The compiler will normally keep it in a register.
             */
            float a = shared_A[ty][i];

            /*
             * Reuse the same A value for two output columns.
             */
            sum0 += a * shared_B[i][tx];

            sum1 += a * shared_B[i][tx + TILE_SIZE];
        }

        /*
         * Wait before any thread overwrites the shared arrays for
         * the next phase.
         */
        __syncthreads();
    }

    /*
     * Write the first result.
     */
    if (row < M && col0 < K)
    {
        C[row * K + col0] = sum0;
    }

    /*
     * Write the second result.
     */
    if (row < M && col1 < K)
    {
        C[row * K + col1] = sum1;
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
     * The block still has 16 × 16 = 256 threads.
     */
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    /*
     * Each block now covers:
     *
     * 16 rows × 32 columns.
     *
     * Therefore grid.x must divide K by 32 rather than 16.
     */
    dim3 blocksPerGrid(
        (K + OUTPUT_TILE_COLUMNS - 1) / OUTPUT_TILE_COLUMNS,
        (M + TILE_SIZE - 1) / TILE_SIZE
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
