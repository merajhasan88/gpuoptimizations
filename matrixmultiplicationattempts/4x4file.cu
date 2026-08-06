#include <cuda_runtime.h>
#include <stddef.h>

#define TILE_SIZE 16
#define MICRO_TILE_SIZE 4

// 16 threads × 4 outputs per thread = 64 output positions.
#define OUTPUT_TILE_SIZE (TILE_SIZE * MICRO_TILE_SIZE)

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
     * Each block calculates a 64 × 64 tile of C.
     */
    int block_row = blockIdx.y * OUTPUT_TILE_SIZE;
    int block_col = blockIdx.x * OUTPUT_TILE_SIZE;

    /*
     * Each thread calculates a 4 × 4 microtile.
     *
     * ty determines which group of four rows this thread owns.
     * tx determines which group of four columns this thread owns.
     */
    int local_row = ty * MICRO_TILE_SIZE;
    int local_col = tx * MICRO_TILE_SIZE;

    /*
     * First global row and column calculated by this thread.
     */
    int global_row = block_row + local_row;
    int global_col = block_col + local_col;

    /*
     * Shared A tile:
     *
     * 64 output rows × 16 reduction elements.
     */
    __shared__ float shared_A[OUTPUT_TILE_SIZE][TILE_SIZE];

    /*
     * Shared B tile:
     *
     * 16 reduction elements × 64 output columns.
     */
    __shared__ float shared_B[TILE_SIZE][OUTPUT_TILE_SIZE];

    /*
     * Each thread calculates 16 output values.
     *
     * acc[r][c] represents:
     *
     * C[global_row + r][global_col + c]
     */
    float acc[MICRO_TILE_SIZE][MICRO_TILE_SIZE] = {0.0f};

    /*
     * Process N in chunks of 16.
     */
    int number_of_phases =
        (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int phase = 0; phase < number_of_phases; ++phase)
    {
        /*
         * ---------------------------------------------------------
         * Load a 64 × 16 chunk of A
         * ---------------------------------------------------------
         *
         * Each thread loads four A values.
         *
         * All four values come from different rows but use the same
         * column in the current reduction tile.
         */
        int a_col = phase * TILE_SIZE + tx;

        #pragma unroll
        for (int r = 0; r < MICRO_TILE_SIZE; ++r)
        {
            int a_row = global_row + r;

            if (a_row < M && a_col < N)
            {
                shared_A[local_row + r][tx] =
                    A[(size_t)a_row * N + a_col];
            }
            else
            {
                shared_A[local_row + r][tx] = 0.0f;
            }
        }

        /*
         * ---------------------------------------------------------
         * Load a 16 × 64 chunk of B
         * ---------------------------------------------------------
         *
         * Each thread loads four B values.
         *
         * All four values come from the same row but use different
         * output columns.
         */
        int b_row = phase * TILE_SIZE + ty;

        #pragma unroll
        for (int c = 0; c < MICRO_TILE_SIZE; ++c)
        {
            int b_col = global_col + c;

            if (b_row < N && b_col < K)
            {
                shared_B[ty][local_col + c] =
                    B[(size_t)b_row * K + b_col];
            }
            else
            {
                shared_B[ty][local_col + c] = 0.0f;
            }
        }

        /*
         * Wait until every thread has finished loading its four
         * A values and four B values.
         */
        __syncthreads();

        /*
         * Process the 16 reduction elements in this phase.
         */
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            /*
             * For this value of i, the thread uses:
             *
             * four values from A
             * four values from B
             *
             * These produce 4 × 4 = 16 multiplications.
             */
            #pragma unroll
            for (int r = 0; r < MICRO_TILE_SIZE; ++r)
            {
                /*
                 * Load one A value and reuse it across four columns.
                 */
                float a = shared_A[local_row + r][i];

                #pragma unroll
                for (int c = 0; c < MICRO_TILE_SIZE; ++c)
                {
                    /*
                     * Each B value is also reused across four rows
                     * by the surrounding loop.
                     */
                    acc[r][c] +=
                        a * shared_B[i][local_col + c];
                }
            }
        }

        /*
         * Wait until every thread has finished reading the current
         * shared tiles before they are overwritten by the next phase.
         */
        __syncthreads();
    }

    /*
     * Write this thread's 4 × 4 output microtile to C.
     */
    #pragma unroll
    for (int r = 0; r < MICRO_TILE_SIZE; ++r)
    {
        int output_row = global_row + r;

        #pragma unroll
        for (int c = 0; c < MICRO_TILE_SIZE; ++c)
        {
            int output_col = global_col + c;

            if (output_row < M && output_col < K)
            {
                C[(size_t)output_row * K + output_col] =
                    acc[r][c];
            }
        }
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
     * 16 × 16 = 256 threads per block.
     */
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    /*
     * Each block now calculates a 64 × 64 tile of C.
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
