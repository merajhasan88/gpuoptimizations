#include <cuda_runtime.h>
#include <stddef.h>

#define TILE_SIZE 16
#define MICRO_TILE_SIZE 8
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int TILE_K  = 16;
constexpr int BLOCK_THREADS = 256;
constexpr int A_TILE_ELEMENTS = BLOCK_M * TILE_K;
constexpr int B_TILE_ELEMENTS = TILE_K * BLOCK_N;

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
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
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
        #pragma unroll
        for (int load = 0;
            load < A_TILE_ELEMENTS / BLOCK_THREADS;
            ++load)
        {
            /*
            * Each thread receives one position during each loop iteration.
            *
            * Iteration 0:
            * threads load positions 0 through 255
            *
            * Iteration 1:
            * threads load positions 256 through 511
            *
            * And so on.
            */
            int index = tid + load * BLOCK_THREADS;

            /*
            * Convert the one-dimensional tile position into a
            * shared-memory row and column.
            *
            * shared_A has shape:
            *
            *     128 rows × 16 columns
            */
            int shared_row = index / TILE_K;
            int shared_col = index % TILE_K;

            /*
            * Convert the shared-memory position into a global-memory
            * position in matrix A.
            */
            int a_row = block_row + shared_row;
            int a_col = phase * TILE_K + shared_col;

            if (a_row < M && a_col < N)
            {
                shared_A[shared_row][shared_col] =
                    A[(size_t)a_row * N + a_col];
            }
            else
            {
                shared_A[shared_row][shared_col] = 0.0f;
            }
        }
        #pragma unroll
        for (int load = 0;
            load < B_TILE_ELEMENTS / BLOCK_THREADS;
            ++load)
        {
            int index = tid + load * BLOCK_THREADS;

            /*
            * shared_B has shape:
            *
            *     16 rows × 128 columns
            */
            int shared_row = index / BLOCK_N;
            int shared_col = index % BLOCK_N;

            /*
            * Convert the shared-memory position into a global-memory
            * position in matrix B.
            */
            int b_row = phase * TILE_K + shared_row;
            int b_col = block_col + shared_col;

            if (b_row < N && b_col < K)
            {
                shared_B[shared_row][shared_col] =
                    B[(size_t)b_row * K + b_col];
            }
            else
            {
                shared_B[shared_row][shared_col] = 0.0f;
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
