#include <cuda_runtime.h>
#include <stddef.h>

/*
 * Thread-block dimensions:
 *
 * 16 × 16 = 256 threads per block.
 */
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

/*
 * Each thread calculates an 8 × 8 microtile of C.
 */
constexpr int MICRO_M = 8;
constexpr int MICRO_N = 8;

/*
 * One block calculates:
 *
 * 16 thread rows × 8 output rows/thread    = 128 rows
 * 16 thread cols × 8 output columns/thread = 128 columns
 */
constexpr int BLOCK_M = THREADS_Y * MICRO_M; // 128
constexpr int BLOCK_N = THREADS_X * MICRO_N; // 128

/*
 * Process the reduction dimension N in chunks of 16.
 */
constexpr int BLOCK_K = 16;

/*
 * Total threads in one block.
 */
constexpr int BLOCK_THREADS = THREADS_X * THREADS_Y; // 256

/*
 * Number of useful elements in each shared-memory tile.
 */
constexpr int A_TILE_ELEMENTS = BLOCK_M * BLOCK_K; // 128 × 16 = 2048
constexpr int B_TILE_ELEMENTS = BLOCK_K * BLOCK_N; // 16 × 128 = 2048


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
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    /*
     * Convert the two-dimensional thread coordinate into a single
     * thread ID from 0 through 255.
     */
    const int tid = ty * THREADS_X + tx;

    /*
     * Starting position of this block's 128 × 128 output tile
     * inside the complete matrix C.
     */
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    /*
     * Starting position of this thread's 8 × 8 microtile
     * inside the block's output tile.
     */
    const int local_row = ty * MICRO_M;
    const int local_col = tx * MICRO_N;

    /*
     * Starting position of this thread's 8 × 8 microtile
     * inside the complete output matrix C.
     */
    const int global_row = block_row + local_row;
    const int global_col = block_col + local_col;

    /*
     * Shared A tile:
     *
     * 128 output rows × 16 reduction elements.
     */
    __shared__ float shared_A[BLOCK_M][BLOCK_K];

    /*
     * Shared B tile:
     *
     * 16 reduction elements × 128 output columns.
     */
    __shared__ float shared_B[BLOCK_K][BLOCK_N];

    /*
     * Each thread computes an 8 × 8 microtile:
     *
     * 8 × 8 = 64 output values.
     *
     * The compiler will normally store these accumulators in registers.
     */
    float acc[MICRO_M][MICRO_N] = {0.0f};

    /*
     * Register fragments.
     *
     * For one reduction position, the thread loads:
     *
     * 8 values from shared_A
     * 8 values from shared_B
     *
     * Those values are reused to calculate 64 multiply-adds.
     */
    float reg_A[MICRO_M];
    float reg_B[MICRO_N];

    /*
     * Process N in chunks of BLOCK_K = 16.
     */
    const int number_of_phases =
        (N + BLOCK_K - 1) / BLOCK_K;

    for (int phase = 0; phase < number_of_phases; ++phase)
    {
        /*
         * ---------------------------------------------------------
         * Cooperatively load the A tile
         * ---------------------------------------------------------
         *
         * The A tile contains:
         *
         * 128 × 16 = 2048 values.
         *
         * There are 256 threads, so each thread loads:
         *
         * 2048 / 256 = 8 values.
         */
        #pragma unroll
        for (int load = 0;
             load < A_TILE_ELEMENTS / BLOCK_THREADS;
             ++load)
        {
            /*
             * Linear position in the 128 × 16 shared-memory tile.
             */
            const int index =
                tid + load * BLOCK_THREADS;

            /*
             * Convert the linear index into:
             *
             * shared_row: 0 through 127
             * shared_col: 0 through 15
             */
            const int shared_row = index / BLOCK_K;
            const int shared_col = index % BLOCK_K;

            /*
             * Corresponding global-memory position in A.
             */
            const int a_row = block_row + shared_row;
            const int a_col =
                phase * BLOCK_K + shared_col;

            /*
             * Load a valid A value or zero-pad the matrix boundary.
             */
            if (a_row < M && a_col < N)
            {
                shared_A[shared_row][shared_col] =
                    A[static_cast<size_t>(a_row) * N + a_col];
            }
            else
            {
                shared_A[shared_row][shared_col] = 0.0f;
            }
        }

        /*
         * ---------------------------------------------------------
         * Cooperatively load the B tile
         * ---------------------------------------------------------
         *
         * The B tile contains:
         *
         * 16 × 128 = 2048 values.
         *
         * Each of the 256 threads again loads eight values.
         */
        #pragma unroll
        for (int load = 0;
             load < B_TILE_ELEMENTS / BLOCK_THREADS;
             ++load)
        {
            /*
             * Linear position in the 16 × 128 shared-memory tile.
             */
            const int index =
                tid + load * BLOCK_THREADS;

            /*
             * Convert the linear index into:
             *
             * shared_row: 0 through 15
             * shared_col: 0 through 127
             */
            const int shared_row = index / BLOCK_N;
            const int shared_col = index % BLOCK_N;

            /*
             * Corresponding global-memory position in B.
             */
            const int b_row =
                phase * BLOCK_K + shared_row;

            const int b_col =
                block_col + shared_col;

            /*
             * Load a valid B value or zero-pad the matrix boundary.
             */
            if (b_row < N && b_col < K)
            {
                shared_B[shared_row][shared_col] =
                    B[static_cast<size_t>(b_row) * K + b_col];
            }
            else
            {
                shared_B[shared_row][shared_col] = 0.0f;
            }
        }

        /*
         * Every thread must finish loading before any thread begins
         * reading the shared-memory tiles.
         */
        __syncthreads();

        /*
         * ---------------------------------------------------------
         * Compute the contribution from this reduction tile
         * ---------------------------------------------------------
         */
        #pragma unroll
        for (int i = 0; i < BLOCK_K; ++i)
        {
            /*
             * Load eight A values from shared memory into registers.
             *
             * These are the eight output rows owned by this thread.
             */
            #pragma unroll
            for (int r = 0; r < MICRO_M; ++r)
            {
                reg_A[r] =
                    shared_A[local_row + r][i];
            }

            /*
             * Load eight B values from shared memory into registers.
             *
             * These are the eight output columns owned by this thread.
             */
            #pragma unroll
            for (int c = 0; c < MICRO_N; ++c)
            {
                reg_B[c] =
                    shared_B[i][local_col + c];
            }

            /*
             * Form an 8 × 8 outer product.
             *
             * Eight A values and eight B values produce:
             *
             * 8 × 8 = 64 multiply-add operations.
             */
            #pragma unroll
            for (int r = 0; r < MICRO_M; ++r)
            {
                #pragma unroll
                for (int c = 0; c < MICRO_N; ++c)
                {
                    acc[r][c] +=
                        reg_A[r] * reg_B[c];
                }
            }
        }

        /*
         * Ensure every thread has finished reading the current
         * shared-memory tiles before the next phase overwrites them.
         */
        __syncthreads();
    }

    /*
     * ---------------------------------------------------------
     * Store this thread's 8 × 8 microtile in C
     * ---------------------------------------------------------
     */
    #pragma unroll
    for (int r = 0; r < MICRO_M; ++r)
    {
        const int output_row = global_row + r;

        #pragma unroll
        for (int c = 0; c < MICRO_N; ++c)
        {
            const int output_col = global_col + c;

            if (output_row < M && output_col < K)
            {
                C[static_cast<size_t>(output_row) * K + output_col] =
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
    dim3 threadsPerBlock(
        THREADS_X,
        THREADS_Y
    );

    /*
     * Each block calculates a 128 × 128 output tile.
     */
    dim3 blocksPerGrid(
        (K + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
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
