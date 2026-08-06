#include <cuda_runtime.h>
#include <stddef.h>

/*
 * Thread-block dimensions:
 *
 * 16 × 16 = 256 threads per block.
 */
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int VECTOR_WIDTH = 4;

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
constexpr int A_TILE_VECTORS =
    A_TILE_ELEMENTS / VECTOR_WIDTH; // 2048 / 4 = 512

constexpr int B_TILE_VECTORS =
    B_TILE_ELEMENTS / VECTOR_WIDTH; // 2048 / 4 = 512

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
    //const int local_col = tx * MICRO_N;

    /*
     * Starting position of this thread's 8 × 8 microtile
     * inside the complete output matrix C.
     */
    const int global_row = block_row + local_row;
    //const int global_col = block_col + local_col;

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
     load < A_TILE_VECTORS / BLOCK_THREADS;
     ++load)
{
    /*
     * This is an index into the tile measured in float4 groups,
     * not individual floats.
     */
    const int vector_index =
        tid + load * BLOCK_THREADS;

    /*
     * A shared-memory row contains:
     *
     * BLOCK_K / 4 = 16 / 4 = 4 float4 groups.
     */
    const int vectors_per_row =
        BLOCK_K / VECTOR_WIDTH;

    const int shared_row =
        vector_index / vectors_per_row;

    const int vector_col =
        vector_index % vectors_per_row;

    /*
     * Convert the float4 column into a normal float column.
     *
     * vector_col 0 -> float column 0
     * vector_col 1 -> float column 4
     * vector_col 2 -> float column 8
     * vector_col 3 -> float column 12
     */
    const int shared_col =
        vector_col * VECTOR_WIDTH;

    const int a_row =
        block_row + shared_row;

    const int a_col =
        phase * BLOCK_K + shared_col;

    /*
     * The vectorized path requires:
     *
     * 1. A valid matrix row.
     * 2. Four valid consecutive columns.
     * 3. Each matrix row to begin at a 16-byte-aligned address.
     *
     * Since cudaMalloc aligns A and a_col is a multiple of four,
     * N being divisible by four keeps every row aligned.
     */
    if (a_row < M &&
        a_col + 3 < N &&
        N % VECTOR_WIDTH == 0)
    {
        const float4 value =
            *reinterpret_cast<const float4*>(
                &A[static_cast<size_t>(a_row) * N + a_col]
            );

        *reinterpret_cast<float4*>(
            &shared_A[shared_row][shared_col]
        ) = value;
    }
    else
    {
        /*
         * Safe scalar fallback for boundary tiles or unaligned rows.
         */
        #pragma unroll
        for (int element = 0;
             element < VECTOR_WIDTH;
             ++element)
        {
            const int current_col =
                a_col + element;

            shared_A[shared_row][shared_col + element] =
                (a_row < M && current_col < N)
                    ? A[
                        static_cast<size_t>(a_row) * N
                        + current_col
                      ]
                    : 0.0f;
        }
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
     load < B_TILE_VECTORS / BLOCK_THREADS;
     ++load)
{
    /*
     * Index measured in float4 groups.
     */
    const int vector_index =
        tid + load * BLOCK_THREADS;

    /*
     * Each B shared-memory row has:
     *
     * BLOCK_N / 4 = 128 / 4 = 32 float4 groups.
     */
    const int vectors_per_row =
        BLOCK_N / VECTOR_WIDTH;

    const int shared_row =
        vector_index / vectors_per_row;

    const int vector_col =
        vector_index % vectors_per_row;

    const int shared_col =
        vector_col * VECTOR_WIDTH;

    const int b_row =
        phase * BLOCK_K + shared_row;

    const int b_col =
        block_col + shared_col;

    /*
     * Vectorized path for four valid, aligned B elements.
     */
    if (b_row < N &&
        b_col + 3 < K &&
        K % VECTOR_WIDTH == 0)
    {
        const float4 value =
            *reinterpret_cast<const float4*>(
                &B[static_cast<size_t>(b_row) * K + b_col]
            );

        *reinterpret_cast<float4*>(
            &shared_B[shared_row][shared_col]
        ) = value;
    }
    else
    {
        /*
         * Safe scalar fallback.
         */
        #pragma unroll
        for (int element = 0;
             element < VECTOR_WIDTH;
             ++element)
        {
            const int current_col =
                b_col + element;

            shared_B[shared_row][shared_col + element] =
                (b_row < N && current_col < K)
                    ? B[
                        static_cast<size_t>(b_row) * K
                        + current_col
                      ]
                    : 0.0f;
        }
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
    /*
     * Threads with consecutive tx values now access consecutive
     * shared-memory columns for each value of c.
     */
    const int shared_col =
        tx + c * THREADS_X;

    reg_B[c] =
        shared_B[i][shared_col];
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
            const int output_col =
    block_col + tx + c * THREADS_X;

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
