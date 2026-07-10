// Include CUDA runtime declarations such as:
// cudaDeviceSynchronize(), dim3, blockIdx, threadIdx, and __global__.
#include <cuda_runtime.h>

// Number of threads along each dimension of the thread block.
// The block will contain 16 x 16 = 256 threads.
#define BLOCK_THREADS 16

// Number of output rows computed by one CUDA block.
// Each of the 16 thread rows computes 4 output rows:
// 16 * 4 = 64.
#define TILE_M 64

// Number of elements processed at once along the shared N dimension.
// A and B are multiplied in chunks of 16 elements.
#define TILE_N 16

// Number of output columns computed by one CUDA block.
// Each of the 16 thread columns computes 4 output columns:
// 16 * 4 = 64.
#define TILE_K 64

// Number of output rows and columns computed by each thread.
// Each thread computes a 4 x 4 output region.
#define MICRO_TILE 4


// CUDA kernel for multiplying:
//
// A: M x N
// B: N x K
// C: M x K
//
// The operation is:
//
// C = A * B
__global__ void matrix_multiplication_kernel(
    // Input matrix A stored in row-major order.
    const float* __restrict__ A,

    // Input matrix B stored in row-major order.
    const float* __restrict__ B,

    // Output matrix C stored in row-major order.
    float* __restrict__ C,

    // Number of rows in A and C.
    int M,

    // Number of columns in A and rows in B.
    int N,

    // Number of columns in B and C.
    int K)
{
    /*
     * Shared-memory tile for matrix A.
     *
     * This block needs 64 rows from A and 16 columns from A.
     *
     * Dimensions:
     *
     *     64 rows x 16 columns
     *
     * Shared-memory usage:
     *
     *     64 * 16 * 4 bytes = 4096 bytes
     */
    __shared__ float ds_A[TILE_M][TILE_N];

    /*
     * Shared-memory tile for matrix B.
     *
     * This block needs 16 rows from B and 64 columns from B.
     *
     * Dimensions:
     *
     *     16 rows x 64 columns
     *
     * Shared-memory usage:
     *
     *     16 * 64 * 4 bytes = 4096 bytes
     */
    __shared__ float ds_B[TILE_N][TILE_K];


    /*
     * Horizontal thread index inside the block.
     *
     * Possible values:
     *
     *     0 through 15
     */
    const int tx = threadIdx.x;

    /*
     * Vertical thread index inside the block.
     *
     * Possible values:
     *
     *     0 through 15
     */
    const int ty = threadIdx.y;


    /*
     * Calculate the first global output row managed by this block.
     *
     * Each block handles 64 output rows.
     *
     * Examples:
     *
     *     blockIdx.y = 0 -> rows 0 through 63
     *     blockIdx.y = 1 -> rows 64 through 127
     *     blockIdx.y = 2 -> rows 128 through 191
     */
    const int block_row = blockIdx.y * TILE_M;

    /*
     * Calculate the first global output column managed by this block.
     *
     * Each block handles 64 output columns.
     *
     * Examples:
     *
     *     blockIdx.x = 0 -> columns 0 through 63
     *     blockIdx.x = 1 -> columns 64 through 127
     *     blockIdx.x = 2 -> columns 128 through 191
     */
    const int block_col = blockIdx.x * TILE_K;


    /*
     * Calculate this thread's first row inside the 64-row output tile.
     *
     * Each thread computes 4 rows.
     *
     * Examples:
     *
     *     ty = 0  -> local rows 0, 1, 2, 3
     *     ty = 1  -> local rows 4, 5, 6, 7
     *     ty = 15 -> local rows 60, 61, 62, 63
     */
    const int local_row = ty * MICRO_TILE;

    /*
     * Calculate this thread's first column inside the 64-column output tile.
     *
     * Each thread computes 4 columns.
     *
     * Examples:
     *
     *     tx = 0  -> local columns 0, 1, 2, 3
     *     tx = 1  -> local columns 4, 5, 6, 7
     *     tx = 15 -> local columns 60, 61, 62, 63
     */
    const int local_col = tx * MICRO_TILE;


    /*
     * Calculate this thread's first global output row.
     *
     * This combines:
     *
     *     1. The block's starting row.
     *     2. The thread's starting row within the block.
     */
    const int global_row = block_row + local_row;

    /*
     * Calculate this thread's first global output column.
     *
     * This combines:
     *
     *     1. The block's starting column.
     *     2. The thread's starting column within the block.
     */
    const int global_col = block_col + local_col;


    /*
     * Register array holding this thread's 4 x 4 output values.
     *
     * Each thread calculates:
     *
     *     acc[0][0] acc[0][1] acc[0][2] acc[0][3]
     *     acc[1][0] acc[1][1] acc[1][2] acc[1][3]
     *     acc[2][0] acc[2][1] acc[2][2] acc[2][3]
     *     acc[3][0] acc[3][1] acc[3][2] acc[3][3]
     *
     * The empty initializer sets all 16 values to 0.0f.
     */
    float acc[MICRO_TILE][MICRO_TILE] = {};


    /*
     * Divide the N dimension into phases of 16 elements.
     *
     * For example, when N = 35:
     *
     *     Phase 0 handles N indexes 0 through 15.
     *     Phase 1 handles N indexes 16 through 31.
     *     Phase 2 handles N indexes 32 through 34.
     *
     * The last phase is padded with zeros where necessary.
     */
    for (
        int phase = 0;
        phase < (N + TILE_N - 1) / TILE_N;
        ++phase)
    {
        /*
         * Each thread loads four elements from matrix A.
         *
         * The four elements are in different rows but share the same
         * N-dimension column.
         *
         * Across all 256 threads:
         *
         *     256 threads * 4 values = 1024 values
         *
         * That exactly fills:
         *
         *     64 * 16 = 1024 elements in ds_A
         */
        #pragma unroll
        for (int r = 0; r < MICRO_TILE; ++r)
        {
            /*
             * Global row of A loaded by this loop iteration.
             *
             * The thread loads:
             *
             *     global_row + 0
             *     global_row + 1
             *     global_row + 2
             *     global_row + 3
             */
            const int a_row = global_row + r;

            /*
             * Global column of A loaded during the current phase.
             *
             * tx selects one of the 16 columns in the current N tile.
             *
             * Example for phase 2:
             *
             *     a_col = 2 * 16 + tx
             *           = 32 + tx
             */
            const int a_col = phase * TILE_N + tx;

            /*
             * Make sure the requested element exists in matrix A.
             *
             * A has:
             *
             *     M rows
             *     N columns
             */
            if (a_row < M && a_col < N)
            {
                /*
                 * Load one A value from global memory into shared memory.
                 *
                 * Row-major A indexing:
                 *
                 *     A[row * N + column]
                 *
                 * The shared-memory row is local_row + r.
                 * The shared-memory column is tx.
                 */
                ds_A[local_row + r][tx] =
                    A[static_cast<size_t>(a_row) * N + a_col];
            }
            else
            {
                /*
                 * The requested A element lies outside the matrix.
                 *
                 * Store zero so the padded area contributes nothing to
                 * the matrix multiplication.
                 */
                ds_A[local_row + r][tx] = 0.0f;
            }
        }


        /*
         * Each thread loads four elements from matrix B.
         *
         * The four elements are in the same row but in four neighboring
         * columns.
         *
         * Across all 256 threads:
         *
         *     256 threads * 4 values = 1024 values
         *
         * That exactly fills:
         *
         *     16 * 64 = 1024 elements in ds_B
         */
        #pragma unroll
        for (int c = 0; c < MICRO_TILE; ++c)
        {
            /*
             * Global row of B for the current phase.
             *
             * ty selects one of the 16 B rows in this N tile.
             */
            const int b_row = phase * TILE_N + ty;

            /*
             * Global B column loaded by this loop iteration.
             *
             * Each thread loads four neighboring columns:
             *
             *     global_col + 0
             *     global_col + 1
             *     global_col + 2
             *     global_col + 3
             */
            const int b_col = global_col + c;

            /*
             * Make sure the requested element exists in matrix B.
             *
             * B has:
             *
             *     N rows
             *     K columns
             */
            if (b_row < N && b_col < K)
            {
                /*
                 * Load one B value from global memory into shared memory.
                 *
                 * Row-major B indexing:
                 *
                 *     B[row * K + column]
                 *
                 * ty determines the shared-memory row.
                 * local_col + c determines the shared-memory column.
                 */
                ds_B[ty][local_col + c] =
                    B[static_cast<size_t>(b_row) * K + b_col];
            }
            else
            {
                /*
                 * The requested B element lies outside the matrix.
                 *
                 * Store zero so the padded region does not affect the
                 * output.
                 */
                ds_B[ty][local_col + c] = 0.0f;
            }
        }


        /*
         * Wait until every thread has finished loading its A and B values.
         *
         * Without this synchronization, some threads could begin computing
         * while other threads are still writing into shared memory.
         */
        __syncthreads();


        /*
         * Compute contributions from the current 16-element N tile.
         *
         * This is the inner dot-product dimension.
         */
        #pragma unroll
        for (int i = 0; i < TILE_N; ++i)
        {
            /*
             * Four A values used by this thread during this iteration.
             *
             * These correspond to four output rows.
             */
            float reg_A[MICRO_TILE];

            /*
             * Four B values used by this thread during this iteration.
             *
             * These correspond to four output columns.
             */
            float reg_B[MICRO_TILE];


            /*
             * Load four A values from shared memory into registers.
             *
             * All four values come from different A rows but use the same
             * inner-dimension index i.
             */
            #pragma unroll
            for (int r = 0; r < MICRO_TILE; ++r)
            {
                reg_A[r] = ds_A[local_row + r][i];
            }


            /*
             * Load four B values from shared memory into registers.
             *
             * All four values come from the same B row i but use different
             * output columns.
             */
            #pragma unroll
            for (int c = 0; c < MICRO_TILE; ++c)
            {
                reg_B[c] = ds_B[i][local_col + c];
            }


            /*
             * Perform a 4 x 4 outer product.
             *
             * Four A values multiplied by four B values produce
             * 16 contributions.
             */
            #pragma unroll
            for (int r = 0; r < MICRO_TILE; ++r)
            {
                #pragma unroll
                for (int c = 0; c < MICRO_TILE; ++c)
                {
                    /*
                     * Add this N-dimension contribution to one output.
                     *
                     * Conceptually:
                     *
                     * C[global_row + r][global_col + c] +=
                     *     A[global_row + r][phase * 16 + i] *
                     *     B[phase * 16 + i][global_col + c]
                     */
                    acc[r][c] += reg_A[r] * reg_B[c];
                }
            }
        }


        /*
         * Wait until every thread has finished reading the current shared
         * memory tiles.
         *
         * This prevents threads from overwriting ds_A or ds_B for the next
         * phase while other threads are still using the current phase.
         */
        __syncthreads();
    }


    /*
     * Write this thread's 4 x 4 result tile to matrix C.
     */
    #pragma unroll
    for (int r = 0; r < MICRO_TILE; ++r)
    {
        #pragma unroll
        for (int c = 0; c < MICRO_TILE; ++c)
        {
            /*
             * Calculate the exact global output row for this accumulator.
             */
            const int output_row = global_row + r;

            /*
             * Calculate the exact global output column for this accumulator.
             */
            const int output_col = global_col + c;

            /*
             * Check matrix boundaries.
             *
             * This is necessary when M or K is not a multiple of 64.
             *
             * For your example:
             *
             *     M = 8
             *     K = 10
             *
             * Most threads in the 64 x 64 block are outside the valid
             * 8 x 10 output and therefore do not write anything.
             */
            if (output_row < M && output_col < K)
            {
                /*
                 * Store the completed output in row-major matrix C.
                 *
                 * Row-major C indexing:
                 *
                 *     C[row * K + column]
                 */
                C[static_cast<size_t>(output_row) * K + output_col] =
                    acc[r][c];
            }
        }
    }
}


// The platform calls solve() with device pointers for A, B, and C.
extern "C" void solve(
    // Device pointer to matrix A.
    const float* A,

    // Device pointer to matrix B.
    const float* B,

    // Device pointer to output matrix C.
    float* C,

    // Number of rows in A and C.
    int M,

    // Number of columns in A and rows in B.
    int N,

    // Number of columns in B and C.
    int K)
{
    /*
     * Launch a two-dimensional block containing:
     *
     *     16 threads in x
     *     16 threads in y
     *
     * Total:
     *
     *     16 * 16 = 256 threads per block
     */
    dim3 threadsPerBlock(
        BLOCK_THREADS,
        BLOCK_THREADS
    );


    /*
     * Calculate the number of blocks required in each direction.
     *
     * Each block computes:
     *
     *     64 rows
     *     64 columns
     *
     * X direction covers the K output columns.
     * Y direction covers the M output rows.
     */
    dim3 blocksPerGrid(
        // Ceiling division of K by 64.
        (K + TILE_K - 1) / TILE_K,

        // Ceiling division of M by 64.
        (M + TILE_M - 1) / TILE_M
    );


    /*
     * Launch the matrix-multiplication kernel.
     *
     * blocksPerGrid determines how many blocks are launched.
     * threadsPerBlock determines how many threads each block contains.
     */
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );


    /*
     * Wait for the GPU kernel to complete before solve() returns.
     *
     * This also causes runtime kernel errors to surface at this point,
     * although production code should explicitly check the returned error.
     */
    cudaDeviceSynchronize();
}
