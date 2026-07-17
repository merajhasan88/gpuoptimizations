#include <cuda_runtime.h>
#include <stddef.h>

// Thread block dimensions:
// 16 x 16 = 256 threads per block.
#define THREADS_X 16
#define THREADS_Y 16

// One block computes a 64 x 64 tile of matrix C.
#define TILE_M 64
#define TILE_N 16
#define TILE_K 64

// Each thread computes a 4 x 4 section of the output tile.
#define MICRO_M 4
#define MICRO_K 4


__global__ void matrix_multiplication_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    /*
     * Shared tile from A:
     *
     *     64 output rows x 16 reduction elements.
     *
     * The extra column can reduce shared-memory bank-conflict risks
     * for some access patterns.
     */
    __shared__ float shared_A[TILE_M][TILE_N + 1];

    /*
     * Shared tile from B:
     *
     *     16 reduction elements x 64 output columns.
     *
     * The extra column changes the shared-memory row stride from
     * 64 floats to 65 floats.
     */
    __shared__ float shared_B[TILE_N][TILE_K + 1];

    // Thread coordinates inside the block.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    /*
     * Starting global output coordinate handled by this block.
     *
     * Each block calculates 64 rows and 64 columns of C.
     */
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_K;

    /*
     * Starting local coordinate handled by this thread.
     *
     * Each thread calculates four consecutive rows and four
     * consecutive columns.
     */
    const int local_row = ty * MICRO_M;
    const int local_col = tx * MICRO_K;

    // Starting global output coordinate for this thread.
    const int global_row = block_row + local_row;
    const int global_col = block_col + local_col;

    /*
     * Sixteen scalar accumulators.
     *
     * Explicit scalar variables avoid dynamic array indexing and
     * give the compiler clearer register lifetimes.
     */
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;

    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;

    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;

    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;

    /*
     * Process the reduction dimension N in chunks of 16.
     */
    const int number_of_phases = (N + TILE_N - 1) / TILE_N;

    for (int phase = 0; phase < number_of_phases; ++phase)
    {
        /*
         * Load the 64 x 16 A tile.
         *
         * Each of the 256 threads loads four A values:
         *
         *     256 x 4 = 1024 values
         *
         * The shared A tile contains:
         *
         *     64 x 16 = 1024 values
         */
        const int a_col = phase * TILE_N + tx;

        const int a_row0 = global_row;
        const int a_row1 = global_row + 1;
        const int a_row2 = global_row + 2;
        const int a_row3 = global_row + 3;

        if (a_col < N)
        {
            shared_A[local_row][tx] =
                (a_row0 < M)
                    ? A[(size_t)a_row0 * N + a_col]
                    : 0.0f;

            shared_A[local_row + 1][tx] =
                (a_row1 < M)
                    ? A[(size_t)a_row1 * N + a_col]
                    : 0.0f;

            shared_A[local_row + 2][tx] =
                (a_row2 < M)
                    ? A[(size_t)a_row2 * N + a_col]
                    : 0.0f;

            shared_A[local_row + 3][tx] =
                (a_row3 < M)
                    ? A[(size_t)a_row3 * N + a_col]
                    : 0.0f;
        }
        else
        {
            shared_A[local_row][tx] = 0.0f;
            shared_A[local_row + 1][tx] = 0.0f;
            shared_A[local_row + 2][tx] = 0.0f;
            shared_A[local_row + 3][tx] = 0.0f;
        }

        /*
         * Load the 16 x 64 B tile.
         *
         * Each thread loads four neighboring B values:
         *
         *     256 x 4 = 1024 values
         *
         * The shared B tile contains:
         *
         *     16 x 64 = 1024 values
         */
        const int b_row = phase * TILE_N + ty;

        const int b_col0 = global_col;
        const int b_col1 = global_col + 1;
        const int b_col2 = global_col + 2;
        const int b_col3 = global_col + 3;

        if (b_row < N)
        {
            shared_B[ty][local_col] =
                (b_col0 < K)
                    ? B[(size_t)b_row * K + b_col0]
                    : 0.0f;

            shared_B[ty][local_col + 1] =
                (b_col1 < K)
                    ? B[(size_t)b_row * K + b_col1]
                    : 0.0f;

            shared_B[ty][local_col + 2] =
                (b_col2 < K)
                    ? B[(size_t)b_row * K + b_col2]
                    : 0.0f;

            shared_B[ty][local_col + 3] =
                (b_col3 < K)
                    ? B[(size_t)b_row * K + b_col3]
                    : 0.0f;
        }
        else
        {
            shared_B[ty][local_col] = 0.0f;
            shared_B[ty][local_col + 1] = 0.0f;
            shared_B[ty][local_col + 2] = 0.0f;
            shared_B[ty][local_col + 3] = 0.0f;
        }

        /*
         * Wait until the entire A and B tiles are available.
         */
        __syncthreads();

        /*
         * Compute the contribution from this 16-element phase.
         *
         * The four B values remain live while A values are loaded and
         * consumed one at a time.
         *
         * This avoids a reg_A[4] temporary array.
         */
        #pragma unroll
        for (int i = 0; i < TILE_N; ++i)
        {
            /*
             * Load four B values.
             *
             * These values are reused across all four output rows.
             */
            const float b0 = shared_B[i][local_col];
            const float b1 = shared_B[i][local_col + 1];
            const float b2 = shared_B[i][local_col + 2];
            const float b3 = shared_B[i][local_col + 3];

            /*
             * Load the first A value and immediately use it for
             * four FMA operations.
             */
            const float a0 = shared_A[local_row][i];

            acc00 = fmaf(a0, b0, acc00);
            acc01 = fmaf(a0, b1, acc01);
            acc02 = fmaf(a0, b2, acc02);
            acc03 = fmaf(a0, b3, acc03);

            /*
             * Load the second A value only after the first has been used.
             */
            const float a1 = shared_A[local_row + 1][i];

            acc10 = fmaf(a1, b0, acc10);
            acc11 = fmaf(a1, b1, acc11);
            acc12 = fmaf(a1, b2, acc12);
            acc13 = fmaf(a1, b3, acc13);

            /*
             * Load and consume the third A value.
             */
            const float a2 = shared_A[local_row + 2][i];

            acc20 = fmaf(a2, b0, acc20);
            acc21 = fmaf(a2, b1, acc21);
            acc22 = fmaf(a2, b2, acc22);
            acc23 = fmaf(a2, b3, acc23);

            /*
             * Load and consume the fourth A value.
             */
            const float a3 = shared_A[local_row + 3][i];

            acc30 = fmaf(a3, b0, acc30);
            acc31 = fmaf(a3, b1, acc31);
            acc32 = fmaf(a3, b2, acc32);
            acc33 = fmaf(a3, b3, acc33);
        }

        /*
         * Ensure every thread has finished reading the current shared
         * tiles before the next phase overwrites them.
         */
        __syncthreads();
    }

    /*
     * Store output row 0.
     */
    if (global_row < M)
    {
        const size_t output_offset = (size_t)global_row * K + global_col;

        if (global_col < K)
            C[output_offset] = acc00;

        if (global_col + 1 < K)
            C[output_offset + 1] = acc01;

        if (global_col + 2 < K)
            C[output_offset + 2] = acc02;

        if (global_col + 3 < K)
            C[output_offset + 3] = acc03;
    }

    /*
     * Store output row 1.
     */
    if (global_row + 1 < M)
    {
        const size_t output_offset =
            (size_t)(global_row + 1) * K + global_col;

        if (global_col < K)
            C[output_offset] = acc10;

        if (global_col + 1 < K)
            C[output_offset + 1] = acc11;

        if (global_col + 2 < K)
            C[output_offset + 2] = acc12;

        if (global_col + 3 < K)
            C[output_offset + 3] = acc13;
    }

    /*
     * Store output row 2.
     */
    if (global_row + 2 < M)
    {
        const size_t output_offset =
            (size_t)(global_row + 2) * K + global_col;

        if (global_col < K)
            C[output_offset] = acc20;

        if (global_col + 1 < K)
            C[output_offset + 1] = acc21;

        if (global_col + 2 < K)
            C[output_offset + 2] = acc22;

        if (global_col + 3 < K)
            C[output_offset + 3] = acc23;
    }

    /*
     * Store output row 3.
     */
    if (global_row + 3 < M)
    {
        const size_t output_offset =
            (size_t)(global_row + 3) * K + global_col;

        if (global_col < K)
            C[output_offset] = acc30;

        if (global_col + 1 < K)
            C[output_offset + 1] = acc31;

        if (global_col + 2 < K)
            C[output_offset + 2] = acc32;

        if (global_col + 3 < K)
            C[output_offset + 3] = acc33;
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
    // 16 x 16 = 256 threads per block.
    dim3 threadsPerBlock(THREADS_X, THREADS_Y);

    /*
     * Each block calculates a 64 x 64 output tile.
     */
    dim3 blocksPerGrid(
        (K + TILE_K - 1) / TILE_K,
        (M + TILE_M - 1) / TILE_M
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
/*
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
*/
