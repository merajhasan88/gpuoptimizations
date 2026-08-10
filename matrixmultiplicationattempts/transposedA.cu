#include <cuda_runtime.h>
#include <stddef.h>

/*
 * ============================================================
 * Configuration
 * ============================================================
 */

constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;

constexpr int BLOCK_THREADS =
    THREADS_X * THREADS_Y;             // 256

constexpr int MICRO_M = 8;
constexpr int MICRO_N = 8;

constexpr int BLOCK_M =
    THREADS_Y * MICRO_M;               // 128

constexpr int BLOCK_N =
    THREADS_X * MICRO_N;               // 128

constexpr int BLOCK_K = 16;

constexpr int VECTOR_WIDTH = 4;


/*
 * Global -> shared cooperative loading.
 */
constexpr int A_TILE_ELEMENTS =
    BLOCK_M * BLOCK_K;                 // 2048

constexpr int B_TILE_ELEMENTS =
    BLOCK_K * BLOCK_N;                 // 2048

constexpr int A_TILE_VECTORS =
    A_TILE_ELEMENTS / VECTOR_WIDTH;    // 512

constexpr int B_TILE_VECTORS =
    B_TILE_ELEMENTS / VECTOR_WIDTH;    // 512


/*
 * ============================================================
 * Transposed shared-A representation
 * ============================================================
 *
 * Logical shared A is:
 *
 *     shared_A[k][output_row]
 *
 * rather than:
 *
 *     shared_A[output_row][k]
 *
 *
 * We want each row of shared_A to remain 16-byte aligned
 * because we will read it using float4.
 *
 * BLOCK_M is 128 floats.
 *
 * Add four padding floats:
 *
 *     128 + 4 = 132 floats
 *
 * 132 floats = 33 float4 groups.
 *
 * 33 × 16 bytes = 528 bytes
 *
 * which is still divisible by 16.
 *
 *
 * Why +4 instead of +1?
 *
 * +1 would give 129 floats per shared row, destroying
 * float4 alignment on following rows.
 *
 * +4 preserves alignment while also changing the bank
 * relationship between consecutive reduction rows.
 */

constexpr int A_SHARED_PADDING = 4;

constexpr int A_SHARED_STRIDE =
    BLOCK_M + A_SHARED_PADDING;        // 132 floats

constexpr int A_SHARED_VEC_STRIDE =
    A_SHARED_STRIDE / VECTOR_WIDTH;    // 33 float4 groups


/*
 * This union lets us:
 *
 * - write individual floats while transposing A
 * - later read complete float4 vectors
 *
 * The object itself is explicitly 16-byte aligned.
 */
union __align__(16) SharedFloat4
{
    float4 v;
    float f[4];
};


/*
 * ============================================================
 * Kernel
 * ============================================================
 */

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    /*
     * Original fast 16×16 thread mapping.
     */
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid =
        ty * THREADS_X + tx;


    /*
     * Starting location of this block's 128×128 C tile.
     */
    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    /*
     * Each thread owns eight consecutive output rows.
     */
    const int local_row =
        ty * MICRO_M;

    const int global_row =
        block_row + local_row;


    /*
     * ========================================================
     * Shared A
     * ========================================================
     *
     * TRANSPOSED relative to the old version.
     *
     * Conceptually:
     *
     *     shared_A[k][output_row]
     *
     * Each logical k row contains:
     *
     *     128 useful floats
     *     + 4 padding floats
     *
     * represented as 33 float4 groups.
     */
    __shared__ SharedFloat4
        shared_A[BLOCK_K][A_SHARED_VEC_STRIDE];


    /*
     * ========================================================
     * Shared B
     * ========================================================
     *
     * Keep B exactly in the layout that worked well before.
     */
    __shared__ float
        shared_B[BLOCK_K][BLOCK_N];


    /*
     * 64 accumulators/thread.
     */
    float acc[MICRO_M][MICRO_N] = {0.0f};


    /*
     * Register double buffering.
     */
    float reg_A[2][MICRO_M];
    float reg_B[2][MICRO_N];


    const int number_of_phases =
        (N + BLOCK_K - 1) / BLOCK_K;


    /*
     * ========================================================
     * Reduction phases
     * ========================================================
     */
    for (int phase = 0;
         phase < number_of_phases;
         ++phase)
    {
        /*
         * ====================================================
         * A: global memory -> TRANSPOSED shared memory
         * ====================================================
         *
         * Global-memory loading remains float4 and coalesced.
         *
         * The difference is where those four values are placed
         * in shared memory.
         */

        #pragma unroll
        for (int load = 0;
             load < A_TILE_VECTORS / BLOCK_THREADS;
             ++load)
        {
            const int vector_index =
                tid + load * BLOCK_THREADS;


            /*
             * A tile logically has:
             *
             * 128 rows × 16 columns.
             *
             * Every logical A row therefore contains:
             *
             * 16 / 4 = 4 float4 groups.
             */
            constexpr int vectors_per_A_row =
                BLOCK_K / VECTOR_WIDTH;        // 4


            /*
             * Which logical A row?
             *
             * 0..127
             */
            const int tile_row =
                vector_index / vectors_per_A_row;


            /*
             * Which float4 within that A row?
             *
             * 0..3
             */
            const int vector_col =
                vector_index % vectors_per_A_row;


            /*
             * First reduction column represented by this float4.
             *
             * 0, 4, 8, or 12.
             */
            const int tile_k =
                vector_col * VECTOR_WIDTH;


            /*
             * Global A coordinates.
             */
            const int a_row =
                block_row + tile_row;

            const int a_col =
                phase * BLOCK_K + tile_k;


            /*
             * Load four adjacent A values from global memory.
             */
            float4 value;


            if (a_row < M &&
                a_col + 3 < N &&
                N % VECTOR_WIDTH == 0)
            {
                value =
                    *reinterpret_cast<const float4*>(
                        &A[
                            static_cast<size_t>(a_row) * N
                            + a_col
                        ]
                    );
            }
            else
            {
                /*
                 * Boundary / unaligned fallback.
                 */
                value.x =
                    (a_row < M && a_col + 0 < N)
                    ? A[
                        static_cast<size_t>(a_row) * N
                        + a_col + 0
                      ]
                    : 0.0f;

                value.y =
                    (a_row < M && a_col + 1 < N)
                    ? A[
                        static_cast<size_t>(a_row) * N
                        + a_col + 1
                      ]
                    : 0.0f;

                value.z =
                    (a_row < M && a_col + 2 < N)
                    ? A[
                        static_cast<size_t>(a_row) * N
                        + a_col + 2
                      ]
                    : 0.0f;

                value.w =
                    (a_row < M && a_col + 3 < N)
                    ? A[
                        static_cast<size_t>(a_row) * N
                        + a_col + 3
                      ]
                    : 0.0f;
            }


            /*
             * =================================================
             * TRANSPOSE while storing into shared memory
             * =================================================
             *
             * Global values:
             *
             *     A[tile_row][tile_k + 0]
             *     A[tile_row][tile_k + 1]
             *     A[tile_row][tile_k + 2]
             *     A[tile_row][tile_k + 3]
             *
             * become:
             *
             *     shared_A[tile_k + 0][tile_row]
             *     shared_A[tile_k + 1][tile_row]
             *     shared_A[tile_k + 2][tile_row]
             *     shared_A[tile_k + 3][tile_row]
             *
             *
             * Find which float4 group contains tile_row.
             */
            const int row_vector =
                tile_row / VECTOR_WIDTH;


            /*
             * Which component within that float4?
             *
             * 0 -> x
             * 1 -> y
             * 2 -> z
             * 3 -> w
             *
             * We use the f[] view so a dynamic component index
             * works naturally.
             */
            const int row_component =
                tile_row % VECTOR_WIDTH;


            shared_A[tile_k + 0]
                    [row_vector]
                    .f[row_component] =
                        value.x;

            shared_A[tile_k + 1]
                    [row_vector]
                    .f[row_component] =
                        value.y;

            shared_A[tile_k + 2]
                    [row_vector]
                    .f[row_component] =
                        value.z;

            shared_A[tile_k + 3]
                    [row_vector]
                    .f[row_component] =
                        value.w;
        }


        /*
         * ====================================================
         * B: global -> shared
         * ====================================================
         *
         * Keep the B loading from the 92.33 ms version.
         */

        #pragma unroll
        for (int load = 0;
             load < B_TILE_VECTORS / BLOCK_THREADS;
             ++load)
        {
            const int vector_index =
                tid + load * BLOCK_THREADS;


            constexpr int vectors_per_B_row =
                BLOCK_N / VECTOR_WIDTH;        // 32


            const int shared_row =
                vector_index / vectors_per_B_row;

            const int vector_col =
                vector_index % vectors_per_B_row;

            const int shared_col =
                vector_col * VECTOR_WIDTH;


            const int b_row =
                phase * BLOCK_K + shared_row;

            const int b_col =
                block_col + shared_col;


            if (b_row < N &&
                b_col + 3 < K &&
                K % VECTOR_WIDTH == 0)
            {
                const float4 value =
                    *reinterpret_cast<const float4*>(
                        &B[
                            static_cast<size_t>(b_row) * K
                            + b_col
                        ]
                    );


                /*
                 * shared_B row width = 128 floats,
                 * preserving float4 alignment.
                 */
                *reinterpret_cast<float4*>(
                    &shared_B[shared_row][shared_col]
                ) = value;
            }
            else
            {
                #pragma unroll
                for (int element = 0;
                     element < VECTOR_WIDTH;
                     ++element)
                {
                    const int current_col =
                        b_col + element;


                    shared_B
                        [shared_row]
                        [shared_col + element] =
                            (
                                b_row < N &&
                                current_col < K
                            )
                            ?
                            B[
                                static_cast<size_t>(b_row) * K
                                + current_col
                            ]
                            :
                            0.0f;
                }
            }
        }


        /*
         * Entire tile must be ready.
         */
        __syncthreads();


        /*
         * ====================================================
         * Preload A reduction position 0
         * ====================================================
         *
         * This is the important new part.
         *
         * Thread ty owns:
         *
         *     local_row
         *     local_row + 1
         *     ...
         *     local_row + 7
         *
         * Because shared_A is transposed, those values are now
         * physically consecutive.
         *
         * local_row is:
         *
         *     ty × 8
         *
         * therefore always divisible by 8 and safely float4
         * aligned.
         */


        /*
         * First four A values:
         *
         * rows local_row + 0..3
         */
        const int A_vector_base =
            local_row / VECTOR_WIDTH;


        float4 a0 =
            shared_A[0][A_vector_base + 0].v;


        /*
         * Next four A values:
         *
         * rows local_row + 4..7
         */
        float4 a1 =
            shared_A[0][A_vector_base + 1].v;


        reg_A[0][0] = a0.x;
        reg_A[0][1] = a0.y;
        reg_A[0][2] = a0.z;
        reg_A[0][3] = a0.w;

        reg_A[0][4] = a1.x;
        reg_A[0][5] = a1.y;
        reg_A[0][6] = a1.z;
        reg_A[0][7] = a1.w;


        /*
         * ====================================================
         * Preload B position 0
         * ====================================================
         *
         * Keep the bank-friendly mapping that produced the
         * large earlier speedup.
         */

        #pragma unroll
        for (int c = 0;
             c < MICRO_N;
             ++c)
        {
            const int shared_col =
                tx + c * THREADS_X;


            reg_B[0][c] =
                shared_B[0][shared_col];
        }


        /*
         * ====================================================
         * Reduction computation
         * ====================================================
         */
        #pragma unroll
        for (int i = 0;
             i < BLOCK_K;
             ++i)
        {
            const int current_buffer =
                i & 1;

            const int next_buffer =
                current_buffer ^ 1;


            /*
             * ------------------------------------------------
             * Prefetch reduction position i+1
             * ------------------------------------------------
             */
            if (i + 1 < BLOCK_K)
            {
                /*
                 * ============================================
                 * A: TWO float4 shared-memory loads
                 * ============================================
                 *
                 * Instead of:
                 *
                 *     8 scalar loads
                 *
                 * we request:
                 *
                 *     2 float4 loads.
                 */

                const float4 next_a0 =
                    shared_A
                        [i + 1]
                        [A_vector_base + 0]
                        .v;


                const float4 next_a1 =
                    shared_A
                        [i + 1]
                        [A_vector_base + 1]
                        .v;


                reg_A[next_buffer][0] =
                    next_a0.x;

                reg_A[next_buffer][1] =
                    next_a0.y;

                reg_A[next_buffer][2] =
                    next_a0.z;

                reg_A[next_buffer][3] =
                    next_a0.w;


                reg_A[next_buffer][4] =
                    next_a1.x;

                reg_A[next_buffer][5] =
                    next_a1.y;

                reg_A[next_buffer][6] =
                    next_a1.z;

                reg_A[next_buffer][7] =
                    next_a1.w;


                /*
                 * ============================================
                 * B: keep existing scalar bank-friendly loads
                 * ============================================
                 */
                #pragma unroll
                for (int c = 0;
                     c < MICRO_N;
                     ++c)
                {
                    const int shared_col =
                        tx + c * THREADS_X;


                    reg_B[next_buffer][c] =
                        shared_B
                            [i + 1]
                            [shared_col];
                }
            }


            /*
             * ------------------------------------------------
             * 8 × 8 outer product
             * ------------------------------------------------
             */
            #pragma unroll
            for (int r = 0;
                 r < MICRO_M;
                 ++r)
            {
                #pragma unroll
                for (int c = 0;
                     c < MICRO_N;
                     ++c)
                {
                    acc[r][c] +=
                        reg_A[current_buffer][r]
                        *
                        reg_B[current_buffer][c];
                }
            }
        }


        /*
         * Single shared-memory buffer.
         */
        __syncthreads();
    }


    /*
     * ========================================================
     * Store C
     * ========================================================
     *
     * Keep exactly the fast baseline output mapping.
     */
    #pragma unroll
    for (int r = 0;
         r < MICRO_M;
         ++r)
    {
        const int output_row =
            global_row + r;


        #pragma unroll
        for (int c = 0;
             c < MICRO_N;
             ++c)
        {
            const int output_col =
                block_col
                + tx
                + c * THREADS_X;


            if (output_row < M &&
                output_col < K)
            {
                C[
                    static_cast<size_t>(output_row) * K
                    + output_col
                ] =
                    acc[r][c];
            }
        }
    }
}


/*
 * ============================================================
 * Launcher
 * ============================================================
 */

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    /*
     * Back to the original 16×16 block.
     */
    dim3 threadsPerBlock(
        THREADS_X,
        THREADS_Y
    );


    /*
     * 128×128 output tile.
     */
    dim3 blocksPerGrid(
        (K + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );


    matrix_multiplication_kernel
        <<<blocksPerGrid, threadsPerBlock>>>(
            A,
            B,
            C,
            M,
            N,
            K
        );


    cudaDeviceSynchronize();
}
