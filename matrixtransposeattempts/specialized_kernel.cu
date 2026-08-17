#include <cuda_runtime.h>

// ============================================================
// TUNING SWITCHES
// ============================================================
//
// Start with configuration 0 below to verify your 2.29 ms.
//
// Then test the configurations listed after the code,
// changing ONE thing at a time.
//
// ============================================================

// 0 = use your exact generic BLOCK_ROWS=16 kernel
// 1 = use compile-time-specialized 7000x6000 kernel
#define USE_SPECIALIZED_7000x6000  1

// Only matters when USE_SPECIALIZED_7000x6000 = 1.
//
// 1 = full 32x32 tiles have zero bounds checks.
//     Edge tiles remain checked, but everything stays
//     in ONE kernel launch.
#define USE_INTERIOR_FAST_PATH      1

// Forces compiler to permit at least two 512-thread blocks/SM.
//
// Test both 0 and 1.
// Do NOT assume 1 is faster.
#define USE_LAUNCH_BOUNDS           0

// L2-oriented/cache-global load policy.
//
// Test separately.
#define USE_LDCG                    0

// Streaming-store cache policy.
//
// Test separately.
#define USE_STCS                    0

// Keep this = 1 unless your benchmark harness synchronizes
// after solve() itself.
#define FORCE_DEVICE_SYNC           1


// ============================================================
// FIXED TILE CONFIGURATION
//
// This is currently your winning geometry:
//     tile       = 32 x 32
//     block      = 32 x 16
//     threads    = 512
//     warps      = 16
//     elements/thread = 2
// ============================================================

#define TILE_DIM    32
#define BLOCK_ROWS  16


// ============================================================
// OPTIONAL LAUNCH BOUNDS
// ============================================================

#if USE_LAUNCH_BOUNDS
#define TRANSPOSE_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#else
#define TRANSPOSE_LAUNCH_BOUNDS
#endif


// ============================================================
// LOAD / STORE POLICY
//
// With both options disabled these reduce to ordinary
// scalar loads/stores.
// ============================================================

#if USE_LDCG

#define LOAD_FLOAT(ptr) \
    __ldcg((const float*)(ptr))

#else

#define LOAD_FLOAT(ptr) \
    (*(const float*)(ptr))

#endif


#if USE_STCS

#define STORE_FLOAT(ptr, value) \
    __stcs((float*)(ptr), (value))

#else

#define STORE_FLOAT(ptr, value) \
    (*(float*)(ptr) = (value))

#endif


// ============================================================
// CURRENT WINNER / GENERIC KERNEL
//
// This deliberately stays extremely close to your 2.29 ms
// version.
//
// rows/cols can be anything in the allowed range.
// ============================================================

__global__ TRANSPOSE_LAUNCH_BOUNDS
void transpose_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    // Padding removes the classic transpose shared-memory
    // bank conflict.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..15

    int x =
        blockIdx.x * TILE_DIM + tx;

    int y =
        blockIdx.y * TILE_DIM + ty;

    // BLOCK_ROWS=16 means exactly two iterations:
    //
    // j = 0
    // j = 16

    #pragma unroll
    for (int j = 0;
         j < TILE_DIM;
         j += BLOCK_ROWS)
    {
        if (x < cols &&
            y + j < rows)
        {
            tile[ty + j][tx] =
                LOAD_FLOAT(
                    input +
                    (y + j) * cols +
                    x
                );
        }
    }

    __syncthreads();

    // Swap block coordinates.

    x =
        blockIdx.y * TILE_DIM + tx;

    y =
        blockIdx.x * TILE_DIM + ty;

    #pragma unroll
    for (int j = 0;
         j < TILE_DIM;
         j += BLOCK_ROWS)
    {
        if (x < rows &&
            y + j < cols)
        {
            STORE_FLOAT(
                output +
                (y + j) * rows +
                x,

                tile[tx][ty + j]
            );
        }
    }
}


// ============================================================
// SPECIALIZED 7000 x 6000 KERNEL
//
// Same transpose algorithm.
//
// Differences:
//
//   - ROWS/COLS are compile-time constants
//   - loop manually expanded
//   - optional uniform interior fast path
//   - still ONE kernel launch
//
// Grid:
//
//   ceil(6000/32) = 188 tiles X
//   ceil(7000/32) = 219 tiles Y
//
// Full tiles:
//
//   floor(6000/32) = 187
//   floor(7000/32) = 218
//
// Therefore 187 * 218 = 40,766 blocks take the hot path.
// ============================================================

__global__ TRANSPOSE_LAUNCH_BOUNDS
void transpose_7000x6000(
    const float* __restrict__ input,
    float* __restrict__ output)
{
    constexpr int ROWS = 7000;
    constexpr int COLS = 6000;

    constexpr int FULL_TILES_X =
        COLS / TILE_DIM;           // 187

    constexpr int FULL_TILES_Y =
        ROWS / TILE_DIM;           // 218

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // --------------------------------------------------------
    // INPUT COORDINATES
    // --------------------------------------------------------

    const int in_x =
        (blockIdx.x << 5) + tx;

    const int in_y =
        (blockIdx.y << 5) + ty;


#if USE_INTERIOR_FAST_PATH

    // Uniform for the entire block.
    //
    // No warp divergence here because every thread in a block
    // sees identical blockIdx values.

    const bool full_tile =
        blockIdx.x < FULL_TILES_X &&
        blockIdx.y < FULL_TILES_Y;

    if (full_tile)
    {
        // ====================================================
        // HOT PATH
        //
        // No bounds checks.
        //
        // More than 99% of the matrix tiles go here.
        // ====================================================

        const float* __restrict__ src =
            input +
            in_y * COLS +
            in_x;

        tile[ty][tx] =
            LOAD_FLOAT(src);

        tile[ty + 16][tx] =
            LOAD_FLOAT(src + 16 * COLS);
    }
    else
    {
        // ====================================================
        // EDGE LOAD
        // ====================================================

        if (in_x < COLS)
        {
            if (in_y < ROWS)
            {
                tile[ty][tx] =
                    LOAD_FLOAT(
                        input +
                        in_y * COLS +
                        in_x
                    );
            }

            if (in_y + 16 < ROWS)
            {
                tile[ty + 16][tx] =
                    LOAD_FLOAT(
                        input +
                        (in_y + 16) * COLS +
                        in_x
                    );
            }
        }
    }

#else

    // ========================================================
    // SPECIALIZED, BUT NO INTERIOR FAST PATH
    // ========================================================

    if (in_x < COLS)
    {
        if (in_y < ROWS)
        {
            tile[ty][tx] =
                LOAD_FLOAT(
                    input +
                    in_y * COLS +
                    in_x
                );
        }

        if (in_y + 16 < ROWS)
        {
            tile[ty + 16][tx] =
                LOAD_FLOAT(
                    input +
                    (in_y + 16) * COLS +
                    in_x
                );
        }
    }

#endif


    // One block-wide synchronization remains essential.
    __syncthreads();


    // --------------------------------------------------------
    // OUTPUT COORDINATES
    //
    // Block coordinates swap.
    // --------------------------------------------------------

    const int out_x =
        (blockIdx.y << 5) + tx;

    const int out_y =
        (blockIdx.x << 5) + ty;


#if USE_INTERIOR_FAST_PATH

    if (full_tile)
    {
        // ====================================================
        // HOT STORE PATH
        //
        // Completely bounds-check-free.
        // ====================================================

        float* __restrict__ dst =
            output +
            out_y * ROWS +
            out_x;

        STORE_FLOAT(
            dst,
            tile[tx][ty]
        );

        STORE_FLOAT(
            dst + 16 * ROWS,
            tile[tx][ty + 16]
        );
    }
    else
    {
        // ====================================================
        // EDGE STORE
        // ====================================================

        if (out_x < ROWS)
        {
            if (out_y < COLS)
            {
                STORE_FLOAT(
                    output +
                    out_y * ROWS +
                    out_x,

                    tile[tx][ty]
                );
            }

            if (out_y + 16 < COLS)
            {
                STORE_FLOAT(
                    output +
                    (out_y + 16) * ROWS +
                    out_x,

                    tile[tx][ty + 16]
                );
            }
        }
    }

#else

    if (out_x < ROWS)
    {
        if (out_y < COLS)
        {
            STORE_FLOAT(
                output +
                out_y * ROWS +
                out_x,

                tile[tx][ty]
            );
        }

        if (out_y + 16 < COLS)
        {
            STORE_FLOAT(
                output +
                (out_y + 16) * ROWS +
                out_x,

                tile[tx][ty + 16]
            );
        }
    }

#endif
}


// ============================================================
// ENTRY POINT
// ============================================================

extern "C" void solve(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    // Current winning block:
    //
    // 32 * 16 = 512 threads
    // 16 warps/block

    dim3 block(
        TILE_DIM,
        BLOCK_ROWS
    );

    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );


#if USE_SPECIALIZED_7000x6000

    if (rows == 7000 &&
        cols == 6000)
    {
        transpose_7000x6000
            <<<grid, block>>>(
                input,
                output
            );
    }
    else
    {
        transpose_generic
            <<<grid, block>>>(
                input,
                output,
                rows,
                cols
            );
    }

#else

    // Exact general structure of your current winner.

    transpose_generic
        <<<grid, block>>>(
            input,
            output,
            rows,
            cols
        );

#endif


#if FORCE_DEVICE_SYNC

    cudaDeviceSynchronize();

#endif
}
