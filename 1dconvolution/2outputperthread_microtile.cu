#include <cuda_runtime.h>

#define THREADS        256
#define MAX_KERNEL     2047

#define OUTPUTS_PER_BLOCK (THREADS * 2)

// For kernel_size = 2047:
//
// 512 output positions
// +
// 2046 halo values
// =
// 2558 shared floats
#define FAST_SHARED_SIZE (OUTPUTS_PER_BLOCK + MAX_KERNEL - 1)


// ============================================================
// FILTER IN CONSTANT MEMORY
// ============================================================

__constant__ float c_kernel[MAX_KERNEL];


// ============================================================
// FAST BENCHMARK KERNEL
//
// kernel_size = 2047
//
// 256 threads/block
// 2 outputs/thread
// 512 outputs/block
//
// Thread tx computes:
//
// output[base + tx]
// output[base + tx + 256]
//
// This is important:
// the second output is 256 positions away, NOT adjacent.
//
// Therefore a warp accesses:
//
// shared[tx + j]
//
// and:
//
// shared[tx + 256 + j]
//
// Both are contiguous across the warp and therefore
// conflict-free in shared memory.
// ============================================================

__global__ void convolution_2047_x2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int output_size)
{
    __shared__ float shared_input[FAST_SHARED_SIZE];

    const int tx = threadIdx.x;

    // Each block computes 512 outputs.
    const int base =
        blockIdx.x * OUTPUTS_PER_BLOCK;


    // ========================================================
    // Is this a completely full output block?
    //
    // This condition is identical for every thread in the
    // block, so it does not create warp divergence.
    // ========================================================

    const bool full_block =
        base + OUTPUTS_PER_BLOCK <= output_size;


    // ========================================================
    // LOAD GLOBAL INPUT -> SHARED MEMORY
    //
    // We need:
    //
    // 512 outputs
    // +
    // 2046-element halo
    //
    // = 2558 input floats.
    //
    // 256 threads cooperatively load them.
    // ========================================================

    for (int s = tx;
         s < FAST_SHARED_SIZE;
         s += THREADS)
    {
        const int global_index =
            base + s;

        if (full_block ||
            global_index < input_size)
        {
            shared_input[s] =
                input[global_index];
        }
    }

    __syncthreads();


    // ========================================================
    // OUTPUT INDICES
    // ========================================================

    const int out0 =
        base + tx;

    const int out1 =
        out0 + THREADS;


    // ========================================================
    // FAST PATH
    //
    // Almost every block in the benchmark goes here.
    //
    // input_size  = 1,500,000
    // kernel_size = 2047
    //
    // output_size = 1,497,954
    //
    // There are:
    //
    // 2925 complete 512-output blocks
    //
    // and only one partial final block.
    // ========================================================

    if (full_block)
    {
        // ----------------------------------------------------
        // Four accumulation chains for output 0.
        // ----------------------------------------------------

        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;


        // ----------------------------------------------------
        // Four accumulation chains for output 1.
        //
        // Eight independent accumulators total.
        // ----------------------------------------------------

        float b0 = 0.0f;
        float b1 = 0.0f;
        float b2 = 0.0f;
        float b3 = 0.0f;


        // ====================================================
        // Process 8 filter values per loop iteration.
        //
        // 2040 / 8 = 255 iterations.
        //
        // This cuts loop-control overhead roughly in half
        // compared with processing four values/iteration.
        // ====================================================

        for (int j = 0;
             j < 2040;
             j += 8)
        {
            // -----------------------------------------------
            // Load each kernel coefficient ONCE.
            //
            // Each value is then reused for BOTH outputs.
            // -----------------------------------------------

            const float k0 = c_kernel[j    ];
            const float k1 = c_kernel[j + 1];
            const float k2 = c_kernel[j + 2];
            const float k3 = c_kernel[j + 3];

            const float k4 = c_kernel[j + 4];
            const float k5 = c_kernel[j + 5];
            const float k6 = c_kernel[j + 6];
            const float k7 = c_kernel[j + 7];


            // -----------------------------------------------
            // First four filter positions
            // -----------------------------------------------

            a0 = fmaf(
                shared_input[tx + j],
                k0,
                a0
            );

            b0 = fmaf(
                shared_input[tx + THREADS + j],
                k0,
                b0
            );


            a1 = fmaf(
                shared_input[tx + j + 1],
                k1,
                a1
            );

            b1 = fmaf(
                shared_input[tx + THREADS + j + 1],
                k1,
                b1
            );


            a2 = fmaf(
                shared_input[tx + j + 2],
                k2,
                a2
            );

            b2 = fmaf(
                shared_input[tx + THREADS + j + 2],
                k2,
                b2
            );


            a3 = fmaf(
                shared_input[tx + j + 3],
                k3,
                a3
            );

            b3 = fmaf(
                shared_input[tx + THREADS + j + 3],
                k3,
                b3
            );


            // -----------------------------------------------
            // Next four filter positions.
            //
            // Reuse the same accumulator chains.
            // -----------------------------------------------

            a0 = fmaf(
                shared_input[tx + j + 4],
                k4,
                a0
            );

            b0 = fmaf(
                shared_input[tx + THREADS + j + 4],
                k4,
                b0
            );


            a1 = fmaf(
                shared_input[tx + j + 5],
                k5,
                a1
            );

            b1 = fmaf(
                shared_input[tx + THREADS + j + 5],
                k5,
                b1
            );


            a2 = fmaf(
                shared_input[tx + j + 6],
                k6,
                a2
            );

            b2 = fmaf(
                shared_input[tx + THREADS + j + 6],
                k6,
                b2
            );


            a3 = fmaf(
                shared_input[tx + j + 7],
                k7,
                a3
            );

            b3 = fmaf(
                shared_input[tx + THREADS + j + 7],
                k7,
                b3
            );
        }


        // ====================================================
        // Remaining seven filter elements:
        //
        // 2040 .. 2046
        // ====================================================

        a0 = fmaf(
            shared_input[tx + 2040],
            c_kernel[2040],
            a0
        );

        b0 = fmaf(
            shared_input[tx + THREADS + 2040],
            c_kernel[2040],
            b0
        );


        a1 = fmaf(
            shared_input[tx + 2041],
            c_kernel[2041],
            a1
        );

        b1 = fmaf(
            shared_input[tx + THREADS + 2041],
            c_kernel[2041],
            b1
        );


        a2 = fmaf(
            shared_input[tx + 2042],
            c_kernel[2042],
            a2
        );

        b2 = fmaf(
            shared_input[tx + THREADS + 2042],
            c_kernel[2042],
            b2
        );


        a3 = fmaf(
            shared_input[tx + 2043],
            c_kernel[2043],
            a3
        );

        b3 = fmaf(
            shared_input[tx + THREADS + 2043],
            c_kernel[2043],
            b3
        );


        a0 = fmaf(
            shared_input[tx + 2044],
            c_kernel[2044],
            a0
        );

        b0 = fmaf(
            shared_input[tx + THREADS + 2044],
            c_kernel[2044],
            b0
        );


        a1 = fmaf(
            shared_input[tx + 2045],
            c_kernel[2045],
            a1
        );

        b1 = fmaf(
            shared_input[tx + THREADS + 2045],
            c_kernel[2045],
            b1
        );


        a2 = fmaf(
            shared_input[tx + 2046],
            c_kernel[2046],
            a2
        );

        b2 = fmaf(
            shared_input[tx + THREADS + 2046],
            c_kernel[2046],
            b2
        );


        // ====================================================
        // REDUCE REGISTER ACCUMULATORS
        // ====================================================

        output[out0] =
            (a0 + a1) +
            (a2 + a3);

        output[out1] =
            (b0 + b1) +
            (b2 + b3);
    }

    else
    {
        // ====================================================
        // FINAL PARTIAL BLOCK
        //
        // Performance here barely matters because there is
        // only one such block for the benchmark.
        // ====================================================

        if (out0 < output_size)
        {
            float sum = 0.0f;

            for (int j = 0;
                 j < MAX_KERNEL;
                 ++j)
            {
                sum = fmaf(
                    shared_input[tx + j],
                    c_kernel[j],
                    sum
                );
            }

            output[out0] = sum;
        }


        if (out1 < output_size)
        {
            float sum = 0.0f;

            for (int j = 0;
                 j < MAX_KERNEL;
                 ++j)
            {
                sum = fmaf(
                    shared_input[tx + THREADS + j],
                    c_kernel[j],
                    sum
                );
            }

            output[out1] = sum;
        }
    }
}


// ============================================================
// GENERIC FALLBACK
//
// Used for every kernel_size other than 2047.
// ============================================================

__global__ void convolution_generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size,
    int output_size)
{
    extern __shared__ float shared_input[];

    const int tx =
        threadIdx.x;

    const int block_start =
        blockIdx.x * blockDim.x;

    const int i =
        block_start + tx;

    const int shared_size =
        blockDim.x +
        kernel_size -
        1;


    // --------------------------------------------------------
    // Load input tile.
    // --------------------------------------------------------

    for (int s = tx;
         s < shared_size;
         s += blockDim.x)
    {
        const int global_index =
            block_start + s;

        if (global_index < input_size)
        {
            shared_input[s] =
                input[global_index];
        }
    }

    __syncthreads();


    if (i < output_size)
    {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        int j = 0;


        // ----------------------------------------------------
        // Four independent accumulator chains.
        // ----------------------------------------------------

        for (;
             j + 3 < kernel_size;
             j += 4)
        {
            sum0 = fmaf(
                shared_input[tx + j],
                c_kernel[j],
                sum0
            );

            sum1 = fmaf(
                shared_input[tx + j + 1],
                c_kernel[j + 1],
                sum1
            );

            sum2 = fmaf(
                shared_input[tx + j + 2],
                c_kernel[j + 2],
                sum2
            );

            sum3 = fmaf(
                shared_input[tx + j + 3],
                c_kernel[j + 3],
                sum3
            );
        }


        float sum =
            (sum0 + sum1) +
            (sum2 + sum3);


        for (;
             j < kernel_size;
             ++j)
        {
            sum = fmaf(
                shared_input[tx + j],
                c_kernel[j],
                sum
            );
        }


        output[i] = sum;
    }
}


// ============================================================
// solve
// ============================================================

extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_size,
    int kernel_size)
{
    const int output_size =
        input_size -
        kernel_size +
        1;


    // ========================================================
    // Copy filter into constant memory.
    // ========================================================

    cudaMemcpyToSymbol(
        c_kernel,
        kernel,
        kernel_size * sizeof(float),
        0,
        cudaMemcpyDeviceToDevice
    );


    // ========================================================
    // PERFORMANCE PATH
    // ========================================================

    if (kernel_size == 2047)
    {
        const int blocks =
            (output_size +
             OUTPUTS_PER_BLOCK -
             1) /
            OUTPUTS_PER_BLOCK;


        convolution_2047_x2
            <<<blocks, THREADS>>>(
                input,
                output,
                input_size,
                output_size
            );
    }

    // ========================================================
    // GENERIC PATH
    // ========================================================

    else
    {
        const int blocks =
            (output_size +
             THREADS -
             1) /
            THREADS;


        const int shared_bytes =
            (THREADS +
             kernel_size -
             1) *
            sizeof(float);


        convolution_generic
            <<<blocks,
               THREADS,
               shared_bytes>>>(
                input,
                output,
                input_size,
                kernel_size,
                output_size
            );
    }


    cudaDeviceSynchronize();
}
