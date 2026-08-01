#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
   int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;

    
    int row = blockIdx.y * blockDim.y + ty;

   /*
Every thread repeatedly reads values from global memory, which is relatively slow.

Neighboring threads also load many of the same values.

For example, threads calculating:

C[0][0]
C[0][1]
C[0][2]

all need the same row of A:

A[0][0], A[0][1], A[0][2], ...

But the basic kernel makes every thread load that row separately.

Similarly, threads calculating:

C[0][0]
C[1][0]
C[2][0]

all need the same column of B, but each thread loads it separately.
   */
float sum = 0.0f;
__shared__ float shared_A[16][16];
__shared__ float shared_B[16][16];
    
        

        
        for (int phase = 0; phase < (N + 15) / 16; ++phase)
        {   

            //Fill up the shared matrices first for each thread.
            //A 16x16 chunk of A and B is filled. 
            int a_col = phase * 16 + tx;
            if (row < M && a_col < N)
        {
            shared_A[ty][tx] =
                A[row * N + a_col];
        }
        else
        {
            /*
             * The thread is outside the valid matrix boundary.
             * Store zero so it does not affect the multiplication.
             */
            shared_A[ty][tx] = 0.0f;
        }
           int b_row = phase * 16 + ty;

        if (b_row < N && col < K)
        {
            shared_B[ty][tx] =
                B[b_row * K + col];
        }
        else
        {
            shared_B[ty][tx] = 0.0f;
        }
        __syncthreads();
            #pragma unroll
            for (int i = 0; i<16; ++i){
            sum += shared_A[ty][i] * shared_B[i][tx];
            //Standard matmul because 
            //the same values were already copied from A and B into faster shared memory.
            }
          __syncthreads();   
        }

             
    
     if (row < M && col < K)
    {
        C[row * K + col] = sum;
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
    
    dim3 threadsPerBlock(16, 16);
/*

one block contains:

16 × 16 = 256 threads

We can make that block compute a:

16 × 16 tile of C

Each thread still computes exactly one output:

Thread (0,0) computes one C element
Thread (1,0) computes one C element
...

The difference is that the threads cooperate 
to load pieces of A and B into shared memory.
*/
    
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
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
