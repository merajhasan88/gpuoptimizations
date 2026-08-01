#include <cuda_runtime.h>


/* Understanding row-major matrix multiplication
#include <stdio.h>
#include <stdlib.h>

#define N 512  // Matrix dimension (N x N)

// Standard row-major multiplication
// Matrix B is accessed by column, causing cache misses
void multiply_standard(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j]; // B[k][j] jumps across memory rows
            }
        }
    }
}

// Optimized row-major multiplication
// Transposing B converts column accesses into linear row accesses
void multiply_optimized(double A[N][N], double B[N][N], double C[N][N]) {
    static double B_T[N][N]; // Temporary matrix for B transpose

    // Step 1: Transpose B so columns become rows in memory
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_T[j][i] = B[i][j];
        }
    }

    // Step 2: Multiply using linear row steps for both A and B_T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B_T[j][k]; // Perfect cache locality for both
            }
            C[i][j] = sum;
        }
    }
}
*/

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {

/*blockIdx.x identifies the block horizontally.
blockDim.x is the number of threads horizontally in each block.
* threadIdx.x identifies the thread horizontally inside its block.
*
* Together, they determine which column of C this thread computes.
*/
int col = blockIdx.x * blockDim.x + threadIdx.x;

/*The same calculation vertically determines which row of C
* this thread computes.
*/

int row = blockIdx.y * blockDim.y + threadIdx.y;
/*
* The grid may contain extra threads when M or K is not divisible
* by the block dimensions.
*
* Only threads corresponding to valid C elements should continue.
*/

if(row<M && col <K){
    float sum= 0.0f;
     /*
         * Matrix A has dimensions M x N.
         * Matrix B has dimensions N x K.
         *
         * To compute C[row][col], multiply:
         *
         * A[row][0] by B[0][col]
         * A[row][1] by B[1][col]
         * ...
         * A[row][N - 1] by B[N - 1][col]
         *
         * Then add all those products.
         */
    /*
C[2][3] =
    A[2][0] * B[0][3]
  + A[2][1] * B[1][3]
  + A[2][2] * B[2][3]
  + ...
  + A[2][N - 1] * B[N - 1][3]
    */
    //Another example:
    //For a 2x3 A and 3x2 B, first row of A (A[0][0]) multiplies to first column 
    //of B (B[0]0])
    //to give C[0][0]. Then first row
    //of A A[0][1] multiplies to second column of B (B[0][1]) to give C[0][1].
    //A 2nd row with B 1st column gives C[1][0]. 
    //2nd row of A with B 2nd column gives C[1][1]
    //C[0][0]=A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
    //C[0][1]=A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
    //C[1][0]=A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0]
    //C[1][1]=A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1]
    /*
The general pattern is:

C[row][col] += A[row][i] * B[i][col];

where i runs from 0 to N - 1. (reducing element)
    */
    for(int i = 0; i< N; i++){
             /*
             * Row-major indexing:
             *
             * A[row][i] becomes:
             *     A[row * N + i]
             *
             * B[i][col] becomes:
             *     B[i * K + col]
             *
            For A that is 2x3, in row-major memory, it is stored consecutively as:

Index:      0        1        2        3        4        5
Value:   A[0][0]  A[0][1]  A[0][2]  A[1][0]  A[1][1]  A[1][2]
MxN = 6 elements. 
To reach row row, you first skip all preceding rows.

Each row has N elements, so the beginning of a row is:

row * N

Then move i columns into that row:

row * N + i

The general row-major formula from the general matrix formula 
(C[row][col] += A[row][i] * B[i][col];)
 is:

matrix[rowindex * number_of_columns + columnindex]
*/

sum+= A[row*N+i]*B[i*K+col];

    }
    C[row*K+col]=sum;
}

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
   /*
Matrix C has dimensions:

M rows x K columns

Therefore:

blocksPerGrid.x

must cover the K columns, while:

blocksPerGrid.y

must cover the M rows.
   */
   
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
//The +15 performs ceiling division, ensuring partially filled edge blocks are also created.
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
