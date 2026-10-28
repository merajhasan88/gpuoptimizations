#include <cuda_runtime.h>
#include <math.h>

constexpr int THREADS = 256;

// Step 1: combine the gradient and momentum buffer.
// X temporarily holds V = momentum * Buf + G.
// G and Buf are read-only; we do not update the buffer itself.
__global__ void make_direction(const float* G, const float* Buf,
                               float* X, int count, float momentum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        X[i] = momentum * Buf[i] + G[i];
    }
}

// Step 2a: compute the normalization denominator.
// Frobenius norm = sqrt(sum of the squares of ALL matrix entries).
// One block handles this reduction to keep the baseline simple.
__global__ void compute_norm(const float* X, float* denominator, int count) {
    __shared__ float sums[THREADS];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < count; i += THREADS) {
        sum += X[i] * X[i];
    }
    sums[tid] = sum;
    __syncthreads();

    // Combine 256 partial sums into sums[0].
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Add epsilon AFTER taking the square root, as specified.
        denominator[0] = sqrtf(sums[0]) + 1.0e-7f;
    }
}

// Step 2b: X = V / (||V||_F + 1e-7).
__global__ void normalize(float* X, const float* denominator, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        X[i] /= denominator[0];
    }
}

// Step 3a: A = X * transpose(X).
// X is M x N. A is M x M.
// One thread computes the dot product of two rows of X.
__global__ void gram_matrix(const float* X, float* A, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * M) return;

    int row = i / M;
    int col = i % M;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum = fmaf(X[row * N + k], X[col * N + k], sum);
    }
    A[i] = sum;
}

// Step 3b: B = b*A + c*(A*A).
// A*A means MATRIX multiplication, not elementwise squaring.
__global__ void polynomial_matrix(const float* A, float* B, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * M) return;

    int row = i / M;
    int col = i % M;

    float sum = 0.0f;
    for (int k = 0; k < M; ++k) {
        sum = fmaf(A[row * M + k], A[k * M + col], sum);
    }
    B[i] = -4.7750f * A[i] + 2.0315f * sum;
}

// Step 3c: next_X = a*X + B*X.
// Use a separate destination: other threads still need the OLD X.
__global__ void newton_schulz_update(const float* X, const float* B,
                                     float* next_X, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;

    int row = i / N;
    int col = i % N;

    float sum = 0.0f;
    for (int k = 0; k < M; ++k) {
        sum = fmaf(B[row * M + k], X[k * N + col], sum);
    }
    next_X[i] = 3.4445f * X[i] + sum;
}

// Step 4: write every output entry with the challenge's final scale.
__global__ void write_output(const float* X, float* output,
                              int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = scale * X[i];
    }
}

// G, Buf, and output are device pointers. Matrices are row-major.
extern "C" void solve(const float* G, const float* Buf, float* output,
                      int M, int N, float momentum) {
    int count = M * N;
    int square = M * M;
    int blocks_X = (count + THREADS - 1) / THREADS;
    int blocks_A = (square + THREADS - 1) / THREADS;

    // One allocation containing two M x N matrices, two M x M
    // matrices, and one scalar. Every temporary is written before use.
    float* workspace = nullptr;
    cudaMalloc(&workspace, (2 * count + 2 * square + 1) * sizeof(float));
    float* X = workspace;
    float* next_X = X + count;
    float* A = next_X + count;
    float* B = A + square;
    float* denominator = B + square;

    make_direction<<<blocks_X, THREADS>>>(G, Buf, X, count, momentum);
    compute_norm<<<1, THREADS>>>(X, denominator, count);
    normalize<<<blocks_X, THREADS>>>(X, denominator, count);

    // Exactly five steps. Each launch uses the same default stream,
    // so its inputs are ready when the preceding kernel finishes.
    for (int step = 0; step < 5; ++step) {
        gram_matrix<<<blocks_A, THREADS>>>(X, A, M, N);
        polynomial_matrix<<<blocks_A, THREADS>>>(A, B, M);
        newton_schulz_update<<<blocks_X, THREADS>>>(X, B, next_X, M, N);

        // Swap pointers, without copying matrix data.
        float* old_X = X;
        X = next_X;
        next_X = old_X;
    }

    float scale = sqrtf(static_cast<float>(M > N ? M : N));
    write_output<<<blocks_X, THREADS>>>(X, output, count, scale);

    // Free the original allocation, not one of the swapped pointers.
    // Per-call allocation/free is deliberately simple for this baseline.
    cudaFree(workspace);
}
