/*
 * Problem: Naive Matrix Multiplication
 * PMPP Chapter 3
 *
 * Compute C = A * B where:
 *   - A is an M x K matrix
 *   - B is a K x N matrix
 *   - C is an M x N matrix (output)
 *
 * All matrices are stored in row-major order.
 *
 * Input:
 *   - A: input matrix (M x K = 1024 x 1024)
 *   - B: input matrix (K x N = 1024 x 1024)
 *   - M, N, K: matrix dimensions
 *
 * Output:
 *   - C: output matrix (M x N = 1024 x 1024)
 *
 * Key concepts:
 *   - 2D thread blocks (typically 16x16 or 32x32)
 *   - Row-major indexing: A[row][col] = A[row * width + col]
 *   - Each thread computes one element of C
 *
 * Optimization ideas:
 *   - This naive version has poor memory access patterns
 *   - See 04_matmul_tiled for the shared memory optimization
 */

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    // Your implementation here
    // Hint: Each thread computes C[row][col] = sum of A[row][i] * B[i][col]
}
