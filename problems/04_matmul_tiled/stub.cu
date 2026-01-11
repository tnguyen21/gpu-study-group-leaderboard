/*
 * Problem: Tiled Matrix Multiplication
 * Compute matrix product `C = A * B` (same math as `02_matmul_naive`).
 *
 * Shapes:
 *   - `A`: M x K (row-major)
 *   - `B`: K x N (row-major)
 *   - `C`: M x N (row-major)
 *
 * Definition:
 *   - `C[row, col] = sum_{k=0..K-1} A[row, k] * B[k, col]`
 *
 * Example:
 *   - A (2x3) = [[1,2,3],[4,5,6]]
 *   - B (3x2) = [[7,8],[9,10],[11,12]]
 *   - C (2x2) = [[58,64],[139,154]]
 */

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    // Your implementation here
}
