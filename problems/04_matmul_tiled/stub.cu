/*
 * Problem: Tiled Matrix Multiplication
 * PMPP Chapter 5-6
 *
 * Compute C = A * B using shared memory tiling.
 *
 * This is the same computation as 02_matmul_naive, but now you should
 * use shared memory to reduce global memory accesses.
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
 *   - Shared memory (__shared__ keyword)
 *   - Tiling: load a tile of A and B into shared memory
 *   - __syncthreads() to synchronize threads within a block
 *   - Boundary checking for tiles at matrix edges
 *
 * The naive implementation reads each element of A and B from global
 * memory K times. With tiling, each element is read once per tile,
 * reducing global memory accesses by a factor of TILE_WIDTH.
 *
 * Optimization ideas:
 *   - Different tile sizes (16x16, 32x32)
 *   - Thread coarsening (each thread computes multiple outputs)
 *   - Register tiling
 */

#define TILE_WIDTH 16

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    // Your implementation here
    // Hint: Use __shared__ float As[TILE_WIDTH][TILE_WIDTH];
}
