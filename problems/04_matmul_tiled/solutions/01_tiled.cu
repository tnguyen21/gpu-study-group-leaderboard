/*
 * Tiled Matrix Multiplication with Shared Memory
 *
 * Uses shared memory to cache tiles of A and B matrices,
 * reducing global memory accesses by a factor of TILE_WIDTH.
 *
 * Each phase:
 * 1. Load one tile of A and one tile of B into shared memory
 * 2. Synchronize threads
 * 3. Compute partial dot product from the tiles
 * 4. Synchronize before loading next tiles
 */

#define TILE_WIDTH 16

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // Load tile of A
        if (row < M && (ph * TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[row * K + ph * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B
        if ((ph * TILE_WIDTH + ty) < K && col < N) {
            Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
