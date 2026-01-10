/*
 * Tiled Matrix Multiplication with 32x32 Tiles
 *
 * Larger tiles mean more data reuse from shared memory,
 * but also require more shared memory per block.
 *
 * 32x32 tiles use 32*32*4*2 = 8KB of shared memory for As and Bs.
 */

#define TILE_WIDTH 32

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

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

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
