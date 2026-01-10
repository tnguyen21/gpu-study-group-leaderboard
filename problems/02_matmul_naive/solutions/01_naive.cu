/*
 * Naive Matrix Multiplication
 *
 * Each thread computes one element of the output matrix C.
 * This version has poor memory access patterns - B is accessed
 * column-wise, which is not coalesced in row-major storage.
 *
 * Complexity: O(M * N * K) work, but high memory bandwidth usage
 */

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
