/*
 * Naive 7-Point 3D Stencil
 *
 * Each thread computes one output point by reading
 * 7 values from global memory (center + 6 neighbors).
 *
 * Uses 3D thread blocks mapping directly to the 3D grid.
 */

__global__ void stencil(float *in, float *out, int N,
                        float c0, float c1, float c2, float c3,
                        float c4, float c5, float c6) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0 * in[i*N*N + j*N + k]
                             + c1 * in[i*N*N + j*N + (k - 1)]
                             + c2 * in[i*N*N + j*N + (k + 1)]
                             + c3 * in[i*N*N + (j - 1)*N + k]
                             + c4 * in[i*N*N + (j + 1)*N + k]
                             + c5 * in[(i - 1)*N*N + j*N + k]
                             + c6 * in[(i + 1)*N*N + j*N + k];
    }
}
