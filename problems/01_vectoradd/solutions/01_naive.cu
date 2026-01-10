/*
 * Naive Vector Addition
 *
 * This is the simplest implementation - one thread per element.
 * Each thread computes exactly one output element.
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
