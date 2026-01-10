/*
 * Grid-Stride Loop Vector Addition
 *
 * Uses a grid-stride loop pattern where each thread processes multiple
 * elements spaced by the total number of threads. This allows using
 * fewer blocks while still processing all elements.
 *
 * Benefits:
 * - Works with any array size regardless of grid configuration
 * - Better for very large arrays (avoids launching millions of blocks)
 * - Can be combined with vectorization for further speedup
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
