/*
 * Naive Histogram with Global Atomics
 *
 * Each thread processes one element and atomically increments
 * the corresponding bin in global memory.
 *
 * Simple but slow due to high atomic contention on popular bins.
 */

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&hist[data[i]], 1);
    }
}
