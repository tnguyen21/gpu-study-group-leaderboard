/*
 * Histogram with Privatization (Shared Memory)
 *
 * Each block maintains a private histogram in shared memory.
 * Atomics are used within shared memory (much faster than global).
 * At the end, private histograms are merged to global memory.
 *
 * This reduces contention on global memory atomics significantly.
 */

#define NUM_BINS 256

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    // Privatized histogram in shared memory
    __shared__ unsigned int hist_s[NUM_BINS];

    // Initialize shared memory histogram to 0
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        hist_s[bin] = 0;
    }
    __syncthreads();

    // Count into private histogram
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&hist_s[data[i]], 1);
    }
    __syncthreads();

    // Merge private histogram to global memory
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        if (hist_s[bin] > 0) {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}
