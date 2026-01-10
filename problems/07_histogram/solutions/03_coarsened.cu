/*
 * Histogram with Privatization and Thread Coarsening
 *
 * Combines privatization with thread coarsening: each thread
 * processes multiple elements using a grid-stride loop.
 *
 * Also uses coalesced memory access pattern where consecutive
 * threads read consecutive memory locations.
 */

#define NUM_BINS 256

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    __shared__ unsigned int hist_s[NUM_BINS];

    // Initialize shared memory
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        hist_s[bin] = 0;
    }
    __syncthreads();

    // Grid-stride loop with coalesced access
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&hist_s[data[i]], 1);
    }
    __syncthreads();

    // Merge to global memory
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        if (hist_s[bin] > 0) {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}
