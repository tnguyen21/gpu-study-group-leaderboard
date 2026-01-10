/*
 * Brent-Kung Parallel Scan
 *
 * Work-efficient: O(N) additions (vs O(N log N) for Kogge-Stone).
 * Two phases:
 *   1. Reduction (up-sweep): build partial sums tree
 *   2. Down-sweep: distribute sums to all elements
 *
 * Note: This version handles one block worth of data (up to 2*blockDim.x).
 */

#define SECTION_SIZE 2048  // Must be 2 * blockDim.x

__global__ void scan(float *input, float *output, int n) {
    __shared__ float XY[SECTION_SIZE];

    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + tid;

    // Load two elements per thread
    XY[tid] = (i < n) ? input[i] : 0.0f;
    XY[tid + blockDim.x] = (i + blockDim.x < n) ? input[i + blockDim.x] : 0.0f;

    // Reduction phase (up-sweep)
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    // Down-sweep phase (reverse tree)
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();

    // Write results
    if (i < n) output[i] = XY[tid];
    if (i + blockDim.x < n) output[i + blockDim.x] = XY[tid + blockDim.x];
}
