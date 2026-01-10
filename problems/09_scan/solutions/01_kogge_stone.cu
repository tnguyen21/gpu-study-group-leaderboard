/*
 * Kogge-Stone Parallel Scan
 *
 * Simple but not work-efficient: O(N log N) additions.
 * Each iteration, element i adds element i-stride.
 *
 * Uses double buffering to avoid WAR race conditions.
 *
 * Note: This version handles one block worth of data.
 * For larger arrays, need hierarchical approach.
 */

__global__ void scan(float *input, float *output, int n) {
    extern __shared__ float shared_mem[];
    float *buffer1 = shared_mem;
    float *buffer2 = &shared_mem[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load into first buffer
    buffer1[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    float *src = buffer1;
    float *dst = buffer2;

    // Kogge-Stone: each iteration doubles the stride
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            dst[tid] = src[tid] + src[tid - stride];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();

        // Swap buffers
        float *tmp = src;
        src = dst;
        dst = tmp;
    }

    // Write result
    if (i < n) {
        output[i] = src[tid];
    }
}
