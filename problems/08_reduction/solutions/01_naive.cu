/*
 * Naive Sum Reduction
 *
 * Uses tree reduction pattern within each block.
 * Final block results are atomically added to global output.
 *
 * Warning: This has control divergence - threads diverge based on
 * whether their index satisfies the stride condition.
 */

__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 adds block result to global output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
