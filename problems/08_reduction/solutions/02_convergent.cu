/*
 * Convergent Sum Reduction
 *
 * Uses convergent reduction pattern: threads with consecutive indices
 * are active, avoiding warp divergence until the final iterations.
 *
 * Key change: stride decreases instead of increases, and we use
 * tid < stride instead of tid % (2*stride) == 0.
 */

__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Convergent tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
