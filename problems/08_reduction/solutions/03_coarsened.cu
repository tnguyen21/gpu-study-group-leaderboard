/*
 * Coarsened Sum Reduction
 *
 * Each thread first sums multiple elements (coarsening factor),
 * then performs tree reduction on the partial sums.
 *
 * This reduces the number of blocks needed and increases
 * arithmetic intensity per thread.
 */

#define COARSE_FACTOR 4

__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;
    int i = segment + tid;

    // Each thread sums COARSE_FACTOR elements
    float sum = 0.0f;
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int idx = i + c * blockDim.x;
        if (idx < n) {
            sum += input[idx];
        }
    }
    sdata[tid] = sum;
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
