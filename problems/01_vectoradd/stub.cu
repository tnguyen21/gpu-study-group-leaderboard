/*
 * Problem: Vector Addition
 * PMPP Chapter 2-3
 *
 * Add two vectors element-wise: c[i] = a[i] + b[i]
 *
 * Input:
 *   - a: input vector of N floats
 *   - b: input vector of N floats
 *   - n: number of elements (N = 1,048,576 = 2^20)
 *
 * Output:
 *   - c: output vector where c[i] = a[i] + b[i]
 *
 * Key concepts:
 *   - Thread indexing: blockIdx.x * blockDim.x + threadIdx.x
 *   - Bounds checking for when N isn't a multiple of block size
 *   - Memory coalescing (threads access consecutive memory)
 *
 * Optimization ideas:
 *   - Vectorized loads/stores (float4) to increase memory throughput
 *   - Loop unrolling to hide latency
 *   - Experiment with different block sizes (128, 256, 512, 1024)
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Your implementation here
}
