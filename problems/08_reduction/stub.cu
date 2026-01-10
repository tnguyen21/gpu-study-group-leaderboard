/*
 * Problem: Parallel Reduction (Sum)
 * PMPP Chapter 10
 *
 * Compute the sum of all elements in an array.
 *
 * Input:
 *   - input: array of N floats (N = 1M = 1,048,576)
 *   - output: single float (sum of all elements)
 *   - n: number of elements
 *
 * Key concepts:
 *   - Tree reduction pattern
 *   - Control divergence and how to avoid it
 *   - Shared memory for block-level reduction
 *   - Thread coarsening to reduce overhead
 *   - Multiple kernel launches or atomic operations for final reduction
 *
 * The challenge is to sum millions of elements efficiently:
 *   - Block-level: each block reduces its portion to one value
 *   - Global: combine block results (atomic or second kernel)
 *
 * Optimization ideas:
 *   - Avoid divergent warps by using convergent reduction pattern
 *   - Thread coarsening: each thread sums multiple elements first
 *   - Warp shuffle instructions (__shfl_down_sync) for final warp
 *   - Sequential addressing vs interleaved addressing
 */

__global__ void reduce(float *input, float *output, int n) {
    // Your implementation here
}
