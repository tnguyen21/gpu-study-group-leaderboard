/*
 * Problem: Histogram
 * PMPP Chapter 9
 *
 * Compute a histogram of an array of unsigned chars.
 * Count how many times each value (0-255) appears in the input.
 *
 * Input:
 *   - data: input array of N unsigned chars
 *   - n: number of elements (N = 1M = 1,048,576)
 *
 * Output:
 *   - hist: array of 256 unsigned ints, where hist[i] = count of value i
 *
 * Key concepts:
 *   - Atomic operations (atomicAdd)
 *   - Privatization: each block maintains a private histogram in shared memory
 *   - Thread coarsening: each thread processes multiple elements
 *   - Coalesced memory access patterns
 *
 * Optimization ideas:
 *   - Use shared memory for privatized histograms per block
 *   - Reduce atomic contention with privatization
 *   - Thread coarsening to process multiple elements per thread
 *   - Coalesced access: threads read consecutive memory locations
 */

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    // Your implementation here
}
