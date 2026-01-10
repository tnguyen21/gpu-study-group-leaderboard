/*
 * Problem: Parallel Sort
 * PMPP Chapter 13
 *
 * Sort an array of unsigned integers in ascending order.
 *
 * Input:
 *   - data: array of N unsigned integers (N = 262144 = 2^18)
 *   - n: number of elements
 *
 * Output:
 *   - data: sorted in-place (ascending order)
 *
 * Key algorithms:
 *   - Merge sort: recursively divide, sort, and merge
 *   - Radix sort: sort by each bit position, LSB to MSB
 *   - Bitonic sort: compare-and-swap network
 *
 * For GPU merge sort:
 *   1. Start with width=1 (each element is sorted)
 *   2. Merge pairs of sorted subarrays
 *   3. Double width, repeat until whole array is sorted
 *
 * For radix sort:
 *   1. For each bit position (LSB to MSB):
 *   2. Partition elements by that bit (0s before 1s)
 *   3. Use scan to compute destination indices
 *
 * Optimization ideas:
 *   - Block-level sorting before merge phases
 *   - Shared memory for small merges
 *   - Avoid bank conflicts in radix sort
 */

// Sort data in-place
// Note: You may define helper kernels above this
__global__ void sort(unsigned int *data, int n) {
    // Your implementation here
}
