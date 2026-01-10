/*
 * Problem: Parallel Prefix Sum (Inclusive Scan)
 * PMPP Chapter 11
 *
 * Compute the inclusive prefix sum of an array:
 *   output[i] = input[0] + input[1] + ... + input[i]
 *
 * Input:
 *   - input: array of N floats (N = 65536 = 2^16)
 *   - output: array of N floats
 *   - n: number of elements
 *
 * Example: [3, 1, 4, 1, 5] -> [3, 4, 8, 9, 14]
 *
 * Key concepts:
 *   - Kogge-Stone algorithm: O(N log N) work, O(log N) depth
 *   - Brent-Kung algorithm: O(N) work, O(log N) depth (more efficient)
 *   - Two-phase approach: reduction tree + reverse tree
 *   - Double buffering to avoid WAR hazards
 *
 * For arrays larger than one block:
 *   - Phase 1: Each block computes its local scan
 *   - Phase 2: Scan the block sums
 *   - Phase 3: Add block prefix to each element
 *
 * Optimization ideas:
 *   - Brent-Kung is more work-efficient than Kogge-Stone
 *   - Double buffering eliminates one syncthreads per iteration
 *   - Minimize bank conflicts in shared memory
 */

__global__ void scan(float *input, float *output, int n) {
    // Your implementation here
}
