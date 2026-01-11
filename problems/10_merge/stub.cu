/*
 * Problem: Parallel Merge
 * Merge two sorted arrays into one sorted array.
 *
 * Inputs:
 *   - `A`: sorted array of length `m`
 *   - `B`: sorted array of length `n`
 *
 * Output:
 *   - `C`: sorted array of length `m + n` containing all elements of `A` and `B`
 *
 * Tie-breaking:
 *   - If `A[i] == B[j]`, place `A[i]` before `B[j]` (stable w.r.t. `A` first).
 *
 * Launch configuration:
 *   - 1D grid, 1D block
 *   - blockDim.x = 256
 *   - gridDim.x = 256 (fixed number of blocks)
 *   - Hint: Use co-rank / merge-path to divide work among threads
 *
 * Example:
 *   - A = [0.5, 2.5, 4.5], B = [0, 2, 4] -> C = [0, 0.5, 2, 2.5, 4, 4.5]
 */

__global__ void merge(float *A, int m, float *B, int n, float *C) {
    // Your implementation here
}
