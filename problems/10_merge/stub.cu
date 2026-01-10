/*
 * Problem: Parallel Merge
 * PMPP Chapter 12
 *
 * Merge two sorted arrays A and B into a single sorted array C.
 *
 * Input:
 *   - A: sorted array of m floats (m = 500,000)
 *   - B: sorted array of n floats (n = 500,000)
 *   - m, n: sizes of input arrays
 *
 * Output:
 *   - C: merged sorted array of (m + n) floats
 *
 * Key concepts:
 *   - Co-rank function: given output index k, find (i, j) where
 *     i + j = k and A[0..i-1], B[0..j-1] are the elements before C[k]
 *   - Binary search to find co-rank in O(log(min(m,n))) time
 *   - Each thread independently merges its portion using co-rank
 *
 * Challenge: dividing work evenly among threads when A and B
 * have different distributions of values.
 *
 * Optimization ideas:
 *   - Tiled merge: load tiles into shared memory
 *   - Reduce global memory binary searches
 *   - Better load balancing
 */

// Helper: find co-rank for output position k
__device__ int co_rank(int k, float *A, int m, float *B, int n);

// Helper: sequential merge of small arrays
__device__ void merge_sequential(float *A, int m, float *B, int n, float *C);

__global__ void merge(float *A, int m, float *B, int n, float *C) {
    // Your implementation here
}
