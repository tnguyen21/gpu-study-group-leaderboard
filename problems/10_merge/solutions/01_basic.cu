/*
 * Basic Parallel Merge
 *
 * Each thread uses co-rank to find its portion of A and B,
 * then sequentially merges those elements.
 *
 * Co-rank(k): Given output index k, find i such that the first
 * k elements of the merged output come from A[0..i-1] and B[0..k-i-1].
 */

#define cdiv(x, y) (((x) + (y) - 1) / (y))

__device__ void merge_sequential(float *A, int m, float *B, int n, float *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

__device__ int co_rank(int k, float *A, int m, float *B, int n) {
    int i = min(k, m);
    int j = k - i;
    int i_low = max(0, k - n);
    int j_low = max(0, k - m);

    while (true) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            int delta = cdiv(i - i_low, 2);
            j_low = j;
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            int delta = cdiv(j - j_low, 2);
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            break;
        }
    }
    return i;
}

__global__ void merge(float *A, int m, float *B, int n, float *C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    int numThreads = blockDim.x * gridDim.x;

    // Each thread handles multiple elements
    int elementsPerThread = cdiv(total, numThreads);
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, total);

    if (k_curr >= total) return;

    // Find co-ranks
    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    // Sequential merge of this thread's portion
    merge_sequential(&A[i_curr], i_next - i_curr,
                     &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}
