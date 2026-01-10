/*
 * Bitonic Sort
 *
 * A comparison-based sorting network that works well on GPUs.
 * Forms bitonic sequences and merges them recursively.
 *
 * A bitonic sequence is one that first increases then decreases
 * (or vice versa). Bitonic merge compares elements at fixed
 * distances and swaps if needed.
 *
 * Complexity: O(n log^2 n) comparisons, highly parallel
 *
 * Note: This is a simplified single-kernel version.
 * For large arrays, use multiple kernels with increasing step sizes.
 */

__global__ void sort(unsigned int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Bitonic sort network
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            __syncthreads();

            for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
                int ixj = i ^ j;

                if (ixj > i) {
                    // Determine sort direction
                    bool ascending = ((i & k) == 0);

                    if (ascending) {
                        if (data[i] > data[ixj]) {
                            unsigned int tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                        }
                    } else {
                        if (data[i] < data[ixj]) {
                            unsigned int tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}
