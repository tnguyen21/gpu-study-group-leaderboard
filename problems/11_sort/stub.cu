/*
 * Problem: Parallel Sort
 * Sort `data[0..n-1]` in ascending order (in-place).
 *
 * Input / output:
 *   - `data`: array of length `n`
 *
 * Launch configuration:
 *   - 1D grid, 1D block
 *   - blockDim.x = 256
 *   - gridDim.x = ceil(n / 256)
 *   - Note: You may need multiple kernel launches or a complex single-pass algorithm
 *
 * Example:
 *   - [5, 1, 5, 2] -> [1, 2, 5, 5]
 */

__global__ void sort(unsigned int *data, int n) {
    // Your implementation here
}
