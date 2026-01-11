/*
 * Problem: Parallel Reduction (Sum)
 * Compute the sum of all elements in `input[0..n-1]`.
 *
 * Inputs:
 *   - `input`: array of length `n`
 *
 * Output:
 *   - `output[0]` is the sum (a single float)
 *   - `output` is pre-initialized to zero
 *
 * Launch configuration:
 *   - 1D grid, 1D block
 *   - blockDim.x = 256
 *   - gridDim.x = ceil(n / 256)
 *   - Hint: Each block reduces its portion, then atomicAdd to output[0]
 *
 * Example:
 *   - input = [1, 2, 3, 4] -> output[0] = 10
 */

__global__ void reduce(float *input, float *output, int n) {
    // Your implementation here
}
