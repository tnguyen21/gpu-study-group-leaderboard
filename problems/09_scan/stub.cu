/*
 * Problem: Parallel Prefix Sum (Inclusive Scan)
 * Compute the inclusive prefix sum (inclusive scan).
 *
 * Definition:
 *   - output[i] = input[0] + input[1] + ... + input[i]
 *
 * Inputs / output:
 *   - `input`: array of length `n`
 *   - `output`: array of length `n`
 *
 * Launch configuration:
 *   - 1D grid, 1D block
 *   - blockDim.x = 1024
 *   - gridDim.x = ceil(n / 1024)
 *   - Shared memory: 8192 bytes (2 * 1024 * sizeof(float))
 *   - Hint: Use Blelloch scan or similar algorithm within each block
 *
 * Example:
 *   - [3, 1, 4, 1, 5] -> [3, 4, 8, 9, 14]
 */

__global__ void scan(float *input, float *output, int n) {
    // Your implementation here
}
