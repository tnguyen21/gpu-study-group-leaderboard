/*
 * Problem: Vector Addition
 * Compute `c[i] = a[i] + b[i]` for `i in [0, n)`.
 *
 * Inputs:
 *   - `a`, `b`: arrays of length `n`
 *   - `n`: number of elements
 *
 * Output:
 *   - `c`: array of length `n`
 *
 * Launch configuration:
 *   - 1D grid, 1D block
 *   - blockDim.x = 256
 *   - gridDim.x = ceil(n / 256)
 *
 * Example:
 *   - a = [1, 2, 3], b = [10, 20, 30]  ->  c = [11, 22, 33]
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Your implementation here
}
