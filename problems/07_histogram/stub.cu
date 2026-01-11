/*
 * Problem: Histogram
 * Count how many times each byte value appears in `data`.
 *
 * Inputs:
 *   - `data`: array of length `n`, values in [0, 255]
 *
 * Output:
 *   - `hist`: array of length 256, where `hist[v] = count(data[i] == v)`
 *
 * Example:
 *   - data = [0, 1, 1, 255] -> hist[0]=1, hist[1]=2, hist[255]=1
 */

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    // Your implementation here
}
