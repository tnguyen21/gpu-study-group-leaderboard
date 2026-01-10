/*
 * Problem: 2D Convolution
 * PMPP Chapter 7
 *
 * Apply a 2D convolution filter to an image.
 *
 * Input:
 *   - input: input image (height x width floats)
 *   - filter: convolution kernel ((2*r+1) x (2*r+1) floats)
 *   - output: output image (height x width floats)
 *   - width, height: image dimensions (1024 x 1024)
 *   - r: filter radius (r=3 means 7x7 filter)
 *
 * For each output pixel, compute the weighted sum of the input pixels
 * in its neighborhood, where weights come from the filter.
 *
 * Boundary handling: pixels outside the image are treated as 0.
 *
 * Key concepts:
 *   - Halo cells: extra border pixels needed for convolution
 *   - Constant memory for filter coefficients (cached, broadcast)
 *   - Shared memory tiling with halo handling
 *
 * Optimization ideas:
 *   - Use constant memory for the filter (__constant__)
 *   - Tile the input into shared memory (including halos)
 *   - L2 cache for halo elements outside the tile
 */

__global__ void conv2d(float *input, float *filter, float *output,
                       int width, int height, int r) {
    // Your implementation here
}
