/*
 * Naive 2D Convolution
 *
 * Each thread computes one output pixel by iterating over
 * all filter elements and corresponding input pixels.
 *
 * Simple but inefficient - filter is read from global memory
 * for every output pixel, and input pixels are read multiple times.
 */

__global__ void conv2d(float *input, float *filter, float *output,
                       int width, int height, int r) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        float sum = 0.0f;
        int filter_width = 2 * r + 1;

        for (int fr = 0; fr < filter_width; fr++) {
            for (int fc = 0; fc < filter_width; fc++) {
                int in_row = row - r + fr;
                int in_col = col - r + fc;

                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                    sum += input[in_row * width + in_col] * filter[fr * filter_width + fc];
                }
            }
        }
        output[row * width + col] = sum;
    }
}
