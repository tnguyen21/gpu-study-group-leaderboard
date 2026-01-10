/*
 * Naive RGB to Grayscale
 *
 * Each thread converts one pixel using the luminosity formula.
 * Uses 2D thread blocks mapping to 2D image coordinates.
 */

__global__ void grayscale(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;

        unsigned char r = rgb[rgbOffset];
        unsigned char g = rgb[rgbOffset + 1];
        unsigned char b = rgb[rgbOffset + 2];

        gray[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}
