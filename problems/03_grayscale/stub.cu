/*
 * Problem: RGB to Grayscale Conversion
 * PMPP Chapter 3
 *
 * Convert an RGB image to grayscale using the luminosity formula:
 *   gray = 0.21 * R + 0.71 * G + 0.07 * B
 *
 * Input:
 *   - rgb: input image as unsigned char array (height x width x 3)
 *         Stored as R,G,B,R,G,B,... (interleaved channels)
 *   - gray: output grayscale image (height x width)
 *   - width: image width (1920)
 *   - height: image height (1080)
 *
 * Image size: 1920 x 1080 (Full HD)
 *
 * Key concepts:
 *   - 2D thread blocks for 2D data
 *   - Memory layout: pixels stored in row-major order
 *   - RGB offset = (row * width + col) * 3
 *
 * Optimization ideas:
 *   - Coalesced memory access patterns
 *   - Vectorized loads (uchar4 to load 4 bytes at once)
 *   - Different block sizes
 */

__global__ void grayscale(unsigned char *rgb, unsigned char *gray, int width, int height) {
    // Your implementation here
}
