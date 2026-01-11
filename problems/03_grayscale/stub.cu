/*
 * Problem: RGB to Grayscale Conversion
 * Convert an RGB image to a single-channel grayscale image.
 *
 * Layout:
 *   - `rgb` is interleaved: [R0,G0,B0, R1,G1,B1, ...] in row-major pixel order
 *   - `gray` has one byte per pixel in row-major order
 *
 * Per pixel:
 *   - `gray = (unsigned char)(0.21*R + 0.71*G + 0.07*B)`
 *
 * Launch configuration:
 *   - 2D grid, 2D block
 *   - blockDim = (32, 32, 1)
 *   - gridDim = (ceil(width/32), ceil(height/32), 1)
 *   - Each thread processes one pixel
 *   - Use blockIdx.x/threadIdx.x for x (column), blockIdx.y/threadIdx.y for y (row)
 *
 * Example:
 *   - (R,G,B) = (100,150,200) -> gray = (unsigned char)141
 */

__global__ void grayscale(unsigned char *rgb, unsigned char *gray, int width, int height) {
    // Your implementation here
}
