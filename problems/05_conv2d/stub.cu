/*
 * Problem: 2D Convolution
 * Apply a 2D convolution to a 2D image.
 *
 * Inputs:
 *   - `input`: height x width (row-major)
 *   - `filter`: (2*r+1) x (2*r+1) (row-major)
 *   - `r`: filter radius
 *
 * Output:
 *   - `output`: height x width (row-major)
 *
 * Definition (out-of-bounds treated as 0):
 *   - output[row,col] = sum_{fr=0..2r} sum_{fc=0..2r}
 *       input[row - r + fr, col - r + fc] * filter[fr,fc]
 *
 * Launch configuration:
 *   - 2D grid, 2D block
 *   - blockDim = (16, 16, 1)
 *   - gridDim = (ceil(width/16), ceil(height/16), 1)
 *   - Each thread computes one output pixel
 *   - Use blockIdx.x/threadIdx.x for column, blockIdx.y/threadIdx.y for row
 *
 * Example (r=1, filter all 1s):
 *   - input (3x3) = [[1,2,3],[4,5,6],[7,8,9]]
 *   - output[1,1] = 45, output[0,0] = 12
 */

__global__ void conv2d(float *input, float *filter, float *output,
                       int width, int height, int r) {
    // Your implementation here
}
