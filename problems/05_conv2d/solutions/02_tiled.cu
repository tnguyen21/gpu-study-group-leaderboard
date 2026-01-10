/*
 * Tiled 2D Convolution with Shared Memory
 *
 * Tiles the input into shared memory to reduce global memory accesses.
 * Each tile includes halo cells needed for the convolution.
 *
 * Note: This version uses a fixed filter radius of 3 (7x7 filter).
 * IN_TILE_SIZE is the size of the input tile loaded into shared memory.
 * OUT_TILE_SIZE is the number of valid output pixels per block.
 */

#define FILTER_RADIUS 3
#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE (IN_TILE_SIZE - 2 * FILTER_RADIUS)

__global__ void conv2d(float *input, float *filter, float *output,
                       int width, int height, int r) {
    // Shared memory for input tile (including halos)
    __shared__ float tile[IN_TILE_SIZE][IN_TILE_SIZE];
    // Cache filter in shared memory too
    __shared__ float f[(2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1)];

    int tx = threadIdx.x, ty = threadIdx.y;

    // Load filter into shared memory (first few threads)
    int filter_size = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
    for (int i = ty * blockDim.x + tx; i < filter_size; i += blockDim.x * blockDim.y) {
        f[i] = filter[i];
    }

    // Calculate input position (accounting for halo offset)
    int in_row = blockIdx.y * OUT_TILE_SIZE + ty - FILTER_RADIUS;
    int in_col = blockIdx.x * OUT_TILE_SIZE + tx - FILTER_RADIUS;

    // Load input tile into shared memory
    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
        tile[ty][tx] = input[in_row * width + in_col];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Calculate output position
    int out_row = blockIdx.y * OUT_TILE_SIZE + ty - FILTER_RADIUS;
    int out_col = blockIdx.x * OUT_TILE_SIZE + tx - FILTER_RADIUS;

    // Only threads in the valid output region compute
    int tile_row = ty - FILTER_RADIUS;
    int tile_col = tx - FILTER_RADIUS;

    if (tile_row >= 0 && tile_row < OUT_TILE_SIZE &&
        tile_col >= 0 && tile_col < OUT_TILE_SIZE &&
        out_row >= 0 && out_row < height &&
        out_col >= 0 && out_col < width) {

        float sum = 0.0f;
        for (int fr = 0; fr < 2 * FILTER_RADIUS + 1; fr++) {
            for (int fc = 0; fc < 2 * FILTER_RADIUS + 1; fc++) {
                sum += tile[ty + fr - FILTER_RADIUS][tx + fc - FILTER_RADIUS] *
                       f[fr * (2 * FILTER_RADIUS + 1) + fc];
            }
        }
        output[out_row * width + out_col] = sum;
    }
}
