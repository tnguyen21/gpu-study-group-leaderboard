/*
 * 7-Point 3D Stencil with Shared Memory Tiling
 *
 * Tiles the input into shared memory to reduce global memory accesses.
 * Each block loads a tile including halo cells (1 cell in each direction).
 *
 * IN_TILE_DIM includes halos, OUT_TILE_DIM is the valid output region.
 */

#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)  // 6

__global__ void stencil(float *in, float *out, int N,
                        float c0, float c1, float c2, float c3,
                        float c4, float c5, float c6) {
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Load tile into shared memory
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    } else {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Only interior threads compute output
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {

            out[i*N*N + j*N + k] =
                c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
              + c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
              + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
              + c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
              + c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
              + c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
              + c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}
