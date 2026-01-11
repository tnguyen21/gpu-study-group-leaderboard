/*
 * Problem: 7-Point 3D Stencil
 * Apply a 7-point stencil to a 3D grid.
 *
 * Layout:
 *   - Flattened index: `idx = z*N*N + y*N + x` (row-major in x, then y, then z)
 *
 * For interior points (1 <= x,y,z <= N-2):
 *   - out[x,y,z] =
 *       c0*in[x,y,z] +
 *       c1*in[x-1,y,z] + c2*in[x+1,y,z] +
 *       c3*in[x,y-1,z] + c4*in[x,y+1,z] +
 *       c5*in[x,y,z-1] + c6*in[x,y,z+1]
 *
 * Boundary points (any index is 0 or N-1):
 *   - out[...] = 0
 */

__global__ void stencil(float *in, float *out, int N,
                        float c0, float c1, float c2, float c3,
                        float c4, float c5, float c6) {
    // Your implementation here
}
