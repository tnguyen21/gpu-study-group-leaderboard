/*
 * Problem: 7-Point 3D Stencil
 * PMPP Chapter 8
 *
 * Apply a 7-point stencil to a 3D grid:
 *   out[i,j,k] = c0*in[i,j,k] + c1*in[i,j,k-1] + c2*in[i,j,k+1]
 *             + c3*in[i,j-1,k] + c4*in[i,j+1,k]
 *             + c5*in[i-1,j,k] + c6*in[i+1,j,k]
 *
 * Input:
 *   - in: input 3D grid (N x N x N floats, N=128)
 *   - out: output 3D grid (N x N x N floats)
 *   - N: grid dimension
 *   - c0-c6: stencil coefficients
 *
 * The stencil is applied to interior points only (not boundaries).
 * Boundary points (where any index is 0 or N-1) should be copied unchanged
 * or set to 0.
 *
 * Key concepts:
 *   - 3D thread blocks and 3D grids
 *   - Halo cells in 3D (1 cell in each of 6 directions)
 *   - Register tiling (keeping prev/curr/next planes in registers)
 *   - Thread coarsening in z-dimension
 *
 * Optimization ideas:
 *   - Shared memory tiling with 3D tiles
 *   - Thread coarsening: each thread computes multiple z-planes
 *   - Register tiling: keep 3 planes in shared memory, slide through z
 */

__global__ void stencil(float *in, float *out, int N,
                        float c0, float c1, float c2, float c3,
                        float c4, float c5, float c6) {
    // Your implementation here
}
