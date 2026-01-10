// Example solution for week01_vectoradd
// This is the naive baseline implementation

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Challenge: Can you make this faster?
// Ideas to explore:
// - Vectorized loads (float4)
// - Loop unrolling
// - Different block sizes
// - Memory coalescing patterns
