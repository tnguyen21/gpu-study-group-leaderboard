/*
 * Vectorized Vector Addition using float4
 *
 * Uses float4 (128-bit) loads and stores to increase memory throughput.
 * Each thread processes 4 elements at once.
 *
 * Note: This assumes N is divisible by 4. A production kernel would
 * need to handle the tail elements separately.
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (i + 3 < n) {
        // Vectorized load
        float4 a4 = *reinterpret_cast<float4*>(a + i);
        float4 b4 = *reinterpret_cast<float4*>(b + i);

        // Compute
        float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;

        // Vectorized store
        *reinterpret_cast<float4*>(c + i) = c4;
    } else if (i < n) {
        // Handle tail elements
        for (int j = i; j < n; j++) {
            c[j] = a[j] + b[j];
        }
    }
}
