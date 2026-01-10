// Harness for 08_reduction
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
const int N = 1 << 20;  // 1M elements
const int BLOCK_SIZE = 256;
const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

float *h_input, *h_output;
float *d_input, *d_output;
float h_ref = 0.0f;

h_input = (float*)malloc(N * sizeof(float));
h_output = (float*)malloc(sizeof(float));

// Initialize with deterministic values that sum nicely
for (int i = 0; i < N; i++) {
    h_input[i] = (i % 100) * 0.001f;
    h_ref += h_input[i];
}

cudaMalloc(&d_input, N * sizeof(float));
cudaMalloc(&d_output, sizeof(float));

cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemset(d_output, 0, sizeof(float));

// === LAUNCH ===
// Note: the kernel should atomically update d_output with block partial sums
reduce<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, N);

// === VALIDATION ===
cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
// Use relative error due to floating point accumulation differences
float rel_error = fabs(*h_output - h_ref) / fabs(h_ref);
if (rel_error > 1e-4) {
    fprintf(stderr, "Validation failed: got %f, expected %f (rel error: %e)\n",
            *h_output, h_ref, rel_error);
    return 1;
}
