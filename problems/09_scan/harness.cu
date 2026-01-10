// Harness for 09_scan
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Use power of 2 for simpler implementations
const int N = 1 << 16;  // 65536 elements (fits in one block for hierarchical scan)
const int BLOCK_SIZE = 1024;

float *h_input, *h_output, *h_ref;
float *d_input, *d_output;

h_input = (float*)malloc(N * sizeof(float));
h_output = (float*)malloc(N * sizeof(float));
h_ref = (float*)malloc(N * sizeof(float));

// Initialize with small values to avoid overflow
for (int i = 0; i < N; i++) {
    h_input[i] = (i % 10) * 0.1f;
}

// Compute reference (inclusive scan) on CPU
h_ref[0] = h_input[0];
for (int i = 1; i < N; i++) {
    h_ref[i] = h_ref[i-1] + h_input[i];
}

cudaMalloc(&d_input, N * sizeof(float));
cudaMalloc(&d_output, N * sizeof(float));
cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

// === LAUNCH ===
// For a simple single-block scan (N <= 2*BLOCK_SIZE), one kernel call suffices
// For larger arrays, implementations need multiple phases
scan<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);

// === VALIDATION ===
cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < N; i++) {
    // Use relative error for floating point comparison
    float rel_error = fabs(h_output[i] - h_ref[i]) / (fabs(h_ref[i]) + 1e-6);
    if (rel_error > 1e-4) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %f, expected %f\n", i, h_output[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
