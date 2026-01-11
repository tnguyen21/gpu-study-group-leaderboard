// Harness for 08_reduction
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_sizes[] = {
        999999,     // Non-power-of-2
        1,          // Single element
        257,        // One above power-of-2
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    const int BLOCK_SIZE = 256;

    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];
        int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float *t_h_input = (float*)malloc(N * sizeof(float));
        float *t_h_output = (float*)malloc(sizeof(float));
        float *t_d_input, *t_d_output;
        float t_h_ref = 0.0f;

        for (int i = 0; i < N; i++) {
            t_h_input[i] = (i % 100) * 0.001f;
            t_h_ref += t_h_input[i];
        }

        cudaMalloc(&t_d_input, N * sizeof(float));
        cudaMalloc(&t_d_output, sizeof(float));

        cudaMemcpy(t_d_input, t_h_input, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(t_d_output, 0, sizeof(float));

        reduce<<<NUM_BLOCKS, BLOCK_SIZE>>>(t_d_input, t_d_output, N);

        cudaMemcpy(t_h_output, t_d_output, sizeof(float), cudaMemcpyDeviceToHost);
        float rel_error = fabs(*t_h_output - t_h_ref) / (fabs(t_h_ref) + 1e-6);
        if (rel_error > 1e-4) {
            fprintf(stderr, "Test %d (N=%d): got %f, expected %f (rel error: %e)\n",
                    t, N, *t_h_output, t_h_ref, rel_error);
            return 1;
        }

        cudaFree(t_d_input); cudaFree(t_d_output);
        free(t_h_input); free(t_h_output);
    }
}

// Main test case for timing
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
