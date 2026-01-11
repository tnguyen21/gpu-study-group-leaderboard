// Harness for 01_vectoradd
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_sizes[] = {
        999999,     // Non-power-of-2
        1,          // Single element
        1025,       // One above common block size
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];

        float *t_h_a = (float*)malloc(N * sizeof(float));
        float *t_h_b = (float*)malloc(N * sizeof(float));
        float *t_h_c = (float*)malloc(N * sizeof(float));
        float *t_h_ref = (float*)malloc(N * sizeof(float));
        float *t_d_a, *t_d_b, *t_d_c;

        for (int i = 0; i < N; i++) {
            t_h_a[i] = i * 0.001f;
            t_h_b[i] = i * 0.002f;
            t_h_ref[i] = t_h_a[i] + t_h_b[i];
        }

        cudaMalloc(&t_d_a, N * sizeof(float));
        cudaMalloc(&t_d_b, N * sizeof(float));
        cudaMalloc(&t_d_c, N * sizeof(float));

        cudaMemcpy(t_d_a, t_h_a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t_d_b, t_h_b, N * sizeof(float), cudaMemcpyHostToDevice);

        vectorAdd<<<(N + 255) / 256, 256>>>(t_d_a, t_d_b, t_d_c, N);

        cudaMemcpy(t_h_c, t_d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < N; i++) {
            if (fabs(t_h_c[i] - t_h_ref[i]) > 1e-5) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (N=%d): Mismatch at %d: got %f, expected %f\n",
                            t, N, i, t_h_c[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (N=%d): Validation failed with %d errors\n", t, N, errors);
            return 1;
        }

        cudaFree(t_d_a); cudaFree(t_d_b); cudaFree(t_d_c);
        free(t_h_a); free(t_h_b); free(t_h_c); free(t_h_ref);
    }
}

// Main test case for timing
const int N = 1 << 20;  // 1,048,576 elements
float *h_a, *h_b, *h_c, *h_ref;
float *d_a, *d_b, *d_c;

h_a = (float*)malloc(N * sizeof(float));
h_b = (float*)malloc(N * sizeof(float));
h_c = (float*)malloc(N * sizeof(float));
h_ref = (float*)malloc(N * sizeof(float));

for (int i = 0; i < N; i++) {
    h_a[i] = i * 0.001f;
    h_b[i] = i * 0.002f;
    h_ref[i] = h_a[i] + h_b[i];
}

cudaMalloc(&d_a, N * sizeof(float));
cudaMalloc(&d_b, N * sizeof(float));
cudaMalloc(&d_c, N * sizeof(float));

cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

// === LAUNCH ===
vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

// === VALIDATION ===
cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < N; i++) {
    if (fabs(h_c[i] - h_ref[i]) > 1e-5) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %f, expected %f\n", i, h_c[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
