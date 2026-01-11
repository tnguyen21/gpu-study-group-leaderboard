// Harness for 04_matmul_tiled
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_cases[][3] = {
        {512, 1024, 768},    // Non-square, all different
        {33, 33, 33},        // Non-multiple of typical tile size (16 or 32)
        {256, 1, 512},       // Single column output
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int t = 0; t < num_tests; t++) {
        int M = test_cases[t][0];
        int N = test_cases[t][1];
        int K = test_cases[t][2];

        float *t_h_A = (float*)malloc(M * K * sizeof(float));
        float *t_h_B = (float*)malloc(K * N * sizeof(float));
        float *t_h_C = (float*)malloc(M * N * sizeof(float));
        float *t_h_ref = (float*)malloc(M * N * sizeof(float));
        float *t_d_A, *t_d_B, *t_d_C;

        for (int i = 0; i < M * K; i++) t_h_A[i] = (i % 100) * 0.01f;
        for (int i = 0; i < K * N; i++) t_h_B[i] = (i % 100) * 0.01f;

        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += t_h_A[row * K + k] * t_h_B[k * N + col];
                }
                t_h_ref[row * N + col] = sum;
            }
        }

        cudaMalloc(&t_d_A, M * K * sizeof(float));
        cudaMalloc(&t_d_B, K * N * sizeof(float));
        cudaMalloc(&t_d_C, M * N * sizeof(float));

        cudaMemcpy(t_d_A, t_h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t_d_B, t_h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        matmul<<<grid, block>>>(t_d_A, t_d_B, t_d_C, M, N, K);

        cudaMemcpy(t_h_C, t_d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < M * N; i++) {
            if (fabs(t_h_C[i] - t_h_ref[i]) > 1e-3) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (M=%d, N=%d, K=%d): Mismatch at %d: got %f, expected %f\n",
                            t, M, N, K, i, t_h_C[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (M=%d, N=%d, K=%d): Validation failed with %d errors\n", t, M, N, K, errors);
            return 1;
        }

        cudaFree(t_d_A); cudaFree(t_d_B); cudaFree(t_d_C);
        free(t_h_A); free(t_h_B); free(t_h_C); free(t_h_ref);
    }
}

// Main test case for timing
const int M = 1024, N = 1024, K = 1024;
float *h_A, *h_B, *h_C, *h_ref;
float *d_A, *d_B, *d_C;

h_A = (float*)malloc(M * K * sizeof(float));
h_B = (float*)malloc(K * N * sizeof(float));
h_C = (float*)malloc(M * N * sizeof(float));
h_ref = (float*)malloc(M * N * sizeof(float));

// Initialize with deterministic values
for (int i = 0; i < M * K; i++) h_A[i] = (i % 100) * 0.01f;
for (int i = 0; i < K * N; i++) h_B[i] = (i % 100) * 0.01f;

// Compute reference on CPU
for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += h_A[row * K + k] * h_B[k * N + col];
        }
        h_ref[row * N + col] = sum;
    }
}

cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

// === LAUNCH ===
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

// === VALIDATION ===
cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < M * N; i++) {
    if (fabs(h_C[i] - h_ref[i]) > 1e-3) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %f, expected %f\n", i, h_C[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
