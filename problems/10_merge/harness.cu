// Harness for 10_merge
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    // {M, N} - sizes of the two arrays to merge
    int test_cases[][2] = {
        {10000, 5000},      // M > N
        {5000, 10000},      // M < N
        {1, 1000},          // Single element in first array
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int t = 0; t < num_tests; t++) {
        int M = test_cases[t][0];
        int N = test_cases[t][1];
        int TOTAL = M + N;

        float *t_h_A = (float*)malloc(M * sizeof(float));
        float *t_h_B = (float*)malloc(N * sizeof(float));
        float *t_h_C = (float*)malloc(TOTAL * sizeof(float));
        float *t_h_ref = (float*)malloc(TOTAL * sizeof(float));
        float *t_d_A, *t_d_B, *t_d_C;

        for (int i = 0; i < M; i++) {
            t_h_A[i] = i * 2.0f + 0.5f;
        }
        for (int i = 0; i < N; i++) {
            t_h_B[i] = i * 2.0f;
        }

        int i = 0, j = 0, k = 0;
        while (i < M && j < N) {
            if (t_h_A[i] <= t_h_B[j]) {
                t_h_ref[k++] = t_h_A[i++];
            } else {
                t_h_ref[k++] = t_h_B[j++];
            }
        }
        while (i < M) t_h_ref[k++] = t_h_A[i++];
        while (j < N) t_h_ref[k++] = t_h_B[j++];

        cudaMalloc(&t_d_A, M * sizeof(float));
        cudaMalloc(&t_d_B, N * sizeof(float));
        cudaMalloc(&t_d_C, TOTAL * sizeof(float));

        cudaMemcpy(t_d_A, t_h_A, M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(t_d_B, t_h_B, N * sizeof(float), cudaMemcpyHostToDevice);

        const int BLOCK_SIZE = 256;
        const int NUM_BLOCKS = 256;
        merge<<<NUM_BLOCKS, BLOCK_SIZE>>>(t_d_A, M, t_d_B, N, t_d_C);

        cudaMemcpy(t_h_C, t_d_C, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int idx = 0; idx < TOTAL; idx++) {
            if (t_h_C[idx] != t_h_ref[idx]) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (M=%d, N=%d): Mismatch at %d: got %f, expected %f\n",
                            t, M, N, idx, t_h_C[idx], t_h_ref[idx]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (M=%d, N=%d): Validation failed with %d errors\n", t, M, N, errors);
            return 1;
        }

        cudaFree(t_d_A); cudaFree(t_d_B); cudaFree(t_d_C);
        free(t_h_A); free(t_h_B); free(t_h_C); free(t_h_ref);
    }
}

// Main test case for timing
const int M = 500000;  // Size of first array
const int N = 500000;  // Size of second array
const int TOTAL = M + N;

float *h_A, *h_B, *h_C, *h_ref;
float *d_A, *d_B, *d_C;

h_A = (float*)malloc(M * sizeof(float));
h_B = (float*)malloc(N * sizeof(float));
h_C = (float*)malloc(TOTAL * sizeof(float));
h_ref = (float*)malloc(TOTAL * sizeof(float));

// Initialize sorted arrays with deterministic values
for (int i = 0; i < M; i++) {
    h_A[i] = i * 2.0f + 0.5f;  // 0.5, 2.5, 4.5, ...
}
for (int i = 0; i < N; i++) {
    h_B[i] = i * 2.0f;  // 0, 2, 4, ...
}

// Compute reference merge on CPU
int i = 0, j = 0, k = 0;
while (i < M && j < N) {
    if (h_A[i] <= h_B[j]) {
        h_ref[k++] = h_A[i++];
    } else {
        h_ref[k++] = h_B[j++];
    }
}
while (i < M) h_ref[k++] = h_A[i++];
while (j < N) h_ref[k++] = h_B[j++];

cudaMalloc(&d_A, M * sizeof(float));
cudaMalloc(&d_B, N * sizeof(float));
cudaMalloc(&d_C, TOTAL * sizeof(float));

cudaMemcpy(d_A, h_A, M * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

// === LAUNCH ===
const int BLOCK_SIZE = 256;
const int NUM_BLOCKS = 256;
merge<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_A, M, d_B, N, d_C);

// === VALIDATION ===
cudaMemcpy(h_C, d_C, TOTAL * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < TOTAL; i++) {
    if (h_C[i] != h_ref[i]) {
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
