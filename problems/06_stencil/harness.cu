// Harness for 06_stencil
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_sizes[] = {
        64,     // Smaller cube
        33,     // Non-multiple of block size (8)
        17,     // Small prime
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    const float c0 = -6.0f, c1 = 1.0f, c2 = 1.0f;
    const float c3 = 1.0f, c4 = 1.0f, c5 = 1.0f, c6 = 1.0f;

    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];
        size_t size = N * N * N * sizeof(float);

        float *t_h_in = (float*)malloc(size);
        float *t_h_out = (float*)malloc(size);
        float *t_h_ref = (float*)malloc(size);
        float *t_d_in, *t_d_out;

        for (int i = 0; i < N * N * N; i++) {
            t_h_in[i] = (i % 100) * 0.01f;
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    int idx = i * N * N + j * N + k;
                    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
                        t_h_ref[idx] = c0 * t_h_in[idx]
                                     + c1 * t_h_in[i*N*N + j*N + (k-1)]
                                     + c2 * t_h_in[i*N*N + j*N + (k+1)]
                                     + c3 * t_h_in[i*N*N + (j-1)*N + k]
                                     + c4 * t_h_in[i*N*N + (j+1)*N + k]
                                     + c5 * t_h_in[(i-1)*N*N + j*N + k]
                                     + c6 * t_h_in[(i+1)*N*N + j*N + k];
                    } else {
                        t_h_ref[idx] = 0.0f;
                    }
                }
            }
        }

        cudaMalloc(&t_d_in, size);
        cudaMalloc(&t_d_out, size);
        cudaMemcpy(t_d_in, t_h_in, size, cudaMemcpyHostToDevice);
        cudaMemset(t_d_out, 0, size);

        dim3 block(8, 8, 8);
        dim3 grid((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);
        stencil<<<grid, block>>>(t_d_in, t_d_out, N, c0, c1, c2, c3, c4, c5, c6);

        cudaMemcpy(t_h_out, t_d_out, size, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < N * N * N; i++) {
            if (fabs(t_h_out[i] - t_h_ref[i]) > 1e-4) {
                errors++;
                if (errors <= 5) {
                    int z = i / (N * N);
                    int y = (i / N) % N;
                    int x = i % N;
                    fprintf(stderr, "Test %d (N=%d): Mismatch at (%d,%d,%d): got %f, expected %f\n",
                            t, N, x, y, z, t_h_out[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (N=%d): Validation failed with %d errors\n", t, N, errors);
            return 1;
        }

        cudaFree(t_d_in); cudaFree(t_d_out);
        free(t_h_in); free(t_h_out); free(t_h_ref);
    }
}

// Main test case for timing
const int N = 128;  // Grid dimension (N x N x N)
const float c0 = -6.0f, c1 = 1.0f, c2 = 1.0f;
const float c3 = 1.0f, c4 = 1.0f, c5 = 1.0f, c6 = 1.0f;

float *h_in, *h_out, *h_ref;
float *d_in, *d_out;

size_t size = N * N * N * sizeof(float);
h_in = (float*)malloc(size);
h_out = (float*)malloc(size);
h_ref = (float*)malloc(size);

// Initialize with deterministic pattern
for (int i = 0; i < N * N * N; i++) {
    h_in[i] = (i % 100) * 0.01f;
}

// Compute reference on CPU
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            int idx = i * N * N + j * N + k;
            if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
                h_ref[idx] = c0 * h_in[idx]
                           + c1 * h_in[i*N*N + j*N + (k-1)]
                           + c2 * h_in[i*N*N + j*N + (k+1)]
                           + c3 * h_in[i*N*N + (j-1)*N + k]
                           + c4 * h_in[i*N*N + (j+1)*N + k]
                           + c5 * h_in[(i-1)*N*N + j*N + k]
                           + c6 * h_in[(i+1)*N*N + j*N + k];
            } else {
                h_ref[idx] = 0.0f;  // Boundary
            }
        }
    }
}

cudaMalloc(&d_in, size);
cudaMalloc(&d_out, size);
cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
cudaMemset(d_out, 0, size);

// === LAUNCH ===
dim3 block(8, 8, 8);
dim3 grid((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);
stencil<<<grid, block>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);

// === VALIDATION ===
cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < N * N * N; i++) {
    if (fabs(h_out[i] - h_ref[i]) > 1e-4) {
        errors++;
        if (errors <= 5) {
            int z = i / (N * N);
            int y = (i / N) % N;
            int x = i % N;
            fprintf(stderr, "Mismatch at (%d,%d,%d): got %f, expected %f\n",
                    x, y, z, h_out[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
