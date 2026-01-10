// Harness for 01_vectoradd
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
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
