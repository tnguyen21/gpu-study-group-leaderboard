// Harness for 10_merge
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
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
