// Harness for 11_sort
// This file is parsed by modal_app.py to build the benchmark
// Note: This problem allows multiple kernel launches

// === SETUP ===
const int N = 1 << 18;  // 262144 elements

unsigned int *h_data, *h_ref;
unsigned int *d_data;

h_data = (unsigned int*)malloc(N * sizeof(unsigned int));
h_ref = (unsigned int*)malloc(N * sizeof(unsigned int));

// Initialize with pseudo-random values (deterministic)
unsigned int seed = 42;
for (int i = 0; i < N; i++) {
    seed = seed * 1103515245 + 12345;  // LCG
    h_data[i] = seed % 1000000;
    h_ref[i] = h_data[i];
}

// Sort reference on CPU
for (int i = 0; i < N - 1; i++) {
    for (int j = 0; j < N - i - 1; j++) {
        if (h_ref[j] > h_ref[j + 1]) {
            unsigned int tmp = h_ref[j];
            h_ref[j] = h_ref[j + 1];
            h_ref[j + 1] = tmp;
        }
    }
}
// Note: In practice, use std::sort, but bubble sort works for reference

cudaMalloc(&d_data, N * sizeof(unsigned int));
cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

// === LAUNCH ===
// Note: Efficient sorts require multiple kernel launches
// This simple harness allows implementations to use multiple passes
sort<<<(N + 255) / 256, 256>>>(d_data, N);

// === VALIDATION ===
cudaMemcpy(h_data, d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
int errors = 0;
// Check sorted order
for (int i = 0; i < N - 1; i++) {
    if (h_data[i] > h_data[i + 1]) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Not sorted at %d: %u > %u\n", i, h_data[i], h_data[i + 1]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: array not sorted (%d violations)\n", errors);
    return 1;
}
