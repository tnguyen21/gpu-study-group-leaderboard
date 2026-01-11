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

// Sort reference on CPU (qsort keeps setup time reasonable at this N)
qsort(
    h_ref,
    N,
    sizeof(unsigned int),
    [](const void *a, const void *b) -> int {
        const unsigned int av = *(const unsigned int *)a;
        const unsigned int bv = *(const unsigned int *)b;
        return (av > bv) - (av < bv);
    });

cudaMalloc(&d_data, N * sizeof(unsigned int));
cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

// === LAUNCH ===
// Note: Efficient sorts require multiple kernel launches
// This simple harness allows implementations to use multiple passes
sort<<<(N + 255) / 256, 256>>>(d_data, N);

// === VALIDATION ===
cudaMemcpy(h_data, d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
int errors = 0;
// Check exact match against reference (sorted order + correct permutation)
for (int i = 0; i < N; i++) {
    if (h_data[i] != h_ref[i]) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %u, expected %u\n", i, h_data[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d mismatches\n", errors);
    return 1;
}
