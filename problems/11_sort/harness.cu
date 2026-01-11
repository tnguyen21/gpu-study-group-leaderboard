// Harness for 11_sort
// This file is parsed by modal_app.py to build the benchmark
// Note: This problem allows multiple kernel launches

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_sizes[] = {
        1000,       // Small non-power-of-2
        1025,       // One above power-of-2
        257,        // One above block size
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];

        unsigned int *t_h_data = (unsigned int*)malloc(N * sizeof(unsigned int));
        unsigned int *t_h_ref = (unsigned int*)malloc(N * sizeof(unsigned int));
        unsigned int *t_d_data;

        unsigned int seed = 42;
        for (int i = 0; i < N; i++) {
            seed = seed * 1103515245 + 12345;
            t_h_data[i] = seed % 1000000;
            t_h_ref[i] = t_h_data[i];
        }

        qsort(
            t_h_ref,
            N,
            sizeof(unsigned int),
            [](const void *a, const void *b) -> int {
                const unsigned int av = *(const unsigned int *)a;
                const unsigned int bv = *(const unsigned int *)b;
                return (av > bv) - (av < bv);
            });

        cudaMalloc(&t_d_data, N * sizeof(unsigned int));
        cudaMemcpy(t_d_data, t_h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

        sort<<<(N + 255) / 256, 256>>>(t_d_data, N);

        cudaMemcpy(t_h_data, t_d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < N; i++) {
            if (t_h_data[i] != t_h_ref[i]) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (N=%d): Mismatch at %d: got %u, expected %u\n",
                            t, N, i, t_h_data[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (N=%d): Validation failed with %d mismatches\n", t, N, errors);
            return 1;
        }

        cudaFree(t_d_data);
        free(t_h_data); free(t_h_ref);
    }
}

// Main test case for timing
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
