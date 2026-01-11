// Harness for 07_histogram
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
    const int NUM_BINS = 256;

    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];

        unsigned char *t_h_data = (unsigned char*)malloc(N);
        unsigned int *t_h_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
        unsigned int *t_h_ref = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
        unsigned char *t_d_data;
        unsigned int *t_d_hist;

        for (int i = 0; i < N; i++) {
            t_h_data[i] = (i * 7 + 13) % 256;
        }

        memset(t_h_ref, 0, NUM_BINS * sizeof(unsigned int));
        for (int i = 0; i < N; i++) {
            t_h_ref[t_h_data[i]]++;
        }

        cudaMalloc(&t_d_data, N);
        cudaMalloc(&t_d_hist, NUM_BINS * sizeof(unsigned int));

        cudaMemcpy(t_d_data, t_h_data, N, cudaMemcpyHostToDevice);
        cudaMemset(t_d_hist, 0, NUM_BINS * sizeof(unsigned int));

        histogram<<<(N + 1023) / 1024, 1024>>>(t_d_data, t_d_hist, N);

        cudaMemcpy(t_h_hist, t_d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < NUM_BINS; i++) {
            if (t_h_hist[i] != t_h_ref[i]) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (N=%d): Bin %d: got %u, expected %u\n",
                            t, N, i, t_h_hist[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (N=%d): Validation failed with %d bins incorrect\n", t, N, errors);
            return 1;
        }

        cudaFree(t_d_data); cudaFree(t_d_hist);
        free(t_h_data); free(t_h_hist); free(t_h_ref);
    }
}

// Main test case for timing
const int N = 1 << 20;  // 1M elements
const int NUM_BINS = 256;

unsigned char *h_data;
unsigned int *h_hist, *h_ref;
unsigned char *d_data;
unsigned int *d_hist;

h_data = (unsigned char*)malloc(N);
h_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
h_ref = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

// Initialize with deterministic pattern
for (int i = 0; i < N; i++) {
    h_data[i] = (i * 7 + 13) % 256;
}

// Compute reference on CPU
memset(h_ref, 0, NUM_BINS * sizeof(unsigned int));
for (int i = 0; i < N; i++) {
    h_ref[h_data[i]]++;
}

cudaMalloc(&d_data, N);
cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));

cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

// === LAUNCH ===
histogram<<<(N + 1023) / 1024, 1024>>>(d_data, d_hist, N);

// === VALIDATION ===
cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < NUM_BINS; i++) {
    if (h_hist[i] != h_ref[i]) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Bin %d: got %u, expected %u\n", i, h_hist[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d bins incorrect\n", errors);
    return 1;
}
