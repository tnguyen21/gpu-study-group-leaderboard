// Harness for 03_grayscale
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
// Validation tests on critical edge cases
{
    int test_cases[][2] = {
        {800, 600},     // Non-square
        {33, 17},       // Small primes (block boundary edge case)
        {1000, 1},      // Single column
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    const int CHANNELS = 3;

    for (int t = 0; t < num_tests; t++) {
        int WIDTH = test_cases[t][0];
        int HEIGHT = test_cases[t][1];

        unsigned char *t_h_rgb = (unsigned char*)malloc(WIDTH * HEIGHT * CHANNELS);
        unsigned char *t_h_gray = (unsigned char*)malloc(WIDTH * HEIGHT);
        unsigned char *t_h_ref = (unsigned char*)malloc(WIDTH * HEIGHT);
        unsigned char *t_d_rgb, *t_d_gray;

        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            t_h_rgb[i * 3 + 0] = (i * 7) % 256;
            t_h_rgb[i * 3 + 1] = (i * 13) % 256;
            t_h_rgb[i * 3 + 2] = (i * 17) % 256;
            t_h_ref[i] = (unsigned char)(0.21f * t_h_rgb[i*3] + 0.71f * t_h_rgb[i*3+1] + 0.07f * t_h_rgb[i*3+2]);
        }

        cudaMalloc(&t_d_rgb, WIDTH * HEIGHT * CHANNELS);
        cudaMalloc(&t_d_gray, WIDTH * HEIGHT);

        cudaMemcpy(t_d_rgb, t_h_rgb, WIDTH * HEIGHT * CHANNELS, cudaMemcpyHostToDevice);

        dim3 block(32, 32);
        dim3 grid((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
        grayscale<<<grid, block>>>(t_d_rgb, t_d_gray, WIDTH, HEIGHT);

        cudaMemcpy(t_h_gray, t_d_gray, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        int errors = 0;
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            if (abs((int)t_h_gray[i] - (int)t_h_ref[i]) > 1) {
                errors++;
                if (errors <= 5) {
                    fprintf(stderr, "Test %d (%dx%d): Mismatch at %d: got %d, expected %d\n",
                            t, WIDTH, HEIGHT, i, t_h_gray[i], t_h_ref[i]);
                }
            }
        }
        if (errors > 0) {
            fprintf(stderr, "Test %d (%dx%d): Validation failed with %d errors\n", t, WIDTH, HEIGHT, errors);
            return 1;
        }

        cudaFree(t_d_rgb); cudaFree(t_d_gray);
        free(t_h_rgb); free(t_h_gray); free(t_h_ref);
    }
}

// Main test case for timing
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int CHANNELS = 3;

unsigned char *h_rgb, *h_gray, *h_ref;
unsigned char *d_rgb, *d_gray;

h_rgb = (unsigned char*)malloc(WIDTH * HEIGHT * CHANNELS);
h_gray = (unsigned char*)malloc(WIDTH * HEIGHT);
h_ref = (unsigned char*)malloc(WIDTH * HEIGHT);

// Initialize with deterministic pattern
for (int i = 0; i < WIDTH * HEIGHT; i++) {
    h_rgb[i * 3 + 0] = (i * 7) % 256;   // R
    h_rgb[i * 3 + 1] = (i * 13) % 256;  // G
    h_rgb[i * 3 + 2] = (i * 17) % 256;  // B
    // Reference: gray = 0.21*R + 0.71*G + 0.07*B
    h_ref[i] = (unsigned char)(0.21f * h_rgb[i*3] + 0.71f * h_rgb[i*3+1] + 0.07f * h_rgb[i*3+2]);
}

cudaMalloc(&d_rgb, WIDTH * HEIGHT * CHANNELS);
cudaMalloc(&d_gray, WIDTH * HEIGHT);

cudaMemcpy(d_rgb, h_rgb, WIDTH * HEIGHT * CHANNELS, cudaMemcpyHostToDevice);

// === LAUNCH ===
dim3 block(32, 32);
dim3 grid((WIDTH + 31) / 32, (HEIGHT + 31) / 32);
grayscale<<<grid, block>>>(d_rgb, d_gray, WIDTH, HEIGHT);

// === VALIDATION ===
cudaMemcpy(h_gray, d_gray, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < WIDTH * HEIGHT; i++) {
    // Allow +/- 1 for rounding differences
    if (abs((int)h_gray[i] - (int)h_ref[i]) > 1) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %d, expected %d\n", i, h_gray[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
