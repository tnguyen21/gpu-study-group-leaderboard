// Harness for 03_grayscale
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
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
