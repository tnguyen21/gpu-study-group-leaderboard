// Harness for 05_conv2d
// This file is parsed by modal_app.py to build the benchmark

// === SETUP ===
const int WIDTH = 1024;
const int HEIGHT = 1024;
const int R = 3;  // Filter radius (7x7 filter)
const int FILTER_SIZE = (2 * R + 1) * (2 * R + 1);

float *h_input, *h_output, *h_filter, *h_ref;
float *d_input, *d_output, *d_filter;

h_input = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
h_output = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
h_filter = (float*)malloc(FILTER_SIZE * sizeof(float));
h_ref = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

// Initialize input with deterministic pattern
for (int i = 0; i < WIDTH * HEIGHT; i++) {
    h_input[i] = (i % 256) / 255.0f;
}

// Initialize filter (Gaussian-like)
float filter_sum = 0.0f;
for (int i = 0; i < 2 * R + 1; i++) {
    for (int j = 0; j < 2 * R + 1; j++) {
        float di = i - R;
        float dj = j - R;
        h_filter[i * (2 * R + 1) + j] = expf(-(di*di + dj*dj) / (2.0f * R));
        filter_sum += h_filter[i * (2 * R + 1) + j];
    }
}
// Normalize filter
for (int i = 0; i < FILTER_SIZE; i++) {
    h_filter[i] /= filter_sum;
}

// Compute reference on CPU
for (int row = 0; row < HEIGHT; row++) {
    for (int col = 0; col < WIDTH; col++) {
        float sum = 0.0f;
        for (int fr = 0; fr < 2 * R + 1; fr++) {
            for (int fc = 0; fc < 2 * R + 1; fc++) {
                int in_row = row - R + fr;
                int in_col = col - R + fc;
                if (in_row >= 0 && in_row < HEIGHT && in_col >= 0 && in_col < WIDTH) {
                    sum += h_input[in_row * WIDTH + in_col] * h_filter[fr * (2 * R + 1) + fc];
                }
            }
        }
        h_ref[row * WIDTH + col] = sum;
    }
}

cudaMalloc(&d_input, WIDTH * HEIGHT * sizeof(float));
cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));
cudaMalloc(&d_filter, FILTER_SIZE * sizeof(float));

cudaMemcpy(d_input, h_input, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_filter, h_filter, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

// === LAUNCH ===
dim3 block(16, 16);
dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
conv2d<<<grid, block>>>(d_input, d_filter, d_output, WIDTH, HEIGHT, R);

// === VALIDATION ===
cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < WIDTH * HEIGHT; i++) {
    if (fabs(h_output[i] - h_ref[i]) > 1e-4) {
        errors++;
        if (errors <= 5) {
            fprintf(stderr, "Mismatch at %d: got %f, expected %f\n", i, h_output[i], h_ref[i]);
        }
    }
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\n", errors);
    return 1;
}
