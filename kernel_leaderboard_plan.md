# GPU Kernel Leaderboard on Modal

## Architecture Overview

The system has three main components: a CUDA compilation/benchmarking environment on Modal, a persistent storage layer for results, and a submission interface. Modal handles the first two nicely—you can build a custom image with CUDA toolkit and use a Volume to persist a SQLite database.

## Core Design Decisions

**Compilation approach**: Modal can absolutely compile and run C/CUDA programs. You define a container image with `nvcc` and the CUDA toolkit, then either compile at runtime or pre-compile kernels. For a learning context where people are iterating on code, runtime compilation makes sense.

**Storage**: Modal Volumes provide persistent filesystem storage across function invocations. A SQLite database stored on a Volume works well for this scale (team learning project). For larger scale, you'd want an external Postgres/Turso, but SQLite keeps things simple and self-contained.

**Submission flow**: Each team member runs a submission script locally that calls a Modal function. The function compiles their kernel, benchmarks it against reference inputs, and records results to the database. A separate endpoint serves the leaderboard.

## Project Structure

```
kernel-leaderboard/
├── modal_app.py          # Main Modal application
├── problems/
│   ├── week01_vectoradd/
│   │   ├── reference.cu   # Reference implementation
│   │   ├── problem.py     # Problem definition (input generation, validation)
│   │   └── README.md      # Problem description
│   └── week02_matmul/
│       └── ...
├── submit.py             # CLI submission script for participants
├── leaderboard.py        # Leaderboard viewer (optional web UI)
└── db/
    └── schema.sql        # SQLite schema
```

## Implementation Details

### 1. Modal Image with CUDA Toolkit

```python
import modal

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04")
    .apt_install("sqlite3", "python3-pip")
    .pip_install("numpy", "pynvml")
)

app = modal.App("kernel-leaderboard")
vol = modal.Volume.from_name("leaderboard-data", create_if_missing=True)
```

### 2. Benchmark Function

```python
@app.function(
    image=cuda_image,
    gpu="T4",  # or A10G, A100 depending on budget
    volumes={"/data": vol},
    timeout=120,
)
def benchmark_kernel(
    problem_id: str,
    user_id: str,
    kernel_source: str,
) -> dict:
    import subprocess
    import time
    import sqlite3
    import tempfile
    import os
    
    # Write kernel source to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "kernel.cu")
        binary_path = os.path.join(tmpdir, "kernel")
        
        # Load problem harness (combines user kernel with timing code)
        harness = load_problem_harness(problem_id)
        full_source = harness.wrap_kernel(kernel_source)
        
        with open(kernel_path, "w") as f:
            f.write(full_source)
        
        # Compile
        compile_result = subprocess.run(
            ["nvcc", "-O3", "-o", binary_path, kernel_path],
            capture_output=True,
            text=True,
        )
        
        if compile_result.returncode != 0:
            return {
                "success": False,
                "error": "compilation_failed",
                "message": compile_result.stderr,
            }
        
        # Run benchmark (multiple iterations for stability)
        times = []
        for _ in range(10):
            result = subprocess.run(
                [binary_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "runtime_error",
                    "message": result.stderr,
                }
            times.append(float(result.stdout.strip()))
        
        # Validate output correctness
        if not harness.validate_output(binary_path):
            return {
                "success": False,
                "error": "incorrect_output",
                "message": "Kernel output doesn't match reference",
            }
        
        median_time = sorted(times)[len(times) // 2]
        
        # Record to database
        conn = sqlite3.connect("/data/leaderboard.db")
        conn.execute("""
            INSERT INTO submissions (problem_id, user_id, time_ms, submitted_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (problem_id, user_id, median_time))
        conn.commit()
        conn.close()
        vol.commit()  # Persist volume changes
        
        return {
            "success": True,
            "time_ms": median_time,
            "times": times,
        }
```

### 3. Problem Harness Pattern

Each problem needs a harness that wraps the user's kernel with timing code and input/output handling:

```cpp
// harness_template.cu
#include <cuda_runtime.h>
#include <stdio.h>

// User's kernel gets inserted here
{USER_KERNEL}

int main() {
    // Problem-specific setup (sizes, allocations)
    {SETUP_CODE}
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        {KERNEL_LAUNCH}
    }
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    {KERNEL_LAUNCH}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms);
    
    // Copy back for validation
    {VALIDATION_CODE}
    
    return 0;
}
```

### 4. Submission Script (for participants)

```python
#!/usr/bin/env python3
# submit.py - run this locally to submit your kernel

import modal
import sys
import os

def submit(problem_id: str, kernel_file: str, user_id: str):
    with open(kernel_file) as f:
        kernel_source = f.read()
    
    # Call the remote Modal function
    benchmark = modal.Function.lookup("kernel-leaderboard", "benchmark_kernel")
    result = benchmark.remote(
        problem_id=problem_id,
        user_id=user_id,
        kernel_source=kernel_source,
    )
    
    if result["success"]:
        print(f"✓ Submission successful!")
        print(f"  Time: {result['time_ms']:.3f} ms")
        print(f"  All runs: {[f'{t:.3f}' for t in result['times']]}")
    else:
        print(f"✗ Submission failed: {result['error']}")
        print(f"  {result['message']}")

if __name__ == "__main__":
    submit(
        problem_id=sys.argv[1],
        kernel_file=sys.argv[2],
        user_id=os.environ.get("LEADERBOARD_USER", "anonymous"),
    )
```

### 5. Leaderboard Endpoint

```python
@app.function(image=cuda_image, volumes={"/data": vol})
@modal.web_endpoint(method="GET")
def leaderboard(problem_id: str = None):
    import sqlite3
    import json
    
    conn = sqlite3.connect("/data/leaderboard.db")
    conn.row_factory = sqlite3.Row
    
    if problem_id:
        # Best submission per user for this problem
        rows = conn.execute("""
            SELECT user_id, MIN(time_ms) as best_time, 
                   COUNT(*) as attempts
            FROM submissions
            WHERE problem_id = ?
            GROUP BY user_id
            ORDER BY best_time ASC
        """, (problem_id,)).fetchall()
    else:
        # Overview of all problems
        rows = conn.execute("""
            SELECT problem_id, COUNT(DISTINCT user_id) as participants,
                   MIN(time_ms) as best_time
            FROM submissions
            GROUP BY problem_id
        """).fetchall()
    
    return [dict(r) for r in rows]
```

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    time_ms REAL NOT NULL,
    submitted_at TEXT NOT NULL
);

CREATE INDEX idx_problem_user ON submissions(problem_id, user_id);
CREATE INDEX idx_problem_time ON submissions(problem_id, time_ms);
```

## Week 1 Example: Vector Add

**problems/week01_vectoradd/problem.py**:
```python
PROBLEM_CONFIG = {
    "id": "week01_vectoradd",
    "name": "Vector Addition",
    "description": "Add two vectors element-wise",
    "input_sizes": [1 << 20],  # 1M elements
    "kernel_signature": "__global__ void vectorAdd(float *a, float *b, float *c, int n)",
}

HARNESS_SETUP = """
const int N = 1 << 20;
float *h_a, *h_b, *h_c;
float *d_a, *d_b, *d_c;

h_a = (float*)malloc(N * sizeof(float));
h_b = (float*)malloc(N * sizeof(float));
h_c = (float*)malloc(N * sizeof(float));

// Initialize with deterministic values
for (int i = 0; i < N; i++) {
    h_a[i] = i * 0.001f;
    h_b[i] = i * 0.002f;
}

cudaMalloc(&d_a, N * sizeof(float));
cudaMalloc(&d_b, N * sizeof(float));
cudaMalloc(&d_c, N * sizeof(float));

cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
"""

KERNEL_LAUNCH = "vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);"
```

**Reference solution for comparison**:
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

## Deployment Workflow

1. **Initial setup** (one-time):
   ```bash
   modal volume create leaderboard-data
   modal deploy modal_app.py
   ```

2. **Initialize database**:
   ```bash
   modal run modal_app.py::init_db
   ```

3. **Participants submit**:
   ```bash
   export LEADERBOARD_USER="tommy"
   python submit.py week01_vectoradd my_vectoradd.cu
   ```

4. **View leaderboard**:
   ```bash
   curl https://your-workspace--kernel-leaderboard-leaderboard.modal.run?problem_id=week01_vectoradd
   ```

## Cost Considerations

Modal charges per-second for GPU time. For a learning project with maybe 10-50 submissions per week, you're looking at very minimal costs—probably under $5/month on T4 instances. The benchmarks themselves run in seconds.

## Extensions to Consider

**Automated correctness checking**: Beyond just timing, validate that outputs match a reference within floating-point tolerance. This catches bugs that might "run fast" but produce garbage.

**Historical tracking**: Store the full kernel source with each submission so people can see their progression and compare approaches.

**GitHub integration**: Have people submit via PR to a shared repo, trigger benchmarks via GitHub Actions that call Modal.

**Metrics beyond time**: Track memory bandwidth utilization, occupancy, etc. using `nvprof` or `nsight` data.

**Multiple GPU tiers**: Let people see how their kernel scales across T4 → A10G → A100.
