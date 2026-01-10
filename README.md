# GPU Kernel Leaderboard

A Modal-based benchmarking system for learning CUDA, inspired by [GPU Mode's kernelboard](https://github.com/gpu-mode/kernelboard).

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or with pip
pip install modal requests
```

Then authenticate with Modal:
```bash
modal setup
```

### 2. Deploy the Leaderboard

```bash
modal deploy modal_app.py
modal run modal_app.py::init_db  # Initialize the database
```

### 3. Submit Your First Kernel

**Option A: CLI (requires Modal SDK)**
```bash
export LEADERBOARD_USER="your_name"
python submit.py 01_vectoradd problems/01_vectoradd/solutions/01_naive.cu
```

**Option B: HTTP POST (no Modal SDK needed)**
```bash
curl -X POST https://your-workspace--kernel-leaderboard-submit.modal.run \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg src "$(cat problems/01_vectoradd/solutions/01_naive.cu)" '{
    "problem_id": "01_vectoradd",
    "user_id": "your_name",
    "kernel_source": $src
  }')"
```

### 4. View the Leaderboard

After deploying, Modal gives you URLs like:
```
https://your-workspace--kernel-leaderboard-leaderboard.modal.run
https://your-workspace--kernel-leaderboard-submit.modal.run
https://your-workspace--kernel-leaderboard-problems.modal.run
```

Query them with:
```bash
# All problems overview
curl https://your-workspace--kernel-leaderboard-leaderboard.modal.run

# Specific problem rankings
curl "https://your-workspace--kernel-leaderboard-leaderboard.modal.run?problem_id=01_vectoradd"

# List available problems
curl https://your-workspace--kernel-leaderboard-problems.modal.run
```

## Problems

Problems are organized by PMPP (Programming Massively Parallel Processors) chapter:

| Problem | Name | PMPP Chapter | Key Concepts |
|---------|------|--------------|--------------|
| `01_vectoradd` | Vector Addition | 2-3 | Thread indexing, memory coalescing |
| `02_matmul_naive` | Naive Matrix Multiply | 3 | 2D thread blocks |
| `03_grayscale` | RGB to Grayscale | 3 | Image processing basics |
| `04_matmul_tiled` | Tiled Matrix Multiply | 5-6 | Shared memory, tiling |
| `05_conv2d` | 2D Convolution | 7 | Constant memory, halos |
| `06_stencil` | 7-Point 3D Stencil | 8 | Register tiling |
| `07_histogram` | Histogram | 9 | Atomics, privatization |
| `08_reduction` | Sum Reduction | 10 | Tree reduction, coarsening |
| `09_scan` | Prefix Sum | 11 | Kogge-Stone, Brent-Kung |
| `10_merge` | Merge Sorted Arrays | 12 | Co-ranking |
| `11_sort` | Parallel Sort | 13 | Bitonic/merge sort |

## Repository Structure

```
modal-leaderboard/
├── modal_app.py              # Main Modal application
├── submit.py                 # CLI submission tool
├── problems/
│   ├── 01_vectoradd/
│   │   ├── stub.cu           # Problem description + signature (start here!)
│   │   ├── harness.cu        # Setup/validation code (used by benchmark)
│   │   └── solutions/
│   │       ├── 01_naive.cu   # Baseline implementation
│   │       ├── 02_float4.cu  # Vectorized version
│   │       └── ...
│   ├── 02_matmul_naive/
│   │   └── ...
│   └── ...
```

## Workflow for Learning

1. **Pick a problem**: Start with `01_vectoradd`
2. **Read the stub**: `problems/01_vectoradd/stub.cu` describes the problem and signature
3. **Implement**: Copy the stub and fill in your implementation
4. **Submit**: `python submit.py 01_vectoradd my_solution.cu`
5. **Iterate**: Optimize your solution, resubmit, check leaderboard
6. **Learn**: When stuck, peek at `solutions/01_naive.cu`, then more optimized versions

## How It Works

When you submit a kernel, the system:

1. Wraps your kernel in a timing harness with predetermined inputs
2. Compiles it with `nvcc -O3 -arch=sm_75` on Modal's T4 GPU instances
3. Runs 3 warmup iterations, then 10 timed runs
4. Validates correctness against reference outputs
5. Records the median time to a SQLite database (persisted on Modal Volume)
6. Returns your results

## Architecture

The Modal app uses:
- **debian_slim** base image with CUDA 12.4 nvcc compiler
- **FastAPI** endpoints for HTTP access (`@modal.fastapi_endpoint`)
- **Modal Volume** for persistent SQLite storage
- **T4 GPUs** for kernel execution (limited to 3 concurrent containers)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/leaderboard` | GET | View rankings (optional `?problem_id=` filter) |
| `/problems` | GET | List available problems with signatures |
| `/submit` | POST | Submit a kernel for benchmarking |

### Submit Endpoint

POST JSON body:
```json
{
  "problem_id": "01_vectoradd",
  "user_id": "your_name",
  "kernel_source": "__global__ void vectorAdd(...) { ... }",
  "api_key": "optional_if_configured"
}
```

## Adding New Problems

1. Create a new directory under `problems/` (e.g., `12_newproblem/`)
2. Create `stub.cu` with problem description and signature
3. Create `harness.cu` with setup, launch, and validation sections
4. Add reference solutions in `solutions/`
5. Redeploy: `modal deploy modal_app.py`

### Harness Format

```cuda
// === SETUP ===
// Allocate memory, initialize inputs, compute reference

// === LAUNCH ===
myKernel<<<grid, block>>>(args);

// === VALIDATION ===
// Check outputs, return 1 on failure
```

## Cost

Modal charges per-second for GPU time. With T4 instances and typical benchmark runs taking ~2-5 seconds total, expect costs under $0.01 per submission.

## Local Testing

```bash
# List all problems
modal run modal_app.py -- --action list

# Test a specific problem
modal run modal_app.py -- --action test --problem 01_vectoradd
```
