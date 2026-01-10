# GPU Kernel Leaderboard

A Modal-based benchmarking system for learning CUDA, inspired by [GPU Mode's kernelboard](https://github.com/gpu-mode/kernelboard).

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal setup  # Authenticate with Modal
```

### 2. Deploy the Leaderboard

```bash
modal deploy modal_app.py
modal run modal_app.py::init_db  # Initialize the database
```

### 3. Submit Your First Kernel

```bash
export LEADERBOARD_USER="your_name"
python submit.py week01_vectoradd examples/week01_vectoradd_baseline.cu
```

### 4. View the Leaderboard

After deploying, Modal gives you a URL like:
```
https://your-workspace--kernel-leaderboard-leaderboard.modal.run
```

Query it with:
```bash
# All problems overview
curl https://your-workspace--kernel-leaderboard-leaderboard.modal.run

# Specific problem rankings
curl "https://your-workspace--kernel-leaderboard-leaderboard.modal.run?problem_id=week01_vectoradd"
```

## How It Works

When you submit a kernel, the system:

1. Wraps your kernel in a timing harness with predetermined inputs
2. Compiles it with `nvcc -O3` on Modal's GPU instances
3. Runs warmup iterations, then 10 timed runs
4. Validates correctness against reference outputs
5. Records the median time to a SQLite database
6. Returns your results

## Problems

Each week corresponds to a PMPP chapter. Problems define:
- Input sizes and data patterns
- Expected kernel signature
- Validation criteria

**Week 1: Vector Addition** (`week01_vectoradd`)
- Signature: `__global__ void vectorAdd(float *a, float *b, float *c, int n)`
- N = 1,048,576 elements

**Week 2: Matrix Multiplication** (`week02_matmul_naive`)
- Signature: `__global__ void matmul(float *A, float *B, float *C, int M, int N, int K)`
- 1024x1024 matrices

## Writing Your Kernels

Your `.cu` file should contain just the kernel function(s). The harness handles:
- Memory allocation and data initialization
- Kernel launch configuration
- Timing with CUDA events
- Result validation

Example submission file:
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

## Adding New Problems

Edit `modal_app.py` and add entries to the `PROBLEMS` dict. Each problem needs:
- `setup`: Allocation and initialization code
- `launch`: Kernel invocation (the harness wraps this in timing)
- `validation`: Correctness checking code
- `expected_signature`: For documentation

Then redeploy: `modal deploy modal_app.py`

## Cost

Modal charges per-second for GPU time. With T4 instances and typical benchmark runs taking ~2-5 seconds total, expect costs under $0.01 per submission. A team doing 100 submissions/week would spend roughly $1-2/month.

## Differences from GPU Mode's Kernelboard

This is a simplified implementation suitable for small teams. The main differences:
- SQLite instead of a proper database (fine for <1000 submissions)
- No GitHub integration for submissions
- No fancy web UI (just JSON endpoints)
- Problems defined in Python rather than separate config files

For a team learning project, this keeps things simple while providing the core benchmark-and-compare functionality.
