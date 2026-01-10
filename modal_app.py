"""
GPU Kernel Leaderboard - Modal Application

Deploy with: modal deploy modal_app.py
Initialize DB: modal run modal_app.py::init_db
"""

import modal
import os

# --- Modal Setup ---

cuda_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "gnupg")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-nvcc-12-4",
    )
    .pip_install("fastapi[standard]")
    .env({"PATH": "/usr/local/cuda-12.4/bin:$PATH"})
)

app = modal.App("kernel-leaderboard")
vol = modal.Volume.from_name("leaderboard-data", create_if_missing=True)

DB_PATH = "/data/leaderboard.db"
PROBLEMS_DIR = "/data/problems"

# Optional: Set this to require an API key for submissions
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
# Leave as None for open access
SUBMIT_API_KEY = os.environ.get("SUBMIT_API_KEY", None)


# --- Problem Definitions ---
# In production, load these from files. Inlining here for simplicity.

PROBLEMS = {
    "week01_vectoradd": {
        "name": "Vector Addition",
        "setup": """
const int N = 1 << 20;
float *h_a, *h_b, *h_c, *h_ref;
float *d_a, *d_b, *d_c;

h_a = (float*)malloc(N * sizeof(float));
h_b = (float*)malloc(N * sizeof(float));
h_c = (float*)malloc(N * sizeof(float));
h_ref = (float*)malloc(N * sizeof(float));

for (int i = 0; i < N; i++) {
    h_a[i] = i * 0.001f;
    h_b[i] = i * 0.002f;
    h_ref[i] = h_a[i] + h_b[i];  // Reference result
}

cudaMalloc(&d_a, N * sizeof(float));
cudaMalloc(&d_b, N * sizeof(float));
cudaMalloc(&d_c, N * sizeof(float));

cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
""",
        "launch": "vectorAdd<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);",
        "validation": """
cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
int errors = 0;
for (int i = 0; i < N; i++) {
    if (fabs(h_c[i] - h_ref[i]) > 1e-5) errors++;
}
if (errors > 0) {
    fprintf(stderr, "Validation failed: %d errors\\n", errors);
    return 1;
}
""",
        "expected_signature": "__global__ void vectorAdd(float *a, float *b, float *c, int n)",
    },
    "week02_matmul_naive": {
        "name": "Matrix Multiplication (Naive)",
        "setup": """
const int M = 1024, N = 1024, K = 1024;
float *h_A, *h_B, *h_C;
float *d_A, *d_B, *d_C;

h_A = (float*)malloc(M * K * sizeof(float));
h_B = (float*)malloc(K * N * sizeof(float));
h_C = (float*)malloc(M * N * sizeof(float));

for (int i = 0; i < M * K; i++) h_A[i] = (i % 100) * 0.01f;
for (int i = 0; i < K * N; i++) h_B[i] = (i % 100) * 0.01f;

cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
""",
        "launch": """
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
""",
        "validation": """
// Simplified validation - check a few elements
cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
// For now just check it ran without error
""",
        "expected_signature": "__global__ void matmul(float *A, float *B, float *C, int M, int N, int K)",
    },
}


def build_harness(problem_id: str, user_kernel: str) -> str:
    """Wrap user's kernel in timing/validation harness."""
    problem = PROBLEMS[problem_id]
    
    return f"""
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============ USER KERNEL ============
{user_kernel}
// =====================================

int main() {{
    // Setup
    {problem["setup"]}
    
    // Warmup runs
    for (int i = 0; i < 3; i++) {{
        {problem["launch"]}
    }}
    cudaDeviceSynchronize();
    
    // Check for errors after warmup
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
        return 1;
    }}
    
    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    {problem["launch"]}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Validation
    {problem["validation"]}
    
    // Output time (parsed by benchmark harness)
    printf("%f\\n", ms);
    
    return 0;
}}
"""


# --- Modal Functions ---

@app.function(image=cuda_image, volumes={"/data": vol})
def init_db():
    """Initialize the SQLite database."""
    import sqlite3
    os.makedirs("/data", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            time_ms REAL NOT NULL,
            kernel_hash TEXT,
            submitted_at TEXT NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_problem_user ON submissions(problem_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_problem_time ON submissions(problem_id, time_ms);
        
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    vol.commit()
    print("Database initialized!")


@app.function(
    image=cuda_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=120,
)
def benchmark_kernel(problem_id: str, user_id: str, kernel_source: str) -> dict:
    """Compile and benchmark a user's kernel submission."""
    import subprocess
    import tempfile
    import sqlite3
    import hashlib
    from datetime import datetime
    
    if problem_id not in PROBLEMS:
        return {
            "success": False,
            "error": "unknown_problem",
            "message": f"Problem '{problem_id}' not found. Available: {list(PROBLEMS.keys())}",
        }
    
    kernel_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "kernel.cu")
        binary_path = os.path.join(tmpdir, "kernel")
        
        # Build full source with harness
        full_source = build_harness(problem_id, kernel_source)
        
        with open(kernel_path, "w") as f:
            f.write(full_source)
        
        # Compile
        compile_result = subprocess.run(
            ["nvcc", "-O3", "-arch=sm_75", "-o", binary_path, kernel_path],
            capture_output=True,
            text=True,
        )
        
        if compile_result.returncode != 0:
            return {
                "success": False,
                "error": "compilation_failed",
                "message": compile_result.stderr,
            }
        
        # Run benchmark (multiple iterations)
        times = []
        for _ in range(10):
            result = subprocess.run(
                [binary_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "runtime_error",
                    "message": result.stderr or "Kernel execution failed",
                }
            
            try:
                times.append(float(result.stdout.strip()))
            except ValueError:
                return {
                    "success": False,
                    "error": "parse_error",
                    "message": f"Could not parse output: {result.stdout}",
                }
        
        median_time = sorted(times)[len(times) // 2]
        
        # Record to database
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO submissions (problem_id, user_id, time_ms, kernel_hash, submitted_at)
            VALUES (?, ?, ?, ?, ?)
        """, (problem_id, user_id, median_time, kernel_hash, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        vol.commit()
        
        return {
            "success": True,
            "time_ms": median_time,
            "times": times,
            "kernel_hash": kernel_hash,
        }


@app.function(image=cuda_image, volumes={"/data": vol})
@modal.fastapi_endpoint(method="GET")
def leaderboard(problem_id: str = None):
    """Get leaderboard data as JSON."""
    import sqlite3

    vol.reload()  # Get latest data from other containers
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    if problem_id:
        rows = conn.execute("""
            SELECT 
                user_id,
                MIN(time_ms) as best_time_ms,
                COUNT(*) as attempts,
                MAX(submitted_at) as last_submission
            FROM submissions
            WHERE problem_id = ?
            GROUP BY user_id
            ORDER BY best_time_ms ASC
        """, (problem_id,)).fetchall()
        
        result = {
            "problem_id": problem_id,
            "problem_name": PROBLEMS.get(problem_id, {}).get("name", problem_id),
            "rankings": [
                {
                    "rank": i + 1,
                    "user_id": r["user_id"],
                    "best_time_ms": round(r["best_time_ms"], 4),
                    "attempts": r["attempts"],
                    "last_submission": r["last_submission"],
                }
                for i, r in enumerate(rows)
            ],
        }
    else:
        rows = conn.execute("""
            SELECT 
                problem_id,
                COUNT(DISTINCT user_id) as participants,
                COUNT(*) as total_submissions,
                MIN(time_ms) as best_time_ms
            FROM submissions
            GROUP BY problem_id
        """).fetchall()
        
        result = {
            "problems": [
                {
                    "problem_id": r["problem_id"],
                    "problem_name": PROBLEMS.get(r["problem_id"], {}).get("name", r["problem_id"]),
                    "participants": r["participants"],
                    "total_submissions": r["total_submissions"],
                    "best_time_ms": round(r["best_time_ms"], 4) if r["best_time_ms"] else None,
                }
                for r in rows
            ],
        }
    
    conn.close()
    return result


@app.function(image=cuda_image, volumes={"/data": vol})
@modal.fastapi_endpoint(method="GET")
def problems():
    """List available problems."""
    return {
        "problems": [
            {
                "id": pid,
                "name": pdata["name"],
                "expected_signature": pdata["expected_signature"],
            }
            for pid, pdata in PROBLEMS.items()
        ]
    }


@app.function(
    image=cuda_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=120,
    concurrency_limit=3,  # Prevent abuse
)
@modal.fastapi_endpoint(method="POST")
def submit(request: dict):
    """
    HTTP endpoint for kernel submissions.

    POST body (JSON):
    {
        "problem_id": "week01_vectoradd",
        "user_id": "tommy",
        "kernel_source": "__global__ void vectorAdd(...) { ... }",
        "api_key": "optional_if_configured"
    }

    This allows anyone to submit without needing Modal SDK access.
    """
    problem_id = request.get("problem_id")
    user_id = request.get("user_id", "anonymous")
    kernel_source = request.get("kernel_source")

    if not problem_id or not kernel_source:
        return {
            "success": False,
            "error": "missing_fields",
            "message": "Required fields: problem_id, kernel_source",
        }

    # Basic validation to prevent abuse
    if len(kernel_source) > 50000:  # 50KB limit
        return {
            "success": False,
            "error": "kernel_too_large",
            "message": "Kernel source must be under 50KB",
        }

    if len(user_id) > 64:
        return {
            "success": False,
            "error": "invalid_user_id",
            "message": "User ID must be under 64 characters",
        }

    # Optional API key check
    if SUBMIT_API_KEY is not None:
        provided_key = request.get("api_key")
        if provided_key != SUBMIT_API_KEY:
            return {
                "success": False,
                "error": "unauthorized",
                "message": "Invalid or missing API key",
            }

    import subprocess
    import tempfile
    import sqlite3
    import hashlib
    from datetime import datetime

    if problem_id not in PROBLEMS:
        return {
            "success": False,
            "error": "unknown_problem",
            "message": f"Problem '{problem_id}' not found. Available: {list(PROBLEMS.keys())}",
        }

    kernel_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]

    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "kernel.cu")
        binary_path = os.path.join(tmpdir, "kernel")

        full_source = build_harness(problem_id, kernel_source)

        with open(kernel_path, "w") as f:
            f.write(full_source)

        compile_result = subprocess.run(
            ["nvcc", "-O3", "-arch=sm_75", "-o", binary_path, kernel_path],
            capture_output=True,
            text=True,
        )

        if compile_result.returncode != 0:
            return {
                "success": False,
                "error": "compilation_failed",
                "message": compile_result.stderr,
            }

        times = []
        for _ in range(10):
            result = subprocess.run(
                [binary_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": "runtime_error",
                    "message": result.stderr or "Kernel execution failed",
                }

            try:
                times.append(float(result.stdout.strip()))
            except ValueError:
                return {
                    "success": False,
                    "error": "parse_error",
                    "message": f"Could not parse output: {result.stdout}",
                }

        median_time = sorted(times)[len(times) // 2]

        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO submissions (problem_id, user_id, time_ms, kernel_hash, submitted_at)
            VALUES (?, ?, ?, ?, ?)
        """, (problem_id, user_id, median_time, kernel_hash, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        vol.commit()

        return {
            "success": True,
            "time_ms": median_time,
            "times": times,
            "kernel_hash": kernel_hash,
            "user_id": user_id,
            "problem_id": problem_id,
        }


# --- Local entrypoint for testing ---

@app.local_entrypoint()
def main(action: str = "test"):
    if action == "init":
        init_db.remote()
    elif action == "test":
        # Test with a simple vector add kernel
        test_kernel = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""
        result = benchmark_kernel.remote("week01_vectoradd", "test_user", test_kernel)
        print(f"Result: {result}")
