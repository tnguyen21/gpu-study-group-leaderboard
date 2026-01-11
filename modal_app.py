"""
GPU Kernel Leaderboard - Modal Application

Deploy with: modal deploy modal_app.py
Initialize DB: modal run modal_app.py::init_db
"""

import os
import re
from pathlib import Path

import modal

# --- Modal Setup ---

# Get the directory containing this file (for local problem loading)
LOCAL_PROBLEMS_DIR = Path(__file__).parent / "problems"
LOCAL_STATIC_DIR = Path(__file__).parent / "static"

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
    .add_local_dir(str(LOCAL_PROBLEMS_DIR), "/app/problems")  # Include problems in image
    .add_local_dir(str(LOCAL_STATIC_DIR), "/app/static")  # Include static files for web UI
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .add_local_dir(str(LOCAL_PROBLEMS_DIR), "/app/problems")
    .add_local_dir(str(LOCAL_STATIC_DIR), "/app/static")
)

app = modal.App("kernel-leaderboard")
vol = modal.Volume.from_name("leaderboard-data", create_if_missing=True)

DB_PATH = "/data/leaderboard.db"
PROBLEMS_DIR = "/app/problems"

# Optional: Set this to require an API key for submissions
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
# Leave as None for open access
SUBMIT_API_KEY = os.environ.get("SUBMIT_API_KEY", None)

MAX_KERNEL_SOURCE_BYTES = 50_000
MAX_USER_ID_LENGTH = 64
MAX_COMPILE_OUTPUT_BYTES = 200_000
MAX_RUN_OUTPUT_BYTES = 100_000


# --- Security: Kernel Source Validation ---

# Patterns that indicate potentially dangerous code
BLOCKED_PATTERNS = [
    (r"#include\s*[<\"](?!cuda|cstdio|cstdlib|cmath|cstring|climits)", "Only CUDA/standard includes allowed"),
    (r"__attribute__\s*\(\s*\(", "Compiler attributes not allowed"),
    (r"\bsystem\s*\(", "system() calls not allowed"),
    (r"\b(execv?[pel]?|popen|fork)\s*\(", "Process spawning not allowed"),
    (r"\b(socket|connect|bind|listen|accept|send|recv)\s*\(", "Network calls not allowed"),
    (r"\b(fopen|freopen)\s*\([^)]*['\"]\/", "Absolute path file access not allowed"),
    (r"\bopen\s*\([^)]*['\"]\/", "Absolute path access not allowed"),
    (r"\bgetenv\s*\(", "Environment variable access not allowed"),
    (r"\/etc\/|\/proc\/|\/data\/", "Sensitive path access not allowed"),
    (r"\basm\s*\(|__asm__", "Inline assembly not allowed"),
    (r"\bdlopen\s*\(|\bdlsym\s*\(", "Dynamic loading not allowed"),
]


def validate_kernel_source(source: str) -> tuple[bool, str]:
    """
    Check kernel source for dangerous patterns.

    Returns (is_valid, error_message).
    """
    for pattern, message in BLOCKED_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            return False, message
    return True, ""


def sanitize_error_message(error: str) -> str:
    """Remove sensitive paths from error messages."""
    error = re.sub(r"/tmp/tmp[a-zA-Z0-9_]+/", "", error)
    error = re.sub(r"/app/", "", error)
    return error


# --- Problem Loading ---


def parse_harness(harness_content: str) -> dict:
    """Parse a harness.cu file into setup, launch, and validation sections."""
    # Find section markers
    setup_match = re.search(r"// === SETUP ===\n(.*?)(?=// === LAUNCH ===)", harness_content, re.DOTALL)
    launch_match = re.search(r"// === LAUNCH ===\n(.*?)(?=// === VALIDATION ===)", harness_content, re.DOTALL)
    validation_match = re.search(r"// === VALIDATION ===\n(.*?)$", harness_content, re.DOTALL)

    return {
        "setup": setup_match.group(1).strip() if setup_match else "",
        "launch": launch_match.group(1).strip() if launch_match else "",
        "validation": validation_match.group(1).strip() if validation_match else "",
    }


def parse_stub(stub_content: str) -> dict:
    """Parse a stub.cu file to extract problem name and signature."""
    # Extract problem name from first comment line like "* Problem: Vector Addition"
    name_match = re.search(r"\* Problem: (.+)", stub_content)
    name = name_match.group(1).strip() if name_match else "Unknown Problem"

    # Extract signature - look for __global__ void function declaration
    sig_match = re.search(r"(__global__\s+void\s+\w+\s*\([^)]*\))", stub_content)
    signature = sig_match.group(1).strip() if sig_match else ""

    return {"name": name, "expected_signature": signature}


def load_problems(problems_dir: str) -> dict:
    """Load all problems from the problems directory."""
    problems = {}
    problems_path = Path(problems_dir)

    if not problems_path.exists():
        return problems

    for problem_dir in sorted(problems_path.iterdir()):
        if not problem_dir.is_dir() or problem_dir.name.startswith("."):
            continue

        harness_path = problem_dir / "harness.cu"
        stub_path = problem_dir / "stub.cu"

        if not harness_path.exists() or not stub_path.exists():
            continue

        harness_content = harness_path.read_text()
        stub_content = stub_path.read_text()

        harness_data = parse_harness(harness_content)
        stub_data = parse_stub(stub_content)

        problem_id = problem_dir.name
        problems[problem_id] = {
            "name": stub_data["name"],
            "expected_signature": stub_data["expected_signature"],
            **harness_data,
        }

    return problems


# Load problems at module level (for local testing)
# In Modal, this will be re-loaded from /app/problems
PROBLEMS = load_problems(str(LOCAL_PROBLEMS_DIR))
_STUBS_BY_ID: dict[str, str] | None = None


def get_problems():
    """Get problems dict, reloading from PROBLEMS_DIR if running in Modal."""
    global PROBLEMS
    # In Modal container, reload from /app/problems
    if Path(PROBLEMS_DIR).exists() and not PROBLEMS:
        PROBLEMS = load_problems(PROBLEMS_DIR)
    return PROBLEMS


def get_stubs_by_id() -> dict[str, str]:
    """Get a {problem_id: stub_text} mapping, lazily loaded per container."""
    global _STUBS_BY_ID
    if _STUBS_BY_ID is not None:
        return _STUBS_BY_ID

    problems = get_problems()
    stubs: dict[str, str] = {}
    base = Path(PROBLEMS_DIR)
    for problem_id in problems.keys():
        stub_path = base / problem_id / "stub.cu"
        if stub_path.exists():
            stubs[problem_id] = stub_path.read_text()
    _STUBS_BY_ID = stubs
    return stubs


def build_harness_tu(problem_id: str) -> str:
    """Build the harness translation unit (host code) without inlining user source."""
    problems = get_problems()
    problem = problems[problem_id]

    expected_signature = (problem.get("expected_signature") or "").strip()
    forward_decl = f"{expected_signature};" if expected_signature else ""

    return f"""
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

{forward_decl}

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

    // Output time with a sentinel to make parsing robust.
    printf("TIME_MS=%f\\n", ms);

    return 0;
}}
"""


def validate_submission_inputs(problem_id: str, user_id: str, kernel_source: str) -> tuple[bool, str, str]:
    if not isinstance(problem_id, str) or not isinstance(kernel_source, str):
        return False, "invalid_payload", "problem_id and kernel_source must be strings"
    if user_id is not None and not isinstance(user_id, str):
        return False, "invalid_payload", "user_id must be a string"
    if not problem_id or not kernel_source:
        return False, "missing_fields", "Required fields: problem_id, kernel_source"
    if len(kernel_source) > MAX_KERNEL_SOURCE_BYTES:
        return False, "kernel_too_large", f"Kernel source must be under {MAX_KERNEL_SOURCE_BYTES} bytes"
    if user_id is None:
        user_id = "anonymous"
    if len(user_id) > MAX_USER_ID_LENGTH:
        return False, "invalid_user_id", f"User ID must be under {MAX_USER_ID_LENGTH} characters"
    return True, "", ""


def run_limited_subprocess(
    *,
    args: list[str],
    cwd: str,
    env: dict[str, str] | None,
    timeout_s: int,
    max_output_bytes: int,
) -> tuple[int, str, bool]:
    """
    Run a subprocess capturing combined stdout/stderr to a file, enforcing an output size cap.

    Returns (returncode, output, timed_out).
    """
    import subprocess

    out_path = os.path.join(cwd, "proc_output.txt")
    try:
        with open(out_path, "wb") as out:
            result = subprocess.run(
                args,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        size = os.path.getsize(out_path)
        with open(out_path, "rb") as f:
            output_bytes = f.read(max_output_bytes + 1)
        output = output_bytes[:max_output_bytes].decode(errors="replace")
        if size > max_output_bytes:
            return 1, f"Process output exceeded {max_output_bytes} bytes (truncated):\n{output}", False
        return result.returncode, output, False
    except subprocess.TimeoutExpired:
        return 124, "Process timed out", True


def parse_time_ms(stdout: str) -> float:
    import math

    time_ms = None
    for line in stdout.splitlines():
        if line.startswith("TIME_MS="):
            value = line.split("=", 1)[1].strip()
            try:
                time_ms = float(value)
            except ValueError:
                continue
    if time_ms is None or not math.isfinite(time_ms):
        raise ValueError("Could not parse TIME_MS from output")
    return time_ms

# --- Modal Functions ---


@app.function(image=web_image, volumes={"/data": vol})
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

@app.function(image=web_image, volumes={"/data": vol}, timeout=30)
def record_submission(problem_id: str, user_id: str, time_ms: float, kernel_hash: str) -> None:
    """Persist a benchmark result to SQLite (separate from untrusted kernel execution)."""
    import sqlite3
    from datetime import datetime

    os.makedirs("/data", exist_ok=True)
    vol.reload()

    conn = sqlite3.connect(DB_PATH, timeout=30)
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
    conn.execute(
        """
        INSERT INTO submissions (problem_id, user_id, time_ms, kernel_hash, submitted_at)
        VALUES (?, ?, ?, ?, ?)
    """,
        (problem_id, user_id, time_ms, kernel_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    vol.commit()


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=120,
    max_containers=3,  # Prevent abuse
)
def benchmark_kernel(problem_id: str, user_id: str, kernel_source: str) -> dict:
    """Compile and benchmark a user's kernel submission."""
    import hashlib
    import tempfile

    ok, err, msg = validate_submission_inputs(problem_id, user_id, kernel_source)
    if not ok:
        return {"success": False, "error": err, "message": msg}
    user_id = user_id or "anonymous"

    problems = get_problems()
    if problem_id not in problems:
        return {
            "success": False,
            "error": "unknown_problem",
            "message": f"Problem '{problem_id}' not found. Available: {list(problems.keys())}",
        }

    # Security: Validate kernel source for dangerous patterns
    is_valid, validation_error = validate_kernel_source(kernel_source)
    if not is_valid:
        return {
            "success": False,
            "error": "blocked_pattern",
            "message": f"Kernel rejected: {validation_error}",
        }

    kernel_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]

    with tempfile.TemporaryDirectory() as tmpdir:
        harness_path = os.path.join(tmpdir, "harness.cu")
        user_path = os.path.join(tmpdir, "user.cu")
        binary_path = os.path.join(tmpdir, "kernel")

        harness_source = build_harness_tu(problem_id)

        with open(harness_path, "w") as f:
            f.write(harness_source)

        with open(user_path, "w") as f:
            f.write(kernel_source)

        # Compile
        returncode, compile_output, timed_out = run_limited_subprocess(
            args=["nvcc", "-O3", "-arch=sm_75", "-o", binary_path, harness_path, user_path],
            cwd=tmpdir,
            env=None,
            timeout_s=60,
            max_output_bytes=MAX_COMPILE_OUTPUT_BYTES,
        )

        if timed_out:
            return {
                "success": False,
                "error": "compilation_timeout",
                "message": "Compilation timed out",
            }

        if returncode != 0:
            return {
                "success": False,
                "error": "compilation_failed",
                "message": sanitize_error_message(compile_output),
            }

        # Run benchmark (multiple iterations)
        # Security: Run with empty environment and isolated working directory
        safe_env = {}
        if "PATH" in os.environ:
            safe_env["PATH"] = os.environ["PATH"]
        if "LD_LIBRARY_PATH" in os.environ:
            safe_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

        times = []
        for _ in range(10):
            returncode, output, timed_out = run_limited_subprocess(
                args=[binary_path],
                cwd=tmpdir,
                env=safe_env,
                timeout_s=30,
                max_output_bytes=MAX_RUN_OUTPUT_BYTES,
            )

            if timed_out:
                return {
                    "success": False,
                    "error": "runtime_timeout",
                    "message": "Kernel execution timed out",
                }

            if returncode != 0:
                return {
                    "success": False,
                    "error": "runtime_error",
                    "message": sanitize_error_message(output) or "Kernel execution failed (no output)",
                }

            try:
                times.append(parse_time_ms(output))
            except ValueError as e:
                return {
                    "success": False,
                    "error": "parse_error",
                    "message": f"{e}. Output:\n{sanitize_error_message(output)}",
                }

        median_time = sorted(times)[len(times) // 2]

        return {
            "success": True,
            "time_ms": median_time,
            "times": times,
            "kernel_hash": kernel_hash,
        }


@app.function(image=web_image, volumes={"/data": vol})
@modal.fastapi_endpoint(method="GET")
def leaderboard(problem_id: str = None):
    """Get leaderboard data as JSON."""
    import sqlite3

    from fastapi.responses import JSONResponse

    problems = get_problems()
    vol.reload()  # Get latest data from other containers
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if problem_id:
        rows = conn.execute(
            """
            SELECT
                user_id,
                MIN(time_ms) as best_time_ms,
                COUNT(*) as attempts,
                MAX(submitted_at) as last_submission
            FROM submissions
            WHERE problem_id = ?
            GROUP BY user_id
            ORDER BY best_time_ms ASC
        """,
            (problem_id,),
        ).fetchall()

        result = {
            "problem_id": problem_id,
            "problem_name": problems.get(problem_id, {}).get("name", problem_id),
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
                    "problem_name": problems.get(r["problem_id"], {}).get("name", r["problem_id"]),
                    "participants": r["participants"],
                    "total_submissions": r["total_submissions"],
                    "best_time_ms": round(r["best_time_ms"], 4) if r["best_time_ms"] else None,
                }
                for r in rows
            ],
        }

    conn.close()
    # Cache for 10 seconds - leaderboard updates frequently
    return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=10"})


@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET")
def problems():
    """List available problems."""
    from fastapi.responses import JSONResponse

    all_problems = get_problems()
    data = {
        "problems": [
            {
                "id": pid,
                "name": pdata["name"],
                "expected_signature": pdata["expected_signature"],
            }
            for pid, pdata in all_problems.items()
        ]
    }
    # Cache for 1 day - problems only change on redeployment
    return JSONResponse(content=data, headers={"Cache-Control": "public, max-age=86400"})


@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET")
def stub(problem_id: str):
    """Get the stub.cu content for a problem (for web editor)."""
    from fastapi.responses import JSONResponse

    probs = get_problems()
    if problem_id not in probs:
        return JSONResponse(
            content={"error": f"Problem '{problem_id}' not found"}, status_code=404
        )

    stubs = get_stubs_by_id()
    if problem_id in stubs:
        # Cache for 7 days - stubs never change between deployments
        return JSONResponse(
            content={"stub": stubs[problem_id]},
            headers={"Cache-Control": "public, max-age=604800"},
        )
    return JSONResponse(content={"error": "Stub file not found"}, status_code=404)

@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET")
def stubs():
    """Get all stub.cu contents for all problems in one request."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content={"stubs": get_stubs_by_id()},
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET")
def index():
    """Serve the main web UI."""
    from fastapi.responses import HTMLResponse

    html_path = Path("/app/static/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), headers={"Cache-Control": "public, max-age=300"})
    return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)


@app.function(
    image=web_image,
    timeout=120,
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

    ok, err, msg = validate_submission_inputs(problem_id, user_id, kernel_source)
    if not ok:
        return {"success": False, "error": err, "message": msg}

    # Optional API key check
    if SUBMIT_API_KEY is not None:
        provided_key = request.get("api_key")
        if provided_key != SUBMIT_API_KEY:
            return {
                "success": False,
                "error": "unauthorized",
                "message": "Invalid or missing API key",
            }

    result = benchmark_kernel.remote(problem_id, user_id, kernel_source)
    if not result.get("success"):
        return result

    record_submission.remote(problem_id, user_id, result["time_ms"], result["kernel_hash"])
    return {
        **result,
        "user_id": user_id,
        "problem_id": problem_id,
    }


# --- Local entrypoint for testing ---


@app.local_entrypoint()
def main(action: str = "test", problem: str = "01_vectoradd"):
    if action == "init":
        init_db.remote()
    elif action == "list":
        # List all available problems
        probs = get_problems()
        print(f"Available problems ({len(probs)}):")
        for pid, pdata in probs.items():
            print(f"  {pid}: {pdata['name']}")
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
        result = benchmark_kernel.remote(problem, "test_user", test_kernel)
        print(f"Result: {result}")
