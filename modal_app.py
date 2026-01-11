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

MAX_PROFILE_RESULTS = 200

USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


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
_DB_SCHEMA_READY = False


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


def ensure_db_schema(conn) -> None:
    """Create/migrate SQLite schema. Safe to call multiple times."""
    global _DB_SCHEMA_READY
    if _DB_SCHEMA_READY:
        return

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            time_ms REAL NOT NULL,
            kernel_hash TEXT,
            kernel_source TEXT,
            submitted_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_problem_user ON submissions(problem_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_problem_time ON submissions(problem_id, time_ms);
        CREATE INDEX IF NOT EXISTS idx_user_submitted_at ON submissions(user_id, submitted_at);

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT,
            password_hash TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
    """)

    # Migration path for existing volumes created before kernel_source existed.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(submissions)").fetchall()}
    if "kernel_source" not in cols:
        conn.execute("ALTER TABLE submissions ADD COLUMN kernel_source TEXT")

    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

    conn.commit()
    _DB_SCHEMA_READY = True


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


def _strip_c_comments(s: str) -> str:
    s = re.sub(r"/\\*.*?\\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    return s


def _split_c_params(params: str) -> list[str]:
    params = params.strip()
    if not params:
        return []
    # Good enough for our problem signatures (no templates/function pointers).
    return [p.strip() for p in params.split(",") if p.strip()]


_C_QUALIFIERS_RE = re.compile(r"\\b(const|volatile|restrict|__restrict__|__restrict)\\b")


def _canonicalize_c_param_type(param: str) -> str:
    param = _strip_c_comments(param)
    param = param.split("=", 1)[0].strip()
    param = _C_QUALIFIERS_RE.sub("", param)
    param = re.sub(r"\\s+", " ", param).strip()

    # Drop the parameter name (keep just the type-ish prefix).
    m = re.match(r"^(.*?)(?:\\b[A-Za-z_]\\w*\\b)\\s*(?:\\[[^\\]]*\\]\\s*)*$", param)
    if m:
        prefix = (m.group(1) or "").strip()
        if prefix:
            param = prefix

    return param.replace(" ", "")


def _parse_expected_signature(expected_signature: str) -> tuple[str, list[str]] | None:
    expected_signature = _strip_c_comments(expected_signature).strip().rstrip(";")
    m = re.search(r"__global__\\s+void\\s+([A-Za-z_]\\w*)\\s*\\(([^)]*)\\)", expected_signature)
    if not m:
        return None
    name = m.group(1)
    params = [_canonicalize_c_param_type(p) for p in _split_c_params(m.group(2))]
    return name, params


def _extract_kernel_signature_from_source(kernel_source: str, kernel_name: str) -> list[str] | None:
    kernel_source = _strip_c_comments(kernel_source)
    m = re.search(rf"__global__\\s+void\\s+{re.escape(kernel_name)}\\s*\\(([^)]*)\\)", kernel_source)
    if not m:
        return None
    return [_canonicalize_c_param_type(p) for p in _split_c_params(m.group(1))]


def validate_kernel_signature(problem_id: str, kernel_source: str) -> tuple[bool, str, str]:
    problems = get_problems()
    expected_signature = (problems.get(problem_id, {}) or {}).get("expected_signature") or ""
    expected_signature = str(expected_signature).strip()
    if not expected_signature:
        return True, "", ""

    parsed = _parse_expected_signature(expected_signature)
    if parsed is None:
        # Don't hard-fail if a problem stub doesn't include a parsable signature.
        return True, "", ""

    expected_name, expected_params = parsed
    actual_params = _extract_kernel_signature_from_source(kernel_source, expected_name)
    if actual_params is None:
        m = re.search(r"__global__\\s+void\\s+([A-Za-z_]\\w*)\\s*\\(", _strip_c_comments(kernel_source))
        found = m.group(1) if m else None
        hint = f" Found kernel '{found}'." if found and found != expected_name else ""
        return False, "signature_mismatch", f"Kernel must define `{expected_signature}`.{hint}"

    if actual_params != expected_params:
        return False, "signature_mismatch", f"Kernel signature does not match stub. Expected `{expected_signature}`."

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


def validate_user_id(user_id: str) -> tuple[bool, str, str]:
    if not isinstance(user_id, str) or not user_id:
        return False, "invalid_user_id", "Username is required"
    if len(user_id) > MAX_USER_ID_LENGTH:
        return False, "invalid_user_id", f"Username must be under {MAX_USER_ID_LENGTH} characters"
    if not USER_ID_RE.fullmatch(user_id):
        return False, "invalid_user_id", "Username must match ^[a-zA-Z0-9_-]{1,64}$"
    return True, "", ""


def hash_password(password: str) -> str:
    import base64
    import hashlib
    import secrets

    if not isinstance(password, str) or not password:
        raise ValueError("Password required")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    if len(password) > 200:
        raise ValueError("Password too long")

    iterations = 200_000
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(dk).decode("ascii").rstrip("="),
    )


def verify_password(password: str, stored: str) -> bool:
    import base64
    import hashlib
    import hmac

    if not isinstance(password, str) or not isinstance(stored, str) or not stored:
        return False
    parts = stored.split("$")
    if len(parts) != 4:
        return False
    algo, iter_s, salt_b64, hash_b64 = parts
    if algo != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iter_s)
        salt = base64.urlsafe_b64decode(salt_b64 + "==")
        expected = base64.urlsafe_b64decode(hash_b64 + "==")
    except Exception:
        return False

    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def create_session_token() -> str:
    import secrets

    return secrets.token_urlsafe(32)


def get_user_id_for_token(conn, token: str) -> str | None:
    from datetime import datetime

    if not isinstance(token, str) or not token:
        return None
    row = conn.execute("SELECT user_id FROM sessions WHERE token = ?", (token,)).fetchone()
    if row is None:
        return None
    conn.execute("UPDATE sessions SET last_used_at = ? WHERE token = ?", (datetime.utcnow().isoformat(), token))
    return row[0]
# --- Modal Functions ---


@app.function(image=web_image, volumes={"/data": vol})
def init_db():
    """Initialize the SQLite database."""
    import sqlite3

    os.makedirs("/data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    ensure_db_schema(conn)
    conn.close()
    vol.commit()
    print("Database initialized!")

@app.function(image=web_image, volumes={"/data": vol}, timeout=30)
def record_submission(problem_id: str, user_id: str, time_ms: float, kernel_hash: str, kernel_source: str) -> None:
    """Persist a benchmark result to SQLite (separate from untrusted kernel execution)."""
    import sqlite3
    from datetime import datetime

    os.makedirs("/data", exist_ok=True)
    vol.reload()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    ensure_db_schema(conn)
    conn.execute(
        """
        INSERT INTO submissions (problem_id, user_id, time_ms, kernel_hash, kernel_source, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (problem_id, user_id, time_ms, kernel_hash, kernel_source, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    vol.commit()


@app.function(
    image=cuda_image,
    gpu="T4",
    timeout=180,
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

    ok, err, msg = validate_kernel_signature(problem_id, kernel_source)
    if not ok:
        return {"success": False, "error": err, "message": msg}

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
        for _ in range(5):
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


@app.function(image=web_image, volumes={"/data": vol}, timeout=180)
@modal.asgi_app()
def web():
    """Single FastAPI app (1 Modal web endpoint) serving both UI and API routes."""
    import sqlite3
    from datetime import datetime

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI()

    @api.get("/")
    def index():
        html_path = Path("/app/static/index.html")
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(), headers={"Cache-Control": "public, max-age=300"})
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

    @api.get("/leaderboard")
    def leaderboard(problem_id: str | None = None):
        problems = get_problems()
        vol.reload()
        conn = sqlite3.connect(DB_PATH)
        ensure_db_schema(conn)
        conn.row_factory = sqlite3.Row

        if problem_id:
            rows = conn.execute(
                """
                SELECT
                    s.user_id,
                    MIN(s.time_ms) as best_time_ms,
                    COUNT(*) as attempts,
                    MAX(s.submitted_at) as last_submission,
                    (
                        SELECT s2.id
                        FROM submissions s2
                        WHERE s2.problem_id = s.problem_id AND s2.user_id = s.user_id
                        ORDER BY s2.time_ms ASC, s2.id ASC
                        LIMIT 1
                    ) as best_submission_id
                FROM submissions s
                WHERE s.problem_id = ?
                GROUP BY s.user_id
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
                        "best_submission_id": r["best_submission_id"],
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
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=10"})

    @api.get("/problems")
    def problems():
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
        return JSONResponse(content=data, headers={"Cache-Control": "public, max-age=86400"})

    @api.get("/stub")
    def stub(problem_id: str):
        probs = get_problems()
        if problem_id not in probs:
            return JSONResponse(content={"error": f"Problem '{problem_id}' not found"}, status_code=404)

        stubs = get_stubs_by_id()
        if problem_id in stubs:
            return JSONResponse(
                content={"stub": stubs[problem_id]},
                headers={"Cache-Control": "public, max-age=604800"},
            )
        return JSONResponse(content={"error": "Stub file not found"}, status_code=404)

    @api.get("/stubs")
    def stubs():
        return JSONResponse(
            content={"stubs": get_stubs_by_id()},
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @api.get("/profile")
    def profile(user_id: str, limit: int = 50):
        if not isinstance(user_id, str) or not user_id:
            return JSONResponse(content={"error": "missing_user_id"}, status_code=400)
        if len(user_id) > MAX_USER_ID_LENGTH:
            return JSONResponse(content={"error": "invalid_user_id"}, status_code=400)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(MAX_PROFILE_RESULTS, limit))

        problems = get_problems()
        vol.reload()
        conn = sqlite3.connect(DB_PATH)
        ensure_db_schema(conn)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                problem_id,
                time_ms,
                kernel_hash,
                submitted_at
            FROM submissions
            WHERE user_id = ?
            ORDER BY submitted_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

        best_rows = conn.execute(
            """
            SELECT
                problem_id,
                MIN(time_ms) AS best_time_ms
            FROM submissions
            WHERE user_id = ?
            GROUP BY problem_id
            """,
            (user_id,),
        ).fetchall()

        conn.close()
        return JSONResponse(
            content={
                "user_id": user_id,
                "recent": [
                    {
                        "submission_id": r["id"],
                        "problem_id": r["problem_id"],
                        "problem_name": problems.get(r["problem_id"], {}).get("name", r["problem_id"]),
                        "time_ms": round(r["time_ms"], 4),
                        "kernel_hash": r["kernel_hash"],
                        "submitted_at": r["submitted_at"],
                    }
                    for r in rows
                ],
                "best_by_problem": [
                    {
                        "problem_id": r["problem_id"],
                        "problem_name": problems.get(r["problem_id"], {}).get("name", r["problem_id"]),
                        "best_time_ms": round(r["best_time_ms"], 4) if r["best_time_ms"] is not None else None,
                    }
                    for r in best_rows
                ],
            },
            headers={"Cache-Control": "private, max-age=5"},
        )

    @api.get("/submission")
    def submission(submission_id: int):
        try:
            submission_id = int(submission_id)
        except (TypeError, ValueError):
            return JSONResponse(content={"error": "invalid_submission_id"}, status_code=400)
        if submission_id <= 0:
            return JSONResponse(content={"error": "invalid_submission_id"}, status_code=400)

        problems = get_problems()
        vol.reload()
        conn = sqlite3.connect(DB_PATH)
        ensure_db_schema(conn)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
                id,
                problem_id,
                user_id,
                time_ms,
                kernel_hash,
                kernel_source,
                submitted_at
            FROM submissions
            WHERE id = ?
            """,
            (submission_id,),
        ).fetchone()
        conn.close()

        if row is None:
            return JSONResponse(content={"error": "not_found"}, status_code=404)

        return JSONResponse(
            content={
                "submission_id": row["id"],
                "problem_id": row["problem_id"],
                "problem_name": problems.get(row["problem_id"], {}).get("name", row["problem_id"]),
                "user_id": row["user_id"],
                "time_ms": round(row["time_ms"], 4),
                "kernel_hash": row["kernel_hash"],
                "kernel_source": row["kernel_source"] or "",
                "submitted_at": row["submitted_at"],
            },
            headers={"Cache-Control": "private, max-age=60"},
        )

    @api.post("/signup")
    def signup(request: dict):
        user_id = request.get("user_id")
        password = request.get("password")

        ok, err, msg = validate_user_id(user_id)
        if not ok:
            return JSONResponse(content={"success": False, "error": err, "message": msg}, status_code=400)
        try:
            password_hash = hash_password(password)
        except ValueError as e:
            return JSONResponse(content={"success": False, "error": "invalid_password", "message": str(e)}, status_code=400)

        os.makedirs("/data", exist_ok=True)
        vol.reload()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        ensure_db_schema(conn)

        existing = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if existing is not None:
            existing_password_hash = existing[0]
            if existing_password_hash:
                conn.close()
                return JSONResponse(
                    content={
                        "success": False,
                        "error": "user_exists",
                        "message": "Username already taken. Try logging in instead, or choose a different username.",
                    },
                    status_code=409,
                )
            conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id))
            conn.commit()
            conn.close()
            vol.commit()
            return JSONResponse(
                content={"success": True, "user_id": user_id, "upgraded": True},
                headers={"Cache-Control": "no-store"},
            )

        conn.execute(
            "INSERT INTO users (user_id, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, None, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        vol.commit()

        return JSONResponse(content={"success": True, "user_id": user_id}, headers={"Cache-Control": "no-store"})

    @api.post("/login")
    def login(request: dict):
        user_id = request.get("user_id")
        password = request.get("password")

        ok, err, msg = validate_user_id(user_id)
        if not ok:
            return JSONResponse(content={"success": False, "error": err, "message": msg}, status_code=400)
        if not isinstance(password, str) or not password:
            return JSONResponse(content={"success": False, "error": "invalid_password", "message": "Password required"}, status_code=400)

        os.makedirs("/data", exist_ok=True)
        vol.reload()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        ensure_db_schema(conn)

        row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            conn.close()
            return JSONResponse(
                content={
                    "success": False,
                    "error": "user_not_found",
                    "message": "No account found for that username. Click Sign up to create it first.",
                },
                status_code=404,
            )
        if not row[0]:
            conn.close()
            return JSONResponse(
                content={
                    "success": False,
                    "error": "password_not_set",
                    "message": "This username exists but doesn't have a password yet. Click Sign up to set one.",
                },
                status_code=401,
            )
        if not verify_password(password, row[0]):
            conn.close()
            return JSONResponse(
                content={"success": False, "error": "incorrect_password", "message": "Incorrect password for this username."},
                status_code=401,
            )

        token = create_session_token()
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, last_used_at) VALUES (?, ?, ?, ?)",
            (token, user_id, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        vol.commit()

        return JSONResponse(content={"success": True, "user_id": user_id, "token": token}, headers={"Cache-Control": "no-store"})

    @api.post("/logout")
    def logout(request: dict):
        token = request.get("token")
        if not isinstance(token, str) or not token:
            return JSONResponse(content={"success": False, "error": "missing_token"}, status_code=400)

        os.makedirs("/data", exist_ok=True)
        vol.reload()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        ensure_db_schema(conn)
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        vol.commit()
        return JSONResponse(content={"success": True}, headers={"Cache-Control": "no-store"})

    @api.get("/my_last_submission")
    def my_last_submission(problem_id: str, token: str):
        if not isinstance(problem_id, str) or not problem_id:
            return JSONResponse(content={"error": "missing_problem_id"}, status_code=400)
        if not isinstance(token, str) or not token:
            return JSONResponse(content={"error": "missing_token"}, status_code=400)

        problems = get_problems()
        if problem_id not in problems:
            return JSONResponse(content={"error": "unknown_problem"}, status_code=404)

        vol.reload()
        conn = sqlite3.connect(DB_PATH)
        ensure_db_schema(conn)
        conn.row_factory = sqlite3.Row

        user_id = get_user_id_for_token(conn, token)
        if user_id is None:
            conn.close()
            return JSONResponse(content={"error": "unauthorized"}, status_code=401)

        row = conn.execute(
            """
            SELECT
                id,
                problem_id,
                user_id,
                time_ms,
                kernel_hash,
                kernel_source,
                submitted_at
            FROM submissions
            WHERE problem_id = ? AND user_id = ?
            ORDER BY submitted_at DESC, id DESC
            LIMIT 1
            """,
            (problem_id, user_id),
        ).fetchone()
        conn.commit()
        conn.close()
        vol.commit()

        if row is None:
            return JSONResponse(content={"error": "not_found"}, status_code=404)

        return JSONResponse(
            content={
                "submission_id": row["id"],
                "problem_id": row["problem_id"],
                "problem_name": problems.get(row["problem_id"], {}).get("name", row["problem_id"]),
                "user_id": row["user_id"],
                "time_ms": round(row["time_ms"], 4),
                "kernel_hash": row["kernel_hash"],
                "kernel_source": row["kernel_source"] or "",
                "submitted_at": row["submitted_at"],
            },
            headers={"Cache-Control": "no-store"},
        )

    @api.post("/submit")
    def submit(request: dict):
        problem_id = request.get("problem_id")
        token = request.get("token")
        kernel_source = request.get("kernel_source")

        ok, err, msg = validate_submission_inputs(problem_id, "placeholder", kernel_source)
        if not ok:
            return {"success": False, "error": err, "message": msg}

        problems = get_problems()
        if problem_id not in problems:
            return {
                "success": False,
                "error": "unknown_problem",
                "message": f"Problem '{problem_id}' not found.",
            }

        if SUBMIT_API_KEY is not None:
            provided_key = request.get("api_key")
            if provided_key != SUBMIT_API_KEY:
                return {
                    "success": False,
                    "error": "unauthorized",
                    "message": "Invalid or missing API key",
                }

        if not isinstance(token, str) or not token:
            return {
                "success": False,
                "error": "unauthorized",
                "message": "Login required (missing token).",
            }

        vol.reload()
        conn = sqlite3.connect(DB_PATH, timeout=30)
        ensure_db_schema(conn)
        user_id = get_user_id_for_token(conn, token)
        conn.commit()
        conn.close()
        vol.commit()
        if user_id is None:
            return {
                "success": False,
                "error": "unauthorized",
                "message": "Login required (invalid token).",
            }

        ok, err, msg = validate_kernel_signature(problem_id, kernel_source)
        if not ok:
            return {"success": False, "error": err, "message": msg}

        result = benchmark_kernel.remote(problem_id, user_id, kernel_source)
        if not result.get("success"):
            return result

        record_submission.remote(problem_id, user_id, result["time_ms"], result["kernel_hash"], kernel_source)
        return {
            **result,
            "user_id": user_id,
            "problem_id": problem_id,
        }

    return api


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
