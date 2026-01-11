#!/usr/bin/env python3
"""
Kernel Leaderboard Submission Script

Usage:
    python submit.py submit <problem_id> <kernel_file> --url <leaderboard_url> [--user <username>]
    python submit.py signup --url <leaderboard_url> --user <username> --password <password>
    python submit.py login --url <leaderboard_url> --user <username> --password <password>
    python submit.py logout --url <leaderboard_url>
    python submit.py whoami --url <leaderboard_url>

Example:
    python submit.py submit week01_vectoradd my_vectoradd.cu --url https://... --user tommy

The leaderboard uses a simple session-token auth system:
- Provide `--token` / `LEADERBOARD_TOKEN`, or
- Provide `--user` + `--password` / `LEADERBOARD_PASSWORD` to log in and fetch a token.

If you omit `--url`, the script will try to resolve it from Modal (requires Modal auth),
or you can set `LEADERBOARD_URL`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

try:
    import modal  # type: ignore
except Exception:  # pragma: no cover
    modal = None


DEFAULT_TOKEN_PATH = pathlib.Path.home() / ".modal_leaderboard_token.json"


def _http_json(url: str, *, method: str = "GET", data: dict[str, Any] | None = None, timeout_s: int = 60) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read().decode("utf-8")
            if not payload:
                return {}
            return json.loads(payload)
    except urllib.error.HTTPError as e:
        payload = (e.read() or b"").decode("utf-8", errors="replace")
        try:
            return json.loads(payload) if payload else {"error": f"http_{e.code}"}
        except json.JSONDecodeError:
            return {"error": f"http_{e.code}", "message": payload}
    except urllib.error.URLError as e:
        return {"error": "network_error", "message": str(e)}


def _resolve_base_url(explicit_url: str | None) -> str | None:
    if explicit_url:
        return explicit_url.rstrip("/")

    env_url = os.environ.get("LEADERBOARD_URL")
    if env_url:
        return env_url.rstrip("/")

    if modal is None:
        return None

    try:
        web_fn = modal.Function.from_name("kernel-leaderboard", "web")
    except Exception:
        return None

    for attr in ("web_url", "url"):
        value = getattr(web_fn, attr, None)
        if isinstance(value, str) and value:
            return value.rstrip("/")

    return None


def _load_saved_token(token_path: pathlib.Path, base_url: str) -> str | None:
    try:
        raw = token_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    if data.get("url") != base_url:
        return None
    token = data.get("token")
    return token if isinstance(token, str) and token else None


def _load_saved_auth(token_path: pathlib.Path, base_url: str) -> dict[str, Any] | None:
    try:
        raw = token_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("url") != base_url:
        return None
    return data  # type: ignore[return-value]


def _save_token(token_path: pathlib.Path, *, base_url: str, user_id: str | None, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"url": base_url, "user_id": user_id, "token": token}
    tmp = token_path.with_suffix(token_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(token_path)


def login(*, base_url: str, user_id: str, password: str, token_path: pathlib.Path | None) -> str:
    resp = _http_json(f"{base_url}/login", method="POST", data={"user_id": user_id, "password": password})
    if not resp.get("success"):
        msg = resp.get("message") or resp.get("error") or "Login failed"
        raise RuntimeError(str(msg))

    token = resp.get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Login response did not include a token")

    if token_path is not None:
        resolved_user_id = resp.get("user_id") if isinstance(resp.get("user_id"), str) else user_id
        _save_token(token_path, base_url=base_url, user_id=resolved_user_id, token=token)

    return token


def signup(*, base_url: str, user_id: str, password: str) -> dict[str, Any]:
    return _http_json(f"{base_url}/signup", method="POST", data={"user_id": user_id, "password": password})


def logout(*, base_url: str, token: str, token_path: pathlib.Path | None) -> None:
    resp = _http_json(f"{base_url}/logout", method="POST", data={"token": token})
    if not resp.get("success"):
        msg = resp.get("message") or resp.get("error") or "Logout failed"
        raise RuntimeError(str(msg))

    if token_path is not None:
        saved = _load_saved_auth(token_path, base_url)
        if saved and saved.get("token") == token:
            try:
                token_path.unlink()
            except FileNotFoundError:
                pass


def whoami(*, base_url: str, token: str | None, token_path: pathlib.Path | None, fallback_user: str | None) -> None:
    user_id = None
    saved_token = None
    if token_path is not None:
        saved = _load_saved_auth(token_path, base_url)
        if saved:
            saved_token = saved.get("token") if isinstance(saved.get("token"), str) else None
            user_id = saved.get("user_id") if isinstance(saved.get("user_id"), str) else None
    if not user_id:
        user_id = fallback_user

    active_token = token or saved_token
    token_hint = f"{active_token[:8]}…" if isinstance(active_token, str) and active_token else "none"
    print(f"url:  {base_url}")
    print(f"user: {user_id or 'unknown'}")
    print(f"token: {token_hint}")


def submit(*, base_url: str, problem_id: str, kernel_file: str, token: str, api_key: str | None) -> None:
    if not os.path.exists(kernel_file):
        print(f"Error: File '{kernel_file}' not found")
        raise SystemExit(1)

    with open(kernel_file, encoding="utf-8") as f:
        kernel_source = f.read()

    payload: dict[str, Any] = {"problem_id": problem_id, "token": token, "kernel_source": kernel_source}
    if api_key:
        payload["api_key"] = api_key

    print(f"Submitting '{kernel_file}' to '{problem_id}' via {base_url} ...")
    print()

    result = _http_json(f"{base_url}/submit", method="POST", data=payload, timeout_s=180)

    if result.get("success"):
        print("✓ Submission successful!")
        if result.get("user_id"):
            print(f"  User:        {result['user_id']}")
        print(f"  Median time: {result['time_ms']:.4f} ms")
        print(f"  All runs:    {', '.join(f'{t:.4f}' for t in result['times'])} ms")
        print(f"  Kernel hash: {result['kernel_hash']}")
        return

    print(f"✗ Submission failed: {result.get('error', 'unknown_error')}")
    print()
    print(result.get("message", "No error message provided"))
    raise SystemExit(1)


def list_problems(*, base_url: str) -> None:
    data = _http_json(f"{base_url}/problems")
    problems = data.get("problems")
    if not isinstance(problems, list):
        print(f"Could not fetch problems: {data.get('error', 'unknown_error')} {data.get('message', '')}".strip())
        raise SystemExit(1)
    for p in problems:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        name = p.get("name")
        if isinstance(pid, str) and isinstance(name, str):
            print(f"{pid}: {name}")


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url",
        default=None,
        help="Leaderboard base URL (or set LEADERBOARD_URL). If omitted, tries to resolve from Modal.",
    )

    common.add_argument(
        "--token-file",
        default=str(DEFAULT_TOKEN_PATH),
        help=f"Path to save/load a session token (default: {DEFAULT_TOKEN_PATH})",
    )

    common.add_argument(
        "--token",
        default=os.environ.get("LEADERBOARD_TOKEN"),
        help="Session token (or set LEADERBOARD_TOKEN). If omitted, may load from --token-file or log in.",
    )

    common.add_argument(
        "--user",
        "-u",
        default=os.environ.get("LEADERBOARD_USER"),
        help="Username for login (or set LEADERBOARD_USER).",
    )
    common.add_argument(
        "--password",
        "-p",
        default=os.environ.get("LEADERBOARD_PASSWORD"),
        help="Password for login (or set LEADERBOARD_PASSWORD).",
    )
    common.add_argument(
        "--api-key",
        default=os.environ.get("LEADERBOARD_API_KEY"),
        help="Optional API key if the server requires SUBMIT_API_KEY (or set LEADERBOARD_API_KEY).",
    )

    parser = argparse.ArgumentParser(description="Submit CUDA kernels to the GPU Kernel Leaderboard", parents=[common])
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    submit_parser = subparsers.add_parser("submit", help="Submit a kernel", parents=[common])
    submit_parser.add_argument("problem_id", help="Problem ID (e.g., week01_vectoradd)")
    submit_parser.add_argument("kernel_file", help="Path to your .cu file")

    subparsers.add_parser("list", help="List available problems", parents=[common])
    subparsers.add_parser("login", help="Log in and save a token", parents=[common])
    subparsers.add_parser("signup", help="Create an account and save a token", parents=[common])
    subparsers.add_parser("logout", help="Invalidate the current token", parents=[common])
    subparsers.add_parser("whoami", help="Show saved auth info", parents=[common], aliases=["me"])

    argv = sys.argv[1:]
    commands = {"submit", "list", "login", "signup", "logout", "whoami", "me"}
    value_flags = {
        "--url",
        "--token-file",
        "--token",
        "--user",
        "-u",
        "--password",
        "-p",
        "--api-key",
    }
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in value_flags:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    if i < len(argv) and argv[i] not in commands:
        argv = argv[:i] + ["submit"] + argv[i:]

    args = parser.parse_args(argv)

    base_url = _resolve_base_url(args.url)
    if not base_url:
        print("Error: leaderboard URL not set.")
        print("Pass `--url https://...-web.modal.run` or set `LEADERBOARD_URL`.")
        print("Tip: after `modal deploy modal_app.py`, run `modal app list` to find the web endpoint URL.")
        raise SystemExit(2)

    token_path = pathlib.Path(args.token_file).expanduser()

    token = args.token or _load_saved_token(token_path, base_url)

    if args.command in (None, "submit", "login") and not token:
        if not args.user or not args.password:
            print("Error: no session token available.")
            print("Provide `--token` / `LEADERBOARD_TOKEN`, or `--user` + `--password` to log in.")
            raise SystemExit(2)
        token = login(base_url=base_url, user_id=args.user, password=args.password, token_path=token_path)

    if args.command == "signup":
        if not args.user or not args.password:
            print("Error: `signup` requires `--user` and `--password` (or LEADERBOARD_USER/LEADERBOARD_PASSWORD).")
            raise SystemExit(2)
        resp = signup(base_url=base_url, user_id=args.user, password=args.password)
        if not resp.get("success"):
            msg = resp.get("message") or resp.get("error") or "Signup failed"
            print(f"Error: {msg}")
            raise SystemExit(1)
        token = login(base_url=base_url, user_id=args.user, password=args.password, token_path=token_path)
        print(token)
        return

    if args.command == "submit":
        assert token is not None
        submit(
            base_url=base_url,
            problem_id=args.problem_id,
            kernel_file=args.kernel_file,
            token=token,
            api_key=args.api_key,
        )
        return

    if args.command == "list":
        list_problems(base_url=base_url)
        return

    if args.command == "login":
        assert token is not None
        print(token)
        return

    if args.command == "logout":
        if not token:
            print("Error: no session token available to log out.")
            print("Provide `--token` / `LEADERBOARD_TOKEN`, or ensure you have a saved token in --token-file.")
            raise SystemExit(2)
        try:
            logout(base_url=base_url, token=token, token_path=token_path)
        except RuntimeError as e:
            print(f"Error: {e}")
            raise SystemExit(1)
        print("ok")
        return

    if args.command in ("whoami", "me"):
        whoami(base_url=base_url, token=args.token, token_path=token_path, fallback_user=args.user)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
