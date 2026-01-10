#!/usr/bin/env python3
"""
Kernel Leaderboard Submission Script

Usage:
    python submit.py <problem_id> <kernel_file> [--user <username>]

Example:
    python submit.py week01_vectoradd my_vectoradd.cu --user tommy

Or set your username via environment variable:
    export LEADERBOARD_USER="tommy"
    python submit.py week01_vectoradd my_vectoradd.cu
"""

import argparse
import os
import sys

import modal


def submit(problem_id: str, kernel_file: str, user_id: str):
    """Submit a kernel for benchmarking."""

    if not os.path.exists(kernel_file):
        print(f"Error: File '{kernel_file}' not found")
        sys.exit(1)

    with open(kernel_file) as f:
        kernel_source = f.read()

    print(f"Submitting '{kernel_file}' to problem '{problem_id}' as '{user_id}'...")
    print()

    # Look up the deployed Modal function
    try:
        benchmark = modal.Function.from_name("kernel-leaderboard", "benchmark_kernel")
    except modal.exception.NotFoundError:
        print("Error: kernel-leaderboard app not found.")
        print("Make sure the Modal app is deployed: modal deploy modal_app.py")
        sys.exit(1)

    # Run the benchmark
    result = benchmark.remote(
        problem_id=problem_id,
        user_id=user_id,
        kernel_source=kernel_source,
    )

    if result["success"]:
        # Persist the result to the leaderboard DB (separate function to keep /data away from untrusted execution).
        try:
            record = modal.Function.from_name("kernel-leaderboard", "record_submission")
            record.remote(
                problem_id=problem_id,
                user_id=user_id,
                time_ms=result["time_ms"],
                kernel_hash=result["kernel_hash"],
            )
        except Exception as e:
            print(f"Warning: benchmark succeeded but failed to record submission: {e}")

        print("✓ Submission successful!")
        print(f"  Median time: {result['time_ms']:.4f} ms")
        print(f"  All runs:    {', '.join(f'{t:.4f}' for t in result['times'])} ms")
        print(f"  Kernel hash: {result['kernel_hash']}")
    else:
        print(f"✗ Submission failed: {result['error']}")
        print()
        print(result["message"])
        sys.exit(1)


def list_problems():
    """List available problems."""
    try:
        problems_fn = modal.Function.from_name("kernel-leaderboard", "problems")
        # For web endpoints, we need to call differently

        # Get the URL from Modal
        print("Fetching problems list...")
        print("(Run `modal app list` to get your leaderboard URL)")
    except Exception as e:
        print(f"Could not fetch problems: {e}")


def main():
    parser = argparse.ArgumentParser(description="Submit CUDA kernels to the GPU Kernel Leaderboard")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a kernel")
    submit_parser.add_argument("problem_id", help="Problem ID (e.g., week01_vectoradd)")
    submit_parser.add_argument("kernel_file", help="Path to your .cu file")
    submit_parser.add_argument(
        "--user", "-u", default=os.environ.get("LEADERBOARD_USER", "anonymous"), help="Your username (or set LEADERBOARD_USER env var)"
    )

    # List command
    subparsers.add_parser("list", help="List available problems")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.problem_id, args.kernel_file, args.user)
    elif args.command == "list":
        list_problems()
    else:
        # Default behavior: treat positional args as submit
        if len(sys.argv) >= 3:
            problem_id = sys.argv[1]
            kernel_file = sys.argv[2]
            user_id = os.environ.get("LEADERBOARD_USER", "anonymous")
            submit(problem_id, kernel_file, user_id)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
