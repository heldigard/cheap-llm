"""CLI entry — argparse + run + report + append JSONL for longitudinal tracking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .candidates import CANDIDATES
from .report import print_leaderboard
from .runner import run_benchmark
from .tasks import TASKS


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark cheap models for the preprocessor slots.")
    p.add_argument("--model", action="append", help="filter to specific model id")
    p.add_argument("--task", action="append", help="filter to specific task name")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".claude" / "state" / "cheap-bench" / "results.jsonl",
    )
    args = p.parse_args()

    model_filter = set(args.model) if args.model else None
    task_filter = set(args.task) if args.task else None

    print(f"Running benchmark: {len(CANDIDATES)} models × {len(TASKS)} tasks")
    if model_filter:
        print(f"  model filter: {model_filter}")
    if task_filter:
        print(f"  task filter: {task_filter}")

    out = run_benchmark(model_filter, task_filter, timeout=args.timeout)
    results = out["results"]
    print_leaderboard(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        ts = int(time.time())
        for r in results:
            r["ts"] = ts
            f.write(json.dumps(r) + "\n")
    print(f"\nResults appended to {args.out}")
    return 0
