"""Benchmark runner — execute every candidate × task and collect scored results."""

from __future__ import annotations

import sys

from .calls import call_candidate
from .candidates import CANDIDATES
from .scoring import score_response
from .tasks import TASKS


def run_benchmark(
    model_filter: set[str] | None = None, task_filter: set[str] | None = None, timeout: float = 30.0
) -> dict:
    results: list[dict] = []
    candidates = [c for c in CANDIDATES if model_filter is None or c["id"] in model_filter]
    tasks = [t for t in TASKS if task_filter is None or t["name"] in task_filter]

    for task in tasks:
        for cand in candidates:
            sys.stdout.write(f"  {task['name']:18} ← {cand['id']:30} ... ")
            sys.stdout.flush()
            raw = call_candidate(cand, task, timeout=timeout)
            score = score_response(task, raw)
            results.append(
                {
                    "task": task["name"],
                    "model": cand["id"],
                    "kind": cand["kind"],
                    "provider": cand["provider"],
                    "latency": round(raw.get("latency", 0), 3),
                    "input_tokens": raw.get("input_tokens", 0),
                    "output_tokens": raw.get("output_tokens", 0),
                    "cost_usd": round(raw.get("cost", 0), 8),
                    "error": raw.get("error"),
                    "raw_text_preview": raw.get("text", "")[:200],
                    **score,
                }
            )
            status = "OK" if raw.get("text") and not raw.get("error") else "FAIL"
            print(
                f"{status:4} score={score['total']:3d} "
                f"lat={raw.get('latency', 0):5.2f}s "
                f"cost=${raw.get('cost', 0):.6f}"
            )
    return {"results": results}
