"""Leaderboard reporting — rank models and print per-task winners."""

from __future__ import annotations


def print_leaderboard(results: list[dict]) -> None:
    # per-model totals
    by_model: dict[str, list[int]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r["total"])
    print("\n" + "=" * 80)
    print("LEADERBOARD (sum of scores across tasks, higher = better)")
    print("=" * 80)
    print(f"{'Model':40} {'Avg':>5} {'Sum':>5} {'Latency':>9} {'Cost':>10} {'Status'}")
    ranked = sorted(
        by_model.items(),
        key=lambda kv: (
            -sum(kv[1]) / len(kv[1]),
            sum(r["latency"] for r in results if r["model"] == kv[0]),
        ),
    )
    for model, scores in ranked:
        recs = [r for r in results if r["model"] == model]
        avg = sum(scores) / len(scores)
        total_lat = sum(r["latency"] for r in recs)
        total_cost = sum(r["cost_usd"] for r in recs)
        any_err = any(r.get("error") for r in recs)
        status = "❌" if any_err else "✓"
        print(
            f"{model:40} {avg:5.1f} {sum(scores):5d} {total_lat:8.2f}s ${total_cost:9.6f}  {status}"
        )

    # per-task winner
    print("\n" + "=" * 80)
    print("PER-TASK WINNERS")
    print("=" * 80)
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    for task, recs in by_task.items():
        winner = max(recs, key=lambda r: r["total"])
        print(
            f"  {task:20} → {winner['model']:35} "
            f"score={winner['total']:3d} lat={winner['latency']:5.2f}s"
        )
