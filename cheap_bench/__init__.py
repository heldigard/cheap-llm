"""Benchmark cheap models for the preprocessor slots.

Runs a fixed task set (intent classify, commit draft, error classify, JSON
extract, diff review) through every candidate model, scores the output, and
prints a leaderboard. The goal is data, not vibes — pick the model that
actually wins on the actual prompts the pipeline will throw at it.

Candidate models (all in our provider catalog — see ``candidates.CANDIDATES``
for the live list):
  - cryptidbleh/gemma4-claude-opus-4.6 (local Ollama, free, private) — T1
  - ling-2.6-flash / ling-2.6-1t / gemini-3.1-flash-lite (OpenRouter cloud)
  - gpt-5.4-nano / deepseek-v4-flash (OpenRouter PAYG)

Scoring (per task, 0-100):
  - json_valid      : 0 or 25  (valid JSON, structure check)
  - field_match     : 0 or 25  (all required fields present + non-empty)
  - content_quality : 0-40     (string similarity vs reference + heuristic check)
  - latency_penalty : 0-10     (subtract up to 10 for >3s responses)
  - cost_estimate   : reported but not scored (low cost = bonus info)

Output: rank-ordered table printed to stdout; raw JSONL appended to
~/.claude/state/cheap-bench/results.jsonl for longitudinal tracking.

Vertical-slice split of the former cheap_bench.py monolith — one responsibility
per submodule (tasks, candidates, calls, scoring, runner, report, cli). The
package stays SELF-CONTAINED on purpose: submodules import only each other and
the stdlib, never the cheap_llm package under benchmark.

Usage:
    python3 -m cheap_bench                  # all models, all tasks
    python3 -m cheap_bench --task intent    # one task
    python3 -m cheap_bench --model cryptidbleh/gemma4-claude-opus-4.6:latest
"""

from __future__ import annotations

from .candidates import CANDIDATES
from .cli import main
from .runner import run_benchmark
from .tasks import TASKS

__all__ = ["main", "run_benchmark", "CANDIDATES", "TASKS"]
