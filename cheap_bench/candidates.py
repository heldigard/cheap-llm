# vs-soft-allow — CANDIDATES rows are intentionally tabular (model | provider |
# pricing) and exceed the 100-char line cap when listing every field on one
# line. Splitting across lines breaks the visual "diff candidates" comparison.
"""Candidate catalog — the models the benchmark exercises.

One row per candidate with its provider, env var, and public-listing pricing
(per 1M tokens, USD). Pricing is the public listing, NOT usage.cost (which
reports $0 for some promo models). Every cloud candidate is PAYG.
"""

from __future__ import annotations

# --- Candidate catalog -----------------------------------------------------
# 2026-06-19 round 3: pruned to the top 7 + local Ollama. Dropped from
# earlier rounds (score <85 or had FAILs in diff_review):
#   - google/gemma-4-31b-it (85.4) — no edge over the local primary (free)
#   - qwen/qwen3.7-plus (84.6) — slow + expensive
#   - qwen/qwen3.6-flash (83.6) — no edge
#   - moonshotai/kimi-k2.7-code (83.6) — duplicates deepseek-v4-flash
#   - cryptidbleh/gemma4-claude-opus-4.6 is the current free-text T1 primary.
#     Keep this candidate aligned with cheap_llm.DEFAULT_LOCAL_PRIMARY so new
#     benchmark rounds measure the model that production actually routes to.
#   - nvidia/nemotron-3-super-120b (80.8) — slow
#   - nvidia/nemotron-3-nano (69.8, FAIL) — variance
#   - xiaomi/mimo-v2.5 (53.0, FAIL) — variance
#   - stepfun/step-3.7-flash (36.4, FAIL) — bad fit for short tasks
# Re-add a model if it shows a meaningful improvement in a future round.
CANDIDATES: list[dict] = [
    # Local (T1) — production free-text default. Keep the benchmark
    # self-contained while matching cheap_llm.DEFAULT_LOCAL_PRIMARY.
    {
        "id": "cryptidbleh/gemma4-claude-opus-4.6:latest",
        "kind": "local",
        "provider": "ollama",
        "input": 0.0,
        "output": 0.0,
    },
    # T2 cloud — pricing per OpenRouter public listing (NOT usage.cost, which
    # reports $0 for some promo models). Every cloud candidate is PAYG.
    # Pruned from this list (see DO-NOT-RE-TEST ledger in
    # topics/tested-models.md): kimi-k2 (superseded by gpt-5.4-nano),
    # gpt-4.1-nano (old 4.1 gen, deprecation risk despite high score),
    # qwen3.6-flash (reasoning tax). Also rejected outright: gpt-5-nano +
    # glm-4.7-flash (reasoning-only, content="" on short tasks).
    {
        "id": "inclusionai/ling-2.6-flash",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.01,
        "output": 0.03,
    },
    {
        "id": "inclusionai/ling-2.6-1t",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.075,
        "output": 0.625,
    },
    {
        "id": "google/gemini-3.1-flash-lite",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.25,
        "output": 1.50,
    },
    {
        "id": "openai/gpt-5.4-nano",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.20,
        "output": 1.25,
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.098,
        "output": 0.196,
    },
]
