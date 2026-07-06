# cheap-llm

Unified cheap-LLM cascade client for preprocessor slots. T1 local (Ollama) →
T2 cloud (ling/gemini/gpt-nano) with cross-provider failover, secret scrub,
and JSON contract.

## Install

```bash
# Dev install (editable)
cd ~/cheap-llm && pip install -e .

# Or just add to PATH via the ecosystem shim (already wired)
# ~/.claude/scripts/cheap_llm.py → imports from ~/cheap-llm/
```

## Quick start

```bash
# Probe: what's available right now?
cheap-llm --probe

# Run the cascade
cheap-llm --system "Classify. JSON only." --prompt "ECONNREFUSED 127.0.0.1:5432"

# Full JSON envelope with field validation
cheap-llm --system "Classify." --prompt "..." --schema category reason --json
```

## Programmatic usage

```python
from cheap_llm import cheap_complete

out = cheap_complete(
    system="Classify the prompt. Reply JSON only.",
    prompt="I'm getting ECONNREFUSED...",
    schema_hint=["category", "reason"],
    timeout_total=20.0,
)
# out: {text, model, tier, latency, cost, json_valid, fields_ok, attempts, error}
```

## Cascade

| Tier | Model | Provider | Cost (per M tokens) | Timeout |
|------|-------|----------|---------------------|---------|
| T1 | qwen3.5:4b | Ollama (local) | $0 | 6s |
| T2 | ling-2.6-flash | OpenRouter → ZenMux | $0.01/$0.03 | 12s |
| T2 | ling-2.6-1t | OpenRouter → ZenMux | $0.075/$0.625 | 12s |
| T2 | gemini-3.1-flash-lite | OpenRouter → ZenMux | $0.25/$1.50 | 12s |
| T2 | gpt-5.4-nano | OpenRouter | $0.20/$1.25 | 12s |
| T2 | deepseek-v4-flash | OpenRouter (BYOK) | $0 | 12s |

Cross-provider failover: OpenRouter primary, ZenMux backup per model.

## Testing

```bash
# Unit + mocked (101 tests, offline, no API keys needed)
python3 tests/test_cheap_llm.py

# Public-API contract gate (SemVer + signature + return shape + require())
python3 tests/test_contract.py

# Live + E2E (real API calls, opt-in)
python3 tests/test_cheap_llm_live.py --live

# Benchmark
python3 cheap_bench.py
```

## Versioning / Contract (ecosystem decoupling)

The public surface consumers depend on is **declared and versioned** so this
project evolves without silently breaking fusion / web-research / the 7
`~/.claude/scripts` consumers:

- `__version__` (SemVer), `__all__`, `RESULT_KEYS`, `CHEAP_COMPLETE_PARAMS`,
  `CONTRACT`. Everything else is `_`-private.
- `require(min_version)` — version gate; consumers fail fast on drift.
- `tests/test_contract.py` — breaking change fails here first → MAJOR bump.
- SemVer: MAJOR=removed/renamed public param or key · MINOR=additive · PATCH=internal.

```python
import cheap_llm
cheap_llm.require("1.1")                      # trips loudly if installed < 1.1
out = cheap_llm.cheap_complete(system=..., prompt=...)
```

## Consumers

7 scripts in `~/.claude/scripts/` import `cheap_llm` for LLM-backed
preprocessing: commit-draft, diff-review, error-classify, extract-tool-output,
pdf-extract-structured, pr-draft, test-triage. Also used by `~/web-research/`
(synthesis cloud fallback) and `~/fusion/` (judge transport).

## Security

All prompts are scrubbed through `scrub_secrets()` before reaching any
third-party API — even the local Ollama path (T1 timeouts cascade to cloud).
Patterns: PEM keys, connection strings, Bearer tokens, API keys, JWTs, cloud
provider credentials.
