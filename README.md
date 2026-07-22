# cheap-llm

Unified cheap-LLM cascade client for preprocessor slots. T1 local (Ollama) →
T2 cloud (DeepSeek/Gemini/Ling/GPT Nano) with cross-provider failover, secret scrub,
and JSON contract.

## Install

> **Ubuntu 26 / PEP 668:** system Python is externally managed. Prefer `uv tool install --force --editable ~/PROJECT` for PATH tools, or `python3 -m pip install --user --break-system-packages -e .` for user-site hooks. Or use a project venv.


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

# Inspect route order, credential availability, and billing class without inference
cheap-llm --route-plan

# Run the cascade
cheap-llm --system "Classify. JSON only." --prompt "ECONNREFUSED 127.0.0.1:5432"

# Full JSON envelope with field validation
cheap-llm --system "Classify." --prompt "..." --schema category reason --json

# Bound a short classifier so local and cloud providers cannot over-generate
cheap-llm --system "Classify." --prompt "..." --schema category --max-tokens 256

# Pin a specific T2 fallback model (with the usual OR → ZenMux failover),
# force cloud-only with --no-local, or choose an explicit T1 local model
cheap-llm --system "Synthesize." --prompt "..." --cloud-model deepseek/deepseek-v4-flash
cheap-llm --no-local --system "Synthesize." --prompt "..." --cloud-model deepseek/deepseek-v4-flash
cheap-llm --no-local --system "Synthesize." --prompt "..." \
  --cloud-model deepseek/deepseek-v4-flash --cloud-provider deepinfra
cheap-llm --system "Classify." --prompt "..." --model my-local-model:latest
```

## Environment variables

| Variable | Effect |
|----------|--------|
| `OLLAMA_URL` | Override the local Ollama endpoint (default `http://localhost:11434`) |
| `CHEAP_LLM_LOCAL_MODEL` | Override the free-text Ollama model |
| `CHEAP_LLM_LOCAL_STRUCTURED_MODEL` | Override the JSON/schema Ollama model independently |
| `CHEAP_LLM_LOCAL_ONLY` | `1/true/yes/on` → T1 only, never call cloud (privacy mode) |
| `CHEAP_LLM_LOCAL_COLD_TIMEOUT` | T1 budget in seconds when the model is not loaded in VRAM yet (default 25) |
| `CHEAP_LLM_LOCAL_WARM_TIMEOUT` | Warm free-text T1 budget in seconds (default 8) |
| `CHEAP_LLM_LOCAL_STRUCTURED_TIMEOUT` | Warm structured/JSON T1 budget in seconds (default 18) |
| `CHEAP_LLM_KEEP_ALIVE` | Ollama `keep_alive` after each T1 call (default `15m`; `0`/`off` disables) |
| `OPENROUTER_API_KEY` / `ZENMUX_API_KEY` / `DEEPSEEK_API_KEY` / `DEEPINFRA_API_KEY` | Enable the respective T2 providers |

## Programmatic usage

```python
from cheap_llm import cheap_complete

out = cheap_complete(
    system="Classify the prompt. Reply JSON only.",
    prompt="I'm getting ECONNREFUSED...",
    schema_hint=["category", "reason"],
    timeout_total=20.0,
    max_output_tokens=256,
    # Optional trust/cost boundary: requires cloud_model and disables
    # cross-provider fallback/cache reuse for T2.
    cloud_provider="deepinfra",
)
# out: {text, model, provider, billing, tier, latency, cost, ...}
```

Set `allow_cloud=False` when a caller needs a hard local-only boundary that
does not depend on process-wide environment state. It requires
`prefer_local=True`; the default remains `True` for backward compatibility.

## Cascade

| Tier | Model | Provider | Cost (per M tokens) | Timeout |
|------|-------|----------|---------------------|---------|
| T1 | cryptidbleh/gemma4-claude-opus-4.6 (text) / SetneufPT/Qwopus3.5-4B-Coder-MTP (JSON/schema) | Ollama (local) | $0 | 8s text / 18s JSON (25s cold) |
| T2 | deepseek-v4-flash | OpenRouter → ZenMux | $0.098/$0.196 | 12s |
| T2 | gemini-3.1-flash-lite | OpenRouter → ZenMux | $0.25/$1.50 | 12s |
| T2 | ling-2.6-1t | OpenRouter → ZenMux | $0.075/$0.625 | 12s |
| T2 | gpt-5.4-nano | OpenRouter | $0.20/$1.25 | 12s |
| T2 | ling-2.6-flash | OpenRouter → ZenMux | $0.01/$0.03 | 12s |
| T2 (optional) | deepseek-v4-flash | DeepInfra | $0.09/$0.18 | 12s |

Default cascade: OpenRouter primary, ZenMux backup per benchmarked cheap model.
The final DeepInfra route is included only when `DEEPINFRA_API_KEY` is present,
so route plans never advertise an unauthenticated attempt.
Forced judgment models use provider-aware failover: DeepSeek first-party for
`deepseek/*`, DeepInfra when that model family is available there, then
OpenRouter and ZenMux. `cheap-llm` distills/classifies signals; it is not an
architecture authority, coder, or substitute for the controller brain.

Provider-specific model IDs are normalized only through explicit catalog
bindings. For example, KAT Coder V2.5 is `kuaishou/kat-coder-*-v2.5` on ZenMux
but `kwaipilot/kat-coder-*-v2.5` on OpenRouter. These aliases make an explicitly
pinned model portable across those transports; they do **not** promote KAT,
Doubao, ERNIE, Grok Build, Step, Hy3, or any other unbenchmarked candidate into
the default cascade.

All four cloud providers are PAYG routes. A provider API key is not a CLI-seat
subscription, even when the account currently has granted/promotional balance.
Subscription-backed workers (`codex-spark`, Antigravity, Kimi, Z.AI) remain in
`cli-orchestration`/`fusion-local`; `cheap-llm` only judges or distills their
evidence. Use `cloud_provider`/`--cloud-provider` when a request must not cross
provider billing or trust boundaries. OpenRouter requests sort backing
endpoints by price; direct DeepSeek disables its default thinking mode and uses
the documented model-specific cache price.

## Output budgets and usage

`max_output_tokens` is a hard per-attempt ceiling shared by every transport:
Ollama receives `num_predict`; OpenRouter, ZenMux, and DeepSeek receive
`max_tokens`. The backward-compatible default is 1024. Callers with bounded
contracts should choose a smaller value; `skill-router classify` uses 256 for
its four-field JSON envelope.

The budget participates in the cache key, so a short response can never be
reused for a request that allowed a longer answer; the default 1024 namespace
retains compatibility with existing cache entries. Response-bearing and cached
`attempts` records include `max_output_tokens`, `input_tokens`, and
`output_tokens`, which makes savings measurable without expanding the
top-level result envelope.

Every HTTP response is also capped at 4 MiB before UTF-8 decoding and JSON
parsing. This protects callers from oversized or malformed provider/proxy
responses independently of the requested token budget.

`timeout_total` must be a positive finite number. Invalid deadlines fail
before Ollama probing, cache access, or provider calls. Cache files are written
atomically with private permissions; new entries retain the source provider
and tier so cross-provider cache hits keep accurate telemetry. Reuse is limited
to the same tier: T1 local and T2 cloud never satisfy one another from cache.
Existing text-only cache entries remain compatible and inherit the lookup tier.

## Testing

```bash
# Behavioral + mocked suite (offline, no API keys needed)
python3 tests/test_cheap_llm.py

# Pytest contract gates (public API/SemVer + ecosystem shim)
python3 -m pytest -q

# Live + E2E (real API calls, opt-in)
python3 tests/test_cheap_llm_live.py --live

# Benchmark
python3 -m cheap_bench
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
cheap_llm.require("1.2")                      # needed for max_output_tokens
out = cheap_llm.cheap_complete(system=..., prompt=..., max_output_tokens=256)

cheap_llm.require("1.3")                      # needed for cloud_provider
```

## Consumers

7 scripts in `~/.claude/scripts/` import `cheap_llm` for LLM-backed
preprocessing: commit-draft, diff-review, error-classify, extract-tool-output,
pdf-extract-structured, pr-draft, test-triage. Also used by `~/web-research/`
(synthesis cloud fallback) and `~/fusion-local/` (judge transport).

## Security

All prompts are scrubbed through `scrub_secrets()` before reaching any
third-party API — even the local Ollama path (T1 timeouts cascade to cloud).
Patterns: PEM keys, connection strings, Bearer tokens, API keys, JWTs, cloud
provider credentials.
