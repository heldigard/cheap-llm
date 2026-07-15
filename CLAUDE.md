# cheap-llm

Unified cheap-LLM cascade client for preprocessor slots. Public repo:
https://github.com/heldigard/cheap-llm

Graduated from `~/.claude/scripts/cheap_llm.py` (756-line monolith) into its
own project, mirroring the `codeq` / `web-research` / `smart-trim` layouts.

## What it is

The **signal-distillation layer** for the big model. The cascade's job is to
remove noise and surface precise context (classify a prompt, extract relevant
lines from a log, triage an error, draft a commit message, flag diff issues)
so the big model (Claude Opus / Codex gpt-5.x) gets clean signal. It must NOT
write/edit code, design, or do security work — its output is advisory/distilled
context, never executed code.

## Architecture

```
cheap_llm/            package (vertical-slice architecture)
  __init__.py         re-exports (backward compat for tests + shim + consumers)
  __main__.py         `python3 -m cheap_llm` entry point
  contract.py         version, RESULT_KEYS, require() gate, _complete_result
  scrub.py            SECRET_PATTERNS, scrub_secrets
  cache.py            CACHE_DIR, _cache_key, _cache_get, _cache_put
  transport.py        provider registry, endpoints, all _call_* functions
  cascade.py          _build_cascade, _try_parse_json, _validate, cheap_complete
  cli.py              _probe, _cache_stats, _cache_clear, main
cheap_bench.py        benchmark harness (self-contained transport layer)
tests/
  test_cheap_llm.py       behavioral + mocked suite (offline, no network)
  test_contract.py        pytest public API/SemVer contract gate
  test_shim.py            pytest ecosystem-shim contract gate
  test_cheap_llm_live.py  live + E2E tests (real API calls, opt-in via --live)
```

## Cascade

```
T1 LOCAL  (free, private)    6s  — cryptidbleh/gemma4-claude-opus-4.6 text /
                                    SetneufPT/Qwopus structured JSON (Ollama)
T2 CHEAP CLOUD              12s  — ling-2.6-flash → ling-2.6-1t → gemini-3.1-flash-lite
                                    (OpenRouter primary, ZenMux failover)
    LEGACY safety net             — gpt-5.4-nano, deepseek-v4-flash (PAYG)
```

## Entry points

- **Wired ecosystem shim**: `~/.claude/scripts/cheap_llm.py` → imports from
  here (env `CHEAP_LLM_HOME`, default `~/cheap-llm`). The 7 consumer scripts
  that do `import cheap_llm` keep resolving untouched.
- **Console script** (`pip install -e .`): `cheap-llm`.
- **Direct**: `python3 -m cheap_llm --probe`.

## CLI

```
cheap-llm --version                            # print SemVer and exit
cheap-llm --probe                              # show what's available
cheap-llm --route-plan                         # no-inference route/billing plan
cheap-llm --system "X" --prompt "Y"            # run cascade, print text
cheap-llm --system "X" --prompt "Y" --json     # full JSON envelope
cheap-llm --system "X" --prompt "Y" --schema f1 f2  # with field validation
cheap-llm --system "X" --prompt "Y" --max-tokens 256 # bound every attempt
cheap-llm --system "X" --prompt "Y" --cloud-model deepseek/deepseek-v4-flash  # pin T2 fallback
cheap-llm --no-local --system "X" --prompt "Y" --cloud-model deepseek/deepseek-v4-flash  # cloud-only
cheap-llm --no-local --system "X" --prompt "Y" --cloud-model deepseek/deepseek-v4-flash --cloud-provider deepinfra
cheap-llm --system "X" --prompt "Y" --model my-model:latest  # explicit T1 local model
```

Env knobs: `OLLAMA_URL` (endpoint override), `CHEAP_LLM_LOCAL_ONLY=1` (never
call cloud), `CHEAP_LLM_LOCAL_MODEL`/`CHEAP_LLM_LOCAL_STRUCTURED_MODEL`
(independent T1 overrides), `CHEAP_LLM_LOCAL_COLD_TIMEOUT` (cold-VRAM T1
budget, default 25s).

## Public API contract (SemVer) — ecosystem decoupling

The surface consumers may depend on is **declared and versioned**, so this
project evolves without silently breaking fusion / web-research / the 7
`~/.claude/scripts` consumers.

- `__version__` (SemVer), `__all__` (public names), `RESULT_KEYS` (stable
  `cheap_complete()` return shape), `CHEAP_COMPLETE_PARAMS` (signature),
  `CONTRACT` (the whole shape as a dict). Everything else is `_`-private.
- `require(min_version)` — version gate. Consumers call it right after import
  to **fail fast** with an actionable message on drift, instead of a cryptic
  mid-run error. `cheap_llm.require("1.1")` → ok or `RuntimeError`.
- **Uniform return shape** — every `cheap_complete()` result has ALL
  `RESULT_KEYS` (success sets `error=None`; failure sets `provider=None`,
  `cached=False`). `_complete_result()` is the single enforcement point.
- **`tests/test_contract.py`** — the evolution gate. A breaking change (removed/
  renamed public param or RESULT_KEY) fails THERE first and forces a MAJOR bump.

**SemVer policy** (how independent evolution stays safe):
- MAJOR = removed/renamed public param or RESULT_KEY (consumers' `require()` trips)
- MINOR = additive (new param with default, new RESULT_KEY, new public fn)
- PATCH = internal refactor, cascade/model changes, bug fixes

**Consumer adoption** (opt-in, backward compatible):
```python
import cheap_llm
cheap_llm.require("1.2")          # needed when using max_output_tokens
out = cheap_llm.cheap_complete(system=..., prompt=..., max_output_tokens=256)
```
`fusion` already adopts this (`judge.py::_CHEAP_LLM_MIN_VERSION`). The 7
`~/.claude/scripts` consumers + web-research can adopt incrementally — the
contract + test prevent silent breaks meanwhile.

## Consumers (7 scripts in ~/.claude/scripts/)

| Script | What it uses cheap_llm for |
|--------|---------------------------|
| commit-draft.py | Draft Conventional Commits message from diff |
| diff-review.py | Flag issues in code diffs |
| error-classify.py | Triage errors to cause/fix |
| extract-tool-output.py | Extract relevant lines from verbose logs |
| pdf-extract-structured.py | Cloud fallback for PDF extraction |
| pr-draft.py | Draft PR descriptions |
| test-triage.py | Synthesize test failure explanations |

Also consumed by `~/web-research/` (synthesis engine cloud fallback) and
`~/fusion-local/` (judge transport via `CHEAP_LLM_HOME`).

## Synergy / Cross-CLI

Two bootstrap conventions, by design:
- **Shim path** (`~/.claude/scripts/cheap_llm.py`) — Claude-ecosystem compat.
  The 7 consumer scripts + web-research `compat.py` import the shim, which
  re-exports the real module. Resolves `ollama_client` too (same dir).
- **`CHEAP_LLM_HOME`** (`~/cheap-llm`, project root) — external consumers
  (`fusion` judge) put this on `sys.path` to import the real module directly.

The 7 signal-distillation consumers live in `~/.claude/scripts/` (Claude-only
by design — each CLI owns its own consumers). Codex / Antigravity / OpenCode
access cheap-llm via the raw `cheap-llm` console script (on the shared
`~/.local/bin` PATH) or by importing through `CHEAP_LLM_HOME` (the fusion
pattern).

## Conventions

- Multi-module package (`cheap_llm/`), vertical-slice architecture — one
  responsibility per module (contract, scrub, cache, transport, cascade, cli).
- **`cheap_llm/` stays at project root** (not `src/`) so `sys.path` bootstrap
  in shim + tests is one dir, no nested package resolution.
- **cheap_bench.py is self-contained** — has its own inline transport layer so
  the benchmark doesn't depend on the module it's benchmarking.
- **Secret scrub is unconditional** — applied even on the local (Ollama) path
  because T1 frequently times out and the same prompt then reaches cloud.
- **Callers own output budgets** — `max_output_tokens` defaults to 1024 for
  compatibility, maps to Ollama `num_predict` and cloud `max_tokens`, and is
  part of the cache key. Bounded classifiers/extractors should request less.
- **Attempts are the token ledger** — live and cached attempt records include
  the requested output ceiling plus actual input/output token counts.

## Commands

- Test (offline behavior): `python3 tests/test_cheap_llm.py`
- Test (contract + shim): `python3 -m pytest -q`
- Test (live): `python3 tests/test_cheap_llm_live.py --live`
- Bench: `python3 cheap_bench.py`
- Lint: `ruff check .`

## Model routing

- **T1 local**: `CHEAP_LLM_LOCAL_MODEL` override; default is
  `cryptidbleh/gemma4-claude-opus-4.6:latest` for free text and
  `SetneufPT/Qwopus3.5-4B-Coder-MTP` for schema/JSON via Ollama.
- **T2 cloud**: cascade order is `TOP3_CASCADE` + `LEGACY_CASCADE` constants.
- **Pinned T2 fallback**: `cheap_complete(cloud_model="deepseek/deepseek-v4-flash")`;
  add `prefer_local=False` when the call must be cloud-only.
  for judgment-heavy tasks (1M ctx, cache-aware cost).
- **Pinned provider boundary**: add `cloud_provider="deepinfra"` (or
  `openrouter`, `zenmux`, `deepseek`) to prevent cross-provider fallback and
  cache reuse. Every API provider is PAYG; subscription seats are consumed by
  `cli-orchestration`/`fusion-local`, not by this signal layer.
