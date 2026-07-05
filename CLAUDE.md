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
cheap_llm.py          core module — cascade, transport, scrub, cache, CLI
cheap_bench.py        benchmark harness (self-contained transport layer)
tests/
  test_cheap_llm.py       unit + mocked tests (86 tests, offline, no network)
  test_cheap_llm_live.py  live + E2E tests (real API calls, opt-in via --live)
```

## Cascade

```
T1 LOCAL  (free, private)    6s  — qwen3.5:4b (Ollama)
T2 CHEAP CLOUD              12s  — ling-2.6-flash → ling-2.6-1t → gemini-3.1-flash-lite
                                    (OpenRouter primary, ZenMux failover)
    LEGACY safety net             — gpt-5.4-nano, deepseek-v4-flash (BYOK $0)
```

## Entry points

- **Wired ecosystem shim**: `~/.claude/scripts/cheap_llm.py` → imports from
  here (env `CHEAP_LLM_HOME`, default `~/cheap-llm`). The 8 consumer scripts
  that do `import cheap_llm` keep resolving untouched.
- **Console script** (`pip install -e .`): `cheap-llm`.
- **Direct**: `python3 cheap_llm.py --probe`.

## CLI

```
cheap-llm --probe                              # show what's available
cheap-llm --system "X" --prompt "Y"            # run cascade, print text
cheap-llm --system "X" --prompt "Y" --json     # full JSON envelope
cheap-llm --system "X" --prompt "Y" --schema f1 f2  # with field validation
```

## Consumers (8 scripts in ~/.claude/scripts/)

| Script | What it uses cheap_llm for |
|--------|---------------------------|
| commit-draft.py | Draft Conventional Commits message from diff |
| diff-review.py | Flag issues in code diffs |
| error-classify.py | Triage errors to cause/fix |
| extract-tool-output.py | Extract relevant lines from verbose logs |
| intent_route.py | Classify developer prompt intent + tier |
| pdf-extract-structured.py | Cloud fallback for PDF extraction |
| pr-draft.py | Draft PR descriptions |
| test-triage.py | Synthesize test failure explanations |

Also consumed by `~/web-research/` (synthesis engine cloud fallback).

## Conventions

- Single-module package (`cheap_llm.py`), not a `src/` layout — the module IS
  the product, no internal splits needed.
- **cheap_llm.py stays at project root** (not `src/`) so `sys.path` bootstrap
  in shim + tests is one dir, no nested package resolution.
- **cheap_bench.py is self-contained** — has its own inline transport layer so
  the benchmark doesn't depend on the module it's benchmarking.
- **Secret scrub is unconditional** — applied even on the local (Ollama) path
  because T1 frequently times out and the same prompt then reaches cloud.

## Commands

- Test (offline): `python3 tests/test_cheap_llm.py`
- Test (live): `python3 tests/test_cheap_llm_live.py --live`
- Bench: `python3 cheap_bench.py`
- Lint: `ruff check .`

## Model routing

- **T1 local**: `CHEAP_LLM_LOCAL_MODEL` (default `qwen3.5:4b`) via Ollama.
- **T2 cloud**: cascade order is `TOP3_CASCADE` + `LEGACY_CASCADE` constants.
- **Forced cloud model**: `cheap_complete(cloud_model="deepseek/deepseek-v4-flash")`
  for judgment-heavy tasks (1M ctx, cache-aware cost).
