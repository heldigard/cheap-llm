---
title: "fix: Strengthen contract and cache reliability"
type: fix
status: completed
date: 2026-07-12
---

# Fix: Strengthen contract and cache reliability

## Enhancement summary

**Deepened on:** 2026-07-12  
**Evidence used:** repository architecture, current CI, 144-check offline baseline
(expanded to 157 checks), git history/blame, Ruff, Pyright, Semgrep, Vulture, and Gitleaks. No external
research is needed because the implementation uses stable Python standard
library primitives and established project patterns.

### Key refinements

1. Preserve the cache namespace and legacy payload compatibility; provenance
   is additive metadata, not a key migration.
2. Use a unique temporary file created inside `CACHE_DIR` so `os.replace` stays
   atomic on the same filesystem. Rely on the creation primitive's `0o600`
   mode and harden the directory to `0o700` best-effort.
3. Validate deadlines with `numbers.Real` plus `math.isfinite`, explicitly
   excluding booleans, and run that validation before `_build_cascade` because
   cascade construction may perform an Ollama `/api/ps` probe.
4. Keep contract failures as `AssertionError` subclasses so pytest recognizes
   them, while teaching the standalone runner not to double-count failures
   already recorded by `check()`.

### Sensor interpretation

- Gitleaks, Ruff, Pyright, and architecture checks are clean (architecture is
  not applicable to a two-module Python project).
- Vulture's single item is the intentionally generic `*a` argument in a test
  lambda; rename it as part of touched test cleanup.
- Semgrep reports four dynamic-urllib warnings. All URLs are either static
  provider constants or the operator-controlled `OLLAMA_URL`; the code already
  documents these trust boundaries. Review suppression placement after edits,
  but do not add a runtime dependency merely to silence the sensor.

## Overview

Harden the small set of guarantees that make `cheap-llm` safe to consume across
multiple CLIs: the SemVer contract gate must actually fail in pytest/CI, cached
responses must retain their true provider provenance, concurrent cache writes
must remain atomic, and invalid public inputs must fail before any cache or
transport work. Align the benchmark's local candidate with the production T1
default while keeping the single-module architecture and zero runtime
dependencies.

## Evidence and motivation

- `.github/workflows/ci.yml` runs `pytest` as the contract gate, but
  `tests/test_contract.py:45` records failed checks without raising. Pytest can
  report green even when a declared API invariant is false.
- `cheap_llm.py:967` caches only `{\"text\": ...}` although the key is shared by
  model across providers. `cheap_llm.py:925` therefore reports the current
  lookup provider rather than the provider that produced the cached response.
- `cheap_llm.py:467` uses one deterministic `.json.tmp` path, so concurrent
  writers for the same key can interfere. Cache data is derived from developer
  prompts and should be private by default.
- `cheap_llm.py:982` validates output budgets but not `timeout_total`; zero,
  negative, boolean, or non-finite values silently degrade into an all-tiers
  failure instead of reporting bad input.
- `cheap_bench.py:66` still benchmarks `gemma4:12b`; production uses
  `DEFAULT_LOCAL_PRIMARY = cryptidbleh/gemma4-claude-opus-4.6:latest` in
  `cheap_llm.py:197`.

## Proposed solution

1. Make contract checks raise a dedicated assertion compatible with both
   pytest and the file's standalone aggregated runner.
2. Store provider provenance in new cache entries. Read legacy text-only cache
   entries without breaking the existing namespace, but report stored
   provenance when available.
3. Replace the shared temporary cache path with a unique same-directory
   temporary file (for example `tempfile.mkstemp`) and atomic `os.replace`; use
   private directory/file modes and always clean up temporary residue in a
   `finally` block. Pruning remains best-effort and must tolerate another
   process removing a candidate between listing and `stat`/`unlink`.
4. Validate `timeout_total` as a positive finite real number before resolving
   models, probing Ollama, touching cache, or invoking providers. Mirror the
   validation in the CLI error path.
5. Update the benchmark's local model and explanatory comments to the current
   production default.

## System-wide impact

- **Interaction graph:** `cheap_complete` validates inputs, builds the cascade,
  checks `_cache_get`, and either returns `_try_cache_hit` or calls a transport;
  successful live responses flow through `_try_live_hit` into `_cache_put`.
- **Error propagation:** invalid caller input raises `ValueError`; transport
  failures remain attempt-ledger entries; cache failures remain advisory and
  must never turn a successful provider call into failure.
- **State lifecycle:** cache writes remain same-filesystem atomic. Unique temp
  files prevent writer collisions and cleanup prevents orphan residue.
- **API parity:** no public name, parameter, or result key changes. This is a
  SemVer patch; cache payload additions are private and backward compatible.
- **Consumers:** the seven script consumers, `web-research`, and `fusion-local`
  keep the same call signature. They gain accurate cache telemetry and clearer
  errors for invalid deadlines.

## Edge cases and SpecFlow analysis

- A legacy cache entry with only `text` must still hit and may use the current
  lookup provider as its best available attribution.
- A new cache entry produced by one provider and looked up first through
  another provider must report the source provider. The attempt ledger should
  retain both facts when they differ: `provider` as source attribution and an
  additive `cache_lookup_provider` for the cascade slot that found the entry.
- Corrupt cache shapes remain misses.
- Two writers targeting the same key must leave a valid final JSON file; losing
  one equivalent write is acceptable, leaking a temp file or partial JSON is
  not.
- Permission hardening must be best effort so unusual filesystems do not break
  completion.
- Existing cache directories may already be permissive; a successful write
  should repair their mode when supported. A chmod failure remains advisory.
- `timeout_total=True`, zero, negative, `NaN`, and infinities must be rejected;
  positive integers/floats remain accepted.
- The contract test's standalone runner must count each failed check once,
  while pytest must receive a real `AssertionError`.
- Live/provider tests remain opt-in; all new regression coverage must be fully
  offline and deterministic.

## Implementation phases

### Phase 1: Regression tests

- Add focused behavioral checks for deadline validation, provider provenance,
  secure cache mode, absence of unique temp residue, and legacy cache payloads.
- Add a contract-harness regression proving a false check raises under pytest
  semantics without polluting the global counters. Keep it outside the
  standalone `TESTS` list so the script does not intentionally fail itself.

### Phase 2: Core fixes

- Update contract check failure behavior and standalone exception handling.
- Harden `_cache_put`, enrich its payload at `_try_live_hit`, and consume source
  provider metadata in `_try_cache_hit`.
- Add shared deadline validation and make the CLI delegate to the same rule or
  issue an equivalent parser error before transport.
- Align `cheap_bench.py` with the live T1 default.

### Phase 3: Documentation and validation

- Document cache provenance/privacy and deadline requirements in `README.md`.
- Run `uv run pytest -q`, the 157-check offline behavioral suite, Ruff, type and
  security/dead-code sensors, and inspect the complete diff.

## Acceptance criteria

- [x] Deliberately false contract checks raise and fail pytest.
- [x] Existing contract and shim tests pass.
- [x] New cache entries preserve source-provider attribution across a
      cross-provider lookup; legacy entries remain readable.
- [x] Cache writes are atomic under concurrent writers, private where POSIX
      modes apply, and leave no temp residue.
- [x] Invalid total timeouts fail before provider/cache work with an actionable
      message; valid deadlines preserve behavior.
- [x] The benchmark local candidate matches the production T1 model.
- [x] Offline behavior, lint, types, secret scan, and SAST gates pass with no
      unresolved relevant findings.

## Post-deploy monitoring and validation

No additional production monitoring is required: this is a local library/CLI
with no server deployment. After editable installation, validate `cheap-llm
--version` reports `1.2.2` and run `cheap-llm --probe`. Healthy behavior is a
valid probe envelope and private cache modes after the first successful write.
Failure signals are contract-test failures, malformed cache JSON, temp-file
residue, or provider attribution drift; mitigation is to reinstall the prior
package revision and remove only the advisory cache directory. Validation
window: the first local completion after upgrade; owner: local operator.

## Risks and mitigations

- **Legacy telemetry ambiguity:** old cache entries lack provenance. Preserve
  compatibility and fall back to the lookup provider only for those entries.
- **Platform permission differences:** perform chmod as best effort inside the
  already non-fatal cache boundary.
- **Concurrent pruning races:** ignore a vanished file and continue pruning;
  never widen the cache failure boundary into the completion path.
- **Test harness global state:** isolate the dedicated failure-semantics check
  or validate it through subprocess so normal contract counters remain clean.
- **Scope creep:** retain the root module layout, cascade ordering, model
  pricing, public API, and live behavior.

## Sources

- `CLAUDE.md` — architecture, public contract, and cache conventions.
- `.github/workflows/ci.yml` — current quality gate sequence.
- `cheap_llm.py` — public API, cache, transport, and cascade implementation.
- `cheap_bench.py` — self-contained benchmark candidate configuration.
- `tests/test_contract.py` and `tests/test_cheap_llm.py` — existing contract and
  offline regression patterns.
