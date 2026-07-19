"""Unified cheap-LLM cascade client for preprocessor slots.

Replaces ad-hoc ollama_client calls in commit-draft / diff-review / error-classify /
extract-tool-output / prompt-improve with one place that knows the cascade,
timeouts, secret-scrub, output contract, and budget.

SCOPE — a SIGNAL-DISTILLATION layer for the big model, NOT a coder. The
cascade's job is to remove noise and surface precise context (classify a
prompt, extract the relevant lines from a log, triage an error to a crisp
cause/fix, draft a commit message, flag diff issues) so the big model —
Claude Opus / Codex gpt-5.x (T3) — gets clean signal, makes the decisions,
and writes the code (higher quality than a cheap general model could). It
must NOT write/edit code, design, or do security work; its output is
advisory/distilled context, never executed code. For code-gen help to the
big model, use the SEPARATE coding tier: codex-coder (GPT-5.6 Terra) or
swarm-coding (kimi-k2 / deepseek-v4 / zm-doubao2-code — code specialists).
Filtering contract each caller enforces — SIGNAL (keep): exact errors, stack
traces, file:line, identifiers, numbers, intent, schema. NOISE (drop):
progress bars, boilerplate, repetition, verbose preamble, raw dumps larger
than the decision needs.

Cascade (T1 → T2 cloud with cross-provider failover), tried in order,
first success wins. Build by cheap_complete() from DEFAULT_LOCAL_PRIMARY +
TOP3_CASCADE + LEGACY_CASCADE (see those constants for the live order):

  T1 LOCAL (free, private)        timeout 8s/18s — cryptidbleh/gemma4-claude-opus-4.6 for text,
                                                    SetneufPT for JSON/schema
  T2 CHEAP CLOUD                  timeout 12s — TOP3_CASCADE:
      deepseek-v4-flash @ openrouter → zenmux
      gemini-3.1-flash-lite @ openrouter → zenmux ($0.25/$1.50 per M)
      ling-2.6-1t    @ openrouter → zenmux      ($0.075/$0.625 per M)
    then LEGACY_CASCADE safety net:
      gpt-5.4-nano @ openrouter, ling-2.6-flash @ openrouter → zenmux
      deepseek-v4-flash @ deepinfra only when its credential is configured

There is NO expensive "T3" fallback — every model above is cheap. If all
tiers fail or return invalid output, cheap_complete returns an empty error
envelope (caller decides what to do, typically fall back to the main model).

Per-call config:
  - timeout: T1=8s text / 18s JSON (25s cold), T2=12s; capped by timeout_total
  - JSON contract: caller passes `schema_hint: list[str]`, we validate
  - secret-scrub: ALWAYS applied before any send (local included) — see
    scrub_secrets. Secrets are redacted even on the prefer_local path,
    because T1 frequently times out and the same prompt then reaches cloud.
  - cache: sha256 of model|effective_system|prompt|schema, per-MODEL (not
    per-provider), so a ZenMux failover after an OpenRouter miss can reuse it

Usage (programmatic):
    from cheap_llm import cheap_complete
    out = cheap_complete(
        system="Classify the prompt. Reply JSON only.",
        prompt="I'm getting ECONNREFUSED...",
        schema_hint=["category", "reason"],
        timeout_total=20.0,
        max_output_tokens=256,
    )
    # out: {text, model, latency, cost, tier, attempts, json_valid, fields_ok}

Usage (CLI):
    python3 -m cheap_llm --system "You are X" --prompt "Y" \\
        --schema field1 field2
    python3 -m cheap_llm --probe   # show what's available

Decisions:
  - The current free-text T1 compatibility default is
    cryptidbleh/gemma4-claude-opus-4.6; structured calls use SetneufPT/Qwopus.
    Keep these comments aligned with DEFAULT_LOCAL_PRIMARY/STRUCTURED below.
  - deepseek-v4-flash is the primary cheap cloud; Gemini and Ling provide
    independent-family fallbacks.
  - Cloud API keys are PAYG capacity, not CLI-seat subscriptions. Subscription
    workers belong to cli-orchestration/fusion-local and stay outside this
    advisory transport.
  - gpt-5.4-nano replaced kimi-k2 in the safety net (R4: 5/5 stable, ~40%
    cheaper + faster, same quality). gpt-4.1-nano benchmarks higher but is an
    older generation → deprecation risk, kept as data only (fall-forward rule).
  - Dropped this round: kimi-k2 (superseded), gpt-5-nano + glm-4.7-flash
    (reasoning-only, content="" on short tasks), qwen3.6-flash (reasoning tax).
"""

from __future__ import annotations

# Public contract surface — matches contract.py's CONTRACT["public_api"].
# Internal symbols are re-exported below with # noqa: F401 for backward compat
# (tests, shim, and consumers access them as cheap_llm.X).
__all__ = ["cheap_complete", "scrub_secrets", "require", "__version__"]

# Re-export cache
from .cache import CACHE_DIR, CACHE_MAX_ENTRIES, _cache_get, _cache_key, _cache_put

# Re-export cascade
from .cascade import (
    _build_cascade,
    _resolve_local_model,
    _try_cache_hit,
    _try_live_hit,
    cheap_complete,
)

# CLI helpers re-exported for tests and programmatic use; main is NOT
# eagerly imported to avoid `-m cheap_llm.cli` conflicts.
from .cli import _cache_clear, _cache_stats, _probe, _probe_url, _route_plan  # noqa: F401
from .contract import (
    _RESULT_DEFAULTS,
    CHEAP_COMPLETE_PARAMS,
    CONTRACT,
    RESULT_KEYS,
    __version__,
    _complete_result,
    _parse_version,
    require,
)

# Re-export scrub
from .scrub import SECRET_PATTERNS, scrub_secrets

# Re-export transport (constants, providers, call functions)
from .transport import (
    _PROVIDER_DISPATCH,
    _PROVIDERS,
    DEEPINFRA_ENDPOINT,
    DEEPINFRA_PRICING,
    DEEPINFRA_URL,
    DEEPSEEK_PRICING,
    DEEPSEEK_URL,
    DEFAULT_LOCAL_PRIMARY,
    DEFAULT_LOCAL_STRUCTURED,
    LEGACY_CASCADE,
    LOCAL_COLD_TIMEOUT,
    LOCAL_KEEP_ALIVE,
    LOCAL_WARM_TIMEOUT_PRIMARY,
    LOCAL_WARM_TIMEOUT_STRUCTURED,
    MAX_RESPONSE_BYTES,
    MODEL_PRICING,
    OLLAMA_URL,
    OPENROUTER_ENDPOINT,
    OPENROUTER_URL,
    REASONING_EFFORT_OVERRIDES,
    TOP3_CASCADE,
    ZENMUX_DEFAULT_MULTIPLIER,
    ZENMUX_ENDPOINT,
    ZENMUX_MODEL_MULTIPLIERS,
    ZENMUX_MODEL_PRICING,
    ZENMUX_URL,
    _call_deepinfra,
    _call_deepseek,
    _call_ollama,
    _call_openrouter,
    _call_provider,
    _call_zenmux,
    _Endpoint,
    _normalize_deepinfra_model,
    _normalize_model_name,
    _ollama_model_loaded,
    _openai_compat_call,
    _provider_billing,
    _ProviderSpec,
    _public_attempt_error,
    _read_json_response,
    _resolve_cost,
    _strip_reasoning,
)

# Re-export validation (extracted from cascade for cohesion)
from .validation import JSON_HINT, _try_parse_json, _validate
