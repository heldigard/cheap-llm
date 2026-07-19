# vs-soft-allow — cheap_complete 9 params = documented public API contract (CHEAP_COMPLETE_PARAMS);
# internal helpers (_try_cache_hit, _try_live_hit) pass through cascade context from cheap_complete.
"""Main cascade — build cascade, resolve models, try hits, cheap_complete.

Orchestrates T1 local → T2 cloud with cross-provider failover. The public
entry point is cheap_complete().
"""

from __future__ import annotations

import math
import os
import time
from numbers import Real

from .cache import _cache_get, _cache_key, _cache_put
from .contract import _complete_result
from .scrub import scrub_secrets
from .transport import (
    _PROVIDERS,
    DEFAULT_LOCAL_PRIMARY,
    DEFAULT_LOCAL_STRUCTURED,
    LEGACY_CASCADE,
    LOCAL_COLD_TIMEOUT,
    LOCAL_WARM_TIMEOUT_PRIMARY,
    LOCAL_WARM_TIMEOUT_STRUCTURED,
    TOP3_CASCADE,
    _provider_billing,
    _public_attempt_error,
)
from .validation import JSON_HINT, _try_parse_json, _validate


def _structured_local_model() -> str:
    """Resolved structured T1 model (env override or package default)."""
    return os.environ.get("CHEAP_LLM_LOCAL_STRUCTURED_MODEL") or DEFAULT_LOCAL_STRUCTURED


def _local_timeout(model: str) -> float:
    """Warm/cold T1 budget for *model*. Structured JSON needs more headroom."""
    # Import at call time so test mocks on cl._ollama_model_loaded take effect.
    import cheap_llm as _pkg

    warm = (
        LOCAL_WARM_TIMEOUT_STRUCTURED
        if model == _structured_local_model()
        else LOCAL_WARM_TIMEOUT_PRIMARY
    )
    if not _pkg._ollama_model_loaded(model):
        return max(warm, LOCAL_COLD_TIMEOUT)
    return warm


def _build_cascade(
    prefer_local: bool,
    local_model: str | None,
    cloud_model: str | None,
    cloud_provider: str | None = None,
) -> list[tuple[str, str, str, float]]:
    """Build the ordered (tier, model, provider, timeout) cascade."""
    cascade: list[tuple[str, str, str, float]] = []
    if prefer_local:
        resolved = local_model or DEFAULT_LOCAL_PRIMARY
        cascade.append(("T1", resolved, "ollama", _local_timeout(resolved)))
    local_only = os.environ.get("CHEAP_LLM_LOCAL_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if local_only:
        if not cascade:
            resolved = local_model or DEFAULT_LOCAL_PRIMARY
            cascade.append(("T1", resolved, "ollama", _local_timeout(resolved)))
        return cascade

    if cloud_provider:
        if not cloud_model:
            raise ValueError("cloud_provider requires cloud_model")
        if cloud_provider not in _PROVIDERS:
            raise ValueError(f"unknown cloud_provider: {cloud_provider}")
        if cloud_provider == "ollama":
            raise ValueError("cloud_provider is a PAYG boundary; use prefer_local for ollama")
        if cloud_provider == "deepseek" and not cloud_model.startswith("deepseek/"):
            raise ValueError("the deepseek provider requires a deepseek/* model")
        cascade.append(("T2", cloud_model, cloud_provider, 18.0))
        return cascade

    has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))
    if cloud_model:
        if cloud_model.startswith("deepseek/"):
            cascade.append(("T2", cloud_model, "deepseek", 18.0))
            if has_deepinfra:
                cascade.append(("T2", cloud_model, "deepinfra", 18.0))
        elif has_deepinfra and any(
            brand in cloud_model.lower() for brand in ("qwen", "glm", "mimo", "kimi")
        ):
            cascade.append(("T2", cloud_model, "deepinfra", 18.0))
        cascade.append(("T2", cloud_model, "openrouter", 18.0))
        cascade.append(("T2", cloud_model, "zenmux", 18.0))
        return cascade
    has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if has_deepseek:
        # 2026-07-17: first-party api.deepseek.com beats resellers — no
        # OpenRouter markup and a much deeper prompt-cache discount. Same
        # credential-gating rule as deepinfra below: never advertise a route
        # that cannot authenticate in the automatic cascade.
        cascade.append(("T2", "deepseek/deepseek-v4-flash", "deepseek", 12.0))
    for m, p in TOP3_CASCADE:
        cascade.append(("T2", m, p, 12.0))
    for m, p in LEGACY_CASCADE:
        cascade.append(("T2", m, p, 12.0))
    if has_deepinfra:
        # Optional distinct billing pool. Do not advertise or attempt a route
        # that cannot authenticate in the automatic cascade.
        cascade.append(("T2", "deepseek/deepseek-v4-flash", "deepinfra", 12.0))
    return cascade


def _resolve_local_model(
    local_model: str | None, require_json: bool, schema_t: tuple[str, ...]
) -> str | None:
    """Pick a local model by output contract while preserving explicit overrides."""
    if local_model:
        return local_model
    if require_json and schema_t:
        return _structured_local_model()
    return os.environ.get("CHEAP_LLM_LOCAL_MODEL") or DEFAULT_LOCAL_PRIMARY


def _try_cache_hit(
    ckey: str,
    tier: str,
    model: str,
    provider: str,
    schema_t: tuple[str, ...],
    require_json: bool,
    max_output_tokens: int,
    attempts: list[dict],
) -> dict | None:
    """Return a cache-hit success envelope, or None on miss / invalid cached value."""
    cached = _cache_get(ckey)
    if not cached:
        return None
    source_provider = cached.get("provider")
    if not isinstance(source_provider, str) or not source_provider:
        source_provider = provider
    source_tier = cached.get("tier")
    if not isinstance(source_tier, str) or not source_tier:
        source_tier = tier
    source_billing = cached.get("billing")
    if source_billing not in {"local", "payg"}:
        source_billing = _provider_billing(source_provider)
    attempt = {
        "tier": source_tier,
        "model": model,
        "provider": source_provider,
        "billing": source_billing,
        "cache_hit": True,
        "latency": 0,
        "cost": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "max_output_tokens": max_output_tokens,
    }
    if source_provider != provider:
        attempt["cache_lookup_provider"] = provider
    if source_tier != tier:
        attempt["cache_lookup_tier"] = tier
    attempts.append(attempt)
    text = cached["text"]
    parsed = _try_parse_json(text) if require_json else None
    ok = _validate(parsed, schema_t) if require_json else True
    if not ok:
        return None
    return {
        "text": text,
        "model": model,
        "provider": source_provider,
        "billing": source_billing,
        "tier": source_tier,
        "latency": 0,
        "cost": 0,
        "json_valid": parsed is not None,
        "fields_ok": _validate(parsed, schema_t),
        "attempts": attempts,
        "cached": True,
    }


def _try_live_hit(
    raw: dict,
    tier: str,
    model: str,
    provider: str,
    ckey: str,
    schema_t: tuple[str, ...],
    require_json: bool,
    max_output_tokens: int,
    attempts: list[dict],
) -> dict | None:
    """Build the live success envelope + cache, or return None if validation fails."""
    text = raw["text"]
    cost = raw.get("api_cost") or 0.0
    billing = raw.get("billing") or _provider_billing(provider)
    parsed = _try_parse_json(text) if require_json else None
    ok = _validate(parsed, schema_t) if require_json else bool(text)
    attempts.append(
        {
            "tier": tier,
            "model": model,
            "provider": provider,
            "billing": billing,
            "latency": round(raw["latency"], 3),
            "cost": cost,
            "json_valid": parsed is not None,
            "input_tokens": raw.get("input_tokens", 0) or 0,
            "output_tokens": raw.get("output_tokens", 0) or 0,
            "max_output_tokens": max_output_tokens,
        }
    )
    if not ok:
        return None
    source_provider = raw.get("provider") or provider
    _cache_put(
        ckey,
        {"text": text, "provider": source_provider, "tier": tier, "billing": billing},
    )
    return {
        "text": text,
        "model": model,
        "provider": source_provider,
        "billing": billing,
        "tier": tier,
        "latency": raw["latency"],
        "cost": cost,
        "json_valid": parsed is not None,
        "fields_ok": _validate(parsed, schema_t),
        "attempts": attempts,
        "cached": False,
    }


def cheap_complete(
    system: str,
    prompt: str,
    schema_hint: list[str] | None = None,
    timeout_total: float = 20.0,
    prefer_local: bool = True,
    require_json: bool = True,
    model: str | None = None,
    cloud_model: str | None = None,
    max_output_tokens: int = 1024,
    cloud_provider: str | None = None,
) -> dict:
    """Try T1 local, then T2 cloud, return the first good result.

    Returns dict with: text, model, provider, billing, tier, latency, cost, json_valid,
    fields_ok, attempts, error.
    """
    # Import at call time so test mocks on cheap_llm._call_provider take effect
    # (direct import binds at module load, before mocks are applied).
    import cheap_llm as _pkg

    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")
    if (
        isinstance(timeout_total, bool)
        or not isinstance(timeout_total, Real)
        or not math.isfinite(timeout_total)
        or timeout_total <= 0
    ):
        raise ValueError("timeout_total must be a positive finite number")
    if cloud_provider is not None:
        if not isinstance(cloud_provider, str) or cloud_provider not in _PROVIDERS:
            raise ValueError(f"cloud_provider must be one of: {', '.join(sorted(_PROVIDERS))}")
        if cloud_provider == "ollama":
            raise ValueError("cloud_provider is a PAYG boundary; use prefer_local for ollama")
        if not cloud_model:
            raise ValueError("cloud_provider requires cloud_model")
        if cloud_provider == "deepseek" and not cloud_model.startswith("deepseek/"):
            raise ValueError("the deepseek provider requires a deepseek/* model")

    schema_t = tuple(schema_hint) if schema_hint else ()
    scrubbed_system = scrub_secrets(system)
    scrubbed_prompt = scrub_secrets(prompt)

    eff_system = scrubbed_system
    if require_json:
        eff_system = scrubbed_system + JSON_HINT
        if schema_t:
            eff_system += f" Required keys: {list(schema_t)}."

    local_model = _resolve_local_model(model, require_json, schema_t)
    cascade = _build_cascade(prefer_local, local_model, cloud_model, cloud_provider)
    attempts: list[dict] = []
    deadline = time.perf_counter() + float(timeout_total)

    for tier, mdl, provider, per_timeout in cascade:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        eff_timeout = min(per_timeout, remaining)

        # An explicit provider is a billing/trust boundary. Do not satisfy that
        # request from another provider's model-level cache entry.
        cache_model = f"{mdl}@{provider}" if cloud_provider and tier == "T2" else mdl
        ckey = _cache_key(cache_model, eff_system, scrubbed_prompt, schema_t, max_output_tokens)
        hit = _try_cache_hit(
            ckey,
            tier,
            mdl,
            provider,
            schema_t,
            require_json,
            max_output_tokens,
            attempts,
        )
        if hit is not None:
            return _complete_result(hit)

        try:
            raw = _pkg._call_provider(
                mdl,
                provider,
                eff_system,
                scrubbed_prompt,
                eff_timeout,
                max_output_tokens,
                require_json,
            )
        except Exception as e:
            attempts.append(
                {
                    "tier": tier,
                    "model": mdl,
                    "provider": provider,
                    "max_output_tokens": max_output_tokens,
                    "error": _public_attempt_error(e),
                }
            )
            continue

        env = _try_live_hit(
            raw,
            tier,
            mdl,
            provider,
            ckey,
            schema_t,
            require_json,
            max_output_tokens,
            attempts,
        )
        if env is not None:
            return _complete_result(env)

    return _complete_result(
        {
            "text": "",
            "model": None,
            "tier": None,
            "latency": 0,
            "cost": 0,
            "json_valid": False,
            "fields_ok": False,
            "attempts": attempts,
            "error": "all tiers failed or returned invalid output",
        }
    )
