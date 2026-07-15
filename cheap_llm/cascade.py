# vs-soft-allow — cheap_complete 9 params = documented public API contract (CHEAP_COMPLETE_PARAMS);
# internal helpers (_try_cache_hit, _try_live_hit) pass through cascade context from cheap_complete.
"""Main cascade — build cascade, resolve models, try hits, cheap_complete.

Orchestrates T1 local → T2 cloud with cross-provider failover. The public
entry point is cheap_complete().
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from numbers import Real

from .cache import _cache_get, _cache_key, _cache_put
from .contract import _complete_result
from .scrub import scrub_secrets
from .transport import (
    DEFAULT_LOCAL_PRIMARY,
    DEFAULT_LOCAL_STRUCTURED,
    LEGACY_CASCADE,
    LOCAL_COLD_TIMEOUT,
    TOP3_CASCADE,
    _public_attempt_error,
)

# JSON contract hint appended to system prompt when require_json=True.
JSON_HINT = (
    "\n\nReply with JSON only — no prose, no code fences, no explanation. "
    "The first character must be `{` and the last must be `}`."
)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        try:
            result = json.loads(re.sub(r",(\s*[}\]])", r"\1", text))
        except json.JSONDecodeError:
            return None
    return result if isinstance(result, dict) else None


def _validate(parsed: dict | None, schema: tuple[str, ...] | None) -> bool:
    """Validate required JSON fields without rejecting valid empty containers."""
    if parsed is None:
        return False
    if not schema:
        return True
    for name in schema:
        if name not in parsed:
            return False
        value = parsed[name]
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def _build_cascade(
    prefer_local: bool,
    local_model: str | None,
    cloud_model: str | None,
) -> list[tuple[str, str, str, float]]:
    """Build the ordered (tier, model, provider, timeout) cascade."""
    # Import at call time so test mocks on cl._ollama_model_loaded take effect.
    import cheap_llm as _pkg

    cascade: list[tuple[str, str, str, float]] = []
    if prefer_local:
        resolved = local_model or DEFAULT_LOCAL_PRIMARY
        local_timeout = 12.0 if resolved == DEFAULT_LOCAL_STRUCTURED else 6.0
        if not _pkg._ollama_model_loaded(resolved):
            local_timeout = max(local_timeout, LOCAL_COLD_TIMEOUT)
        cascade.append(("T1", resolved, "ollama", local_timeout))
    local_only = os.environ.get("CHEAP_LLM_LOCAL_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if local_only:
        if not cascade:
            resolved = local_model or DEFAULT_LOCAL_PRIMARY
            local_timeout = 12.0 if resolved == DEFAULT_LOCAL_STRUCTURED else 6.0
            if not _pkg._ollama_model_loaded(resolved):
                local_timeout = max(local_timeout, LOCAL_COLD_TIMEOUT)
            cascade.append(("T1", resolved, "ollama", local_timeout))
        return cascade

    if cloud_model:
        has_deepinfra = bool(os.environ.get("DEEPINFRA_API_KEY"))
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
    for m, p in TOP3_CASCADE:
        cascade.append(("T2", m, p, 12.0))
    for m, p in LEGACY_CASCADE:
        cascade.append(("T2", m, p, 12.0))
    return cascade


def _resolve_local_model(
    local_model: str | None, require_json: bool, schema_t: tuple[str, ...]
) -> str | None:
    """Pick a local model by output contract while preserving explicit overrides."""
    if local_model:
        return local_model
    if require_json and schema_t:
        return DEFAULT_LOCAL_STRUCTURED
    return DEFAULT_LOCAL_PRIMARY


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
    attempt = {
        "tier": source_tier,
        "model": model,
        "provider": source_provider,
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
    parsed = _try_parse_json(text) if require_json else None
    ok = _validate(parsed, schema_t) if require_json else bool(text)
    attempts.append(
        {
            "tier": tier,
            "model": model,
            "provider": provider,
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
    _cache_put(ckey, {"text": text, "provider": source_provider, "tier": tier})
    return {
        "text": text,
        "model": model,
        "provider": source_provider,
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
) -> dict:
    """Try T1 local, then T2 cloud, return the first good result.

    Returns dict with: text, model, tier, latency, cost, json_valid,
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

    schema_t = tuple(schema_hint) if schema_hint else ()
    scrubbed_system = scrub_secrets(system)
    scrubbed_prompt = scrub_secrets(prompt)

    eff_system = scrubbed_system
    if require_json:
        eff_system = scrubbed_system + JSON_HINT
        if schema_t:
            eff_system += f" Required keys: {list(schema_t)}."

    local_model = _resolve_local_model(model, require_json, schema_t)
    cascade = _build_cascade(prefer_local, local_model, cloud_model)
    attempts: list[dict] = []
    deadline = time.perf_counter() + float(timeout_total)

    for tier, mdl, provider, per_timeout in cascade:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        eff_timeout = min(per_timeout, remaining)

        ckey = _cache_key(mdl, eff_system, scrubbed_prompt, schema_t, max_output_tokens)
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
