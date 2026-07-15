# vs-soft-allow — transport dispatch functions need
# model+provider+system+prompt+timeout+max_output_tokens
"""Transport layer — providers, endpoints, and API call functions.

Constants, provider registry, OpenAI-compatible call helper, and per-provider
call functions. Adding a new provider = one new _PROVIDERS entry + one _call_*
function + one _PROVIDER_DISPATCH entry.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

# _strip_reasoning is defined below in this module

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Local T1 defaults. cryptidbleh/gemma4-claude-opus-4.6 is the free-text compatibility default and
# matches ollama_client.DEFAULT_GEN_MODEL. JSON/schema calls use the measured
# structured-output specialist unless callers pass an explicit `model=...`.
DEFAULT_LOCAL_PRIMARY = "cryptidbleh/gemma4-claude-opus-4.6:latest"
DEFAULT_LOCAL_STRUCTURED = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"

# T1 budget when the local model is NOT loaded in VRAM yet (cold start).
# Warm budgets stay 6s/12s; eff_timeout always clamps to the caller's
# timeout_total, so callers with tight deadlines are unaffected.
LOCAL_COLD_TIMEOUT = float(os.environ.get("CHEAP_LLM_LOCAL_COLD_TIMEOUT", "25"))

# External responses are untrusted input. Token ceilings constrain model
# generation but do not constrain a broken/proxied HTTP response, so every
# transport also enforces a byte ceiling before decoding or JSON parsing.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Reasoning control: empty (all cascade models are non-reasoning or
# already have reasoning disabled by default).
REASONING_EFFORT_OVERRIDES: dict[str, str] = {}

# Public listing price per 1M tokens (input, output) in USD — used to
# estimate cost when a provider returns usage.cost=None (ZenMux always;
# OpenRouter for some promo/preview models). Source: OpenRouter catalog +
# tested-models.md. ZenMux real price is ~4-10x higher for the same model,
# so this is a conservative LOWER-bound for ZenMux calls.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "inclusionai/ling-2.6-flash": (0.01, 0.03),
    "inclusionai/ling-2.6-1t": (0.075, 0.625),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "moonshotai/kimi-k2": (0.57, 2.30),  # kept for cost lookup only
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
}

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ZENMUX_URL = "https://zenmux.ai/api/v1"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Per-model ZenMux pricing multipliers (verified 2026-06-19 against
# zenmux.ai/pricing). Default 5x for any model not listed; ling-2.6-flash is
# 10x OR, ling-2.6-1t is 4x OR.
ZENMUX_MODEL_MULTIPLIERS: dict[str, float] = {
    "inclusionai/ling-2.6-flash": 10.0,
    "inclusionai/ling-2.6-1t": 4.0,
}
ZENMUX_DEFAULT_MULTIPLIER = 5.0

# Cascade as (model, provider) pairs. For each top model we try OpenRouter
# first, then ZenMux as backup.
TOP3_CASCADE: list[tuple[str, str]] = [
    ("inclusionai/ling-2.6-flash", "openrouter"),
    ("inclusionai/ling-2.6-flash", "zenmux"),
    ("inclusionai/ling-2.6-1t", "openrouter"),
    ("inclusionai/ling-2.6-1t", "zenmux"),
    ("google/gemini-3.1-flash-lite", "openrouter"),
    ("google/gemini-3.1-flash-lite", "zenmux"),
]

LEGACY_CASCADE: list[tuple[str, str]] = [
    ("openai/gpt-5.4-nano", "openrouter"),
    ("deepseek/deepseek-v4-flash", "openrouter"),
]


# ---------------------------------------------------------------------------
# Response reading
# ---------------------------------------------------------------------------


class _ReadableResponse(Protocol):
    def read(self, amount: int = -1) -> bytes: ...


def _read_json_response(response: _ReadableResponse) -> dict:
    """Read one bounded UTF-8 JSON object from an HTTP response."""
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("provider response exceeds 4 MiB limit")
    body = json.loads(raw.decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("provider response must be a JSON object")
    return body


def _public_attempt_error(exc: Exception) -> str:
    """Return bounded provider diagnostics without URLs, bodies, or prompts."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError: HTTP {exc.code}"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return f"{type(exc).__name__}: request failed"
    if isinstance(exc, RuntimeError) and str(exc).endswith(" not set"):
        return f"RuntimeError: {str(exc)[:260]}"
    return f"{type(exc).__name__}: provider attempt failed"


def _normalize_model_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip()
    if ":" not in name:
        name = f"{name}:latest"
    return name


def _ollama_model_loaded(model: str) -> bool:
    """True if `model` is currently loaded (GET /api/ps). Unknown -> assume warm."""
    norm_target = _normalize_model_name(model)
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = _read_json_response(resp)
    except Exception:
        return True
    models = data.get("models") or []
    for m in models:
        if not isinstance(m, dict):
            continue
        for key in ("name", "model"):
            val = m.get(key)
            if val and _normalize_model_name(val) == norm_target:
                return True
    return False


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Endpoint:
    """OpenAI-compatible chat-completions endpoint config."""

    url: str
    key_env: str
    provider_label: str
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _ProviderSpec:
    """A unified provider spec — extends _Endpoint with slug_map and probe URL."""

    endpoint: _Endpoint
    slug_map: dict[str, str] = field(default_factory=dict)
    probe_url: str | None = None


_PROVIDERS: dict[str, _ProviderSpec] = {
    "openrouter": _ProviderSpec(
        endpoint=_Endpoint(
            url=OPENROUTER_URL,
            key_env="OPENROUTER_API_KEY",
            provider_label="openrouter",
            extra_headers={"X-Title": "cheap-llm-cascade"},
        ),
        probe_url=f"{OPENROUTER_URL}/models",
    ),
    "zenmux": _ProviderSpec(
        endpoint=_Endpoint(
            url=ZENMUX_URL,
            key_env="ZENMUX_API_KEY",
            provider_label="zenmux",
        ),
        probe_url=f"{ZENMUX_URL}/models",
    ),
    "deepinfra": _ProviderSpec(
        endpoint=_Endpoint(
            url=DEEPINFRA_URL,
            key_env="DEEPINFRA_API_KEY",
            provider_label="deepinfra",
        ),
        slug_map={
            "deepseek/deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek/deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
            "qwen3.7-max": "Qwen/Qwen3.7-Max",
            "glm-5.2": "zai-org/GLM-5.2",
            "mimo-v2.5-pro": "XiaomiMiMo/MiMo-V2.5-Pro",
            "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
        },
    ),
    "deepseek": _ProviderSpec(
        endpoint=_Endpoint(
            url=DEEPSEEK_URL,
            key_env="DEEPSEEK_API_KEY",
            provider_label="deepseek",
        ),
        probe_url=f"{DEEPSEEK_URL}/models",
    ),
}


def _provider_spec(name: str) -> _ProviderSpec:
    spec = _PROVIDERS.get(name)
    if spec is None:
        raise ValueError(f"unknown provider: {name}")
    return spec


def _normalize_deepinfra_model(model: str) -> str:
    """Map generic/OpenRouter model slugs to DeepInfra-specific slugs."""
    low = model.lower()
    for needle, slug in _PROVIDERS["deepinfra"].slug_map.items():
        if needle in low:
            return slug
    return model


# Derived endpoints (single source of truth = _PROVIDERS)
OPENROUTER_ENDPOINT = _provider_spec("openrouter").endpoint
ZENMUX_ENDPOINT = _provider_spec("zenmux").endpoint
DEEPINFRA_ENDPOINT = _provider_spec("deepinfra").endpoint


# ---------------------------------------------------------------------------
# Cost resolution
# ---------------------------------------------------------------------------


def _resolve_cost(model: str, usage: dict, provider: str = "openrouter") -> float | None:
    """Reported API cost if present, else estimate from the listing price."""
    reported = usage.get("cost")
    if reported is not None and reported > 0 and provider != "zenmux":
        return reported
    price = MODEL_PRICING.get(model)
    if not price:
        return reported
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    in_per_m, out_per_m = price
    raw_cost = (in_tok * in_per_m + out_tok * out_per_m) / 1_000_000.0
    if provider == "zenmux":
        multiplier = ZENMUX_DEFAULT_MULTIPLIER
        for needle, mult in ZENMUX_MODEL_MULTIPLIERS.items():
            if needle in model:
                multiplier = mult
                break
        return raw_cost * multiplier
    return raw_cost


# ---------------------------------------------------------------------------
# Call functions
# ---------------------------------------------------------------------------


def _strip_reasoning(text: str) -> str:
    """Conservatively remove recoverable local/cloud reasoning traces."""
    if not text:
        return ""
    text = re.sub(r"<think\b[^>]*>.*?(</think\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"<reasoning\b[^>]*>.*?(</reasoning\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"<reflection\b[^>]*>.*?(</reflection\s*>|$)", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</(?:think|reasoning|reflection)\s*>", "", text, flags=re.S | re.I)
    text = re.sub(r"<output\b[^>]*>(.*?)</output\s*>", r"\1", text, flags=re.S | re.I)
    text = re.sub(r"<\|channel\|>.*?(<\|channel\|>|$)", "", text, flags=re.S | re.I)
    visible = re.search(
        r"^\s*(thinking process|let me think)[: ].*?(final answer|answer|output)\s*:\s*",
        text,
        flags=re.S | re.I,
    )
    if visible:
        text = text[visible.end() :]
    else:
        low = text.lstrip().lower()
        if low.startswith(("thinking process:", "let me think:")):
            parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
            text = parts[1] if len(parts) == 2 else ""
    return text.strip()


def _call_ollama(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    total = len(system) + len(prompt)
    if total < 4000:
        num_ctx = 2048
    elif total < 24000:
        num_ctx = 8192
    else:
        num_ctx = 32768
    est_input_tokens = total // 3
    needed = est_input_tokens + max_output_tokens + 256
    num_ctx = max(num_ctx, min(needed, 32768))
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": num_ctx,
            "num_predict": max_output_tokens,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = _read_json_response(r)
    latency = time.perf_counter() - t0
    return {
        "text": _strip_reasoning(str(body.get("response", "")).strip()),
        "latency": latency,
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
        "api_cost": 0.0,
    }


def _openai_compat_call(
    endpoint: _Endpoint,
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    """Shared OpenAI-compatible chat-completions POST."""
    api_key = os.environ.get(endpoint.key_env, "")
    if not api_key:
        raise RuntimeError(f"{endpoint.key_env} not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    if model in REASONING_EFFORT_OVERRIDES:
        payload["reasoning_effort"] = REASONING_EFFORT_OVERRIDES[model]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    headers.update(endpoint.extra_headers)
    req = urllib.request.Request(
        f"{endpoint.url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = _read_json_response(r)
    latency = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    text = _strip_reasoning((msg.get("content") or "").strip())
    usage = body.get("usage", {})
    return {
        "text": text,
        "latency": latency,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "api_cost": _resolve_cost(model, usage, provider=endpoint.provider_label),
        "provider": endpoint.provider_label,
    }


def _call_deepinfra(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    resolved_model = _normalize_deepinfra_model(model)
    return _openai_compat_call(
        DEEPINFRA_ENDPOINT, resolved_model, system, prompt, timeout, max_output_tokens
    )


def _call_openrouter(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    return _openai_compat_call(
        OPENROUTER_ENDPOINT, model, system, prompt, timeout, max_output_tokens
    )


def _call_zenmux(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    return _openai_compat_call(ZENMUX_ENDPOINT, model, system, prompt, timeout, max_output_tokens)


def _call_deepseek(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    """DeepSeek FIRST-PARTY call (api.deepseek.com). OpenAI-compatible with
    cache-aware pricing (prompt_cache_hit_tokens)."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    slug = model.split("/", 1)[1] if "/" in model else model
    payload = {
        "model": slug,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{DEEPSEEK_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = _read_json_response(r)
    latency = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    text = _strip_reasoning((msg.get("content") or "").strip())
    usage = body.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    cached = usage.get("prompt_cache_hit_tokens") or usage.get("prompt_cached_tokens") or 0
    price = MODEL_PRICING.get(model, (0.14, 0.28))
    fresh_in = max(in_tok - cached, 0)
    cached_rate = price[0] / 10.0
    cost = (fresh_in * price[0] + cached * cached_rate + out_tok * price[1]) / 1_000_000.0
    return {
        "text": text,
        "latency": latency,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "api_cost": cost,
        "provider": "deepseek",
    }


# Dispatch table: provider label -> call function.
_PROVIDER_DISPATCH: dict[str, "Callable[..., dict]"] = {
    "ollama": _call_ollama,
    "openrouter": _call_openrouter,
    "zenmux": _call_zenmux,
    "deepseek": _call_deepseek,
    "deepinfra": _call_deepinfra,
}


def _call_provider(
    model: str,
    provider: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    """Dispatch a (model, provider) call. Raises on transport error or
    unknown provider."""
    fn = _PROVIDER_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"unknown provider: {provider}")
    return fn(model, system, prompt, timeout, max_output_tokens)
