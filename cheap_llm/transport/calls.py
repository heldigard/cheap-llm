# vs-soft-allow — provider call functions need
# model+system+prompt+timeout+max_output_tokens+require_json
"""Provider call functions and the dispatch table.

One ``_call_*`` per provider (sharing an OpenAI-compatible helper), plus the
``_PROVIDER_DISPATCH`` table and the ``_call_provider`` entry point. Adding a
new provider = one ``_call_*`` here + one ``_PROVIDER_DISPATCH`` entry.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Callable

from .constants import DEEPSEEK_URL, LOCAL_KEEP_ALIVE, OLLAMA_URL, REASONING_EFFORT_OVERRIDES
from .httpio import _normalize_model_name, _read_json_response
from .pricing import DEEPSEEK_PRICING, _resolve_cost
from .providers import (
    DEEPINFRA_ENDPOINT,
    OPENROUTER_ENDPOINT,
    ZENMUX_ENDPOINT,
    _Endpoint,
    _normalize_deepinfra_model,
)


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


def _ollama_model_loaded(model: str) -> bool:
    """True if `model` is currently loaded (GET /api/ps). Unknown -> assume warm."""
    norm_target = _normalize_model_name(model)
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/ps", method="GET")
        # nosemgrep — OLLAMA_URL is operator config, not user input
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


def _call_ollama(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
    require_json: bool = False,
) -> dict:
    # Smaller ctx for short tasks (faster on 16GB-class VRAM e.g. RTX 5080,
    # where memory bandwidth is the bottleneck — 9B at q8 leaves little room
    # for a big KV cache). Heuristic: under 4K total input chars → 2048 ctx;
    # else 8192 or 32768 for large tasks.
    total = len(system) + len(prompt)
    if total < 4000:
        num_ctx = 2048
    elif total < 24000:
        num_ctx = 8192
    else:
        num_ctx = 32768
    # num_ctx must hold the prompt AND the generated tokens. The char-based
    # bracket above sizes for VRAM but ignores max_output_tokens, so a
    # near-bracket-edge input + a large output budget silently truncates
    # generation (Ollama caps num_predict at the remaining context). Floor on
    # an input-token estimate + the requested output + slack. The //3 divisor
    # is dense-conservative (code/JSON/logs run ~3 chars/token, not 4) so the
    # floor is not under-sized on the very content this layer distills. The
    # floor is capped at the largest bracket (32768) so a pathological very
    # large input cannot push num_ctx past the pre-existing ceiling into OOM
    # territory on a 16GB-VRAM 9B model — such inputs truncated before too.
    est_input_tokens = total // 3
    needed = est_input_tokens + max_output_tokens + 256
    num_ctx = max(num_ctx, min(needed, 32768))
    payload: dict = {
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
    # Pin residency so successive preprocessor slots on a native desktop GPU
    # skip cold VRAM loads. Disabled when CHEAP_LLM_KEEP_ALIVE=0/off.
    if LOCAL_KEEP_ALIVE is not None:
        payload["keep_alive"] = LOCAL_KEEP_ALIVE
    if require_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    # nosemgrep — OLLAMA_URL is operator config, not user input
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
    require_json: bool = False,
) -> dict:
    """Shared OpenAI-compatible chat-completions POST.

    Used by _call_openrouter and _call_zenmux — both are OpenAI-shaped, only
    URL/key/headers differ. Both providers have been observed returning
    message.content=None for certain reasoning-style responses, so we coerce
    None → "" rather than letting .strip() raise AttributeError on a fresh
    model rollout.
    """
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
    # DeepInfra documents native JSON-object mode and OpenRouter's current
    # catalog advertises response_format for every benchmark-selected model.
    # ZenMux remains prompt-only because its catalog omits parameter support.
    if require_json and endpoint.provider_label in {"deepinfra", "openrouter"}:
        payload["response_format"] = {"type": "json_object"}
    if endpoint.provider_label == "openrouter":
        # cheap-llm optimizes signal-per-dollar; make the aggregator choose the
        # lowest-price healthy endpoint for the already benchmarked model.
        payload["provider"] = {"sort": "price"}
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
    # nosemgrep — endpoint.url is a frozen in-module constant
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
    require_json: bool = False,
) -> dict:
    resolved_model = _normalize_deepinfra_model(model)
    return _openai_compat_call(
        DEEPINFRA_ENDPOINT,
        resolved_model,
        system,
        prompt,
        timeout,
        max_output_tokens,
        require_json,
    )


def _call_openrouter(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
    require_json: bool = False,
) -> dict:
    return _openai_compat_call(
        OPENROUTER_ENDPOINT,
        model,
        system,
        prompt,
        timeout,
        max_output_tokens,
        require_json,
    )


def _call_zenmux(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
    require_json: bool = False,
) -> dict:
    return _openai_compat_call(
        ZENMUX_ENDPOINT, model, system, prompt, timeout, max_output_tokens, require_json
    )


def _call_deepseek(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
    require_json: bool = False,
) -> dict:
    """DeepSeek FIRST-PARTY call (api.deepseek.com). OpenAI-compatible.

    OpenRouter currently lists V4 Flash below the first-party fresh rate, but
    direct DeepSeek exposes a much deeper cache discount ($0.0028/M versus
    $0.14/M fresh). Repeated-prefix workloads may therefore be cheaper direct,
    so it stays first in the automatic pinned-model order. DeepSeek reports
    prompt_cache_hit_tokens and pricing is model-specific. Slug mapping: our
    catalog uses the OpenRouter form "deepseek/deepseek-v4-flash"; the direct
    API wants "deepseek-v4-flash" (strip the "deepseek/" provider prefix).
    """
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
        # V4 defaults to thinking=enabled. This layer distills short signals;
        # hidden reasoning adds latency and spend without authority or value.
        "thinking": {"type": "disabled"},
    }
    if require_json:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{DEEPSEEK_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    # nosemgrep — DEEPSEEK_URL is a frozen in-module constant
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = _read_json_response(r)
    latency = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    text = _strip_reasoning((msg.get("content") or "").strip())
    usage = body.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    cached = usage.get("prompt_cache_hit_tokens") or usage.get("prompt_cached_tokens") or 0
    fresh_rate, cached_rate, output_rate = DEEPSEEK_PRICING.get(
        model, DEEPSEEK_PRICING["deepseek/deepseek-v4-flash"]
    )
    fresh_in = max(in_tok - cached, 0)
    cost = (fresh_in * fresh_rate + cached * cached_rate + out_tok * output_rate) / 1_000_000.0
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
    require_json: bool = False,
) -> dict:
    """Dispatch a (model, provider) call. Raises on transport error or
    unknown provider."""
    fn = _PROVIDER_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"unknown provider: {provider}")
    return fn(model, system, prompt, timeout, max_output_tokens, require_json)
