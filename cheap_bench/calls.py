# vs-soft-allow — call_openai_compat needs base_url+api_key+model+system+prompt
# +timeout+extra to mirror the OpenAI chat-completions wire shape.
"""Inline transport layer — call_local + call_openai_compat + call_candidate.

Self-contained on purpose — the benchmark measures the cheap_llm package by
exercising the same wire shape WITHOUT importing it (otherwise we can't measure
the module from inside itself). urllib imports sit below the rationale comment
intentionally.
"""

from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.parse
import urllib.request

PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
}


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _ollama_generate_url() -> str:
    raw = os.environ.get("OLLAMA_URL", "http://localhost:11434").strip().rstrip("/")
    try:
        parsed = urllib.parse.urlparse(raw)
        port = parsed.port
    except (UnicodeError, ValueError):
        return "http://localhost:11434/api/generate"
    if (
        parsed.scheme != "http"
        or not _is_loopback_host(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        return "http://localhost:11434/api/generate"
    assert parsed.hostname is not None  # established by _is_loopback_host above
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    suffix = f":{port}" if port is not None else ""
    return f"http://{host}{suffix}/api/generate"


def call_local(model: str, system: str, prompt: str, timeout: float = 30.0) -> dict:
    """Call local Ollama. Returns {text, latency, input_tokens, output_tokens}."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        _ollama_generate_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    # URL is normalized to plain-HTTP loopback by _ollama_generate_url.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosemgrep
        body = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - t0
    return {
        "text": body.get("response", "").strip(),
        "latency": latency,
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
    }


def call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    timeout: float = 30.0,
    extra: dict | None = None,
) -> dict:
    """Call any OpenAI-compatible chat-completions endpoint."""
    normalized_base = base_url.rstrip("/")
    if normalized_base not in PROVIDER_URLS.values():
        raise ValueError("unsupported benchmark provider endpoint")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{normalized_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    # normalized_base is selected from the static HTTPS PROVIDER_URLS allowlist.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosemgrep
        body = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - t0
    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    api_cost = usage.get("cost")  # openrouter/deepinfra report this
    return {
        "text": text,
        "latency": latency,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "api_cost": api_cost,
    }


def call_candidate(cand: dict, task: dict, timeout: float) -> dict:
    """Dispatch a candidate against a task. Returns the raw response dict
    plus a 'cost' estimate in USD for cloud models."""
    try:
        if cand["kind"] == "local":
            out = call_local(cand["id"], task["system"], task["prompt"], timeout=timeout)
        else:
            env = cand.get("env")
            if not isinstance(env, str) or not os.environ.get(env):
                return {
                    "error": f"missing env {env}",
                    "text": "",
                    "latency": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0,
                }
            api_key = os.environ[env]
            base_url = PROVIDER_URLS[cand["provider"]]
            out = call_openai_compat(
                base_url,
                api_key,
                cand["id"],
                task["system"],
                task["prompt"],
                timeout=timeout,
                extra=cand.get("extra"),
            )
        inp = out.get("input_tokens", 0)
        out_t = out.get("output_tokens", 0)
        # Trust the PUBLIC LISTING price, not usage.cost — OpenRouter returns
        # usage.cost=0 for some promo/preview models (e.g. gemini-3.1-flash-lite
        # is $0.25/$1.50 real, API reports $0). API keys consume PAYG or granted
        # balance; they are never treated as a zero-cost subscription seat.
        out["cost"] = (inp * cand["input"] + out_t * cand["output"]) / 1_000_000
        return out
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "text": "",
            "latency": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0,
        }
