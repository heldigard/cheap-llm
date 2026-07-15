# vs-soft-allow — call_openai_compat needs base_url+api_key+model+system+prompt
# +timeout+extra to mirror the OpenAI chat-completions wire shape.
"""Inline transport layer — call_local + call_openai_compat + call_candidate.

Self-contained on purpose — the benchmark measures the cheap_llm package by
exercising the same wire shape WITHOUT importing it (otherwise we can't measure
the module from inside itself). urllib imports sit below the rationale comment
intentionally.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
}


def call_local(model: str, system: str, prompt: str, timeout: float = 30.0) -> dict:
    """Call local Ollama. Returns {text, latency, input_tokens, output_tokens}."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    # Operator-controlled local endpoint; Request above constrains the shape.
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp
    ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
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
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    # base_url is selected from the static PROVIDER_URLS map.
    with (
        urllib.request.urlopen(req, timeout=timeout) as resp
    ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
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
