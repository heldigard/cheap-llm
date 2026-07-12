#!/usr/bin/env python3
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

  T1 LOCAL (free, private)        timeout 6s  — cryptidbleh/gemma4-claude-opus-4.6 for text,
                                                    SetneufPT for JSON/schema
  T2 CHEAP CLOUD                  timeout 12s — TOP3_CASCADE:
      ling-2.6-flash @ openrouter → zenmux      ($0.01/$0.03 per M)
      ling-2.6-1t    @ openrouter → zenmux      ($0.075/$0.625 per M)
      gemini-3.1-flash-lite @ openrouter → zenmux ($0.25/$1.50 per M)
    then LEGACY_CASCADE safety net:
      gpt-5.4-nano, deepseek-v4-flash  @ openrouter  (deepseek BYOK = $0)

There is NO expensive "T3" fallback — every model above is cheap. If all
tiers fail or return invalid output, cheap_complete returns an empty error
envelope (caller decides what to do, typically fall back to the main model).

Per-call config:
  - timeout: T1=6s, T2=12s, capped by timeout_total (a deadline, not per-tier)
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
    python3 cheap_llm.py --system "You are X" --prompt "Y" \\
        --schema field1 field2
    python3 cheap_llm.py --probe   # show what's available

Decisions:
  - The current free-text T1 compatibility default is
    cryptidbleh/gemma4-claude-opus-4.6; structured calls use SetneufPT/Qwopus.
    Keep these comments aligned with DEFAULT_LOCAL_PRIMARY/STRUCTURED below.
  - ling-2.6-flash is the primary cheap cloud (wins 4/5 tasks, $0.000018/call).
  - deepseek-v4-flash via OpenRouter BYOK is free ($0, our key) — the cascade's
    cost floor.
  - gpt-5.4-nano replaced kimi-k2 in the safety net (R4: 5/5 stable, ~40%
    cheaper + faster, same quality). gpt-4.1-nano benchmarks higher but is an
    older generation → deprecation risk, kept as data only (fall-forward rule).
  - Dropped this round: kimi-k2 (superseded), gpt-5-nano + glm-4.7-flash
    (reasoning-only, content="" on short tasks), qwen3.6-flash (reasoning tax).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --- Public API contract ----------------------------------------------------
# The surface consumers may depend on. Everything else is private (_-prefixed)
# and may change without notice. ``tests/test_contract.py`` is the evolution
# gate: a breaking change fails there first and forces a SemVer MAJOR bump.
# SemVer policy for independent evolution across the ecosystem:
#   - MAJOR = removed/renamed public param or RESULT_KEY (consumers' require() gate trips)
#   - MINOR = additive (new param with default, new RESULT_KEY, new public fn)
#   - PATCH = internal refactor, model/cascade changes, bug fixes
__version__ = "1.2.1"
__all__ = ["cheap_complete", "scrub_secrets", "require", "__version__"]

# Stable shape of the dict returned by cheap_complete(). Additive-only: a new
# key is MINOR; removing/renaming is MAJOR.
RESULT_KEYS: tuple[str, ...] = (
    "text",
    "model",
    "provider",
    "tier",
    "latency",
    "cost",
    "json_valid",
    "fields_ok",
    "attempts",
    "error",
    "cached",
)

# Documented cheap_complete() signature — test_contract.py pins the real one.
CHEAP_COMPLETE_PARAMS: tuple[str, ...] = (
    "system",
    "prompt",
    "schema_hint",
    "timeout_total",
    "prefer_local",
    "require_json",
    "model",
    "cloud_model",
    "max_output_tokens",
)

CONTRACT: dict[str, object] = {
    "version": __version__,
    "public_api": list(__all__),
    "result_keys": list(RESULT_KEYS),
    "cheap_complete_params": list(CHEAP_COMPLETE_PARAMS),
}


def _parse_version(v: str) -> tuple[int, ...]:
    """``"1.2.3"`` → ``(1, 2, 3)`` for ordering; non-numeric parts ignored."""
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


def require(min_version: str | None = None) -> str:
    """Version gate consumers call right after ``import cheap_llm``.

    Fails FAST with an actionable message on version drift, instead of a
    cryptic mid-run error when a needed param/key is absent. Returns the
    current version.

        import cheap_llm
        cheap_llm.require("1.1")   # RuntimeError if installed cheap_llm < 1.1
    """
    if min_version and _parse_version(__version__) < _parse_version(min_version):
        raise RuntimeError(
            f"cheap_llm {__version__} is older than required {min_version}. "
            f"Upgrade: cd ~/cheap-llm && pip install -e . --user"
        )
    return __version__


# Default values for RESULT_KEYS that a partial envelope may omit, so every
# cheap_complete() return has the FULL contract shape regardless of outcome
# (success paths omit ``error``; the all-failed path omits ``provider``/
# ``cached``). Centralizing this means adding a RESULT_KEY auto-propagates.
_RESULT_DEFAULTS: dict[str, object] = {
    "text": "",
    "model": None,
    "provider": None,
    "tier": None,
    "latency": 0,
    "cost": 0.0,
    "json_valid": False,
    "fields_ok": False,
    "attempts": [],
    "error": None,
    "cached": False,
}


def _complete_result(env: dict) -> dict:
    """Return ``env`` with every RESULT_KEY present (uniform contract shape).

    Fills absent keys with ``_RESULT_DEFAULTS``. Mutation is in-place on a
    fresh envelope, so callers can still build partial dicts at each path.
    """
    for _k in RESULT_KEYS:
        if _k not in env:
            env[_k] = _RESULT_DEFAULTS[_k]
    return env


# Local T1 defaults. cryptidbleh/gemma4-claude-opus-4.6 is the free-text compatibility default and
# matches ollama_client.DEFAULT_GEN_MODEL. JSON/schema calls use the measured
# structured-output specialist unless callers pass an explicit `model=...`.
DEFAULT_LOCAL_PRIMARY = "cryptidbleh/gemma4-claude-opus-4.6:latest"
DEFAULT_LOCAL_STRUCTURED = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"

# T1 budget when the local model is NOT loaded in VRAM yet (cold start).
# Warm budgets stay 6s/12s; eff_timeout always clamps to the caller's
# timeout_total, so callers with tight deadlines are unaffected.
LOCAL_COLD_TIMEOUT = float(os.environ.get("CHEAP_LLM_LOCAL_COLD_TIMEOUT", "25"))


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
        # OLLAMA_URL is an explicit operator setting, not request/user input.
        # nosemgrep
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Ollama down or unreadable: keep the fast budget — a dead server
        # fails the T1 attempt instantly anyway.
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


# Cascade order (2026-06-19 round 3, top 5 cloud + 1 local).
# CORRECTED PRICING (per OpenRouter's own listing — the API's usage.cost
# field returns $0 for some models in preview/promo but the public list
# price is the truth):
#
#   1. ling-2.6-flash        $0.01 in / $0.03 out  per M  (truly cheap)
#   2. gemini-3.1-flash-lite $0.25 in / $1.50 out  per M  (NOT free despite
#                                                   API returning cost=0;
#                                                   see screenshot 2026-06-19)
#   3. ling-2.6-1t           $0.075 in / $0.625 out per M
#   4. kimi-k2               $0.57 in / $2.30 out  per M  (proven reliable)
#   5. deepseek-v4-flash     $0 in / $0 out (BYOK via our own DEEPSEEK_API_KEY)
#
# All scored 5/5 in dedicated stability test. Dropped: gemma-4-31b,
# qwen3.7-plus, qwen3.6-flash, kimi-k2.7-code, gemma4:12b local,
# nemotron-3-{nano,super}, mimo-v2.5, step-3.7-flash, ring-2.6-1t
# (the last is a reasoning-only model that returned content:None on
# short classification tasks). The live cascade order is TOP3_CASCADE +
# LEGACY_CASCADE below; there are no separate DEFAULT_CLOUD_* constants.

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
# ZenMux is BACK (2026-06-19) at https://zenmux.ai/api/v1 (NOT api.zenmux.ai
# which is NXDOMAIN). Their /api/v1/models endpoint returns pricing data
# showing ZenMux is 4-10x MORE EXPENSIVE than OpenRouter for inclusionai
# ling models (e.g. ling-2.6-flash: OR $0.01/$0.03 vs ZenMux $0.10/$0.30).
# Use as failover only, not primary.
ZENMUX_URL = "https://zenmux.ai/api/v1"
# DeepSeek FIRST-PARTY API (api.deepseek.com) — direct, no intermediary.
# Cheapest route for deepseek models (first-party price, no OR/ZenMux markup;
# Western hosts charge ~2x). Bonus: reports prompt_cache_hit_tokens so cached
# input is priced at the discounted $0.029/M (Flash) vs $0.14/M fresh — a
# saving OpenRouter obscures. Caveat: DeepSeek retains data for training;
# scrub_secrets (always applied) keeps it safe for non-secret prep work.
DEEPSEEK_URL = "https://api.deepseek.com/v1"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _strip_ollama_reasoning(text: str) -> str:
    """Conservatively remove recoverable local reasoning traces."""
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


# Cascade as (model, provider) pairs. For each top model we try OpenRouter
# first, then ZenMux as backup. DeepInfra is intentionally NOT in the cascade
# for the ling/gemini top-3 because (a) it doesn't host ling-2.6 models,
# (b) gemini-3.1-flash-lite on DeepInfra returns <thinking> content instead
# of the actual response, unusable for JSON extraction.
# Order rationale (2026-06-19, per user): ling-2.6-flash + ling-2.6-1t first
# (cheaper + better quality than gemini in WORST case of ZenMux failover —
# gemini-3.1-flash-lite on ZenMux = $0.25/$1.50, vs ling-2.6-1t on ZenMux =
# $0.30/$2.50 and ling-2.6-flash on ZenMux = $0.10/$0.30; on OpenRouter the
# ling models are MUCH cheaper than gemini too). Gemini-3.1-flash-lite demoted
# to TERTIARY: high quality (88.4 avg) but worst $/quality ratio of the 3.
TOP3_CASCADE: list[tuple[str, str]] = [
    # PRIMARY: ling-2.6-flash (89.0 avg, $0.01/$0.03 OR / $0.10/$0.30 ZenMux)
    ("inclusionai/ling-2.6-flash", "openrouter"),
    ("inclusionai/ling-2.6-flash", "zenmux"),  # failover: 10x cost
    # SECONDARY: ling-2.6-1t (87.8 avg, $0.075/$0.625 OR / $0.30/$2.50 ZenMux)
    # 1T model for harder tasks. On OpenRouter it's ~5x cheaper than gemini
    # output. On ZenMux the worst case is comparable but quality is higher.
    ("inclusionai/ling-2.6-1t", "openrouter"),
    ("inclusionai/ling-2.6-1t", "zenmux"),  # failover: 4x cost
    # TERTIARY: gemini-3.1-flash-lite (88.4 avg, $0.25/$1.50 on both providers)
    # Demoted from secondary. Highest benchmark score but worst $/quality of
    # the 3 — even on OR it's 25x more expensive than ling-2.6-flash output
    # ($1.50 vs $0.03). Use only if both ling models fail.
    ("google/gemini-3.1-flash-lite", "openrouter"),
    ("google/gemini-3.1-flash-lite", "zenmux"),  # failover: same price
]

# Quaternary/Quinary safety net (still on OpenRouter BYOK / standard):
# gpt-5.4-nano replaced kimi-k2 (2026-06-19): 5/5 stable on the stability
# protocol, $0.20/$1.25 vs kimi-k2 $0.57/$2.30, ~2.2s vs ~3.6s, same JSON
# quality. gpt-5-nano and glm-4.7-flash were REJECTED — both are
# reasoning-only and return content="" on short classification tasks.
LEGACY_CASCADE: list[tuple[str, str]] = [
    ("openai/gpt-5.4-nano", "openrouter"),  # 5/5 stable, cheap+fast
    ("deepseek/deepseek-v4-flash", "openrouter"),  # $0 BYOK (our key)
]

CACHE_DIR = Path.home() / ".claude" / "state" / "cheap-llm-cache"
CACHE_MAX_ENTRIES = 2000

# --- Secret scrub ---------------------------------------------------------
# Patterns we never send to a third-party. Scrubbed in both system and prompt.
SECRET_PATTERNS = [
    # PEM private-key block (RSA/EC/OPENSSH/PGP) — full block first, then a
    # dangling BEGIN for truncated logs/diffs that never reach an END line.
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]*)PRIVATE KEY-----.*?"
            r"-----END (?:[A-Z0-9 ]*)PRIVATE KEY-----",
            re.S,
        ),
        "<REDACTED_PEM_KEY>",
    ),
    (re.compile(r"-----BEGIN (?:[A-Z0-9 ]*)PRIVATE KEY-----[^\n]*"), "<REDACTED_PEM_KEY>"),
    # DB / message-broker connection strings with embedded creds:
    # postgres://user:pass@host, mongodb+srv://.., redis://.., amqp(s)://..
    (
        re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^:/@\s]+):([^@\s]+)@"),
        r"\1<REDACTED_USER>:<REDACTED_PASS>@",
    ),
    # env-style assignment with quote
    (
        re.compile(
            r"(?i)(password|passwd|secret|api[_-]?key|api[_-]?token|access[_-]?token|"
            r'auth[_-]?token|private[_-]?key)\s*[:=]\s*["\'][^"\']{6,}["\']'
        ),
        r"\1=<REDACTED>",
    ),
    # bare bearer / sk- / ghp_ / gho_ style
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-+/=]{16,}"), r"\1<REDACTED_TOKEN>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "<REDACTED_SK>"),
    # GitHub tokens: classic (ghp/gho/ghu/ghs/ghr, 36+ payload) + fine-grained PAT
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"), "<REDACTED_GH>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "<REDACTED_GH>"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "<REDACTED_XOX>"),
    # cloud provider keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS>"),  # AWS access-key id
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "<REDACTED_GCP>"),  # Google API key
    (re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{20,}\b"), "<REDACTED_STRIPE>"),
    # JWT-ish (3 base64 segments)
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
        "<REDACTED_JWT>",
    ),
]


def scrub_secrets(text: str) -> str:
    """Remove secret-looking strings before sending to a third-party model."""
    out = text
    for pat, repl in SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


# --- Cache ----------------------------------------------------------------
def _cache_key(
    model: str,
    system: str,
    prompt: str,
    schema: tuple[str, ...] | None,
    max_output_tokens: int = 1024,
) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\0")
    h.update(system.encode())
    h.update(b"\0")
    h.update(prompt.encode())
    h.update(b"\0")
    if schema:
        h.update("|".join(schema).encode())
    if max_output_tokens != 1024:
        # Preserve the pre-1.2 cache namespace for the backward-compatible
        # default; only explicitly different budgets need a new namespace.
        h.update(b"\0")
        h.update(str(max_output_tokens).encode())
    return h.hexdigest()


def _cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            value = json.loads(p.read_text())
        except Exception:
            return None
        # Shape guard: a corrupted/foreign cache file that parses as JSON but
        # isn't {"text": str} would raise KeyError/TypeError inside
        # _try_cache_hit and crash the whole cascade. Treat it as a miss.
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value
    return None


def _cache_put(key: str, value: dict) -> None:
    # Atomic write (temp + rename) so a mid-write crash never leaves a partial
    # cache file that the next _cache_get would try to parse. Cache writes are
    # best-effort: failure here MUST NOT propagate and break a successful
    # cascade — caller already returned the value to the user.
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # prune oldest beyond CACHE_MAX_ENTRIES
        files = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(files) >= CACHE_MAX_ENTRIES:
            files.pop(0).unlink(missing_ok=True)
        target = CACHE_DIR / f"{key}.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(target)
    except Exception:
        pass  # cache is advisory; never break the cascade on a write error


# --- Transport ------------------------------------------------------------
def _resolve_cost(model: str, usage: dict, provider: str = "openrouter") -> float | None:
    """Reported API cost if present, else estimate from the listing price.

    ZenMux returns usage.cost=None and OpenRouter returns $0 for some promo
    models; without an estimate those calls show $0 in telemetry, which hides
    real spend (ZenMux is 4-10x OpenRouter for the same model). Returns None
    only when we have neither a reported cost nor a known price.
    """
    reported = usage.get("cost")
    if reported is not None and reported > 0 and provider != "zenmux":
        return reported
    price = MODEL_PRICING.get(model)
    if not price:
        return reported  # may be None/0 — caller treats falsy as 0.0
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    in_per_m, out_per_m = price
    raw_cost = (in_tok * in_per_m + out_tok * out_per_m) / 1_000_000.0
    if provider == "zenmux":
        # ZenMux has a premium pricing multiplier over OpenRouter
        multiplier = 10.0 if "ling-2.6-flash" in model else (4.0 if "ling-2.6-1t" in model else 5.0)
        return raw_cost * multiplier
    return raw_cost


def _call_ollama(
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    # Smaller ctx for short tasks (faster on 16GB VRAM, where memory bandwidth
    # is the bottleneck — 9B at q8 leaves little room for big KV cache).
    # Heuristic: under 4K total input chars → 2048 ctx; else 8192 or 32768 for large tasks.
    total = len(system) + len(prompt)
    if total < 4000:
        num_ctx = 2048
    elif total < 24000:
        num_ctx = 8192
    else:
        num_ctx = 32768
    payload = {
        "model": model,
        "prompt": f"{system}\n\n{prompt}",
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
    # nosemgrep: OLLAMA_URL is explicit local operator configuration.
    with (
        urllib.request.urlopen(req, timeout=timeout) as r
    ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
        body = json.loads(r.read().decode())
    latency = time.perf_counter() - t0
    return {
        "text": _strip_ollama_reasoning(str(body.get("response", "")).strip()),
        "latency": latency,
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
        "api_cost": 0.0,
    }


@dataclass(frozen=True)
class _Endpoint:
    """OpenAI-compatible chat-completions endpoint config.

    Bundles url + key_env + provider_label + headers so the call-site helper
    only sees one endpoint token instead of 4-5 positional params.
    """

    url: str
    key_env: str
    provider_label: str
    extra_headers: dict[str, str] = field(default_factory=dict)


def _openai_compat_call(
    endpoint: _Endpoint,
    model: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
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
    # endpoint.url comes only from frozen in-module provider constants.
    # nosemgrep
    with (
        urllib.request.urlopen(req, timeout=timeout) as r
    ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
        body = json.loads(r.read().decode())
    latency = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    usage = body.get("usage", {})
    return {
        "text": text,
        "latency": latency,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "api_cost": _resolve_cost(model, usage, provider=endpoint.provider_label),
        "provider": endpoint.provider_label,
    }


# OpenRouter endpoint — X-Title is the app-attribution field shown on their
# leaderboard. Set honestly to the internal tool name; no fake repo URL —
# HTTP-Referer omitted rather than pointing at a non-existent project.
OPENROUTER_ENDPOINT = _Endpoint(
    url=OPENROUTER_URL,
    key_env="OPENROUTER_API_KEY",
    provider_label="openrouter",
    extra_headers={"X-Title": "cheap-llm-cascade"},
)

# ZenMux endpoint. Same OpenAI-compatible shape as OpenRouter. ZenMux returns
# usage.cost=None and pricing may show $0 for promo models; trust the public
# ZenMux pricing page (zenmux.ai/pricing) as production cost. ZenMux pricing
# is 4-10x HIGHER than OpenRouter for inclusionai ling models (verified
# 2026-06-19). Use as failover, not primary.
ZENMUX_ENDPOINT = _Endpoint(
    url=ZENMUX_URL,
    key_env="ZENMUX_API_KEY",
    provider_label="zenmux",
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
    """DeepSeek FIRST-PARTY call (api.deepseek.com). OpenAI-compatible.

    NOTE 2026-07-02: OpenRouter now lists v4-flash at $0.089/$0.18 (below the
    first-party $0.14/$0.28 fresh rate), but first-party keeps the 1/10
    cache-hit discount ($0.014/M cached input) — for repeated-prefix workloads
    (fixed system prompts, iterative synthesis) first-party still wins overall,
    so it stays FIRST in the forced-cloud_model order. Cache-aware cost:
    DeepSeek reports prompt_cache_hit_tokens,
    so cached prompt input is priced at the discounted rate ($0.029/M for Flash
    vs $0.14/M fresh) — a saving intermediaries obscure. Slug mapping: our
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
    }
    req = urllib.request.Request(
        f"{DEEPSEEK_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    # nosemgrep: DEEPSEEK_URL is a fixed in-module provider constant.
    with (
        urllib.request.urlopen(req, timeout=timeout) as r
    ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
        body = json.loads(r.read().decode())
    latency = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    usage = body.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    cached = usage.get("prompt_cache_hit_tokens") or usage.get("prompt_cached_tokens") or 0
    # Cache-aware cost: fresh input @ list rate, cached input @ 1/10 of input
    # (V4 pricing 2026-04: Flash $0.14 fresh / $0.014 cached — verified
    # 2026-07-02 vs cloudzero/apidog pricing pages). Pro follows the same
    # 1/10 cache ratio per the published V4 pricing.
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


def _call_provider(
    model: str,
    provider: str,
    system: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int = 1024,
) -> dict:
    """Dispatch a (model, provider) call. Raises on transport error."""
    if provider == "ollama":
        return _call_ollama(model, system, prompt, timeout, max_output_tokens)
    elif provider == "openrouter":
        return _call_openrouter(model, system, prompt, timeout, max_output_tokens)
    elif provider == "zenmux":
        return _call_zenmux(model, system, prompt, timeout, max_output_tokens)
    elif provider == "deepseek":
        return _call_deepseek(model, system, prompt, timeout, max_output_tokens)
    else:
        raise ValueError(f"unknown provider: {provider}")


# --- JSON contract --------------------------------------------------------
JSON_HINT = (
    "\n\nReply with JSON only — no prose, no code fences, no explanation. "
    "The first character must be `{` and the last must be `}`."
)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Lenient retry: models occasionally emit trailing commas. Only
        # applied when strict parse FAILS, so valid JSON is never altered.
        try:
            result = json.loads(re.sub(r",(\s*[}\]])", r"\1", text))
        except json.JSONDecodeError:
            return None
    # Preprocessor slots always expect an object with fields; reject arrays
    # and primitives so callers can rely on dict semantics downstream.
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


# --- Main cascade ---------------------------------------------------------
# vs-soft-allow — cheap_complete signature has 9 kwargs (system, prompt,
# schema_hint, timeout_total, prefer_local, require_json, model, cloud_model,
# max_output_tokens). These are the cascade resolver's feature toggles; the 7
# consumer scripts in ~/.claude/scripts/ (commit-draft, diff-review, error-classify,
# extract-tool-output, pdf-extract-structured, pr-draft, test-triage) depend on
# this public signature.


def _build_cascade(
    prefer_local: bool,
    local_model: str | None,
    cloud_model: str | None,
) -> list[tuple[str, str, str, float]]:
    """Build the ordered (tier, model, provider, timeout) cascade.

    Cascade order (2026-06-19, round 3 — kept stable; cloud_model=None path):
      PRIMARY  ling-2.6-flash  OR → ZenMux
      SECONDARY ling-2.6-1t     OR → ZenMux
      TERTIARY  gemini-3.1-flash-lite  OR → ZenMux
      SAFETY NET gpt-5.4-nano   OR
      SAFETY NET deepseek-v4-flash OR

    Forced `cloud_model` (judgment-heavy tasks — e.g. web-research synthesis):
      - deepseek/... → FIRST-PARTY (cheapest + cache-aware) → OR → ZenMux
      - other        → OR → ZenMux
    """
    cascade: list[tuple[str, str, str, float]] = []
    if prefer_local:
        # T1 timeout 6s (not 3s): lets short tasks (classify/commit/error)
        # resolve FREE + PRIVATE, leaving only heavier extract/review to cloud.
        resolved = local_model or DEFAULT_LOCAL_PRIMARY
        local_timeout = 12.0 if resolved == DEFAULT_LOCAL_STRUCTURED else 6.0
        if not _ollama_model_loaded(resolved):
            # Cold start: the first call of a session must not silently leak
            # to cloud PAYG just because the local model needs to load into
            # VRAM (observed 2026-07-08: warm T1 answers in <6s, cold falls
            # to T2). eff_timeout still clamps to the caller's timeout_total,
            # so tight-budget hooks keep their own ceiling.
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
            if not _ollama_model_loaded(resolved):
                local_timeout = max(local_timeout, LOCAL_COLD_TIMEOUT)
            cascade.append(("T1", resolved, "ollama", local_timeout))
        return cascade

    if cloud_model:
        if cloud_model.startswith("deepseek/"):
            cascade.append(("T2", cloud_model, "deepseek", 18.0))
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
    """Return a cache-hit success envelope, or None on miss / invalid cached value.

    Cache is keyed per-MODEL (not per-provider): a ZenMux failover after an
    OpenRouter miss reuses the same answer, saving cost.
    """
    cached = _cache_get(ckey)
    if not cached:
        return None
    attempts.append(
        {
            "tier": tier,
            "model": model,
            "provider": provider,
            "cache_hit": True,
            "latency": 0,
            "cost": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "max_output_tokens": max_output_tokens,
        }
    )
    text = cached["text"]
    parsed = _try_parse_json(text) if require_json else None
    ok = _validate(parsed, schema_t) if require_json else True
    if not ok:
        return None
    return {
        "text": text,
        "model": model,
        "provider": provider,
        "tier": tier,
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
    _cache_put(ckey, {"text": text})
    return {
        "text": text,
        "model": model,
        "provider": raw.get("provider", provider),
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

    cloud_model: force a SPECIFIC cloud model for the T2 tier (with the usual
    OpenRouter→ZenMux failover) instead of the default ling/gemini cascade.
    Use for judgment-heavy tasks where a frontier-class economical model beats
    the signal-distillation ling tier — e.g. web-research cited synthesis
    (deepseek-v4-flash: 1M ctx, $0.14/$0.28). Default None = current cascade.

    max_output_tokens: hard output budget propagated to Ollama ``num_predict``
    and OpenAI-compatible ``max_tokens``. Keep the 1024 default for backward
    compatibility; bounded classifiers/extractors should request less.

    Returns dict with: text, model, tier, latency, cost, json_valid,
    fields_ok, attempts, error.
    """
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    schema_t = tuple(schema_hint) if schema_hint else ()
    # ALWAYS scrub, even on the prefer_local path: T1 (Ollama) frequently
    # times out on 16GB VRAM and the SAME prompt then reaches a cloud tier.
    # Scrubbing is harmless locally and is the only thing standing between
    # an error-log/diff full of creds and a third-party API (+ the on-disk
    # cache, which stores this text verbatim).
    scrubbed_system = scrub_secrets(system)
    scrubbed_prompt = scrub_secrets(prompt)

    # JSON contract: whenever the output will be VALIDATED as JSON, the model
    # must be TOLD to emit JSON — schema or not. Before 1.2.1, require_json
    # without schema_hint validated JSON but never instructed the model
    # (observed live: pdf-extract-structured), so prose replies were rejected
    # as "all tiers failed". The schema path keeps the exact pre-existing
    # string so its cache namespace is preserved.
    eff_system = scrubbed_system
    if require_json:
        eff_system = scrubbed_system + JSON_HINT
        if schema_t:
            eff_system += f" Required keys: {list(schema_t)}."

    local_model = _resolve_local_model(model, require_json, schema_t)
    cascade = _build_cascade(prefer_local, local_model, cloud_model)
    attempts: list[dict] = []
    deadline = time.perf_counter() + timeout_total

    for tier, mdl, provider, per_timeout in cascade:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        eff_timeout = min(per_timeout, remaining)

        # cache lookup (per-model, NOT per-provider: same model on different
        # providers usually gives the same answer for short tasks; saves cost
        # if OpenRouter fails and we try ZenMux, ZenMux call hits cache).
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
            raw = _call_provider(
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
                    "error": f"{type(e).__name__}: {e}",
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

    # all tiers failed or returned invalid output
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


# --- CLI ------------------------------------------------------------------
def _probe() -> dict:
    """Report what's available right now."""
    out: dict = {
        "ollama_alive": False,
        "local_models": [],
        "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        "zenmux_key_set": bool(os.environ.get("ZENMUX_API_KEY")),
        "deepseek_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "local_only": os.environ.get("CHEAP_LLM_LOCAL_ONLY", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "cache_entries": len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0,
    }
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        # nosemgrep: OLLAMA_URL is explicit local operator configuration.
        with (
            urllib.request.urlopen(req, timeout=2) as r
        ):  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
            data = json.loads(r.read())
        out["ollama_alive"] = True
        out["local_models"] = [
            m["name"] for m in data.get("models", []) if "embed" not in m["name"]
        ]
    except Exception as e:
        out["ollama_error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Cheap-LLM cascade client")
    # --system/--prompt are required for completion but NOT for --probe;
    # enforced manually below so `cheap_llm.py --probe` works standalone.
    p.add_argument("--system", help="system prompt")
    p.add_argument("--prompt", help="user prompt")
    p.add_argument("--schema", nargs="*", default=None, help="required JSON keys")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum output tokens per provider attempt (default: 1024)",
    )
    p.add_argument("--no-local", action="store_true", help="skip T1 local")
    p.add_argument("--model", help="explicit T1 local model (Ollama tag)")
    p.add_argument(
        "--cloud-model",
        help="force a specific T2 cloud model (e.g. deepseek/deepseek-v4-flash)",
    )
    p.add_argument("--no-json", action="store_true", help="don't require JSON output")
    p.add_argument("--probe", action="store_true", help="report availability")
    p.add_argument("--json", action="store_true", help="output JSON envelope")
    p.add_argument("--version", action="store_true", help="print version and exit")
    args = p.parse_args()

    if args.version:
        print(__version__)
        return 0
    if args.probe:
        print(json.dumps(_probe(), indent=2))
        return 0
    if not args.system or not args.prompt:
        p.error("--system and --prompt are required (unless --probe)")
    if args.max_tokens < 1:
        p.error("--max-tokens must be a positive integer")

    result = cheap_complete(
        system=args.system,
        prompt=args.prompt,
        schema_hint=args.schema,
        timeout_total=args.timeout,
        prefer_local=not args.no_local,
        require_json=not args.no_json,
        model=args.model,
        cloud_model=args.cloud_model,
        max_output_tokens=args.max_tokens,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["text"])
        if result.get("error"):
            print(f"\n[cheap_llm] error: {result['error']}", file=sys.stderr)
            return 1
        if result.get("model"):
            meta = (
                f"\n[cheap_llm] model={result['model']} tier={result['tier']} "
                f"lat={result['latency']:.2f}s cost=${result['cost']:.6f} "
                f"json_valid={result['json_valid']} cached={result['cached']}"
            )
            print(meta, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
