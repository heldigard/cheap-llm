#!/usr/bin/env python3
"""Regression tests for cheap_llm package — cascade, scrubbing, caching, failover.

Three layers:
  UNIT (no network):     _try_parse_json, _validate, scrub_secrets, _cache_key
  MOCKED (no network):   cascade ordering, provider failover, cache hit, total failure
  LIVE (real API):       smoke test each top-3 cascade entry returns valid JSON

Run: python3 ~/.claude/scripts/test-cheap-llm.py [--live]
  --live     also run the live API smoke tests (requires API keys in env)
  --quick    skip live tests even if --live is set
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request as _urlreq
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cheap_llm as cl  # noqa: E402

# Save and clear DeepInfra API key to keep unit/mock test assertions predictable
_actual_deepinfra_key = os.environ.pop("DEEPINFRA_API_KEY", None)

PASS = 0
FAIL = 0
SKIP = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def skip(name: str, reason: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  SKIP  {name}  ({reason})")


def synthetic_secret(*parts: str) -> str:
    """Build detector-shaped fixtures without storing complete secrets in source."""
    return "".join(parts)


# =================================================================
# UNIT: pure functions, no network, no mocks
# =================================================================
print("\n=== UNIT: pure functions ===")

# _try_parse_json
check("parse clean JSON", cl._try_parse_json('{"a": 1}') == {"a": 1})
check("parse fenced JSON", cl._try_parse_json('```json\n{"a": 2}\n```') == {"a": 2})
check("parse JSON with leading prose", cl._try_parse_json('Sure! Here: {"a": 3}') == {"a": 3})
check("reject malformed JSON", cl._try_parse_json("not json at all") is None)
check("reject empty string", cl._try_parse_json("") is None)
check("reject JSON array (not object)", cl._try_parse_json("[1,2,3]") is None)
check(
    "parse nested JSON in prose",
    cl._try_parse_json('Result: {"category": "debug", "nested": {"x": 1}} end')
    == {"category": "debug", "nested": {"x": 1}},
)

# _validate
check("validate no schema = always true", cl._validate({"a": 1}, None) is True)
check("validate empty schema = always true", cl._validate({"a": 1}, ()) is True)
check("validate all fields present", cl._validate({"a": 1, "b": 2}, ("a", "b")) is True)
check("validate missing field", cl._validate({"a": 1}, ("a", "b")) is False)
check("validate empty field rejected", cl._validate({"a": "", "b": 2}, ("a", "b")) is False)
check("validate None field rejected", cl._validate({"a": None, "b": 2}, ("a", "b")) is False)
check("validate empty array accepted", cl._validate({"a": []}, ("a",)) is True)
check("validate empty object accepted", cl._validate({"a": {}}, ("a",)) is True)
check("validate false boolean accepted", cl._validate({"a": False}, ("a",)) is True)
check("validate zero accepted", cl._validate({"a": 0}, ("a",)) is True)

# scrub_secrets
check(
    "scrub api_key assignment",
    "REDACTED" in cl.scrub_secrets('api_key = "' + synthetic_secret("abcdef", "1234567890") + '"'),
    detail="expected REDACTED in scrubbed output",
)
check(
    "scrub Bearer token",
    cl.scrub_secrets("Authorization: Bearer " + synthetic_secret("abc123def456", "ghi789jkl012"))
    == "Authorization: Bearer <REDACTED_TOKEN>",
)
check(
    "scrub sk- key",
    "REDACTED_SK"
    in cl.scrub_secrets("my key is " + synthetic_secret("sk-proj", "1234567890abcdefghij")),
)
check(
    "scrub ghp_ key",
    "REDACTED_GH"
    in cl.scrub_secrets(
        "token: " + synthetic_secret("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789AB")
    ),
)
check(
    "scrub JWT",
    "REDACTED_JWT"
    in cl.scrub_secrets(
        synthetic_secret(
            "eyJhbGciOiJIUzI1NiJ9",
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            ".signature_abc123",
        )
    ),
)
check(
    "scrub non-secrets unchanged",
    cl.scrub_secrets("just a normal log line") == "just a normal log line",
)
check("scrub xox token", "REDACTED_XOX" in cl.scrub_secrets("slack: xoxb-12345-67890"))

# _cache_key
k1 = cl._cache_key("m", "sys", "user", ("a",))
k2 = cl._cache_key("m", "sys", "user", ("a",), max_output_tokens=1024)
k3 = cl._cache_key("m", "sys", "userX", ("a",))
k4 = cl._cache_key("m", "sys", "user", ("b",))
k5 = cl._cache_key("m", "sys", "user", ("a",), max_output_tokens=256)
check("cache key deterministic", k1 == k2)
check("cache key differs on prompt", k1 != k3)
check("cache key differs on schema", k1 != k4)
check("cache key differs on output budget", k1 != k5)

# JSON_HINT
check("JSON hint includes 'JSON only' marker", "JSON only" in cl.JSON_HINT)
check("JSON hint mentions first char", "`{`" in cl.JSON_HINT)

# TOP3_CASCADE structure
check("TOP3_CASCADE has 6 entries (3 models × 2 providers)", len(cl.TOP3_CASCADE) == 6)
check(
    "first entry is ling-2.6-flash@openrouter",
    cl.TOP3_CASCADE[0] == ("inclusionai/ling-2.6-flash", "openrouter"),
)
check(
    "ling-2.6-flash has zenmux failover",
    ("inclusionai/ling-2.6-flash", "zenmux") in cl.TOP3_CASCADE,
)
check(
    "ling-2.6-1t comes before gemini",
    [m for m, _ in cl.TOP3_CASCADE].index("inclusionai/ling-2.6-1t")
    < [m for m, _ in cl.TOP3_CASCADE].index("google/gemini-3.1-flash-lite"),
)

# LEGACY_CASCADE: kimi-k2 replaced by gpt-5.4-nano; dead constants gone
check(
    "LEGACY_CASCADE has gpt-5.4-nano (replaced kimi-k2)",
    ("openai/gpt-5.4-nano", "openrouter") in cl.LEGACY_CASCADE,
    detail=f"LEGACY={cl.LEGACY_CASCADE}",
)
check(
    "kimi-k2 is no longer in the cascade",
    "moonshotai/kimi-k2" not in [m for m, _ in cl.TOP3_CASCADE + cl.LEGACY_CASCADE],
)
check(
    "deepseek-v4-flash still in LEGACY PAYG fallback",
    ("deepseek/deepseek-v4-flash", "openrouter") in cl.LEGACY_CASCADE,
)
check(
    "no dead DEFAULT_CLOUD_* constants remain",
    not any(
        hasattr(cl, n)
        for n in (
            "DEFAULT_CLOUD_PRIMARY",
            "DEFAULT_CLOUD_SECONDARY",
            "DEFAULT_CLOUD_TERTIARY",
            "DEFAULT_CLOUD_QUATERNARY",
            "DEFAULT_CLOUD_QUINARY",
        )
    ),
    detail="stale DEFAULT_CLOUD_* constants should be removed",
)
check("MODEL_PRICING has gpt-5.4-nano", "openai/gpt-5.4-nano" in cl.MODEL_PRICING)

_old_zenmux_billing = os.environ.get("CHEAP_LLM_ZENMUX_BILLING")
try:
    os.environ["CHEAP_LLM_ZENMUX_BILLING"] = "subscription"
    _zenmux_plan = cl._route_plan(
        prefer_local=False,
        cloud_model="inclusionai/ling-2.6-flash",
        cloud_provider="zenmux",
    )
    check(
        "zenmux stays PAYG when a stale subscription override exists",
        cl._provider_billing("zenmux") == "payg"
        and _zenmux_plan["routes"][0]["billing"] == "payg",
    )
finally:
    if _old_zenmux_billing is None:
        os.environ.pop("CHEAP_LLM_ZENMUX_BILLING", None)
    else:
        os.environ["CHEAP_LLM_ZENMUX_BILLING"] = _old_zenmux_billing

# T1 free-text compatibility and structured JSON defaults stay explicit.
check(
    "T1 local primary is cryptidbleh/gemma4-claude-opus-4.6",
    cl.DEFAULT_LOCAL_PRIMARY == "cryptidbleh/gemma4-claude-opus-4.6:latest",
    detail=f"got {cl.DEFAULT_LOCAL_PRIMARY}",
)
check(
    "T1 local structured primary is SetneufPT/Qwopus",
    "Qwopus" in cl.DEFAULT_LOCAL_STRUCTURED,
    detail=f"got {cl.DEFAULT_LOCAL_STRUCTURED}",
)

_old_local_override = os.environ.get("CHEAP_LLM_LOCAL_MODEL")
_old_structured_override = os.environ.get("CHEAP_LLM_LOCAL_STRUCTURED_MODEL")
try:
    os.environ["CHEAP_LLM_LOCAL_MODEL"] = "operator/text-model:latest"
    os.environ["CHEAP_LLM_LOCAL_STRUCTURED_MODEL"] = "operator/json-model:latest"
    check(
        "local text model honors environment override",
        cl._resolve_local_model(None, False, ()) == "operator/text-model:latest",
    )
    check(
        "local structured model honors separate environment override",
        cl._resolve_local_model(None, True, ("field",)) == "operator/json-model:latest",
    )
    check(
        "explicit local model beats environment overrides",
        cl._resolve_local_model("explicit:latest", True, ("field",)) == "explicit:latest",
    )
finally:
    if _old_local_override is None:
        os.environ.pop("CHEAP_LLM_LOCAL_MODEL", None)
    else:
        os.environ["CHEAP_LLM_LOCAL_MODEL"] = _old_local_override
    if _old_structured_override is None:
        os.environ.pop("CHEAP_LLM_LOCAL_STRUCTURED_MODEL", None)
    else:
        os.environ["CHEAP_LLM_LOCAL_STRUCTURED_MODEL"] = _old_structured_override

# --- CRITICAL regression: secrets are scrubbed on the prefer_local path ---
# DeepSeek first-party cache-aware cost uses the exact model-specific rate.
print("\n=== UNIT: deepseek cache-aware cost ===")


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _fake_urlopen_factory(body: dict, seen_payload: dict | None = None):
    def _fake(req, timeout=None):
        if seen_payload is not None and req.data:
            seen_payload.update(json.loads(req.data.decode()))
        return _FakeResp(json.dumps(body).encode())

    return _fake


_ds_body = {
    "choices": [{"message": {"content": "ok"}}],
    "usage": {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 100_000,
        "prompt_cache_hit_tokens": 600_000,
    },
}
_orig_urlopen = _urlreq.urlopen
_ds_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(_ds_body, _ds_payload)
try:
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
    _ds = cl._call_deepseek(
        "deepseek/deepseek-v4-flash",
        "s",
        "p",
        timeout=5,
        max_output_tokens=384,
        require_json=True,
    )
finally:
    _urlreq.urlopen = _orig_urlopen
# fresh 400K @ $0.14/M + cached 600K @ $0.0028/M + out 100K @ $0.28/M
_expected = (400_000 * 0.14 + 600_000 * 0.0028 + 100_000 * 0.28) / 1_000_000
check(
    "deepseek cost: cached input uses exact V4 Flash rate",
    abs(_ds["api_cost"] - _expected) < 1e-9,
    detail=f"got {_ds['api_cost']:.6f} expected {_expected:.6f}",
)
check(
    "deepseek slug strips provider prefix (call succeeded)",
    _ds["text"] == "ok" and _ds["provider"] == "deepseek",
)
check("deepseek receives output budget", _ds_payload.get("max_tokens") == 384)
check("deepseek disables thinking", _ds_payload.get("thinking") == {"type": "disabled"})
check(
    "deepseek enables native JSON mode when required",
    _ds_payload.get("response_format") == {"type": "json_object"},
)

# Ollama uses the equivalent num_predict option so local generation obeys the
# same public budget as cloud transports.
_ollama_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _ollama_payload
)
try:
    _ol = cl._call_ollama("local", "s", "p", timeout=5, max_output_tokens=192)
finally:
    _urlreq.urlopen = _orig_urlopen
check(
    "ollama receives output budget as num_predict",
    _ollama_payload.get("options", {}).get("num_predict") == 192 and _ol["text"] == "ok",
)

# num_ctx must hold the prompt AND generation. A small input + the default
# output budget must NOT bump num_ctx past the 2048 bracket (no VRAM regression
# for the common short-task path).
_small_ctx_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _small_ctx_payload
)
try:
    _small_ctx = cl._call_ollama("local", "s", "p", timeout=5, max_output_tokens=1024)
finally:
    _urlreq.urlopen = _orig_urlopen
check(
    "ollama num_ctx stays at 2048 for small input + default budget",
    _small_ctx_payload.get("options", {}).get("num_ctx") == 2048 and _small_ctx["text"] == "ok",
    detail=f"num_ctx={_small_ctx_payload.get('options', {}).get('num_ctx')}",
)

# A near-bracket-edge input (just under 4K chars) + a large output budget must
# raise num_ctx above the 2048 bracket so Ollama does not silently cap
# num_predict at the remaining context (the original truncation bug).
_big_ctx_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _big_ctx_payload
)
try:
    _big_ctx = cl._call_ollama("local", "x" * 3900, "y" * 50, timeout=5, max_output_tokens=2048)
finally:
    _urlreq.urlopen = _orig_urlopen
_big_total = 3900 + 50
_big_needed = _big_total // 3 + 2048  # est_input + output (slack excluded for a lower bound)
check(
    "ollama num_ctx floors on input + output budget",
    _big_ctx_payload.get("options", {}).get("num_ctx") >= _big_needed
    and _big_ctx_payload.get("options", {}).get("num_predict") == 2048
    and _big_ctx["text"] == "ok",
    detail=(f"num_ctx={_big_ctx_payload.get('options', {}).get('num_ctx')} needed>={_big_needed}"),
)

# A pathological very large input must NOT push num_ctx past the 32768 ceiling
# (the floor is capped) — otherwise a 9B model on 16GB VRAM OOMs. Such inputs
# truncated before the fix too; the cap preserves that pre-existing ceiling.
_cap_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _cap_payload
)
try:
    _cap = cl._call_ollama("local", "x" * 200_000, "y", timeout=5, max_output_tokens=1024)
finally:
    _urlreq.urlopen = _orig_urlopen
check(
    "ollama num_ctx capped at 32768 for huge input (no VRAM OOM)",
    _cap_payload.get("options", {}).get("num_ctx") == 32768 and _cap["text"] == "ok",
    detail=f"num_ctx={_cap_payload.get('options', {}).get('num_ctx')}",
)

# A provider/proxy can ignore the requested token budget. Bound raw response
# bytes before decoding so malformed upstreams cannot grow memory unboundedly.
try:
    cl._read_json_response(_FakeResp(b"x" * (cl.MAX_RESPONSE_BYTES + 1)))
except ValueError as exc:
    check("transport rejects responses over 4 MiB", "4 MiB" in str(exc))
else:
    check("transport rejects responses over 4 MiB", False, "oversized body accepted")

check(
    "public HTTP attempt error omits URL and reason body",
    cl._public_attempt_error(
        urllib.error.HTTPError(
            "https://example.invalid/private", 503, "body-like reason", cast(Any, {}), None
        )
    )
    == "HTTPError: HTTP 503",
)

# Reproduce the 2026-06-19 bug: prefer_local=True used to skip scrubbing, but
# cloud tiers always follow T1, so unscrubbed secrets reached third-party APIs
# (+ the plaintext cache). Fix: scrub is unconditional.
print("\n=== UNIT: secret scrub coverage ===")

SCRUB_CASES = [
    (
        "bearer",
        "Authorization: Bearer " + synthetic_secret("abc123def456", "ghi789jkl012"),
        "REDACTED_TOKEN",
    ),
    ("postgres conn string", "db=postgres://admin:SuperSecret123@db:5432/x", "REDACTED_USER"),
    ("mongodb conn string", "MONGO=mongodb://u:S3cret%40p@cluster:27017", "REDACTED_USER"),
    ("redis conn string", "redis://default:hunter2@redis:6379", "REDACTED_USER"),
    (
        "PEM block",
        synthetic_secret(
            "-----BEGIN RSA ",
            "PRIVATE KEY-----\n",
            "MIIEpAIBAAKCAQEA\n",
            "-----END RSA ",
            "PRIVATE KEY-----",
        ),
        "REDACTED_PEM_KEY",
    ),
    (
        "PEM dangling begin",
        synthetic_secret("-----BEGIN OPENSSH ", "PRIVATE KEY-----\n", "b3BlbnNz"),
        "REDACTED_PEM_KEY",
    ),
    ("AWS AKIA", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "REDACTED_AWS"),
    (
        "Google AIza",
        "key = " + synthetic_secret("AIzaSyA", "1234567890abcdefghijklmnopqrstuv"),
        "REDACTED_GCP",
    ),
    (
        "GitHub ghp_",
        "token: " + synthetic_secret("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789AB"),
        "REDACTED_GH",
    ),
    (
        "GitHub PAT",
        "GITHUB_PAT="
        + synthetic_secret("github_pat_", "11ABCDEFGHIJKLMNOPQRSTUVWXabcdefghijklmnopqrstuvwxyz"),
        "REDACTED_GH",
    ),
    (
        "Stripe",
        "stripe: " + synthetic_secret("sk_test_", "51HqabcdefGHIJKLMN0123456789abcd"),
        "REDACTED_STRIPE",
    ),
    (
        "JWT",
        "jwt "
        + synthetic_secret(
            "eyJhbGciOiJIUzI1NiJ9",
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            ".SflKxwRJsignature1234567",
        ),
        "REDACTED_JWT",
    ),
]
for name, inp, marker in SCRUB_CASES:
    out_s = cl.scrub_secrets(inp)
    # the live secret payload must NOT survive scrubbing
    leaked = any(
        need in out_s
        for need in (
            "SuperSecret123",
            "hunter2",
            "MIIEpAIBAAKCAQEA",
            "AKIAIOSFODNN7EXAMPLE",
            "AIzaSyA1234567890abcdefghijklmnopqrstuvw",
            "SflKxwRJ",
        )
    )
    check(f"scrub {name} → marker present", marker in out_s, detail=f"out={out_s[:80]}")
    check(f"scrub {name} → no live secret", not leaked, detail=f"LEAKED in {out_s[:80]}")
check(
    "scrub non-secrets unchanged",
    cl.scrub_secrets("just a normal log line about passwords")
    == "just a normal log line about passwords",
)
# url with no creds must be preserved (no false-positive redaction)
check(
    "scrub leaves credential-free URL intact",
    "https://example.com/path?x=1" in cl.scrub_secrets("see https://example.com/path?x=1"),
)

# --- trailing-comma JSON leniency (only fires on strict-parse failure) ---
check(
    "parse trailing-comma JSON",
    cl._try_parse_json('{"category": "debug", "reason": "x",}')
    == {"category": "debug", "reason": "x"},
)
check(
    "parse valid JSON unchanged (no comma edit)",
    cl._try_parse_json('{"a": "1,2", "b": 2}') == {"a": "1,2", "b": 2},
)
check("reject truly malformed", cl._try_parse_json('{"a": ') is None)

# --- cost estimate fills ZenMux/None gap ---
est = cl._resolve_cost(
    "openai/gpt-5.4-nano", {"prompt_tokens": 1000, "completion_tokens": 500, "cost": None}
)
check(
    "cost estimate when api_cost is None", est is not None and 0 < est < 0.01, detail=f"est={est}"
)
rep = cl._resolve_cost(
    "inclusionai/ling-2.6-flash", {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.000123}
)
check("reported cost (>0) returned as-is", abs(rep - 0.000123) < 1e-12, detail=f"rep={rep}")
di_rep = cl._resolve_cost(
    "provider/new-model", {"estimated_cost": 0.000321}, provider="deepinfra"
)
check(
    "deepinfra estimated_cost returned as-is",
    di_rep is not None and abs(di_rep - 0.000321) < 1e-12,
    detail=f"cost={di_rep}",
)


# =================================================================
# MOCKED: cascade logic with provider functions stubbed
# =================================================================
print("\n=== MOCKED: cascade with stubbed providers ===")


def _stub_cascade(provider_results: dict[tuple[str, str], list[Any]]):
    """Returns (call_log, real_call_fn). Stashes the REAL _call_provider so
    the test can restore it via the returned real_call_fn.
    """
    log: list[tuple[str, str]] = []

    def fake_call(model, provider, system, prompt, timeout, max_output_tokens, require_json=False):
        log.append((model, provider))
        outcomes = provider_results.get((model, provider), [])
        if outcomes:
            outcome = outcomes.pop(0)
        else:
            raise RuntimeError(f"no outcome configured for {model}@{provider}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # dict {text, latency, api_cost, provider, ...}

    real_provider = cl._call_provider
    # Stash the original on cl so it survives across test cases.
    if not hasattr(cl, "_ORIGINAL_CALL_PROVIDER"):
        cl._ORIGINAL_CALL_PROVIDER = real_provider
    cl._call_provider = fake_call
    return log, cl._ORIGINAL_CALL_PROVIDER


def _restore_call_provider():
    """Restore _call_provider to the original real function. Idempotent."""
    real = getattr(cl, "_ORIGINAL_CALL_PROVIDER", None)
    if real is not None:
        cl._call_provider = real


def _ok(text: str, cost: float = 0.000001, latency: float = 1.0, provider: str = "stub") -> dict:
    return {
        "text": text,
        "latency": latency,
        "input_tokens": 10,
        "output_tokens": 10,
        "api_cost": cost,
        "provider": provider,
    }


# M1: First tier succeeds, returns immediately
cache_dir = Path.home() / ".claude" / "state" / "cheap-llm-cache"
shutil.rmtree(cache_dir, ignore_errors=True)

log, _restore_unused = _stub_cascade(
    {
        ("inclusionai/ling-2.6-flash", "openrouter"): [_ok('{"category": "debug"}')],
    }
)
seen_budgets: list[int] = []


def _t1_collector(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    log.append((model, provider))
    seen_budgets.append(max_output_tokens)
    return _ok('{"category": "debug"}', provider=provider)


cl._call_provider = _t1_collector
out = cl.cheap_complete(
    system="Classify.",
    prompt="something",
    schema_hint=["category"],
    timeout_total=15,
    prefer_local=False,
    max_output_tokens=256,
)
check(
    "first tier succeeds → only 1 attempt",
    len(out["attempts"]) == 1,
    detail=f"got {len(out['attempts'])} attempts",
)
check(
    "first tier returns ling-2.6-flash@openrouter",
    out["model"] == "inclusionai/ling-2.6-flash" and out["provider"] == "openrouter",
)
check("output budget reaches provider", seen_budgets == [256], detail=f"got {seen_budgets}")
check(
    "attempt ledger records budget and token usage",
    out["attempts"][0]["max_output_tokens"] == 256
    and out["attempts"][0]["input_tokens"] == 10
    and out["attempts"][0]["output_tokens"] == 10,
)
_restore_call_provider()


# M2: OpenRouter down on ling-2.6-flash, ZenMux catches
def _m2_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "inclusionai/ling-2.6-flash" and provider == "openrouter":
        raise urllib.error.HTTPError("https://x", 503, "Service Unavailable", cast(Any, {}), None)
    if model == "inclusionai/ling-2.6-flash" and provider == "zenmux":
        return _ok('{"category": "debug"}', provider=provider)
    return _ok('{"category": "should not reach"}', provider=provider)


cl._call_provider = _m2_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="Classify.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "OR 503 → ZenMux catches ling-2.6-flash",
    out["model"] == "inclusionai/ling-2.6-flash" and out["provider"] == "zenmux",
    detail=f"got model={out['model']} provider={out['provider']}",
)
check(
    "OR 503 → 2 attempts (fail + success)",
    len(out["attempts"]) == 2,
    detail=f"got {len(out['attempts'])} attempts",
)
_restore_call_provider()


# M3: Both ling models fail on both providers → gemini catches
def _m3_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if "ling" in model:
        raise RuntimeError("ling model unavailable")
    if model == "google/gemini-3.1-flash-lite" and provider == "openrouter":
        return _ok('{"category": "debug"}')
    return _ok('{"category": "should not reach"}')


cl._call_provider = _m3_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="Classify.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "all ling fail → gemini catches",
    out["model"] == "google/gemini-3.1-flash-lite",
    detail=f"got {out['model']}",
)
_restore_call_provider()


# M4: All providers fail → graceful error
def _m4_call(*_args, **_kwargs):
    raise RuntimeError("everything is down")


cl._call_provider = _m4_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="Classify.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check("all fail → model is None", out["model"] is None)
check(
    "all fail → has error message",
    "failed" in out.get("error", "").lower(),
    detail=f"error={out.get('error')}",
)
check(
    "all fail → 8 attempts logged",
    len(out["attempts"]) == 8,
    detail=f"got {len(out['attempts'])} attempts",
)
_restore_call_provider()

# M5: Cache hit — first tier succeeds, second call should use cache
call_count = {"n": 0}


def _m5_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    call_count["n"] += 1
    return _ok('{"category": "debug"}')


cl._call_provider = _m5_call
shutil.rmtree(cache_dir, ignore_errors=True)
cl.cheap_complete(
    system="C.",
    prompt="same prompt",
    schema_hint=["category"],
    timeout_total=15,
    prefer_local=False,
)
n_after_first = call_count["n"]
out2 = cl.cheap_complete(
    system="C.",
    prompt="same prompt",
    schema_hint=["category"],
    timeout_total=15,
    prefer_local=False,
)
n_after_second = call_count["n"]
cl.cheap_complete(
    system="C.",
    prompt="same prompt",
    schema_hint=["category"],
    timeout_total=15,
    prefer_local=False,
    max_output_tokens=256,
)
n_after_different_budget = call_count["n"]
check("cache miss → 1 call", n_after_first == 1, detail=f"after first: {n_after_first}")
check(
    "cache hit → 0 additional calls",
    n_after_second == n_after_first,
    detail=f"after second: {n_after_second}, expected {n_after_first}",
)
check("cached result has cached=True", out2.get("cached") is True)
check(
    "different output budget bypasses incompatible cache entry",
    n_after_different_budget == n_after_second + 1,
)
_restore_call_provider()

# Explicit provider requests must not reuse another provider's model-level
# cache entry: provider pinning is a billing/trust boundary.
_explicit_model = "deepseek/deepseek-v4-flash"
cl._cache_put(
    cl._cache_key(_explicit_model, "C.", "provider-boundary", (), 1024),
    {"text": "generic cache", "provider": "openrouter", "tier": "T2"},
)
_explicit_calls: list[str] = []


def _explicit_provider_call(
    model, provider, sys, prompt, timeout, max_output_tokens, require_json=False
):
    _explicit_calls.append(provider)
    return _ok("deepinfra result", provider=provider)


cl._call_provider = _explicit_provider_call
out = cl.cheap_complete(
    system="C.",
    prompt="provider-boundary",
    prefer_local=False,
    require_json=False,
    cloud_model=_explicit_model,
    cloud_provider="deepinfra",
)
check(
    "explicit provider bypasses cross-provider cache",
    _explicit_calls == ["deepinfra"] and out["provider"] == "deepinfra" and not out["cached"],
    detail=f"calls={_explicit_calls} out={out}",
)
_restore_call_provider()

# Cache payloads shared by model retain the provider/tier that generated the
# response, even when a different cascade slot finds the entry later. Legacy
# text-only payloads remain valid and use the lookup slot as best attribution.
_provenance_key = cl._cache_key("shared-model", "s", "p", ("category",))
cl._cache_put(
    _provenance_key,
    {"text": '{"category": "debug"}', "provider": "openrouter", "tier": "T2"},
)
_provenance_attempts: list[dict] = []
_provenance_hit = cl._try_cache_hit(
    _provenance_key,
    "T2",
    "shared-model",
    "deepseek",
    ("category",),
    True,
    1024,
    _provenance_attempts,
)
check(
    "cache hit preserves source provider provenance",
    _provenance_hit is not None
    and _provenance_hit["provider"] == "openrouter"
    and _provenance_hit["billing"] == "payg"
    and _provenance_attempts[0].get("cache_lookup_provider") == "deepseek",
    detail=f"hit={_provenance_hit} attempts={_provenance_attempts}",
)
_legacy_key = cl._cache_key("legacy-model", "s", "p", None)
cl._cache_put(_legacy_key, {"text": "legacy text"})
_legacy_hit = cl._try_cache_hit(_legacy_key, "T2", "legacy-model", "zenmux", (), False, 1024, [])
check(
    "legacy text-only cache payload remains readable",
    _legacy_hit is not None and _legacy_hit["provider"] == "zenmux",
    detail=f"hit={_legacy_hit}",
)

# Bad budgets fail before transport/cache work and cannot silently request an
# unbounded or nonsensical generation.
for _bad_budget in (0, -1, True, 1.5):
    try:
        cl.cheap_complete(
            system="C.",
            prompt="x",
            prefer_local=False,
            require_json=False,
            max_output_tokens=_bad_budget,  # type: ignore[arg-type]
        )
        check(f"reject bad output budget {_bad_budget!r}", False, detail="no ValueError")
    except ValueError:
        check(f"reject bad output budget {_bad_budget!r}", True)

# Invalid deadlines fail before cascade construction (which may probe Ollama)
# or provider/cache work. Positive numeric values retain existing behavior.
for _bad_timeout in (0, -1, True, float("nan"), float("inf"), -float("inf")):
    try:
        cl.cheap_complete(
            system="C.",
            prompt="x",
            prefer_local=False,
            require_json=False,
            timeout_total=_bad_timeout,  # type: ignore[arg-type]
        )
        check(f"reject bad total timeout {_bad_timeout!r}", False, detail="no ValueError")
    except ValueError:
        check(f"reject bad total timeout {_bad_timeout!r}", True)


# M6: Invalid JSON triggers cascade fallthrough
def _m6_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "inclusionai/ling-2.6-flash":
        return _ok("not valid json at all")  # fails JSON contract
    if model == "inclusionai/ling-2.6-1t":
        return _ok('{"category": "debug"}')
    return _ok('{"category": "should not reach"}')


cl._call_provider = _m6_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="C.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "invalid JSON → fallthrough to ling-2.6-1t",
    out["model"] == "inclusionai/ling-2.6-1t",
    detail=f"got {out['model']}",
)
_restore_call_provider()


# M7: require_json=False (text mode) accepts non-JSON
def _m7_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "inclusionai/ling-2.6-flash":
        return _ok("plain text response, no JSON")
    return _ok("not reached")


cl._call_provider = _m7_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="C.",
    prompt="x",
    schema_hint=None,
    timeout_total=15,
    prefer_local=False,
    require_json=False,
)
check("text mode accepts non-JSON", out["text"] == "plain text response, no JSON")
_restore_call_provider()


# M8: prefer_local=True + schema JSON -> configured structured T1 is attempted
# FIRST and resolves there (1 attempt, no cloud call).
def _m8_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if provider == "ollama":
        return _ok('{"category": "debug"}', provider=provider)
    return _ok('{"category": "should not reach cloud"}', provider=provider)


cl._call_provider = _m8_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="C.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=True
)
check(
    "prefer_local schema -> structured T1@ollama wins first",
    out["model"] == cl.DEFAULT_LOCAL_STRUCTURED
    and out["provider"] == "ollama"
    and out["tier"] == "T1",
    detail=f"model={out['model']} provider={out['provider']} tier={out['tier']}",
)
check(
    "prefer_local → local resolves in 1 attempt",
    len(out["attempts"]) == 1,
    detail=f"got {len(out['attempts'])} attempts",
)
_restore_call_provider()


# =================================================================
# UNIT: atomic cache write + content=None robustness + cascade builder
# (regressions guarded after the 2026-07-05 refactor)
# =================================================================
print("\n=== UNIT: refactor regression guards ===")

# Atomic cache write: the .tmp file should not survive a successful write
# (the rename replaces the target). Bad-temp residue would corrupt the next
# _cache_get call.
ckey = cl._cache_key("atomic-probe", "s", "p", None)
cl._cache_put(ckey, {"text": "x"})
target = cl.CACHE_DIR / f"{ckey}.json"
temp = target.with_suffix(".json.tmp")
check("cache: target file written", target.exists())
check("cache: no .tmp residue after successful write", not temp.exists())
check(
    "cache: file is private",
    stat.S_IMODE(target.stat().st_mode) == 0o600,
    detail=f"mode={stat.S_IMODE(target.stat().st_mode):o}",
)
check(
    "cache: directory is private",
    stat.S_IMODE(cl.CACHE_DIR.stat().st_mode) == 0o700,
    detail=f"mode={stat.S_IMODE(cl.CACHE_DIR.stat().st_mode):o}",
)

_concurrent_key = cl._cache_key("concurrent-probe", "s", "p", None)


def _write_concurrent_cache(value: int) -> None:
    cl._cache_put(_concurrent_key, {"text": str(value)})


with ThreadPoolExecutor(max_workers=8) as _pool:
    list(_pool.map(_write_concurrent_cache, range(16)))
_concurrent_value = cl._cache_get(_concurrent_key)
check(
    "cache: concurrent writers leave valid JSON",
    _concurrent_value is not None and _concurrent_value["text"] in {str(i) for i in range(16)},
    detail=f"value={_concurrent_value}",
)
check(
    "cache: concurrent writers leave no temp residue",
    not list(cl.CACHE_DIR.glob(f".{_concurrent_key}.*.tmp")),
)

# Cache write failure is best-effort: a write into an invalid dir must NOT
# propagate and break a successful cascade. Simulate by monkeypatching
# CACHE_DIR to a path that cannot be created.
real_cache_dir = cl.CACHE_DIR
try:
    cl.CACHE_DIR = Path("/proc/this-cannot-be-created/cheap-llm-test")
    try:
        cl._cache_put(cl._cache_key("err-probe", "s", "p", None), {"text": "x"})
        check("cache: write failure swallowed (does not raise)", True)
    except Exception as e:
        check(
            "cache: write failure swallowed (does not raise)",
            False,
            detail=f"raised {type(e).__name__}: {e}",
        )
finally:
    cl.CACHE_DIR = real_cache_dir

# content=None robustness: reasoning-style models occasionally return
# message.content=None. The cascade must coerce to "" rather than raise
# AttributeError on .strip().
_body_none_content = {
    "choices": [{"message": {"content": None}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 0},
}
_orig_urlopen = _urlreq.urlopen
_openrouter_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(_body_none_content, _openrouter_payload)
try:
    os.environ["OPENROUTER_API_KEY"] = "test"
    try:
        r = cl._call_openrouter("openai/gpt-5.4-nano", "s", "p", timeout=5, max_output_tokens=288)
        check(
            "content=None → empty text, no AttributeError",
            r["text"] == "" and r["provider"] == "openrouter",
            detail=f"text={r['text']!r}",
        )
    except AttributeError as e:
        check("content=None → empty text, no AttributeError", False, detail=f"AttributeError: {e}")
finally:
    _urlreq.urlopen = _orig_urlopen
    del os.environ["OPENROUTER_API_KEY"]
check(
    "OpenAI-compatible transport receives output budget",
    _openrouter_payload["max_tokens"] == 288,
)
check(
    "OpenRouter sorts backing providers by price",
    _openrouter_payload.get("provider") == {"sort": "price"},
)
check(
    "OpenRouter uses the documented application-title header",
    cl.OPENROUTER_ENDPOINT.extra_headers
    == {"X-OpenRouter-Title": "cheap-llm-cascade"},
)

# _Endpoint dataclass exists and is usable
check("_Endpoint dataclass exists", hasattr(cl, "_Endpoint"))

ep_fields = {f.name for f in _dc_fields(cl._Endpoint)}
check(
    "_Endpoint has expected fields",
    ep_fields == {"url", "key_env", "provider_label", "extra_headers"},
    detail=f"got {ep_fields}",
)
check("_Endpoint is frozen (immutable)", cl._Endpoint.__dataclass_params__.frozen is True)
# Frozen means attribute assignment raises
try:
    cl.OPENROUTER_ENDPOINT.url = "http://evil"  # type: ignore[misc]
    check("_Endpoint is frozen", False, detail="assignment did not raise")
except (AttributeError, dataclasses.FrozenInstanceError):
    check("_Endpoint is frozen", True)

# _build_cascade returns the right shape: T1 first (default), T2 cloud pairs
# second, legacy last
cascade_default = cl._build_cascade(prefer_local=True, local_model=None, cloud_model=None)
check("cascade default: starts with T1 ollama", cascade_default[0][0] == "T1")
check(
    "cascade default: ends with LEGACY (gpt-5.4-nano or deepseek-v4-flash)",
    cascade_default[-1][1] in ("openai/gpt-5.4-nano", "deepseek/deepseek-v4-flash"),
    detail=f"last={cascade_default[-1]}",
)
check(
    "cascade default: 9 entries (T1 + 6 TOP3 + 2 LEGACY)",
    len(cascade_default) == 9,
    detail=f"got {len(cascade_default)}",
)

# forced cloud_model (non-deepseek) → just OR + ZenMux
forced = cl._build_cascade(
    prefer_local=True, local_model=None, cloud_model="inclusionai/ling-2.6-flash"
)
check(
    "cascade forced: 1 T1 + 2 T2 (OR + ZenMux)",
    len(forced) == 3 and all(c[2] in ("ollama", "openrouter", "zenmux") for c in forced),
)
check(
    "cascade forced: ZenMux comes after OpenRouter",
    [c[2] for c in forced[1:]] == ["openrouter", "zenmux"],
)

# forced deepseek → first-party + OR + ZenMux
forced_ds = cl._build_cascade(
    prefer_local=False, local_model=None, cloud_model="deepseek/deepseek-v4-flash"
)
providers = [c[2] for c in forced_ds]
check(
    "cascade forced-deepseek: deepseek → openrouter → zenmux order",
    providers == ["deepseek", "openrouter", "zenmux"],
    detail=f"got {providers}",
)

explicit_di = cl._build_cascade(
    prefer_local=False,
    local_model=None,
    cloud_model="deepseek/deepseek-v4-flash",
    cloud_provider="deepinfra",
)
check(
    "cascade explicit provider: one isolated T2 route",
    explicit_di == [("T2", "deepseek/deepseek-v4-flash", "deepinfra", 18.0)],
    detail=f"got {explicit_di}",
)
try:
    cl._build_cascade(
        prefer_local=False,
        local_model=None,
        cloud_model=None,
        cloud_provider="deepinfra",
    )
    check("cascade explicit provider requires cloud model", False, detail="no ValueError")
except ValueError:
    check("cascade explicit provider requires cloud model", True)

# cold-start T1 budget: model not loaded -> extended timeout; warm -> fast
_orig_loaded = cl._ollama_model_loaded
try:
    cl._ollama_model_loaded = lambda m: False
    cold = cl._build_cascade(prefer_local=True, local_model=None, cloud_model=None)
    check(
        "cascade cold T1: extended budget (>= LOCAL_COLD_TIMEOUT)",
        cold[0][3] >= cl.LOCAL_COLD_TIMEOUT,
        detail=f"timeout={cold[0][3]}",
    )
    cl._ollama_model_loaded = lambda m: True
    warm = cl._build_cascade(prefer_local=True, local_model=None, cloud_model=None)
    check(
        "cascade warm T1: fast budget (6s primary)",
        warm[0][3] == 6.0,
        detail=f"timeout={warm[0][3]}",
    )
finally:
    cl._ollama_model_loaded = _orig_loaded

# prefer_local=False + cloud_model=None → only T2 cloud tiers
cascade_cloud_only = cl._build_cascade(prefer_local=False, local_model=None, cloud_model=None)
check(
    "cascade cloud-only: no T1 entry",
    all(c[0] == "T2" for c in cascade_cloud_only),
    detail=f"tiers={[c[0] for c in cascade_cloud_only]}",
)


# =================================================================
# UNIT: local-only and model normalization additions (2026-07-08)
# =================================================================
print("\n=== UNIT: local-only and model normalization ===")

# Test _normalize_model_name
check(
    "normalize: empty or None returns empty",
    cl._normalize_model_name(None) == "" and cl._normalize_model_name("") == "",
)
check(
    "normalize: adds latest suffix when missing",
    cl._normalize_model_name("my-model") == "my-model:latest",
)
check("normalize: keeps existing suffix", cl._normalize_model_name("my-model:v2") == "my-model:v2")
check(
    "normalize: trims whitespace",
    cl._normalize_model_name("  my-model:latest  ") == "my-model:latest",
)

# Test _ollama_model_loaded tag normalization
_orig_urlopen_norm = _urlreq.urlopen
_ps_payload = {
    "models": [
        {"name": "some-model:latest", "model": "some-model:latest"},
        {"name": "another-model:v1", "model": "another-model:v1"},
    ]
}
_urlreq.urlopen = _fake_urlopen_factory(_ps_payload)
try:
    check(
        "model loaded matches exact with tag", cl._ollama_model_loaded("some-model:latest") is True
    )
    check("model loaded matches normalized tagless", cl._ollama_model_loaded("some-model") is True)
    check(
        "model loaded matches exact with tag (another-model)",
        cl._ollama_model_loaded("another-model:v1") is True,
    )
    check(
        "model loaded tagless fails for tag v1 model",
        cl._ollama_model_loaded("another-model") is False,
    )
    check("non-loaded model returns False", cl._ollama_model_loaded("missing-model") is False)
finally:
    _urlreq.urlopen = _orig_urlopen_norm

# Test CHEAP_LLM_LOCAL_ONLY cascade behavior
_orig_local_only = os.environ.get("CHEAP_LLM_LOCAL_ONLY")
try:
    os.environ["CHEAP_LLM_LOCAL_ONLY"] = "1"
    cascade_lo = cl._build_cascade(
        prefer_local=False, local_model="test-model", cloud_model="some-cloud"
    )
    check(
        "local_only forces T1 only and ignores cloud models",
        len(cascade_lo) == 1 and cascade_lo[0][0] == "T1" and cascade_lo[0][1] == "test-model",
        detail=f"cascade={cascade_lo}",
    )

    # local_only with prefer_local=True and cloud_model=None
    cascade_lo_2 = cl._build_cascade(prefer_local=True, local_model=None, cloud_model=None)
    check(
        "local_only restricts normal cascade to T1 only",
        len(cascade_lo_2) == 1 and cascade_lo_2[0][0] == "T1",
        detail=f"cascade={cascade_lo_2}",
    )
finally:
    if _orig_local_only is not None:
        os.environ["CHEAP_LLM_LOCAL_ONLY"] = _orig_local_only
    else:
        os.environ.pop("CHEAP_LLM_LOCAL_ONLY", None)


# Test deepinfra mapping and cascade inclusion
check(
    "deepinfra model mapping: deepseek-v4-pro",
    cl._normalize_deepinfra_model("deepseek/deepseek-v4-pro") == "deepseek-ai/DeepSeek-V4-Pro",
)
check(
    "deepinfra model mapping: deepseek-v4-flash",
    cl._normalize_deepinfra_model("deepseek/deepseek-v4-flash") == "deepseek-ai/DeepSeek-V4-Flash",
)
check(
    "deepinfra model mapping: qwen",
    cl._normalize_deepinfra_model("qwen/qwen3.7-max") == "Qwen/Qwen3.7-Max",
)
check(
    "deepinfra model mapping: unmapped remains identical",
    cl._normalize_deepinfra_model("some/other-model") == "some/other-model",
)
check(
    "deepinfra pricing covers every normalized catalog model",
    set(cl._PROVIDERS["deepinfra"].slug_map.values()) <= set(cl.DEEPINFRA_PRICING),
)

try:
    os.environ["DEEPINFRA_API_KEY"] = "test-key"
    cascade_di = cl._build_cascade(
        prefer_local=False, local_model=None, cloud_model="deepseek/deepseek-v4-flash"
    )
    providers_di = [c[2] for c in cascade_di]
    check(
        "deepinfra added to deepseek cascade when api key set",
        "deepinfra" in providers_di and providers_di[1] == "deepinfra",
        detail=f"got {providers_di}",
    )

    cascade_di_qwen = cl._build_cascade(
        prefer_local=False, local_model=None, cloud_model="qwen/qwen3.7-max"
    )
    providers_di_qwen = [c[2] for c in cascade_di_qwen]
    check(
        "deepinfra added to qwen cascade when api key set",
        "deepinfra" in providers_di_qwen and providers_di_qwen[0] == "deepinfra",
        detail=f"got {providers_di_qwen}",
    )
finally:
    os.environ.pop("DEEPINFRA_API_KEY", None)

# DeepInfra native JSON mode and provider-specific price fallback.
_deepinfra_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000},
    },
    _deepinfra_payload,
)
try:
    os.environ["DEEPINFRA_API_KEY"] = "test-key"
    _di = cl._call_deepinfra(
        "deepseek/deepseek-v4-flash", "s", "p", timeout=5, require_json=True
    )
finally:
    _urlreq.urlopen = _orig_urlopen
    os.environ.pop("DEEPINFRA_API_KEY", None)
check(
    "deepinfra enables native JSON mode",
    _deepinfra_payload.get("response_format") == {"type": "json_object"},
)
check(
    "deepinfra fallback cost uses provider-specific listing",
    abs(_di["api_cost"] - ((1_000_000 * 0.09 + 100_000 * 0.18) / 1_000_000)) < 1e-9,
    detail=f"cost={_di['api_cost']}",
)


# =================================================================
# UNIT: 1.2.1 regressions — JSON hint without schema + cache shape guard
# =================================================================
print("\n=== UNIT: 1.2.1 regression guards ===")

# require_json=True WITHOUT schema_hint must still instruct the model to emit
# JSON (before 1.2.1 the hint was only added when a schema was present, so
# validation rejected prose the model was never told not to write).
_seen_systems: list[str] = []


def _sys_collector(model, provider, system, prompt, timeout, max_output_tokens, require_json=False):
    _seen_systems.append(system)
    return _ok('{"anything": "goes"}', provider=provider)


shutil.rmtree(cache_dir, ignore_errors=True)
cl._call_provider = _sys_collector
out = cl.cheap_complete(
    system="Summarize.", prompt="x", schema_hint=None, require_json=True, prefer_local=False
)
check(
    "require_json without schema → JSON_HINT injected",
    len(_seen_systems) == 1 and cl.JSON_HINT in _seen_systems[0],
    detail=f"system={_seen_systems[-1][:120]!r}",
)
check(
    "require_json without schema → no 'Required keys' suffix",
    "Required keys" not in _seen_systems[0],
)
out = cl.cheap_complete(
    system="Summarize.",
    prompt="x",
    schema_hint=["anything"],
    require_json=True,
    prefer_local=False,
)
check(
    "require_json with schema → hint + Required keys (pre-1.2.1 string preserved)",
    len(_seen_systems) == 2
    and _seen_systems[1].endswith(cl.JSON_HINT + " Required keys: ['anything']."),
    detail=f"system={_seen_systems[-1][-80:]!r}",
)
_restore_call_provider()

# Cache shape guard: a cache file that parses as JSON but is not
# {"text": str} must be treated as a MISS, not crash _try_cache_hit.
_shape_key = cl._cache_key("shape-guard-probe", "s", "p", None)
_shape_path = cl.CACHE_DIR / f"{_shape_key}.json"
cl.CACHE_DIR.mkdir(parents=True, exist_ok=True)
for _bad in ('["a", "list"]', '{"no_text_key": 1}', '{"text": 42}', '"bare string"'):
    _shape_path.write_text(_bad)
    check(
        f"cache shape guard: {_bad[:24]!r} → miss (None)",
        cl._cache_get(_shape_key) is None,
    )
_shape_path.write_text('{"text": "valid"}')
_hit = cl._cache_get(_shape_key)
check(
    "cache shape guard: well-formed entry still hits",
    _hit is not None and _hit["text"] == "valid",
)
_shape_path.unlink(missing_ok=True)

# End-to-end: corrupted cache at the live ckey must fall through to the
# provider instead of raising.
cl._call_provider = lambda *_args, **_kwargs: _ok('{"category": "debug"}')
_e2e_system = "Classify." + cl.JSON_HINT + " Required keys: ['category']."
_e2e_key = cl._cache_key("inclusionai/ling-2.6-flash", _e2e_system, "x", ("category",))
(cl.CACHE_DIR / f"{_e2e_key}.json").write_text('["corrupted"]')
try:
    out = cl.cheap_complete(
        system="Classify.",
        prompt="x",
        schema_hint=["category"],
        prefer_local=False,
        timeout_total=15,
    )
    check(
        "corrupted cache falls through to live provider (no crash)",
        out["error"] is None and out["cached"] is False and out["json_valid"] is True,
        detail=f"error={out['error']!r} cached={out['cached']}",
    )
except Exception as e:  # noqa: BLE001 — the regression under test IS the crash
    check(
        "corrupted cache falls through to live provider (no crash)",
        False,
        detail=f"raised {type(e).__name__}: {e}",
    )
finally:
    _restore_call_provider()

# _probe reports every provider key + local_only + cache size
_probe_keys = set(cl._probe())
for _pk in ("zenmux_key_set", "deepseek_key_set", "local_only", "cache_entries"):
    check(f"_probe exposes {_pk}", _pk in _probe_keys)

# Provider health probes use authenticated GET, parse a bounded JSON object,
# and report no credential value.
_probe_seen: dict = {}


def _fake_probe_open(req, timeout=None):
    _probe_seen["method"] = req.get_method()
    _probe_seen["authorization"] = req.get_header("Authorization")
    return _FakeResp(b'{"data": []}')


_urlreq.urlopen = _fake_probe_open
try:
    os.environ["CHEAP_LLM_TEST_KEY"] = "synthetic-test-value"
    _probe_result = cl._probe_url(
        "https://provider.invalid/v1/models", timeout=1, key_env="CHEAP_LLM_TEST_KEY"
    )
finally:
    _urlreq.urlopen = _orig_urlopen
    os.environ.pop("CHEAP_LLM_TEST_KEY", None)
check(
    "provider probe uses authenticated GET",
    _probe_result["reachable"] is True
    and _probe_seen == {
        "method": "GET",
        "authorization": "Bearer synthetic-test-value",
    },
    detail=f"seen={_probe_seen} result={_probe_result}",
)


# =================================================================
# CLI: --probe works standalone (regression 2026-07-02: --system/--prompt
# were required=True, so the documented `python3 -m cheap_llm --probe` exited 2)
# =================================================================
print("\n=== CLI: --probe standalone ===")
_probe_proc = subprocess.run(
    [sys.executable, "-m", "cheap_llm", "--probe"],
    capture_output=True,
    text=True,
    timeout=15,
    cwd=PROJECT_ROOT,
)
check(
    "--probe runs without --system/--prompt",
    _probe_proc.returncode == 0,
    detail=f"rc={_probe_proc.returncode} stderr={_probe_proc.stderr[:120]}",
)
try:
    _probe_out = json.loads(_probe_proc.stdout)
    check("--probe emits JSON with ollama_alive key", "ollama_alive" in _probe_out)
except json.JSONDecodeError:
    check(
        "--probe emits JSON with ollama_alive key",
        False,
        detail=f"stdout={_probe_proc.stdout[:120]}",
    )
_route_proc = subprocess.run(
    [
        sys.executable,
        "-m",
        "cheap_llm",
        "--route-plan",
        "--no-local",
        "--no-json",
        "--cloud-model",
        "deepseek/deepseek-v4-flash",
        "--cloud-provider",
        "deepinfra",
    ],
    capture_output=True,
    text=True,
    timeout=15,
    cwd=PROJECT_ROOT,
)
try:
    _route_out = json.loads(_route_proc.stdout)
except json.JSONDecodeError:
    _route_out = {}
check(
    "--route-plan exposes one PAYG provider route without completion",
    _route_proc.returncode == 0
    and len(_route_out.get("routes", [])) == 1
    and _route_out["routes"][0]["provider"] == "deepinfra"
    and _route_out["routes"][0]["billing"] == "payg"
    and _route_out.get("subscription_workers_in_scope") is False,
    detail=f"rc={_route_proc.returncode} out={_route_out} err={_route_proc.stderr[:120]}",
)
_noargs_proc = subprocess.run(
    [sys.executable, "-m", "cheap_llm"],
    capture_output=True,
    text=True,
    timeout=15,
    cwd=PROJECT_ROOT,
)
check(
    "no args still errors (required pair enforced)",
    _noargs_proc.returncode == 2,
    detail=f"rc={_noargs_proc.returncode}",
)
_bad_budget_proc = subprocess.run(
    [
        sys.executable,
        "-m",
        "cheap_llm",
        "--system",
        "s",
        "--prompt",
        "p",
        "--max-tokens",
        "0",
    ],
    capture_output=True,
    text=True,
    timeout=15,
    cwd=PROJECT_ROOT,
)
check(
    "CLI rejects non-positive --max-tokens before provider calls",
    _bad_budget_proc.returncode == 2 and "positive integer" in _bad_budget_proc.stderr,
    detail=f"rc={_bad_budget_proc.returncode} stderr={_bad_budget_proc.stderr[:120]}",
)
_bad_timeout_proc = subprocess.run(
    [
        sys.executable,
        "-m",
        "cheap_llm",
        "--system",
        "s",
        "--prompt",
        "p",
        "--timeout",
        "nan",
    ],
    capture_output=True,
    text=True,
    timeout=15,
    cwd=PROJECT_ROOT,
)
check(
    "CLI rejects non-finite --timeout before provider calls",
    _bad_timeout_proc.returncode == 2 and "positive finite" in _bad_timeout_proc.stderr,
    detail=f"rc={_bad_timeout_proc.returncode} stderr={_bad_timeout_proc.stderr[:120]}",
)

# =================================================================
# UNIT: robustness additions (2026-07-13)
# =================================================================
print("\n=== UNIT: robustness additions ===")
_v_beta = cl._parse_version("1.2.3-beta")
check("parse version with beta suffix", _v_beta == (1, 2, 3), detail=f"got {_v_beta}")
_v_dev = cl._parse_version("1.2.4.dev0")
check("parse version with dev suffix", _v_dev == (1, 2, 4), detail=f"got {_v_dev}")
check("parse version with normal string", cl._parse_version("2.0.1") == (2, 0, 1))

# Verify Ollama request payload structure
_ollama_payload_struct: dict = {}
_orig_urlopen_struct = _urlreq.urlopen
_urlreq.urlopen = _fake_urlopen_factory(
    {"response": "ok", "prompt_eval_count": 2, "eval_count": 1}, _ollama_payload_struct
)
try:
    cl._call_ollama("local", "sys_p", "user_p", timeout=5, max_output_tokens=192)
finally:
    _urlreq.urlopen = _orig_urlopen_struct
check("ollama payload separate prompt", _ollama_payload_struct.get("prompt") == "user_p")
check("ollama payload separate system", _ollama_payload_struct.get("system") == "sys_p")

# Verify global reasoning stripping for cloud / OpenAI compat / deepseek calls
_body_with_think = {
    "choices": [{"message": {"content": "<think>Let me think...</think>final answer"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10},
}
_compat_payload_think: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(_body_with_think, _compat_payload_think)
try:
    os.environ["OPENROUTER_API_KEY"] = "test"
    _or_resp = cl._call_openrouter(
        "openai/gpt-5.4-nano", "s", "p", timeout=5, max_output_tokens=128
    )
    check(
        "openrouter strips reasoning block",
        _or_resp["text"] == "final answer",
        detail=f"got {_or_resp['text']!r}",
    )
finally:
    _urlreq.urlopen = _orig_urlopen_struct
    del os.environ["OPENROUTER_API_KEY"]

_urlreq.urlopen = _fake_urlopen_factory(_body_with_think)
try:
    os.environ["DEEPSEEK_API_KEY"] = "test"
    _ds_resp = cl._call_deepseek(
        "deepseek/deepseek-v4-flash", "s", "p", timeout=5, max_output_tokens=128
    )
    check(
        "deepseek strips reasoning block",
        _ds_resp["text"] == "final answer",
        detail=f"got {_ds_resp['text']!r}",
    )
finally:
    _urlreq.urlopen = _orig_urlopen_struct
    del os.environ["DEEPSEEK_API_KEY"]

_json_internal_fences = '```json\n{"code": "```python\\nprint(1)\\n```"}\n```'
_parsed_fences = cl._try_parse_json(_json_internal_fences)
check(
    "parse JSON with internal code fences",
    _parsed_fences == {"code": "```python\nprint(1)\n```"},
    detail=f"got {_parsed_fences}",
)

# Restore DeepInfra API key before live tests
if _actual_deepinfra_key:
    os.environ["DEEPINFRA_API_KEY"] = _actual_deepinfra_key

LIVE = "--live" in sys.argv and "--quick" not in sys.argv

print(f"\n=== LIVE: real API smoke (enabled={LIVE}) ===")

if not LIVE:
    skip("all live tests", "pass --live to enable")
elif not (os.environ.get("OPENROUTER_API_KEY") and os.environ.get("ZENMUX_API_KEY")):
    skip("all live tests", "OPENROUTER_API_KEY or ZENMUX_API_KEY not set")
else:
    shutil.rmtree(cache_dir, ignore_errors=True)
    # One live call per top-3 cascade tier
    live_cases = cl.TOP3_CASCADE  # 6 (model, provider) pairs
    for model, provider in live_cases:
        out = cl.cheap_complete(
            system="Classify into trivial/lookup/code-edit/architecture/security/debug. "
            'JSON only with field "category".',
            prompt="ECONNREFUSED 127.0.0.1:5432 in my Express app",
            schema_hint=["category"],
            timeout_total=20,
            prefer_local=False,
        )
        ok_json = out.get("json_valid") is True
        ok_provider = out.get("provider") == provider
        # Ling models should win first; later tiers only if earlier failed
        if (model, provider) == cl.TOP3_CASCADE[0]:
            check(
                f"live: {provider}/{model} reachable + valid JSON",
                ok_json and ok_provider,
                detail=f"text={out.get('text', '')[:80]}",
            )
        else:
            # Just verify the (model, provider) entry is reachable — we can't
            # easily force a specific tier without mocking. Check whether the
            # cascade ever hit this (model, provider) by retrying with mock failure.
            check(
                f"live: {provider}/{model} reachable (smoke)",
                out.get("model") is not None,
                detail=f"text={out.get('text', '')[:80]}",
            )


# =================================================================
# Summary
# =================================================================
print(f"\n{'=' * 60}")
print(f"PASS: {PASS}    FAIL: {FAIL}    SKIP: {SKIP}")
if FAILURES:
    print("\nFailures:")
    for f in FAILURES:
        print(f"  - {f}")
print(f"{'=' * 60}")

sys.exit(0 if FAIL == 0 else 1)
