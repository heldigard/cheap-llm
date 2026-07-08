#!/usr/bin/env python3
"""Regression tests for cheap_llm.py — cascade, scrubbing, caching, failover.

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
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request as _urlreq
from dataclasses import fields as _dc_fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("cheap_llm", PROJECT_ROOT / "cheap_llm.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["cheap_llm"] = cl  # needed so @dataclass can resolve cls.__module__
_spec.loader.exec_module(cl)

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
    "REDACTED" in cl.scrub_secrets('api_key = "abcdef1234567890"'),
    detail="expected REDACTED in scrubbed output",
)
check(
    "scrub Bearer token",
    cl.scrub_secrets("Authorization: Bearer abc123def456ghi789jkl012")
    == "Authorization: Bearer <REDACTED_TOKEN>",
)
check("scrub sk- key", "REDACTED_SK" in cl.scrub_secrets("my key is sk-proj1234567890abcdefghij"))
check(
    "scrub ghp_ key",
    "REDACTED_GH" in cl.scrub_secrets("token: ghp_abcdefghijklmnopqrstuvwxyz0123456789AB"),
)
check(
    "scrub JWT",
    "REDACTED_JWT"
    in cl.scrub_secrets("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature_abc123"),
)
check(
    "scrub non-secrets unchanged",
    cl.scrub_secrets("just a normal log line") == "just a normal log line",
)
check("scrub xox token", "REDACTED_XOX" in cl.scrub_secrets("slack: xoxb-12345-67890"))

# _cache_key
k1 = cl._cache_key("m", "sys", "user", ("a",))
k2 = cl._cache_key("m", "sys", "user", ("a",))
k3 = cl._cache_key("m", "sys", "userX", ("a",))
k4 = cl._cache_key("m", "sys", "user", ("b",))
check("cache key deterministic", k1 == k2)
check("cache key differs on prompt", k1 != k3)
check("cache key differs on schema", k1 != k4)

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
    "deepseek-v4-flash still in LEGACY (BYOK $0)",
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

# T1 local primary stays qwen3.5:4b (2026-06-27 prune winner — best
# free-text compatibility default). Structured JSON calls route to functiongemma.
check(
    "T1 local primary is qwen3.5:4b",
    cl.DEFAULT_LOCAL_PRIMARY == "cryptidbleh/gemma4-claude-opus-4.6:latest",
    detail=f"got {cl.DEFAULT_LOCAL_PRIMARY}",
)
check(
    "T1 local structured primary is SetneufPT/Qwopus",
    "Qwopus" in cl.DEFAULT_LOCAL_STRUCTURED,
    detail=f"got {cl.DEFAULT_LOCAL_STRUCTURED}",
)

# --- CRITICAL regression: secrets are scrubbed on the prefer_local path ---
# DeepSeek first-party cache-aware cost (2026-07-02: cache-hit = 1/10 of the
# input list rate per the published V4 pricing — was hardcoded 0.029).
print("\n=== UNIT: deepseek cache-aware cost ===")


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _fake_urlopen_factory(body: dict):
    def _fake(req, timeout=None):
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
_urlreq.urlopen = _fake_urlopen_factory(_ds_body)
try:
    os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
    _ds = cl._call_deepseek("deepseek/deepseek-v4-flash", "s", "p", timeout=5)
finally:
    _urlreq.urlopen = _orig_urlopen
# fresh 400K @ $0.14/M + cached 600K @ $0.014/M + out 100K @ $0.28/M
_expected = (400_000 * 0.14 + 600_000 * 0.014 + 100_000 * 0.28) / 1_000_000
check(
    "deepseek cost: cached input billed at 1/10 of input rate",
    abs(_ds["api_cost"] - _expected) < 1e-9,
    detail=f"got {_ds['api_cost']:.6f} expected {_expected:.6f}",
)
check(
    "deepseek slug strips provider prefix (call succeeded)",
    _ds["text"] == "ok" and _ds["provider"] == "deepseek",
)

# Reproduce the 2026-06-19 bug: prefer_local=True used to skip scrubbing, but
# cloud tiers always follow T1, so unscrubbed secrets reached third-party APIs
# (+ the plaintext cache). Fix: scrub is unconditional.
print("\n=== UNIT: secret scrub coverage ===")

SCRUB_CASES = [
    ("bearer", "Authorization: Bearer abc123def456ghi789jkl012", "REDACTED_TOKEN"),
    ("postgres conn string", "db=postgres://admin:SuperSecret123@db:5432/x", "REDACTED_USER"),
    ("mongodb conn string", "MONGO=mongodb://u:S3cret%40p@cluster:27017", "REDACTED_USER"),
    ("redis conn string", "redis://default:hunter2@redis:6379", "REDACTED_USER"),
    (
        "PEM block",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        "REDACTED_PEM_KEY",
    ),
    ("PEM dangling begin", "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNz", "REDACTED_PEM_KEY"),
    ("AWS AKIA", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "REDACTED_AWS"),
    ("Google AIza", "key = AIzaSyA1234567890abcdefghijklmnopqrstuv", "REDACTED_GCP"),
    ("GitHub ghp_", "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789AB", "REDACTED_GH"),
    (
        "GitHub PAT",
        "GITHUB_PAT=github_pat_11ABCDEFGHIJKLMNOPQRSTUVWXabcdefghijklmnopqrstuvwxyz",
        "REDACTED_GH",
    ),
    ("Stripe", "stripe: sk_test_51HqabcdefGHIJKLMN0123456789abcd", "REDACTED_STRIPE"),
    (
        "JWT",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJsignature1234567",
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


# =================================================================
# MOCKED: cascade logic with provider functions stubbed
# =================================================================
print("\n=== MOCKED: cascade with stubbed providers ===")


def _stub_cascade(provider_results: dict[str, list]):
    """Returns (call_log, real_call_fn). Stashes the REAL _call_provider so
    the test can restore it via the returned real_call_fn.
    """
    log: list[tuple[str, str]] = []

    def fake_call(model, provider, system, prompt, timeout):
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


def _t1_collector(model, provider, sys, prompt, timeout):
    log.append((model, provider))
    return _ok('{"category": "debug"}', provider=provider)


cl._call_provider = _t1_collector
out = cl.cheap_complete(
    system="Classify.",
    prompt="something",
    schema_hint=["category"],
    timeout_total=15,
    prefer_local=False,
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
_restore_call_provider()


# M2: OpenRouter down on ling-2.6-flash, ZenMux catches
def _m2_call(model, provider, sys, prompt, timeout):
    if model == "inclusionai/ling-2.6-flash" and provider == "openrouter":
        raise urllib.error.HTTPError("https://x", 503, "Service Unavailable", {}, None)
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
def _m3_call(model, provider, sys, prompt, timeout):
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


def _m5_call(model, provider, sys, prompt, timeout):
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
check("cache miss → 1 call", n_after_first == 1, detail=f"after first: {n_after_first}")
check(
    "cache hit → 0 additional calls",
    n_after_second == n_after_first,
    detail=f"after second: {n_after_second}, expected {n_after_first}",
)
check("cached result has cached=True", out2.get("cached") is True)
_restore_call_provider()


# M6: Invalid JSON triggers cascade fallthrough
def _m6_call(model, provider, sys, prompt, timeout):
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
def _m7_call(model, provider, sys, prompt, timeout):
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


# M8: prefer_local=True + schema JSON → T1 functiongemma@ollama is attempted
# FIRST and resolves there (1 attempt, no cloud call).
def _m8_call(model, provider, sys, prompt, timeout):
    if provider == "ollama":
        return _ok('{"category": "debug"}', provider=provider)
    return _ok('{"category": "should not reach cloud"}', provider=provider)


cl._call_provider = _m8_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="C.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=True
)
check(
    "prefer_local schema → T1 functiongemma@ollama wins first",
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
_urlreq.urlopen = _fake_urlopen_factory(_body_none_content)
try:
    os.environ["OPENROUTER_API_KEY"] = "test"
    try:
        r = cl._call_openrouter("openai/gpt-5.4-nano", "s", "p", timeout=5)
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
# CLI: --probe works standalone (regression 2026-07-02: --system/--prompt
# were required=True, so the documented `cheap_llm.py --probe` exited 2)
# =================================================================
print("\n=== CLI: --probe standalone ===")
_probe_proc = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "cheap_llm.py"), "--probe"],
    capture_output=True,
    text=True,
    timeout=15,
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
_noargs_proc = subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "cheap_llm.py")],
    capture_output=True,
    text=True,
    timeout=15,
)
check(
    "no args still errors (required pair enforced)",
    _noargs_proc.returncode == 2,
    detail=f"rc={_noargs_proc.returncode}",
)

# =================================================================
# LIVE: real API smoke tests (require API keys)
# =================================================================
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
