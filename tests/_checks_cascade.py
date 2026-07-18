from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

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
        ("deepseek/deepseek-v4-flash", "openrouter"): [_ok('{"category": "debug"}')],
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
    "first tier returns deepseek-v4-flash@openrouter",
    out["model"] == "deepseek/deepseek-v4-flash" and out["provider"] == "openrouter",
)
check("output budget reaches provider", seen_budgets == [256], detail=f"got {seen_budgets}")
check(
    "attempt ledger records budget and token usage",
    out["attempts"][0]["max_output_tokens"] == 256
    and out["attempts"][0]["input_tokens"] == 10
    and out["attempts"][0]["output_tokens"] == 10,
)
_restore_call_provider()


# M2: OpenRouter down on deepseek-v4-flash, ZenMux catches
def _m2_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "deepseek/deepseek-v4-flash" and provider == "openrouter":
        raise urllib.error.HTTPError("https://x", 503, "Service Unavailable", cast(Any, {}), None)
    if model == "deepseek/deepseek-v4-flash" and provider == "zenmux":
        return _ok('{"category": "debug"}', provider=provider)
    return _ok('{"category": "should not reach"}', provider=provider)


cl._call_provider = _m2_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="Classify.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "OR 503 → ZenMux catches deepseek-v4-flash",
    out["model"] == "deepseek/deepseek-v4-flash" and out["provider"] == "zenmux",
    detail=f"got model={out['model']} provider={out['provider']}",
)
check(
    "OR 503 → 2 attempts (fail + success)",
    len(out["attempts"]) == 2,
    detail=f"got {len(out['attempts'])} attempts",
)
_restore_call_provider()


# M3: DeepSeek fails on both providers → Gemini catches
def _m3_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "deepseek/deepseek-v4-flash":
        raise RuntimeError("deepseek model unavailable")
    if model == "google/gemini-3.1-flash-lite" and provider == "openrouter":
        return _ok('{"category": "debug"}')
    return _ok('{"category": "should not reach"}')


cl._call_provider = _m3_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="Classify.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "deepseek fails → gemini catches",
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
    "all fail → every configured route is logged",
    len(out["attempts"]) == len(cl._build_cascade(False, None, None)),
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
    if model == "deepseek/deepseek-v4-flash":
        return _ok("not valid json at all")  # fails JSON contract
    if model == "google/gemini-3.1-flash-lite":
        return _ok('{"category": "debug"}')
    return _ok('{"category": "should not reach"}')


cl._call_provider = _m6_call
shutil.rmtree(cache_dir, ignore_errors=True)
out = cl.cheap_complete(
    system="C.", prompt="x", schema_hint=["category"], timeout_total=15, prefer_local=False
)
check(
    "invalid JSON → fallthrough to gemini",
    out["model"] == "google/gemini-3.1-flash-lite",
    detail=f"got {out['model']}",
)
_restore_call_provider()


# M7: require_json=False (text mode) accepts non-JSON
def _m7_call(model, provider, sys, prompt, timeout, max_output_tokens, require_json=False):
    if model == "deepseek/deepseek-v4-flash":
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
