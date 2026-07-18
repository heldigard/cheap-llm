from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

print("\n=== UNIT: deepseek cache-aware cost ===")


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
_old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
try:
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
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
    if _old_deepseek_key is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = _old_deepseek_key
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
