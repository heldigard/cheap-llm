from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

from cheap_bench import calls as bench_calls

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
    "first entry is deepseek-v4-flash@openrouter",
    cl.TOP3_CASCADE[0] == ("deepseek/deepseek-v4-flash", "openrouter"),
)
check(
    "deepseek-v4-flash has zenmux failover",
    ("deepseek/deepseek-v4-flash", "zenmux") in cl.TOP3_CASCADE,
)
check(
    "gemini comes before ling-2.6-1t",
    [m for m, _ in cl.TOP3_CASCADE].index("google/gemini-3.1-flash-lite")
    < [m for m, _ in cl.TOP3_CASCADE].index("inclusionai/ling-2.6-1t"),
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
    "legacy cascade retains ling-2.6-flash provider failover",
    ("inclusionai/ling-2.6-flash", "openrouter") in cl.LEGACY_CASCADE
    and ("inclusionai/ling-2.6-flash", "zenmux") in cl.LEGACY_CASCADE,
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

# Benchmark transport keeps dynamic URLs inside explicit trust boundaries.
_old_ollama_url = os.environ.get("OLLAMA_URL")
try:
    os.environ["OLLAMA_URL"] = "file:///etc/passwd"
    check(
        "benchmark rejects non-HTTP Ollama endpoint",
        bench_calls._ollama_generate_url() == "http://localhost:11434/api/generate",
    )
    os.environ["OLLAMA_URL"] = "http://127.0.0.1:11500"
    check(
        "benchmark accepts loopback Ollama endpoint",
        bench_calls._ollama_generate_url() == "http://127.0.0.1:11500/api/generate",
    )
    os.environ["OLLAMA_URL"] = "http://127.0.0.2:11500"
    check(
        "benchmark accepts full IPv4 loopback range",
        bench_calls._ollama_generate_url() == "http://127.0.0.2:11500/api/generate",
    )
    os.environ["OLLAMA_URL"] = "http://[::1]:11500"
    check(
        "benchmark accepts IPv6 loopback",
        bench_calls._ollama_generate_url() == "http://[::1]:11500/api/generate",
    )
    os.environ["OLLAMA_URL"] = "http://192.0.2.1:11500"
    check(
        "benchmark rejects remote IP endpoint",
        bench_calls._ollama_generate_url() == "http://localhost:11434/api/generate",
    )
finally:
    if _old_ollama_url is None:
        os.environ.pop("OLLAMA_URL", None)
    else:
        os.environ["OLLAMA_URL"] = _old_ollama_url

try:
    bench_calls.call_openai_compat("file:///tmp", "unused", "m", "s", "p")
    check("benchmark rejects unlisted provider endpoint", False, detail="no ValueError")
except ValueError:
    check("benchmark rejects unlisted provider endpoint", True)

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
        cl._provider_billing("zenmux") == "payg" and _zenmux_plan["routes"][0]["billing"] == "payg",
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
