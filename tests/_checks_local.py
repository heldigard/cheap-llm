from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

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
    "OpenRouter maps the ZenMux KAT Air organization slug",
    cl._normalize_provider_model("openrouter", "kuaishou/kat-coder-air-v2.5")
    == "kwaipilot/kat-coder-air-v2.5",
)
check(
    "ZenMux maps the OpenRouter KAT Pro organization slug",
    cl._normalize_provider_model("zenmux", "kwaipilot/kat-coder-pro-v2.5")
    == "kuaishou/kat-coder-pro-v2.5",
)
check(
    "provider model mapping leaves unrelated slugs unchanged",
    cl._normalize_provider_model("zenmux", "qwen/qwen3.7-max") == "qwen/qwen3.7-max",
)
check(
    "provider model mapping rejects substring collisions",
    cl._normalize_provider_model("openrouter", "prefix-kuaishou/kat-coder-air-v2.5")
    == "prefix-kuaishou/kat-coder-air-v2.5",
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
    _di = cl._call_deepinfra("deepseek/deepseek-v4-flash", "s", "p", timeout=5, require_json=True)
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

# KAT aliases are normalized at the shared OpenAI-compatible boundary, while
# cost estimation uses the resolved provider wire id.
_kat_or_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 100},
    },
    _kat_or_payload,
)
try:
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    _kat_or = cl._call_openrouter(
        "kuaishou/kat-coder-air-v2.5", "s", "p", timeout=5, require_json=False
    )
finally:
    _urlreq.urlopen = _orig_urlopen
    os.environ.pop("OPENROUTER_API_KEY", None)
check(
    "OpenRouter KAT request uses kwaipilot wire id",
    _kat_or_payload.get("model") == "kwaipilot/kat-coder-air-v2.5",
)
check(
    "OpenRouter KAT fallback cost is nonzero",
    _kat_or["api_cost"] is not None and _kat_or["api_cost"] > 0,
)

_kat_zm_payload: dict = {}
_urlreq.urlopen = _fake_urlopen_factory(
    {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 100, "cost": None},
    },
    _kat_zm_payload,
)
try:
    os.environ["ZENMUX_API_KEY"] = "test-key"
    _kat_zm = cl._call_zenmux(
        "kwaipilot/kat-coder-pro-v2.5", "s", "p", timeout=5, require_json=False
    )
finally:
    _urlreq.urlopen = _orig_urlopen
    os.environ.pop("ZENMUX_API_KEY", None)
check(
    "ZenMux KAT request uses kuaishou wire id",
    _kat_zm_payload.get("model") == "kuaishou/kat-coder-pro-v2.5",
)
check(
    "ZenMux KAT fallback cost is nonzero",
    _kat_zm["api_cost"] is not None and _kat_zm["api_cost"] > 0,
)


# =================================================================
# UNIT: 1.2.1 regressions — JSON hint without schema + cache shape guard
# =================================================================
