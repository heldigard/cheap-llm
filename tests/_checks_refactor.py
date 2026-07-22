from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

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
    cl.OPENROUTER_ENDPOINT.extra_headers == {"X-OpenRouter-Title": "cheap-llm-cascade"},
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
_environment_deepinfra_key = os.environ.pop("DEEPINFRA_API_KEY", None)
try:
    cascade_default = cl._build_cascade(prefer_local=True, local_model=None, cloud_model=None)
finally:
    if _environment_deepinfra_key is not None:
        os.environ["DEEPINFRA_API_KEY"] = _environment_deepinfra_key
check("cascade default: starts with T1 ollama", cascade_default[0][0] == "T1")
check(
    "cascade default: ends with legacy ling failover",
    cascade_default[-1][1:] == ("inclusionai/ling-2.6-flash", "zenmux", 12.0),
    detail=f"last={cascade_default[-1]}",
)
check(
    "cascade default: 10 entries (T1 + 6 TOP3 + 3 LEGACY)",
    len(cascade_default) == 10,
    detail=f"got {len(cascade_default)}",
)
check(
    "cascade default omits unavailable deepinfra fallback",
    all(route[2] != "deepinfra" for route in cascade_default),
)

local_only_by_contract = cl._build_cascade(
    prefer_local=True,
    local_model=None,
    cloud_model="deepseek/deepseek-v4-flash",
    allow_cloud=False,
)
check(
    "allow_cloud=False keeps the cascade local-only",
    len(local_only_by_contract) == 1 and local_only_by_contract[0][2] == "ollama",
    detail=f"routes={local_only_by_contract}",
)

_old_deepinfra_key = os.environ.get("DEEPINFRA_API_KEY")
try:
    os.environ["DEEPINFRA_API_KEY"] = "test-key"
    cascade_with_deepinfra = cl._build_cascade(
        prefer_local=False, local_model=None, cloud_model=None
    )
finally:
    if _old_deepinfra_key is None:
        os.environ.pop("DEEPINFRA_API_KEY", None)
    else:
        os.environ["DEEPINFRA_API_KEY"] = _old_deepinfra_key
check(
    "automatic cascade adds deepinfra only when configured",
    cascade_with_deepinfra[-1] == ("T2", "deepseek/deepseek-v4-flash", "deepinfra", 12.0),
)

# 2026-07-17: first-party deepseek leg leads the automatic T2 order when the
# credential exists, and stays out when it does not (no unauthenticated route).
_old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
try:
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    cascade_with_deepseek = cl._build_cascade(
        prefer_local=False, local_model=None, cloud_model=None
    )
    os.environ.pop("DEEPSEEK_API_KEY", None)
    cascade_without_deepseek = cl._build_cascade(
        prefer_local=False, local_model=None, cloud_model=None
    )
finally:
    if _old_deepseek_key is None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    else:
        os.environ["DEEPSEEK_API_KEY"] = _old_deepseek_key
check(
    "automatic cascade leads with first-party deepseek when configured",
    cascade_with_deepseek[0] == ("T2", "deepseek/deepseek-v4-flash", "deepseek", 12.0),
    detail=f"first={cascade_with_deepseek[0]}",
)
check(
    "automatic cascade omits first-party deepseek without credential",
    all(route[2] != "deepseek" for route in cascade_without_deepseek),
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
        "cascade warm T1: fast budget (primary warm timeout)",
        warm[0][3] == cl.LOCAL_WARM_TIMEOUT_PRIMARY,
        detail=f"timeout={warm[0][3]}",
    )
    warm_struct = cl._build_cascade(
        prefer_local=True, local_model=cl.DEFAULT_LOCAL_STRUCTURED, cloud_model=None
    )
    check(
        "cascade warm T1: structured budget (schema JSON headroom)",
        warm_struct[0][3] == cl.LOCAL_WARM_TIMEOUT_STRUCTURED,
        detail=f"timeout={warm_struct[0][3]}",
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
