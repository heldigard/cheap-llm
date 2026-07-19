from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

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

# _probe reports every provider key + local_only + cache size + local defaults
_probe_snapshot = cl._probe()
_probe_keys = set(_probe_snapshot)
for _pk in (
    "zenmux_key_set",
    "deepseek_key_set",
    "local_only",
    "cache_entries",
    "defaults",
    "loaded_models",
):
    check(f"_probe exposes {_pk}", _pk in _probe_keys)
_defaults = _probe_snapshot.get("defaults") or {}
check(
    "_probe defaults names primary + structured models",
    _defaults.get("primary") == cl.DEFAULT_LOCAL_PRIMARY
    and _defaults.get("structured") == cl.DEFAULT_LOCAL_STRUCTURED
    and "primary_installed" in _defaults
    and "structured_installed" in _defaults,
    detail=f"defaults={_defaults}",
)
check(
    "_probe loaded_models is a list",
    isinstance(_probe_snapshot.get("loaded_models"), list),
    detail=f"loaded={_probe_snapshot.get('loaded_models')!r}",
)
# route-plan: local Ollama needs no API key — credential_set must not false-negative
_route_local = cl._route_plan(prefer_local=True, require_json=False)
_t1 = next((r for r in _route_local.get("routes", []) if r.get("provider") == "ollama"), None)
check(
    "route-plan marks ollama credential_set true (no key required)",
    _t1 is not None and _t1.get("credential_set") is True and _t1.get("billing") == "local",
    detail=f"t1={_t1}",
)

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
    and _probe_seen
    == {
        "method": "GET",
        "authorization": "Bearer synthetic-test-value",
    },
    detail=f"seen={_probe_seen} result={_probe_result}",
)


# =================================================================
# CLI: --probe works standalone (regression 2026-07-02: --system/--prompt
# were required=True, so the documented `python3 -m cheap_llm --probe` exited 2)
# =================================================================
