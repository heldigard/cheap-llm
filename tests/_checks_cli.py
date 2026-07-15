from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

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
