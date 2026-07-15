# vs-soft-allow — main() is a CLI entry point: argparse + dispatch, one responsibility
"""CLI entry point — probe, cache ops, and main().

Provides --probe (reachability), --cache stats/clear, and the main
completion CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

from .cache import CACHE_DIR
from .cascade import _build_cascade, _resolve_local_model, cheap_complete
from .contract import __version__
from .transport import (
    _PROVIDERS,
    OLLAMA_URL,
    _provider_billing,
    _public_attempt_error,
    _read_json_response,
)


def _probe_url(url: str, timeout: float = 2.0, key_env: str | None = None) -> dict:
    """Authenticated, bounded model-list probe that never exposes key values."""
    t0 = time.perf_counter()
    try:
        headers = {"Accept": "application/json"}
        if key_env:
            key = os.environ.get(key_env, "")
            if not key:
                raise RuntimeError(f"{key_env} not set")
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        # nosemgrep — url comes from _PROVIDERS registry (frozen constants)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _read_json_response(r)
            return {
                "reachable": True,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "status": getattr(r, "status", None) or 200,
                "error": None,
            }
    except Exception as e:  # noqa: BLE001
        return {
            "reachable": False,
            "latency_ms": None,
            "status": None,
            "error": _public_attempt_error(e),
        }


def _route_plan(
    *,
    prefer_local: bool = True,
    model: str | None = None,
    cloud_model: str | None = None,
    cloud_provider: str | None = None,
    require_json: bool = True,
    schema_hint: list[str] | None = None,
) -> dict:
    """Return the effective route order without making a completion request."""
    local_model = _resolve_local_model(model, require_json, tuple(schema_hint or ()))
    cascade = _build_cascade(prefer_local, local_model, cloud_model, cloud_provider)
    routes: list[dict] = []
    for index, (tier, route_model, provider, timeout) in enumerate(cascade, start=1):
        spec = _PROVIDERS.get(provider)
        key_env = spec.endpoint.key_env if spec else None
        routes.append(
            {
                "position": index,
                "tier": tier,
                "model": route_model,
                "provider": provider,
                "timeout": timeout,
                "billing": _provider_billing(provider),
                "credential_env": key_env,
                "credential_set": bool(key_env and os.environ.get(key_env)),
            }
        )
    return {
        "routes": routes,
        "explicit_provider": cloud_provider,
        "local_only": all(route["provider"] == "ollama" for route in routes),
        "subscription_workers_in_scope": False,
        "note": (
            "CLI-seat subscriptions are routed by cli-orchestration/fusion-local; "
            "cheap-llm cloud providers are PAYG even when using granted balance."
        ),
    }


def _probe() -> dict:
    """Report what's available right now (key set + per-provider reachability)."""
    out: dict = {
        "ollama_alive": False,
        "local_models": [],
        "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        "zenmux_key_set": bool(os.environ.get("ZENMUX_API_KEY")),
        "deepseek_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "deepinfra_key_set": bool(os.environ.get("DEEPINFRA_API_KEY")),
        "local_only": os.environ.get("CHEAP_LLM_LOCAL_ONLY", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "cache_entries": len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0,
        "providers": {},
    }
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        # nosemgrep — OLLAMA_URL is operator config, not user input
        with urllib.request.urlopen(req, timeout=2) as r:
            data = _read_json_response(r)
        out["ollama_alive"] = True
        out["local_models"] = [
            m["name"] for m in data.get("models", []) if "embed" not in m["name"]
        ]
    except Exception as e:
        out["ollama_error"] = f"{type(e).__name__}: {e}"

    for name, spec in _PROVIDERS.items():
        if not spec.probe_url:
            continue
        if not os.environ.get(spec.endpoint.key_env):
            out["providers"][name] = {
                "reachable": False,
                "skipped": "key_not_set",
                "latency_ms": None,
                "status": None,
                "error": None,
                "billing": _provider_billing(name),
            }
            continue
        out["providers"][name] = _probe_url(
            spec.probe_url, key_env=spec.endpoint.key_env
        )
        out["providers"][name]["billing"] = _provider_billing(name)
    return out


def _cache_stats() -> dict:
    """Return cache health: entry count, total bytes, oldest/newest mtime."""
    if not CACHE_DIR.exists():
        return {"exists": False, "entries": 0, "bytes": 0, "oldest": None, "newest": None}
    files = list(CACHE_DIR.glob("*.json"))
    if not files:
        return {"exists": True, "entries": 0, "bytes": 0, "oldest": None, "newest": None}
    sizes: list[int] = []
    mtimes: list[float] = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        sizes.append(st.st_size)
        mtimes.append(st.st_mtime)
    return {
        "exists": True,
        "entries": len(files),
        "bytes": sum(sizes),
        "oldest": min(mtimes) if mtimes else None,
        "newest": max(mtimes) if mtimes else None,
        "path": str(CACHE_DIR),
    }


def _cache_clear() -> dict:
    """Remove all cached entries. Returns counts (removed / kept)."""
    if not CACHE_DIR.exists():
        return {"removed": 0, "kept": 0, "path": str(CACHE_DIR)}
    files = [p for p in CACHE_DIR.glob("*.json") if p.is_file()]
    removed = 0
    for p in files:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return {"removed": removed, "kept": len(files) - removed, "path": str(CACHE_DIR)}


def main() -> int:
    p = argparse.ArgumentParser(description="Cheap-LLM cascade client")
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
        help=(
            "pin the T2 fallback model; combine with --no-local for cloud-only "
            "(e.g. deepseek/deepseek-v4-flash)"
        ),
    )
    p.add_argument(
        "--cloud-provider",
        choices=sorted(_PROVIDERS),
        help="pin the T2 model to exactly one PAYG provider (requires --cloud-model)",
    )
    p.add_argument("--no-json", action="store_true", help="don't require JSON output")
    p.add_argument("--probe", action="store_true", help="report availability")
    p.add_argument(
        "--route-plan",
        action="store_true",
        help="print effective routes and billing classes without making a completion",
    )
    p.add_argument(
        "--cache",
        choices=["stats", "clear"],
        help="operate on the local cache (stats or clear)",
    )
    p.add_argument("--json", action="store_true", help="output JSON envelope")
    p.add_argument("--version", action="store_true", help="print version and exit")
    args = p.parse_args()

    if args.version:
        print(__version__)
        return 0
    if args.probe:
        print(json.dumps(_probe(), indent=2))
        return 0
    if args.route_plan:
        if args.cloud_provider and not args.cloud_model:
            p.error("--cloud-provider requires --cloud-model")
        print(
            json.dumps(
                _route_plan(
                    prefer_local=not args.no_local,
                    model=args.model,
                    cloud_model=args.cloud_model,
                    cloud_provider=args.cloud_provider,
                    require_json=not args.no_json,
                    schema_hint=args.schema,
                ),
                indent=2,
            )
        )
        return 0
    if args.cache == "stats":
        print(json.dumps(_cache_stats(), indent=2))
        return 0
    if args.cache == "clear":
        result = _cache_clear()
        print(json.dumps(result, indent=2))
        return 0
    if not args.system or not args.prompt:
        p.error("--system and --prompt are required (unless --probe)")
    if args.max_tokens < 1:
        p.error("--max-tokens must be a positive integer")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("--timeout must be a positive finite number")

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
        cloud_provider=args.cloud_provider,
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
