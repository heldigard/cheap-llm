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

# --- shared cross-section fixtures (moved here so every _checks_* sees them) ---
# Saved once at import: sections that monkeypatch _urlreq.urlopen restore this.
_orig_urlopen = _urlreq.urlopen
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

# cache dir reused by cascade / regression / live sections
cache_dir = Path.home() / ".claude" / "state" / "cheap-llm-cache"

# Cross-section cascade helpers (defined in the cascade section, used in regression).
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

# Live-smoke gate (parsed once at import).
LIVE = "--live" in sys.argv and "--quick" not in sys.argv

# Star-import surface: export EVERYTHING (incl. _FakeResp / _urlreq / _actual_deepinfra_key)
# so section bodies stay verbatim under `from _testlib import *`.
__all__ = [n for n in dir() if not n.startswith("__")]
