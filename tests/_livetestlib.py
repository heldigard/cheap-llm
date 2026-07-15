#!/usr/bin/env python3
"""Live + end-to-end tests for the cheap-LLM cascade (real API calls).

Companion to test-cheap-llm.py (which is unit + mocked, offline, 81 tests).
This file exercises the REAL stack:

  LIVE  — cheap_llm.cheap_complete() against real Ollama + OpenRouter + ZenMux.
           Each cascade tier reachable, full resolve, cache hit, scrub on the
           live path (the 2026-06-19 critical-fix regression).
  E2E   — the 5 migrated scripts (intent_route, error-classify, commit-draft,
           diff-review, extract-tool-output) run as SUBPROCESSES, proving the
           whole CLI → cascade → provider → output contract holds end to end.

Cost: a few cents (<= $0.02). Time: ~1-3 min (the local T1 call is the slow part).
Requires: OPENROUTER_API_KEY (+ ZENMUX_API_KEY for failover tests), Ollama up.

Run:
    python3 test-cheap-llm-live.py            # live + e2e
    python3 test-cheap-llm-live.py --live-only
    python3 test-cheap-llm-live.py --e2e-only
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM_SCRIPTS = Path.home() / ".claude" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))

import cheap_llm as cl  # noqa: E402

PASS = 0
FAIL = 0
SKIP = 0
FAILURES: list[str] = []
RESULTS: list[tuple[str, str, str]] = []  # (group, name, detail-line)


def check(group: str, name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    line = f"  {tag}  {name}" + (f"  {detail}" if detail else "")
    print(line)
    RESULTS.append((group, name, f"{tag} {detail}"))
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")


def skip(group: str, name: str, reason: str) -> None:
    global SKIP
    print(f"  SKIP  {name}  ({reason})")
    RESULTS.append((group, name, f"SKIP ({reason})"))
    SKIP += 1


HAVE_OR = bool(os.environ.get("OPENROUTER_API_KEY"))
HAVE_ZM = bool(os.environ.get("ZENMUX_API_KEY"))
HAVE_OLLAMA = False
try:
    import urllib.request

    req = urllib.request.Request(f"{cl.OLLAMA_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=2) as r:
        HAVE_OLLAMA = r.status == 200
except Exception:
    HAVE_OLLAMA = False

CLASSIFY_SYS = (
    "Classify this developer prompt into one of: trivial, lookup, "
    "code-edit, refactor, feature, debug, architecture, security, "
    'meta. Reply JSON only with keys "category" and "reason".'
)
CLASSIFY_PROMPT = "I'm getting ECONNREFUSED 127.0.0.1:5432 in my Express app after adding TypeORM."

# intent_route category set (defined in intent_route.py, mirrored here for e2e checks)
IR_CATEGORIES = {
    "trivial",
    "lookup",
    "code-edit",
    "refactor",
    "feature",
    "debug",
    "architecture",
    "security",
    "meta",
}


def _safe_get_category(text: str) -> str:
    d = cl._try_parse_json(text)
    if isinstance(d, dict):
        return str(d.get("category", "")).strip().lower()
    return ""


# Opt-in gate (2026-07-02): this file is a LIVE integration test — every case
# makes real cascade calls (network). It flakes under the deterministic unit
# battery when a third-party API hiccups. The UNIT gate is test-cheap-llm.py
# (86/86, mocked). To run THIS one explicitly: pass --live / --live-only /
# --e2e-only, or set CHEAP_LLM_LIVE=1. Plain invocation (the battery) now
# skips cleanly with exit 0 instead of making 14 flaky network calls.
_EXPLICIT = (
    "--live" in sys.argv
    or "--live-only" in sys.argv
    or "--e2e-only" in sys.argv
    or bool(os.environ.get("CHEAP_LLM_LIVE"))
)
LIVE = ("--e2e-only" not in sys.argv) and _EXPLICIT
E2E = ("--live-only" not in sys.argv) and _EXPLICIT

# Star-import surface: export everything so section bodies stay verbatim.
__all__ = [n for n in dir() if not n.startswith("__")]
