#!/usr/bin/env python3
"""Live + E2E integration test runner for cheap_llm.

Opt-in (every case hits the network): pass --live / --live-only / --e2e-only,
or set CHEAP_LLM_LIVE=1. Plain invocation skips cleanly with exit 0.

Harness + gate live in _livetestlib.py; LIVE cascade checks in
_checks_liveapi.py; E2E subprocess-script checks in _checks_e2e.py. This runner
imports them in order and prints the shared summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _livetestlib  # noqa: E402  -- state, check/skip, gate (LIVE/E2E), cl
import _checks_liveapi  # noqa: E402,F401  -- runs the LIVE section on import
import _checks_e2e  # noqa: E402,F401  -- runs the E2E section on import

# =================================================================
# Summary
# =================================================================
print(f"\n{'=' * 64}")
print(
    f"LIVE+E2E  PASS: {_livetestlib.PASS}   FAIL: {_livetestlib.FAIL}   SKIP: {_livetestlib.SKIP}"
)
if _livetestlib.FAILURES:
    print("\nFailures:")
    for _f in _livetestlib.FAILURES:
        print(f"  - {_f}")
print(f"{'=' * 64}")
sys.exit(0 if _livetestlib.FAIL == 0 else 1)
