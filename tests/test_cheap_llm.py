#!/usr/bin/env python3
"""Regression test runner for the cheap_llm package.

The suite is split by concern: harness + shared fixtures live in _testlib.py,
and each _checks_*.py module runs its section's checks at import time. This
runner imports them in order, then prints the shared summary.

Run: python3 tests/test_cheap_llm.py [--live] [--quick]
  --live     also run the live API smoke tests (requires API keys in env)
  --quick    skip live tests even if --live is set
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _testlib  # noqa: E402  -- sets up state, check/skip, cl, shared fixtures

# Import each group in original section order; top-level checks run on import.
# DEEPINFRA_API_KEY stays POPPED (by _testlib) through all unit/mocked sections so
# cascade assertions are predictable; restore it only before the live section.
import _checks_pure  # noqa: E402,F401
import _checks_cost  # noqa: E402,F401
import _checks_scrub  # noqa: E402,F401
import _checks_cascade  # noqa: E402,F401
import _checks_refactor  # noqa: E402,F401
import _checks_local  # noqa: E402,F401
import _checks_regression  # noqa: E402,F401
import _checks_cli  # noqa: E402,F401
import _checks_robustness  # noqa: E402,F401

# Restore DeepInfra API key before the (optional) live section.
if _testlib._actual_deepinfra_key:
    os.environ["DEEPINFRA_API_KEY"] = _testlib._actual_deepinfra_key

import _checks_live  # noqa: E402,F401

# =================================================================
# Summary
# =================================================================
print(f"\n{'=' * 60}")
print(f"PASS: {_testlib.PASS}    FAIL: {_testlib.FAIL}    SKIP: {_testlib.SKIP}")
if _testlib.FAILURES:
    print("\nFailures:")
    for _f in _testlib.FAILURES:
        print(f"  - {_f}")
print(f"{'=' * 60}")

sys.exit(0 if _testlib.FAIL == 0 else 1)
