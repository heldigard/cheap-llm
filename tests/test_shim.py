#!/usr/bin/env python3
"""Verify the ecosystem shim at ~/.claude/scripts/cheap_llm.py re-exports correctly."""

import importlib.util
import sys
from pathlib import Path

SHIM = Path.home() / ".claude" / "scripts" / "cheap_llm.py"
spec = importlib.util.spec_from_file_location("cheap_llm_shim", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

errors = []
if not callable(mod.cheap_complete):
    errors.append("cheap_complete not callable")
if not callable(mod.scrub_secrets):
    errors.append("scrub_secrets not callable")
if not callable(mod.require):
    errors.append("require not callable")
if mod.__version__ != mod.require():
    errors.append(f"__version__ {mod.__version__} != require() {mod.require()}")
if mod.DEFAULT_LOCAL_PRIMARY != "qwen3.5:4b":
    errors.append(f"DEFAULT_LOCAL_PRIMARY={mod.DEFAULT_LOCAL_PRIMARY}")
if "Qwopus" not in mod.DEFAULT_LOCAL_STRUCTURED:
    errors.append(f"DEFAULT_LOCAL_STRUCTURED={mod.DEFAULT_LOCAL_STRUCTURED}")
if len(mod.TOP3_CASCADE) != 6:
    errors.append(f"TOP3_CASCADE len={len(mod.TOP3_CASCADE)}")
if not hasattr(mod, "CACHE_DIR"):
    errors.append("CACHE_DIR missing")
if not hasattr(mod, "MODEL_PRICING"):
    errors.append("MODEL_PRICING missing")
if not hasattr(mod, "LEGACY_CASCADE"):
    errors.append("LEGACY_CASCADE missing")

if errors:
    for e in errors:
        print(f"  FAIL  {e}")
    sys.exit(1)
else:
    print("  PASS  shim re-exports all critical symbols")
    sys.exit(0)
