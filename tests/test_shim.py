#!/usr/bin/env python3
"""Verify the ecosystem shim at ~/.claude/scripts/cheap_llm.py re-exports correctly."""

import importlib.util
import sys
from pathlib import Path

SHIM = Path.home() / ".claude" / "scripts" / "cheap_llm.py"


def _shim_errors() -> list[str]:
    spec = importlib.util.spec_from_file_location("cheap_llm_shim", SHIM)
    assert spec and spec.loader
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
    if mod.DEFAULT_LOCAL_PRIMARY != "cryptidbleh/gemma4-claude-opus-4.6:latest":
        errors.append(f"DEFAULT_LOCAL_PRIMARY={mod.DEFAULT_LOCAL_PRIMARY}")
    if "Qwopus" not in mod.DEFAULT_LOCAL_STRUCTURED:
        errors.append(f"DEFAULT_LOCAL_STRUCTURED={mod.DEFAULT_LOCAL_STRUCTURED}")
    if len(mod.TOP3_CASCADE) != 6:
        errors.append(f"TOP3_CASCADE len={len(mod.TOP3_CASCADE)}")
    for name in ("CACHE_DIR", "MODEL_PRICING", "LEGACY_CASCADE"):
        if not hasattr(mod, name):
            errors.append(f"{name} missing")
    return errors


def test_shim_reexports_critical_symbols() -> None:
    assert not _shim_errors()


if __name__ == "__main__":
    failures = _shim_errors()
    if failures:
        for error in failures:
            print(f"  FAIL  {error}")
        sys.exit(1)
    print("  PASS  shim re-exports all critical symbols")
