#!/usr/bin/env python3
"""Public-API contract test — the ecosystem decoupling gate.

Pins the **declared** public surface of cheap_llm so the project can evolve
independently of its consumers (fusion, web-research, the 7 ~/.claude/scripts
consumers). A breaking change (removed/renamed public param or RESULT_KEY)
fails HERE first and forces a SemVer MAJOR bump; consumers' ``require()`` gate
then trips loudly on version drift instead of cryptic mid-run ImportErrors.

Differs from ``test_cheap_llm.py``: that suite pins BEHAVIOR (cascade routing,
cache, scrub). This one pins the CONTRACT (names, signature, return shape,
version) — what consumers depend on across project boundaries.

Run: python3 tests/test_contract.py
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable

# Resolve the real module regardless of cwd (mirrors the consumer bootstrap).
_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import cheap_llm as cl  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: list[str] = []


class ContractCheckFailure(AssertionError):
    """A recorded contract violation that pytest must treat as a failure."""


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        failure = f"{name}: {detail}" if detail else name
        FAILURES.append(failure)
        raise ContractCheckFailure(failure)


def test_failed_check_raises_assertion() -> None:
    """Guard the guard: pytest must see a deliberately false check fail."""
    global PASS, FAIL
    original_pass, original_fail = PASS, FAIL
    original_failures = list(FAILURES)
    try:
        try:
            check("contract harness probe", False)
        except ContractCheckFailure:
            pass
        else:
            raise AssertionError("a false contract check did not raise")
    finally:
        PASS, FAIL = original_pass, original_fail
        FAILURES[:] = original_failures


def _sig_params(fn: Callable[..., Any]) -> list[str]:
    return list(inspect.signature(fn).parameters.keys())


def test_version_is_semver_and_matches_pyproject() -> None:
    v = cl.__version__
    check("version is SemVer", bool(re.fullmatch(r"\d+\.\d+\.\d+", v)), v)
    pyproject = (_PROJECT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    check(
        "pyproject version matches module",
        bool(m and m.group(1) == v),
        f"{m.group(1) if m else None} vs {v}",
    )


def test_all_symbols_exist() -> None:
    for name in cl.__all__:
        check(f"__all__ entry {name} resolves", hasattr(cl, name), name)
    check("cheap_complete callable", callable(cl.cheap_complete))
    check("scrub_secrets callable", callable(cl.scrub_secrets))
    check("require callable", callable(cl.require))


def test_cheap_complete_signature_pinned() -> None:
    # The documented contract vs the real signature. If this fails, either
    # update CHEAP_COMPLETE_PARAMS (MINOR) or, if a param was removed/renamed,
    # bump MAJOR and notify consumers.
    real = _sig_params(cl.cheap_complete)
    check(
        "cheap_complete params match CHEAP_COMPLETE_PARAMS",
        real == list(cl.CHEAP_COMPLETE_PARAMS),
        f"real={real} declared={list(cl.CHEAP_COMPLETE_PARAMS)}",
    )


def test_result_keys_present_on_success() -> None:
    # Mock-driven: force one success and assert every documented RESULT_KEY is
    # in the returned dict (the contract consumers rely on).
    real_provider = cl._call_provider

    def fake_call(
        model: str,
        provider: str,
        system: str,
        prompt: str,
        timeout: float,
        max_output_tokens: int = 1024,
        require_json: bool = False,
    ) -> dict:
        return {"text": '{"ok": 1}', "latency": 0.1, "api_cost": 0.0, "provider": provider}

    cl._call_provider = fake_call
    try:
        out = cl.cheap_complete(
            system="test",
            prompt="test",
            schema_hint=None,
            prefer_local=False,
            require_json=False,
        )
    finally:
        cl._call_provider = real_provider
    missing = [k for k in cl.RESULT_KEYS if k not in out]
    check("success result has all RESULT_KEYS", not missing, f"missing={missing}")


def test_require_gate() -> None:
    check("require() returns version", cl.require() == cl.__version__)
    check("require(current) passes", cl.require(cl.__version__) == cl.__version__)
    check("require(older) passes", cl.require("1.0.0") == cl.__version__)
    try:
        cl.require("99.0.0")
        check("require(future) raises", False, "no RuntimeError")
    except RuntimeError as e:
        check("require(future) raises RuntimeError", "99.0.0" in str(e), str(e))


def test_contract_dict_self_consistent() -> None:
    c = cl.CONTRACT
    check("CONTRACT.version matches __version__", c["version"] == cl.__version__)
    check("CONTRACT.public_api == __all__", c["public_api"] == list(cl.__all__))
    check("CONTRACT.result_keys == RESULT_KEYS", c["result_keys"] == list(cl.RESULT_KEYS))
    check(
        "CONTRACT.cheap_complete_params == CHEAP_COMPLETE_PARAMS",
        c["cheap_complete_params"] == list(cl.CHEAP_COMPLETE_PARAMS),
    )


TESTS = [
    ("version_is_semver_and_matches_pyproject", test_version_is_semver_and_matches_pyproject),
    ("all_symbols_exist", test_all_symbols_exist),
    ("cheap_complete_signature_pinned", test_cheap_complete_signature_pinned),
    ("result_keys_present_on_success", test_result_keys_present_on_success),
    ("require_gate", test_require_gate),
    ("contract_dict_self_consistent", test_contract_dict_self_consistent),
]


def main() -> int:
    print(f"cheap_llm contract tests — {len(TESTS)} cases (v{cl.__version__})\n")
    for name, fn in TESTS:
        try:
            fn()
        except ContractCheckFailure:
            # check() already recorded the failure. Avoid double-counting it
            # while retaining the standalone runner's aggregated report.
            pass
        except Exception as exc:  # noqa: BLE001
            global FAIL
            FAIL += 1
            FAILURES.append(f"{name}: raised {type(exc).__name__}: {exc}")
        else:
            print(f"  [ok] {name}")
    print(f"\nPASS={PASS} FAIL={FAIL}")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
