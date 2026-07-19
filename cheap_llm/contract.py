"""Public API contract — version, result shape, require() gate.

The surface consumers may depend on. Everything else is private (_-prefixed)
and may change without notice. ``tests/test_contract.py`` is the evolution
gate: a breaking change fails there first and forces a SemVer MAJOR bump.
SemVer policy for independent evolution across the ecosystem:
  - MAJOR = removed/renamed public param or RESULT_KEY (consumers' require() gate trips)
  - MINOR = additive (new param with default, new RESULT_KEY, new public fn)
  - PATCH = internal refactor, model/cascade changes, bug fixes
"""

# This module stores the package manifest; implementations are re-exported by
# cheap_llm.__init__, so the names intentionally do not resolve in this module.
# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

import re

__version__ = "1.3.2"
__all__ = [  # noqa: F822
    "cheap_complete",
    "scrub_secrets",
    "require",
    "__version__",
]

# Stable shape of the dict returned by cheap_complete(). Additive-only: a new
# key is MINOR; removing/renaming is MAJOR.
RESULT_KEYS: tuple[str, ...] = (
    "text",
    "model",
    "provider",
    "billing",
    "tier",
    "latency",
    "cost",
    "json_valid",
    "fields_ok",
    "attempts",
    "error",
    "cached",
)

# Documented cheap_complete() signature — test_contract.py pins the real one.
CHEAP_COMPLETE_PARAMS: tuple[str, ...] = (
    "system",
    "prompt",
    "schema_hint",
    "timeout_total",
    "prefer_local",
    "require_json",
    "model",
    "cloud_model",
    "max_output_tokens",
    "cloud_provider",
)

CONTRACT: dict[str, object] = {
    "version": __version__,
    "public_api": list(__all__),
    "result_keys": list(RESULT_KEYS),
    "cheap_complete_params": list(CHEAP_COMPLETE_PARAMS),
}


def _parse_version(v: str) -> tuple[int, ...]:
    """``"1.2.3-beta"`` → ``(1, 2, 3)`` for ordering; non-numeric parts ignored."""
    parts = []
    for p in v.split(".")[:3]:
        m = re.match(r"\d+", p)
        if m:
            parts.append(int(m.group(0)))
        else:
            parts.append(0)
    return tuple(parts)


def require(min_version: str | None = None) -> str:
    """Version gate consumers call right after ``import cheap_llm``.

    Fails FAST with an actionable message on version drift, instead of a
    cryptic mid-run error when a needed param/key is absent. Returns the
    current version.

        import cheap_llm
        cheap_llm.require("1.1")   # RuntimeError if installed cheap_llm < 1.1
    """
    if min_version and _parse_version(__version__) < _parse_version(min_version):
        raise RuntimeError(
            f"cheap_llm {__version__} is older than required {min_version}. "
            f"Upgrade: cd ~/cheap-llm && pip install -e . --user"
        )
    return __version__


# Default values for RESULT_KEYS that a partial envelope may omit, so every
# cheap_complete() return has the FULL contract shape regardless of outcome
# (success paths omit ``error``; the all-failed path omits ``provider``/
# ``cached``). Centralizing this means adding a RESULT_KEY auto-propagates.
_RESULT_DEFAULTS: dict[str, object] = {
    "text": "",
    "model": None,
    "provider": None,
    "billing": None,
    "tier": None,
    "latency": 0,
    "cost": 0.0,
    "json_valid": False,
    "fields_ok": False,
    "attempts": [],
    "error": None,
    "cached": False,
}


def _complete_result(env: dict) -> dict:
    """Return ``env`` with every RESULT_KEY present (uniform contract shape).

    Fills absent keys with ``_RESULT_DEFAULTS``. Mutation is in-place on a
    fresh envelope, so callers can still build partial dicts at each path.
    """
    for _k in RESULT_KEYS:
        if _k not in env:
            env[_k] = _RESULT_DEFAULTS[_k]
    return env
