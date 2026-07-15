"""Low-level HTTP response safety — bounded reads, sanitized errors.

These helpers keep untrusted provider responses and diagnostics out of telemetry:
responses are size-capped before JSON parsing, and public error strings omit
URLs, bodies, and prompts.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Protocol

from .constants import MAX_RESPONSE_BYTES


class _ReadableResponse(Protocol):
    def read(self, _amount: int = -1) -> bytes: ...


def _read_json_response(response: _ReadableResponse) -> dict:
    """Read one bounded UTF-8 JSON object from an HTTP response."""
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("provider response exceeds 4 MiB limit")
    body = json.loads(raw.decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("provider response must be a JSON object")
    return body


def _public_attempt_error(exc: Exception) -> str:
    """Return bounded provider diagnostics without URLs, bodies, or prompts."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTPError: HTTP {exc.code}"
    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return f"{type(exc).__name__}: request failed"
    if isinstance(exc, RuntimeError) and str(exc).endswith(" not set"):
        return f"RuntimeError: {str(exc)[:260]}"
    return f"{type(exc).__name__}: provider attempt failed"


def _normalize_model_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.strip()
    if ":" not in name:
        name = f"{name}:latest"
    return name
