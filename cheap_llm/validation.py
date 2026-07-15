"""Output validation — parse model output to JSON and validate it.

Pure helpers the cascade uses to decide whether an attempt produced good
output: ``_try_parse_json`` tolerates fences / leading prose / trailing
commas; ``_validate`` checks required fields without rejecting valid empty
containers. ``JSON_HINT`` is the system-prompt addendum that asks for
JSON-only output. Extracted from the cascade so the orchestrator stays
orchestration-only.
"""

from __future__ import annotations

import json
import re

# JSON contract hint appended to system prompt when require_json=True.
JSON_HINT = (
    "\n\nReply with JSON only — no prose, no code fences, no explanation. "
    "The first character must be `{` and the last must be `}`."
)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        try:
            result = json.loads(re.sub(r",(\s*[}\]])", r"\1", text))
        except json.JSONDecodeError:
            return None
    return result if isinstance(result, dict) else None


def _validate(parsed: dict | None, schema: tuple[str, ...] | None) -> bool:
    """Validate required JSON fields without rejecting valid empty containers."""
    if parsed is None:
        return False
    if not schema:
        return True
    for name in schema:
        if name not in parsed:
            return False
        value = parsed[name]
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True
