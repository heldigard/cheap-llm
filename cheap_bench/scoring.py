"""Scoring — parse model output and score it 0-100 per task.

try_parse_json tolerates fences/leading prose; score_response combines JSON
validity (25) + field match (25) + content quality (0-40) + latency (0-10).
"""

# vs-soft-allow — _content_quality is one cohesive per-task scoring dispatch:
# a flat if/elif chain over the 5 benchmark tasks, each a 1-2 level field check.
# The 66-line shape and 4-deep nesting come from the task branches themselves,
# not from tangled logic; splitting into 5 functions + a registry would add
# indirection without clarifying a benchmark heuristic.
from __future__ import annotations

import json


def try_parse_json(text: str) -> dict | None:
    """Extract JSON from model output. Tolerate ```json fences, leading prose."""
    text = text.strip()
    # strip code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # first { ... last }
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def score_response(task: dict, raw: dict) -> dict:
    """Score 0-100 per task. Higher = better."""
    text = raw.get("text", "")
    parsed = try_parse_json(text)

    # 1. JSON validity: 0 or 25
    json_score = 25 if parsed is not None else 0

    # 2. Field match: 0 or 25
    field_score = 0
    if parsed:
        missing = [f for f in task["schema"] if f not in parsed or not parsed[f]]
        if not missing:
            field_score = 25
        else:
            # partial credit
            present = len(task["schema"]) - len(missing)
            field_score = int(25 * present / len(task["schema"]))

    # 3. Content quality: 0-40
    content_score = 0
    if parsed:
        content_score = _content_quality(task, parsed)

    # 4. Latency penalty: 0-10
    lat = raw.get("latency", 0)
    if lat <= 1.0:
        lat_score = 10
    elif lat <= 3.0:
        lat_score = 7
    elif lat <= 6.0:
        lat_score = 4
    elif lat <= 15.0:
        lat_score = 1
    else:
        lat_score = 0

    return {
        "json_valid": json_score,
        "field_match": field_score,
        "content": content_score,
        "latency_penalty": lat_score,
        "total": json_score + field_score + content_score + lat_score,
    }


def _content_quality(task: dict, parsed: dict) -> int:
    """Task-specific quality heuristic. 0-40."""
    name = task["name"]
    if name == "intent_classify":
        cat = str(parsed.get("category", "")).lower().strip()
        if cat == task.get("ref_category", "").lower():
            return 40
        if cat in ("trivial", "lookup", "code-edit", "architecture", "security", "debug"):
            return 20  # valid category, wrong one
        return 5  # garbage
    if name == "commit_draft":
        t = str(parsed.get("type", "")).lower()
        s = str(parsed.get("scope", "")).lower()
        sub = str(parsed.get("subject", ""))
        score = 0
        if t in ("feat", "fix", "chore", "docs", "test", "refactor", "build", "perf"):
            score += 15
        if t == task.get("ref_type"):
            score += 10
        if s == task.get("ref_scope"):
            score += 5
        if 5 <= len(sub) <= 70:
            score += 5
        if any(
            sub.lower().startswith(p)
            for p in ("feat:", "fix:", "chore:", "docs:", "test:", "refactor:", "build:", "perf:")
        ):
            score += 5
        return min(score, 40)
    if name == "error_classify":
        sys_l = str(parsed.get("system", "")).lower()
        if "dynamics" in sys_l or "dataverse" in sys_l or "n8n" in sys_l:
            return 30
        cause = str(parsed.get("cause", ""))
        fix = str(parsed.get("fix", ""))
        # any plausible cause + non-empty fix
        if 10 < len(cause) < 200 and 10 < len(fix) < 300:
            return 20
        return 5
    if name == "json_extract":
        # check exact field values
        score = 0
        if parsed.get("name") == "@openai/codex":
            score += 12
        if parsed.get("version") == "0.42.0":
            score += 12
        if "node" in str(parsed.get("deps", "")).lower():
            score += 8
        if (
            "experimental" in str(parsed.get("warnings", "")).lower()
            or "streaming" in str(parsed.get("warnings", "")).lower()
        ):
            score += 8
        return min(score, 40)
    if name == "diff_review":
        findings = parsed.get("findings", [])
        if not isinstance(findings, list):
            return 5
        msgs = " ".join(str(f.get("message", "")).lower() for f in findings).lower()
        score = 0
        if any(t in msgs for t in ("sql injection", "sql", "injection", "concat", "string concat")):
            score += 20
        if any(t in msgs for t in ("todo", "unfinished", "incomplete")):
            score += 15
        if len(findings) >= 1:
            score += 5
        return min(score, 40)
    return 0
