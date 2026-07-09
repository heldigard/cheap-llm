#!/usr/bin/env python3
"""Benchmark cheap models for the preprocessor slots.

Runs a fixed task set (intent classify, commit draft, error classify, JSON
extract, diff review) through every candidate model, scores the output, and
prints a leaderboard. The goal is data, not vibes — pick the model that
actually wins on the actual prompts the pipeline will throw at it.

Candidate models (all in our provider catalog — see CANDIDATES below for the live list):
  - gemma4:12b            (local Ollama, 7.6 GB, free, private) — T1
  - ling-2.6-flash / ling-2.6-1t / gemini-3.1-flash-lite (OpenRouter cloud)
  - gpt-5.4-nano / deepseek-v4-flash (OpenRouter; deepseek BYOK = $0)

Scoring (per task, 0-100):
  - json_valid      : 0 or 25  (valid JSON, structure check)
  - field_match     : 0 or 25  (all required fields present + non-empty)
  - content_quality : 0-40     (string similarity vs reference + heuristic check)
  - latency_penalty : 0-10     (subtract up to 10 for >3s responses)
  - cost_estimate   : reported but not scored (low cost = bonus info)

Output: rank-ordered table printed to stdout; raw JSONL appended to
~/.claude/state/cheap-bench/results.jsonl for longitudinal tracking.

Usage:
    python3 cheap_bench.py                  # all models, all tasks
    python3 cheap_bench.py --task intent    # one task
    python3 cheap_bench.py --model gemma4:12b google/gemma-4-31b-it
    python3 cheap_bench.py --no-cost        # skip cost reporting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# vs-soft-allow — CANDIDATES rows are intentionally tabular (model | provider |
# pricing) and exceed the 100-char line cap when listing every field on one
# line. Splitting across lines breaks the visual "diff candidates" comparison.


# add parent dir so we can import cheap_llm
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Candidate catalog -----------------------------------------------------
# 2026-06-19 round 3: pruned to the top 7 + local Ollama. Dropped from
# earlier rounds (score <85 or had FAILs in diff_review):
#   - google/gemma-4-31b-it (85.4) — no edge over local gemma4:12b (free)
#   - qwen/qwen3.7-plus (84.6) — slow + expensive
#   - qwen/qwen3.6-flash (83.6) — no edge
#   - moonshotai/kimi-k2.7-code (83.6) — duplicates deepseek-v4-flash
#   - gemma4:12b local is the T1 primary. Free, private, fast. R4 head-to-head:
#     gemma4:12b avg 84.2 vs qwen3.5:9b 78.4, 3.2× faster. qwen3.5:9b was pruned
#     (deleted locally 2026-06-19) — gemma4:12b wins on every axis.
#   - nvidia/nemotron-3-super-120b (80.8) — slow
#   - nvidia/nemotron-3-nano (69.8, FAIL) — variance
#   - xiaomi/mimo-v2.5 (53.0, FAIL) — variance
#   - stepfun/step-3.7-flash (36.4, FAIL) — bad fit for short tasks
# Re-add a model if it shows a meaningful improvement in a future round.
CANDIDATES: list[dict] = [
    # Local (T1) — gemma4:12b. Free, private, fast. Validated as primary by
    # cheap_llm.py R4 head-to-head (avg 84.2 on 5 preprocessor tasks) and
    # ousted the previous qwen3.5:9b fallback (deleted 2026-06-19).
    {"id": "gemma4:12b", "kind": "local", "provider": "ollama", "input": 0.0, "output": 0.0},
    # T2 cloud — pricing per OpenRouter public listing (NOT usage.cost, which
    # reports $0 for gemini/promo models). deepseek is BYOK = genuinely $0.
    # Pruned from this list (see DO-NOT-RE-TEST ledger in
    # topics/tested-models.md): kimi-k2 (superseded by gpt-5.4-nano),
    # gpt-4.1-nano (old 4.1 gen, deprecation risk despite high score),
    # qwen3.6-flash (reasoning tax). Also rejected outright: gpt-5-nano +
    # glm-4.7-flash (reasoning-only, content="" on short tasks).
    {
        "id": "inclusionai/ling-2.6-flash",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.01,
        "output": 0.03,
    },
    {
        "id": "inclusionai/ling-2.6-1t",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.075,
        "output": 0.625,
    },
    {
        "id": "google/gemini-3.1-flash-lite",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.25,
        "output": 1.50,
    },
    {
        "id": "openai/gpt-5.4-nano",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.20,
        "output": 1.25,
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "kind": "cloud",
        "provider": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "input": 0.14,
        "output": 0.28,
        "byok": True,
    },
]

# --- Task set --------------------------------------------------------------
# Each task: {name, system, prompt, schema, scorer}
# schema: list of required fields (top-level)
# scorer: callable(text) -> (field_match, content_quality 0-40)
TASKS: list[dict] = [
    {
        "name": "intent_classify",
        "system": (
            "You classify developer prompts into one of: "
            "trivial, lookup, code-edit, architecture, security, debug. "
            "Reply with JSON only — no prose, no code fences. Use field name "
            '"category" (not "classification" or "label").'
        ),
        "prompt": (
            "I'm getting 'ECONNREFUSED 127.0.0.1:5432' when starting my "
            "Express server after adding TypeORM. It's been working for weeks."
        ),
        "schema": ["category", "reason"],
        # heuristic: correct category is "debug"
        "ref_category": "debug",
    },
    {
        "name": "commit_draft",
        "system": (
            "Write a Conventional Commits message from a diff. "
            "Reply with JSON only: {subject, body, type, scope}."
        ),
        "prompt": """\
diff --git a/src/auth/middleware.ts b/src/auth/middleware.ts
+import jwt from 'jsonwebtoken';
+export function requireAuth(req, res, next) {
+  const token = req.headers.authorization?.split(' ')[1];
+  if (!token) return res.status(401).json({ error: 'no token' });
+  try {
+    const payload = jwt.verify(token, process.env.JWT_SECRET!);
+    (req as any).user = payload;
+    next();
+  } catch (err) {
+    return res.status(401).json({ error: 'bad token' });
+  }
+}""",
        "schema": ["subject", "body", "type", "scope"],
        "ref_type": "feat",
        "ref_scope": "auth",
    },
    {
        "name": "error_classify",
        "system": (
            "You classify error messages. Reply with JSON only: {system, cause, fix, confidence}."
        ),
        "prompt": (
            "n8n workflow failing with: 'Request failed with status code 401 "
            'and message: {\\"error\\":\\"unauthorized\\"}\'. The Dataverse '
            "node was working yesterday."
        ),
        "schema": ["system", "cause", "fix", "confidence"],
        "ref_system": "n8n / Dynamics 365",
        # expected: expired/rotated credentials OR URL typo
    },
    {
        "name": "json_extract",
        "system": (
            "Extract structured data from text. Reply with JSON only: "
            "{name, version, deps, warnings}."
        ),
        "prompt": """\
Package: @openai/codex v0.42.0
Requires: node >= 20
Warnings: experimental streaming API, may change in 0.43.0
Optional deps: ripgrep (for --search)
""",
        "schema": ["name", "version", "deps", "warnings"],
    },
    {
        "name": "diff_review",
        "system": (
            "Review a code diff. Reply with JSON only: "
            "{findings: [{severity, line, message}]}. Empty list if clean."
        ),
        "prompt": """\
+function login(user, pass) {
+  const query = \"SELECT * FROM users WHERE name='\" + user + \"' AND pwd='\" + pass + \"'\";
+  return db.query(query);
+}
+function logout() { /* TODO */ }
""",
        "schema": ["findings"],
        # Should flag SQL injection + bare TODO
    },
]


# --- cheap_llm client (inline to keep benchmark self-contained) ----------
# Self-contained on purpose — the benchmark measures cheap_llm.py by
# exercising the same wire shape without importing it (otherwise we can't
# measure the module from inside itself). urllib imports sit below the
# rationale comment intentionally.
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def call_local(model: str, system: str, prompt: str, timeout: float = 30.0) -> dict:
    """Call local Ollama. Returns {text, latency, input_tokens, output_tokens}."""
    payload = {
        "model": model,
        "prompt": f"{system}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    # Benchmark endpoints come from local operator config/static candidates.
    # nosemgrep
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
        body = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - t0
    return {
        "text": body.get("response", "").strip(),
        "latency": latency,
        "input_tokens": body.get("prompt_eval_count", 0),
        "output_tokens": body.get("eval_count", 0),
    }


def call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    timeout: float = 30.0,
    extra: dict | None = None,
) -> dict:
    """Call any OpenAI-compatible chat-completions endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    # nosemgrep: base_url is selected from the static PROVIDER_URLS map.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: E501
        body = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - t0
    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage", {})
    api_cost = usage.get("cost")  # openrouter/deepinfra report this
    return {
        "text": text,
        "latency": latency,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "api_cost": api_cost,
    }


PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
}


def call_candidate(cand: dict, task: dict, timeout: float) -> dict:
    """Dispatch a candidate against a task. Returns the raw response dict
    plus a 'cost' estimate in USD for cloud models."""
    try:
        if cand["kind"] == "local":
            out = call_local(cand["id"], task["system"], task["prompt"], timeout=timeout)
        else:
            env = cand.get("env")
            if not isinstance(env, str) or not os.environ.get(env):
                return {
                    "error": f"missing env {env}",
                    "text": "",
                    "latency": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0,
                }
            api_key = os.environ[env]
            base_url = PROVIDER_URLS[cand["provider"]]
            out = call_openai_compat(
                base_url,
                api_key,
                cand["id"],
                task["system"],
                task["prompt"],
                timeout=timeout,
                extra=cand.get("extra"),
            )
        inp = out.get("input_tokens", 0)
        out_t = out.get("output_tokens", 0)
        # Trust the PUBLIC LISTING price, not usage.cost — OpenRouter returns
        # usage.cost=0 for some promo/preview models (e.g. gemini-3.1-flash-lite
        # is $0.25/$1.50 real, API reports $0) and for BYOK models. Listing
        # estimate is the production truth EXCEPT for BYOK models (we supply
        # our own key → genuinely $0 regardless of listing). This keeps the
        # leaderboard's Cost column honest.
        if cand.get("byok"):
            out["cost"] = 0.0
        else:
            out["cost"] = (inp * cand["input"] + out_t * cand["output"]) / 1_000_000
        return out
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "text": "",
            "latency": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0,
        }


# --- Scoring ---------------------------------------------------------------


def try_parse_json(text: str) -> dict | None:
    """Extract JSON from model output. Tolerate ```json fences, leading prose."""
    text = text.strip()
    # strip code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last ```
        lines = [line for line in lines if not line.strip().startswith("```")]
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


# --- Main loop -------------------------------------------------------------


def run_benchmark(
    model_filter: set[str] | None = None, task_filter: set[str] | None = None, timeout: float = 30.0
) -> dict:
    results: list[dict] = []
    candidates = [c for c in CANDIDATES if model_filter is None or c["id"] in model_filter]
    tasks = [t for t in TASKS if task_filter is None or t["name"] in task_filter]

    for task in tasks:
        for cand in candidates:
            sys.stdout.write(f"  {task['name']:18} ← {cand['id']:30} ... ")
            sys.stdout.flush()
            raw = call_candidate(cand, task, timeout=timeout)
            score = score_response(task, raw)
            results.append(
                {
                    "task": task["name"],
                    "model": cand["id"],
                    "kind": cand["kind"],
                    "provider": cand["provider"],
                    "latency": round(raw.get("latency", 0), 3),
                    "input_tokens": raw.get("input_tokens", 0),
                    "output_tokens": raw.get("output_tokens", 0),
                    "cost_usd": round(raw.get("cost", 0), 8),
                    "error": raw.get("error"),
                    "raw_text_preview": raw.get("text", "")[:200],
                    **score,
                }
            )
            status = "OK" if raw.get("text") and not raw.get("error") else "FAIL"
            print(
                f"{status:4} score={score['total']:3d} "
                f"lat={raw.get('latency', 0):5.2f}s "
                f"cost=${raw.get('cost', 0):.6f}"
            )
    return {"results": results}


def print_leaderboard(results: list[dict]) -> None:
    # per-model totals
    by_model: dict[str, list[int]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r["total"])
    print("\n" + "=" * 80)
    print("LEADERBOARD (sum of scores across tasks, higher = better)")
    print("=" * 80)
    print(f"{'Model':40} {'Avg':>5} {'Sum':>5} {'Latency':>9} {'Cost':>10} {'Status'}")
    ranked = sorted(
        by_model.items(),
        key=lambda kv: (
            -sum(kv[1]) / len(kv[1]),
            sum(r["latency"] for r in results if r["model"] == kv[0]),
        ),
    )
    for model, scores in ranked:
        recs = [r for r in results if r["model"] == model]
        avg = sum(scores) / len(scores)
        total_lat = sum(r["latency"] for r in recs)
        total_cost = sum(r["cost_usd"] for r in recs)
        any_err = any(r.get("error") for r in recs)
        status = "❌" if any_err else "✓"
        print(
            f"{model:40} {avg:5.1f} {sum(scores):5d} {total_lat:8.2f}s ${total_cost:9.6f}  {status}"
        )

    # per-task winner
    print("\n" + "=" * 80)
    print("PER-TASK WINNERS")
    print("=" * 80)
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    for task, recs in by_task.items():
        winner = max(recs, key=lambda r: r["total"])
        print(
            f"  {task:20} → {winner['model']:35} "
            f"score={winner['total']:3d} lat={winner['latency']:5.2f}s"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", help="filter to specific model id")
    p.add_argument("--task", action="append", help="filter to specific task name")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / ".claude" / "state" / "cheap-bench" / "results.jsonl",
    )
    args = p.parse_args()

    model_filter = set(args.model) if args.model else None
    task_filter = set(args.task) if args.task else None

    print(f"Running benchmark: {len(CANDIDATES)} models × {len(TASKS)} tasks")
    if model_filter:
        print(f"  model filter: {model_filter}")
    if task_filter:
        print(f"  task filter: {task_filter}")

    out = run_benchmark(model_filter, task_filter, timeout=args.timeout)
    results = out["results"]
    print_leaderboard(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        ts = int(time.time())
        for r in results:
            r["ts"] = ts
            f.write(json.dumps(r) + "\n")
    print(f"\nResults appended to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
