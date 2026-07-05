#!/usr/bin/env python3
"""Live + end-to-end tests for the cheap-LLM cascade (real API calls).

Companion to test-cheap-llm.py (which is unit + mocked, offline, 81 tests).
This file exercises the REAL stack:

  LIVE  — cheap_llm.cheap_complete() against real Ollama + OpenRouter + ZenMux.
           Each cascade tier reachable, full resolve, cache hit, scrub on the
           live path (the 2026-06-19 critical-fix regression).
  E2E   — the 5 migrated scripts (intent_route, error-classify, commit-draft,
           diff-review, extract-tool-output) run as SUBPROCESSES, proving the
           whole CLI → cascade → provider → output contract holds end to end.

Cost: a few cents (≤$0.02). Time: ~1-3 min (local qwen3.5:4b is the slow part).
Requires: OPENROUTER_API_KEY (+ ZENMUX_API_KEY for failover tests), Ollama up.

Run:
    python3 test-cheap-llm-live.py            # live + e2e
    python3 test-cheap-llm-live.py --live-only
    python3 test-cheap-llm-live.py --e2e-only
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM_SCRIPTS = Path.home() / ".claude" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("cheap_llm", PROJECT_ROOT / "cheap_llm.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

PASS = 0
FAIL = 0
SKIP = 0
FAILURES: list[str] = []
RESULTS: list[tuple[str, str, str]] = []  # (group, name, detail-line)


def check(group: str, name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    line = f"  {tag}  {name}" + (f"  {detail}" if detail else "")
    print(line)
    RESULTS.append((group, name, f"{tag} {detail}"))
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")


def skip(group: str, name: str, reason: str) -> None:
    global SKIP
    print(f"  SKIP  {name}  ({reason})")
    RESULTS.append((group, name, f"SKIP ({reason})"))
    SKIP += 1


HAVE_OR = bool(os.environ.get("OPENROUTER_API_KEY"))
HAVE_ZM = bool(os.environ.get("ZENMUX_API_KEY"))
HAVE_OLLAMA = False
try:
    import urllib.request
    req = urllib.request.Request(f"{cl.OLLAMA_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=2) as r:
        HAVE_OLLAMA = r.status == 200
except Exception:
    HAVE_OLLAMA = False

CLASSIFY_SYS = ("Classify this developer prompt into one of: trivial, lookup, "
                "code-edit, refactor, feature, debug, architecture, security, "
                'meta. Reply JSON only with keys "category" and "reason".')
CLASSIFY_PROMPT = "I'm getting ECONNREFUSED 127.0.0.1:5432 in my Express app after adding TypeORM."

# intent_route category set (defined in intent_route.py, mirrored here for e2e checks)
IR_CATEGORIES = {"trivial", "lookup", "code-edit", "refactor", "feature",
                 "debug", "architecture", "security", "meta"}


def _safe_get_category(text: str) -> str:
    d = cl._try_parse_json(text)
    if isinstance(d, dict):
        return str(d.get("category", "")).strip().lower()
    return ""

# Opt-in gate (2026-07-02): this file is a LIVE integration test — every case
# makes real cascade calls (network). It flakes under the deterministic unit
# battery when a third-party API hiccups. The UNIT gate is test-cheap-llm.py
# (86/86, mocked). To run THIS one explicitly: pass --live / --live-only /
# --e2e-only, or set CHEAP_LLM_LIVE=1. Plain invocation (the battery) now
# skips cleanly with exit 0 instead of making 14 flaky network calls.
_EXPLICIT = ("--live" in sys.argv or "--live-only" in sys.argv
             or "--e2e-only" in sys.argv or bool(os.environ.get("CHEAP_LLM_LIVE")))
LIVE = ("--e2e-only" not in sys.argv) and _EXPLICIT
E2E = ("--live-only" not in sys.argv) and _EXPLICIT


# =================================================================
# LIVE: real cascade calls
# =================================================================
if LIVE:
    print("\n=== LIVE: real cascade (Ollama + OpenRouter) ===")
    if not (HAVE_OR or HAVE_OLLAMA):
        skip("LIVE", "all live tests", "no OPENROUTER_API_KEY and no Ollama")
    else:
        # L1: T1 local (Ollama qwen3.5:4b) directly reachable
        if HAVE_OLLAMA:
            try:
                r = cl._call_ollama(cl.DEFAULT_LOCAL_PRIMARY, CLASSIFY_SYS,
                                    CLASSIFY_PROMPT, timeout=15)
                txt = r.get("text", "")
                check("LIVE", "T1 qwen3.5:4b@ollama reachable + non-empty",
                      bool(txt) and len(txt) > 5,
                      detail=f"lat={r.get('latency',0):.1f}s out_tok={r.get('output_tokens',0)}")
            except Exception as e:
                check("LIVE", "T1 qwen3.5:4b@ollama reachable", False,
                      detail=f"{type(e).__name__}: {str(e)[:70]}")
        else:
            skip("LIVE", "T1 qwen3.5:4b@ollama reachable", "Ollama down")

        # L2: full cascade, prefer_local=True (the default every caller uses)
        try:
            t0 = time.perf_counter()
            out = cl.cheap_complete(system=CLASSIFY_SYS, prompt=CLASSIFY_PROMPT,
                                    schema_hint=["category", "reason"],
                                    timeout_total=20, prefer_local=True)
            wall = time.perf_counter() - t0
            check("LIVE", "cascade prefer_local=True resolves",
                  out.get("model") is not None and out.get("json_valid") is True,
                  detail=f"model={out.get('model')} tier={out.get('tier')} wall={wall:.1f}s")
            check("LIVE", "cascade returns valid category field",
                  bool(_safe_get_category(out.get("text", ""))),
                  detail=f"category={_safe_get_category(out.get('text',''))}")
        except Exception as e:
            check("LIVE", "cascade prefer_local=True resolves", False,
                  detail=f"{type(e).__name__}: {str(e)[:70]}")

        # L3: full cascade, prefer_local=False (cloud-first)
        try:
            out = cl.cheap_complete(system=CLASSIFY_SYS, prompt=CLASSIFY_PROMPT,
                                    schema_hint=["category", "reason"],
                                    timeout_total=20, prefer_local=False)
            check("LIVE", "cascade prefer_local=False resolves on cloud",
                  out.get("model") is not None and out.get("provider") != "ollama",
                  detail=f"model={out.get('model')} provider={out.get('provider')} cost=${out.get('cost',0):.6f}")
        except Exception as e:
            check("LIVE", "cascade prefer_local=False resolves", False,
                  detail=f"{type(e).__name__}: {str(e)[:70]}")

        # L4: cache hit — repeat identical call → cached=True, no provider call
        try:
            cl._cache_put  # ensure present
            out1 = cl.cheap_complete(system="Cache probe.", prompt="identical-cache-key-prompt-xyz",
                                     schema_hint=["category"], timeout_total=20, prefer_local=False)
            # record how many attempts the first call made, then second must be cached
            out2 = cl.cheap_complete(system="Cache probe.", prompt="identical-cache-key-prompt-xyz",
                                     schema_hint=["category"], timeout_total=20, prefer_local=False)
            check("LIVE", "repeat call hits cache",
                  out2.get("cached") is True and out2.get("latency", 99) < 0.05,
                  detail=f"cached={out2.get('cached')} lat={out2.get('latency',0):.3f}s")
        except Exception as e:
            check("LIVE", "repeat call hits cache", False,
                  detail=f"{type(e).__name__}: {str(e)[:70]}")

        # L5: scrub confirmed on the LIVE path (critical-fix regression).
        # Spy on _call_provider: assert a planted secret never reaches it,
        # while the call still resolves normally.
        try:
            seen: dict = {}
            orig = cl._call_provider

            def spy(model, provider, system, prompt, timeout):
                seen["prompt"] = prompt
                seen["system"] = system
                return orig(model, provider, system, prompt, timeout)

            cl._call_provider = spy
            try:
                out = cl.cheap_complete(
                    system=CLASSIFY_SYS,
                    prompt="DEBUG: Authorization: Bearer eyJhbGc.iO.SflKx; "
                           "db=postgres://admin:Hunter2Secret@db:5432 — classify this",
                    schema_hint=["category", "reason"],
                    timeout_total=20, prefer_local=True)
            finally:
                cl._call_provider = orig
            leaked = any(s in (seen.get("prompt", "") + seen.get("system", ""))
                         for s in ("eyJhbGc", "Hunter2Secret"))
            check("LIVE", "secret scrubbed before live provider call",
                  not leaked and out.get("model") is not None,
                  detail=f"leaked={leaked} resolved={out.get('model') is not None}")
        except Exception as e:
            check("LIVE", "secret scrubbed before live provider call", False,
                  detail=f"{type(e).__name__}: {str(e)[:70]}")

        # L6: ZenMux reachable independently (failover path is real)
        if HAVE_ZM:
            try:
                r = cl._call_zenmux(cl.TOP3_CASCADE[0][0], CLASSIFY_SYS,
                                    CLASSIFY_PROMPT, timeout=15)
                check("LIVE", "ZenMux failover tier reachable",
                      bool(r.get("text")),
                      detail=f"provider={r.get('provider')} cost_est=${r.get('api_cost',0) or 0:.6f}")
            except Exception as e:
                check("LIVE", "ZenMux failover tier reachable", False,
                      detail=f"{type(e).__name__}: {str(e)[:70]}")
        else:
            skip("LIVE", "ZenMux failover tier reachable", "ZENMUX_API_KEY not set")
else:
    print("\n=== LIVE: (--e2e-only, skipped) ===")


# =================================================================
# E2E: the 5 migrated scripts as subprocesses
# =================================================================
def run_script(name: str, args: list[str], stdin: str | None = None,
               timeout: int = 90) -> tuple[int, str, str]:
    """Run a script under ECOSYSTEM_SCRIPTS; return (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            [sys.executable, str(ECOSYSTEM_SCRIPTS / name)] + args,
            input=stdin, capture_output=True, text=True, timeout=timeout,
            cwd=str(ECOSYSTEM_SCRIPTS),
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


if E2E:
    print("\n=== E2E: migrated scripts via subprocess ===")
    if not (HAVE_OR or HAVE_OLLAMA):
        skip("E2E", "all e2e tests", "no API key and no Ollama")
    else:
        # E1: intent_route — trivial prompt → cheap tier
        rc, out, err = run_script("intent_route.py",
                                  ["--prompt", "fix a typo in the README title", "--json", "--no-log"])
        cat = ""
        try:
            cat = json.loads(out).get("category", "") if rc == 0 else ""
        except Exception:
            pass
        check("E2E", "intent_route: trivial prompt classified",
              rc == 0 and cat in IR_CATEGORIES,
              detail=f"rc={rc} category={cat!r} {err.strip()[:60]}")

        # E2: intent_route — architecture prompt → T3 tier hint
        rc, out, err = run_script("intent_route.py",
                                  ["--prompt", "design a zero-trust OAuth2 flow for a multi-tenant SaaS with per-tenant key isolation",
                                   "--json", "--no-log"])
        tier = ""
        try:
            tier = json.loads(out).get("tier", "") if rc == 0 else ""
        except Exception:
            pass
        check("E2E", "intent_route: architecture prompt → T3 hint",
              rc == 0 and tier == "T3",
              detail=f"rc={rc} tier={tier!r} {err.strip()[:60]}")

        # E3: error-classify — deterministic catalog match (no LLM needed)
        rc, out, err = run_script("error-classify.py",
                                  ["--text", "n8n D365 request failed 0x80072530 on PATCH"])
        check("E2E", "error-classify: catalog match (0x80072530 → bodyless PATCH)",
              rc == 0 and "body" in out.lower() and "patch" in out.lower(),
              detail=f"rc={rc} {out.strip()[:80]!r}")

        # E4: error-classify — novel error → LLM hypothesis (System/Cause/Fix block)
        rc, out, err = run_script("error-classify.py",
                                  ["--text", "Frobnicator exceeded quantum throughput in the wibbly subsystem at sector 7G"])
        has_block = ("system:" in out.lower() or "cause:" in out.lower()) and "fix:" in out.lower()
        check("E2E", "error-classify: novel error → LLM hypothesis block",
              rc == 0 and has_block,
              detail=f"rc={rc} {out.strip()[:80]!r}")

        # E5: commit-draft — sample diff → valid Conventional Commits message
        sample_diff = (
            "diff --git a/src/auth/middleware.ts b/src/auth/middleware.ts\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/src/auth/middleware.ts\n"
            "@@ -0,0 +1,4 @@\n"
            "+import jwt from 'jsonwebtoken';\n"
            "+export function requireAuth(req, res, next) {\n"
            "+  const token = req.headers.authorization?.split(' ')[1];\n"
            "+}\n"
        )
        diff_path = PROJECT_ROOT / "_e2e_sample.diff"
        diff_path.write_text(sample_diff)
        try:
            rc, out, err = run_script("commit-draft.py", ["--file", str(diff_path)])
            subject = out.strip().splitlines()[0] if out.strip() else ""
            conv = re.match(r"^(feat|fix|chore|docs|test|build|refactor|perf|style|ci)(\([^)]+\))?: .+",
                            subject)
            check("E2E", "commit-draft: conventional commit subject",
                  rc == 0 and conv is not None,
                  detail=f"rc={rc} subject={subject[:60]!r}")
        finally:
            diff_path.unlink(missing_ok=True)

        # E6: diff-review — SQL injection diff → flagged
        vuln_diff = (
            "diff --git a/src/db/query.ts b/src/db/query.ts\n"
            "--- a/src/db/query.ts\n"
            "+++ b/src/db/query.ts\n"
            "@@ -0,0 +1,3 @@\n"
            "+function findUser(user, pass) {\n"
            '+  return db.query("SELECT * FROM users WHERE name=\'" + user + "\'");\n'
            "+}\n"
        )
        vuln_path = PROJECT_ROOT / "_e2e_vuln.diff"
        vuln_path.write_text(vuln_diff)
        try:
            rc, out, err = run_script("diff-review.py", ["--file", str(vuln_path)])
            low = out.lower()
            flagged = ("sql" in low or "injection" in low or "concat" in low)
            check("E2E", "diff-review: flags SQL injection in diff",
                  rc == 0 and flagged,
                  detail=f"rc={rc} flagged={flagged} {err.strip()[:50]}")
        finally:
            vuln_path.unlink(missing_ok=True)

        # E7: extract-tool-output — sample log → extraction header
        log_lines = ["[INFO] server starting on port 3000",
                     "[ERROR] ECONNREFUSED 127.0.0.1:5432 — postgres unavailable"]
        log_lines += [f"[DEBUG] tick {i} ok" for i in range(400)]  # pad past token threshold
        log_path = PROJECT_ROOT / "_e2e_sample.log"
        log_path.write_text("\n".join(log_lines))
        try:
            rc, out, err = run_script("extract-tool-output.py",
                                      ["--file", str(log_path), "--query", "ECONNREFUSED postgres error",
                                       "--threshold", "1"])
            check("E2E", "extract-tool-output: produces extraction header",
                  rc == 0 and "extract-tool-output:" in out,
                  detail=f"rc={rc} has_header={'extract-tool-output:' in out} {err.strip()[:50]}")
        finally:
            log_path.unlink(missing_ok=True)
else:
    print("\n=== E2E: (--live-only, skipped) ===")


# =================================================================
# Summary
# =================================================================
print(f"\n{'='*64}")
print(f"LIVE+E2E  PASS: {PASS}   FAIL: {FAIL}   SKIP: {SKIP}")
if FAILURES:
    print("\nFailures:")
    for f in FAILURES:
        print(f"  - {f}")
print(f"{'='*64}")
sys.exit(0 if FAIL == 0 else 1)
