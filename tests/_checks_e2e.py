from _livetestlib import *  # noqa: E402,F401,F403  -- harness + gate


# =================================================================
# E2E: the 5 migrated scripts as subprocesses
# =================================================================
def run_script(
    name: str, args: list[str], stdin: str | None = None, timeout: int = 90
) -> tuple[int, str, str]:
    """Run a script under ECOSYSTEM_SCRIPTS; return (rc, stdout, stderr)."""
    try:
        p = subprocess.run(
            [sys.executable, str(ECOSYSTEM_SCRIPTS / name)] + args,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
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
        rc, out, err = run_script(
            "intent_route.py", ["--prompt", "fix a typo in the README title", "--json", "--no-log"]
        )
        cat = ""
        try:
            cat = json.loads(out).get("category", "") if rc == 0 else ""
        except Exception:
            pass
        check(
            "E2E",
            "intent_route: trivial prompt classified",
            rc == 0 and cat in IR_CATEGORIES,
            detail=f"rc={rc} category={cat!r} {err.strip()[:60]}",
        )

        # E2: intent_route — architecture prompt → T3 tier hint
        rc, out, err = run_script(
            "intent_route.py",
            [
                "--prompt",
                (
                    "design a zero-trust OAuth2 flow for a multi-tenant SaaS "
                    "with per-tenant key isolation"
                ),
                "--json",
                "--no-log",
            ],
        )
        tier = ""
        try:
            tier = json.loads(out).get("tier", "") if rc == 0 else ""
        except Exception:
            pass
        check(
            "E2E",
            "intent_route: architecture prompt → T3 hint",
            rc == 0 and tier == "T3",
            detail=f"rc={rc} tier={tier!r} {err.strip()[:60]}",
        )

        # E3: error-classify — deterministic catalog match (no LLM needed)
        rc, out, err = run_script(
            "error-classify.py", ["--text", "n8n D365 request failed 0x80072530 on PATCH"]
        )
        check(
            "E2E",
            "error-classify: catalog match (0x80072530 → bodyless PATCH)",
            rc == 0 and "body" in out.lower() and "patch" in out.lower(),
            detail=f"rc={rc} {out.strip()[:80]!r}",
        )

        # E4: error-classify — novel error → LLM hypothesis (System/Cause/Fix block)
        rc, out, err = run_script(
            "error-classify.py",
            [
                "--text",
                "Frobnicator exceeded quantum throughput in the wibbly subsystem at sector 7G",
            ],
        )
        has_block = ("system:" in out.lower() or "cause:" in out.lower()) and "fix:" in out.lower()
        check(
            "E2E",
            "error-classify: novel error → LLM hypothesis block",
            rc == 0 and has_block,
            detail=f"rc={rc} {out.strip()[:80]!r}",
        )

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
            conv = re.match(
                r"^(feat|fix|chore|docs|test|build|refactor|perf|style|ci)(\([^)]+\))?: .+", subject
            )
            check(
                "E2E",
                "commit-draft: conventional commit subject",
                rc == 0 and conv is not None,
                detail=f"rc={rc} subject={subject[:60]!r}",
            )
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
            flagged = "sql" in low or "injection" in low or "concat" in low
            check(
                "E2E",
                "diff-review: flags SQL injection in diff",
                rc == 0 and flagged,
                detail=f"rc={rc} flagged={flagged} {err.strip()[:50]}",
            )
        finally:
            vuln_path.unlink(missing_ok=True)

        # E7: extract-tool-output — sample log → extraction header
        log_lines = [
            "[INFO] server starting on port 3000",
            "[ERROR] ECONNREFUSED 127.0.0.1:5432 — postgres unavailable",
        ]
        log_lines += [f"[DEBUG] tick {i} ok" for i in range(400)]  # pad past token threshold
        log_path = PROJECT_ROOT / "_e2e_sample.log"
        log_path.write_text("\n".join(log_lines))
        try:
            rc, out, err = run_script(
                "extract-tool-output.py",
                [
                    "--file",
                    str(log_path),
                    "--query",
                    "ECONNREFUSED postgres error",
                    "--threshold",
                    "1",
                ],
            )
            check(
                "E2E",
                "extract-tool-output: produces extraction header",
                rc == 0 and "extract-tool-output:" in out,
                detail=f"rc={rc} has_header={'extract-tool-output:' in out} {err.strip()[:50]}",
            )
        finally:
            log_path.unlink(missing_ok=True)
else:
    print("\n=== E2E: (--live-only, skipped) ===")
