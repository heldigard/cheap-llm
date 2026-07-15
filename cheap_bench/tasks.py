"""Benchmark task set — the fixed prompts every candidate is scored against.

Each task: {name, system, prompt, schema, ...reference fields for scoring}.
schema: list of required top-level JSON fields.
Reference fields (ref_category / ref_type / ref_scope / ref_system) drive the
task-specific content-quality heuristic in ``scoring``.
"""

from __future__ import annotations

# intent_classify, commit_draft, error_classify, json_extract, diff_review — the
# five preprocessor slots cheap-llm distills signal for.
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
+""",
        "schema": ["findings"],
        # Should flag SQL injection + bare TODO
    },
]
