"""Secret scrubbing — remove secret-looking strings before sending to models.

Patterns we never send to a third-party. Scrubbed in both system and prompt.
"""

from __future__ import annotations

import re

SECRET_PATTERNS = [
    # PEM private-key block (RSA/EC/OPENSSH/PGP) — full block first, then a
    # dangling BEGIN for truncated logs/diffs that never reach an END line.
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]*)PRIVATE KEY-----.*?"
            r"-----END (?:[A-Z0-9 ]*)PRIVATE KEY-----",
            re.S,
        ),
        "<REDACTED_PEM_KEY>",
    ),
    (re.compile(r"-----BEGIN (?:[A-Z0-9 ]*)PRIVATE KEY-----[^\n]*"), "<REDACTED_PEM_KEY>"),
    # DB / message-broker connection strings with embedded creds:
    # postgres://user:pass@host, mongodb+srv://.., redis://.., amqp(s)://..
    (
        re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^:/@\s]+):([^@\s]+)@"),
        r"\1<REDACTED_USER>:<REDACTED_PASS>@",
    ),
    # env-style assignment with quote
    (
        re.compile(
            r"(?i)(password|passwd|secret|api[_-]?key|api[_-]?token|access[_-]?token|"
            r'auth[_-]?token|private[_-]?key)\s*[:=]\s*["\'][^"\']{6,}["\']'
        ),
        r"\1=<REDACTED>",
    ),
    # bare bearer / sk- / ghp_ / gho_ style
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-+/=]{16,}"), r"\1<REDACTED_TOKEN>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "<REDACTED_SK>"),
    # GitHub tokens: classic (ghp/gho/ghu/ghs/ghr, 36+ payload) + fine-grained PAT
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"), "<REDACTED_GH>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"), "<REDACTED_GH>"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "<REDACTED_XOX>"),
    # cloud provider keys
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS>"),  # AWS access-key id
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "<REDACTED_GCP>"),  # Google API key
    # Google OAuth access token (used by Drive/Gmail/etc SDKs). Format:
    # ya29.<base64url ~80-150 chars>. Distinguished from AIza (API key) by the
    # "ya29." prefix and is widely present in Gmail/Drive stack traces.
    (re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}"), "<REDACTED_GOOGLE_OAUTH>"),
    (re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9a-zA-Z]{20,}\b"), "<REDACTED_STRIPE>"),
    # SendGrid API key (SG.<22 chars>.<43 chars>) and Mailgun (key-<32 hex>)
    (re.compile(r"\bSG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"), "<REDACTED_SENDGRID>"),
    (re.compile(r"\bkey-[a-f0-9]{32}\b"), "<REDACTED_MAILGUN>"),
    # Slack incoming webhook URL — workspace + channel + token. Common in
    # CI logs and bot incident dumps.
    (
        re.compile(r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"),
        "<REDACTED_SLACK_WEBHOOK>",
    ),
    # PEM certificate block (sometimes embedded in logs alongside private keys
    # by accident — redacting the cert block too avoids leaking SANs/CN).
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]*)CERTIFICATE-----.*?-----END (?:[A-Z0-9 ]*)CERTIFICATE-----",
            re.S,
        ),
        "<REDACTED_CERTIFICATE>",
    ),
    # JWT-ish (3 base64 segments). Also catch the standalone JWT header
    # `eyJ...` (no dots) which appears when partial tokens are logged.
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
        "<REDACTED_JWT>",
    ),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}"), "<REDACTED_JWT_HEADER>"),
]


def scrub_secrets(text: str) -> str:
    """Remove secret-looking strings before sending to a third-party model."""
    out = text
    for pat, repl in SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out
