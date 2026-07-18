from _testlib import *  # noqa: E402,F401,F403  -- harness + shared fixtures

print("\n=== UNIT: secret scrub coverage ===")

SCRUB_CASES = [
    (
        "bearer",
        "Authorization: Bearer " + synthetic_secret("abc123def456", "ghi789jkl012"),
        "REDACTED_TOKEN",
    ),
    ("postgres conn string", "db=postgres://admin:SuperSecret123@db:5432/x", "REDACTED_USER"),
    ("mongodb conn string", "MONGO=mongodb://u:S3cret%40p@cluster:27017", "REDACTED_USER"),
    ("redis conn string", "redis://default:hunter2@redis:6379", "REDACTED_USER"),
    (
        "PEM block",
        synthetic_secret(
            "-----BEGIN RSA ",
            "PRIVATE KEY-----\n",
            "MIIEpAIBAAKCAQEA\n",
            "-----END RSA ",
            "PRIVATE KEY-----",
        ),
        "REDACTED_PEM_KEY",
    ),
    (
        "PEM dangling begin",
        synthetic_secret("-----BEGIN OPENSSH ", "PRIVATE KEY-----\n", "b3BlbnNz"),
        "REDACTED_PEM_KEY",
    ),
    ("AWS AKIA", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "REDACTED_AWS"),
    (
        "Google AIza",
        "key = " + synthetic_secret("AIzaSyA", "1234567890abcdefghijklmnopqrstuv"),
        "REDACTED_GCP",
    ),
    (
        "GitHub ghp_",
        "token: " + synthetic_secret("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789AB"),
        "REDACTED_GH",
    ),
    (
        "GitHub PAT",
        "GITHUB_PAT="
        + synthetic_secret("github_pat_", "11ABCDEFGHIJKLMNOPQRSTUVWXabcdefghijklmnopqrstuvwxyz"),
        "REDACTED_GH",
    ),
    (
        "Stripe",
        "stripe: " + synthetic_secret("sk_test_", "51HqabcdefGHIJKLMN0123456789abcd"),
        "REDACTED_STRIPE",
    ),
    (
        "JWT",
        "jwt "
        + synthetic_secret(
            "eyJhbGciOiJIUzI1NiJ9",
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            ".SflKxwRJsignature1234567",
        ),
        "REDACTED_JWT",
    ),
]
for name, inp, marker in SCRUB_CASES:
    out_s = cl.scrub_secrets(inp)
    # the live secret payload must NOT survive scrubbing
    leaked = any(
        need in out_s
        for need in (
            "SuperSecret123",
            "hunter2",
            "MIIEpAIBAAKCAQEA",
            "AKIAIOSFODNN7EXAMPLE",
            "AIzaSyA1234567890abcdefghijklmnopqrstuvw",
            "SflKxwRJ",
        )
    )
    check(f"scrub {name} → marker present", marker in out_s, detail=f"out={out_s[:80]}")
    check(f"scrub {name} → no live secret", not leaked, detail=f"LEAKED in {out_s[:80]}")
check(
    "scrub non-secrets unchanged",
    cl.scrub_secrets("just a normal log line about passwords")
    == "just a normal log line about passwords",
)
# url with no creds must be preserved (no false-positive redaction)
check(
    "scrub leaves credential-free URL intact",
    "https://example.com/path?x=1" in cl.scrub_secrets("see https://example.com/path?x=1"),
)

# --- trailing-comma JSON leniency (only fires on strict-parse failure) ---
check(
    "parse trailing-comma JSON",
    cl._try_parse_json('{"category": "debug", "reason": "x",}')
    == {"category": "debug", "reason": "x"},
)
check(
    "parse valid JSON unchanged (no comma edit)",
    cl._try_parse_json('{"a": "1,2", "b": 2}') == {"a": "1,2", "b": 2},
)
check("reject truly malformed", cl._try_parse_json('{"a": ') is None)

# --- cost estimate fills ZenMux/None gap ---
est = cl._resolve_cost(
    "openai/gpt-5.4-nano", {"prompt_tokens": 1000, "completion_tokens": 500, "cost": None}
)
check(
    "cost estimate when api_cost is None", est is not None and 0 < est < 0.01, detail=f"est={est}"
)
rep = cl._resolve_cost(
    "inclusionai/ling-2.6-flash", {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.000123}
)
check("reported cost (>0) returned as-is", abs(rep - 0.000123) < 1e-12, detail=f"rep={rep}")
di_rep = cl._resolve_cost("provider/new-model", {"estimated_cost": 0.000321}, provider="deepinfra")
check(
    "deepinfra estimated_cost returned as-is",
    di_rep is not None and abs(di_rep - 0.000321) < 1e-12,
    detail=f"cost={di_rep}",
)


# =================================================================
# MOCKED: cascade logic with provider functions stubbed
# =================================================================
