"""Transport constants — timeouts, endpoints URLs, and local-model defaults.

Pure configuration scalars. Pricing tables live in ``pricing``; the provider
registry lives in ``providers``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Local T1 defaults
# ---------------------------------------------------------------------------

# cryptidbleh/gemma4-claude-opus-4.6 is the free-text compatibility default and
# matches ollama_client.DEFAULT_GEN_MODEL. JSON/schema calls use the measured
# structured-output specialist unless callers pass an explicit `model=...`.
DEFAULT_LOCAL_PRIMARY = "cryptidbleh/gemma4-claude-opus-4.6:latest"
DEFAULT_LOCAL_STRUCTURED = "SetneufPT/Qwopus3.5-4B-Coder-MTP_Q4_64k_8GB-GPU:latest"

# T1 budget when the local model is NOT loaded in VRAM yet (cold start).
# Warm budgets stay 6s/12s; eff_timeout always clamps to the caller's
# timeout_total, so callers with tight deadlines are unaffected.
LOCAL_COLD_TIMEOUT = float(os.environ.get("CHEAP_LLM_LOCAL_COLD_TIMEOUT", "25"))

# External responses are untrusted input. Token ceilings constrain model
# generation but do not constrain a broken/proxied HTTP response, so every
# transport also enforces a byte ceiling before decoding or JSON parsing.
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Reasoning control for OpenAI-compatible aggregators. Direct DeepSeek uses
# its provider-specific ``thinking`` toggle (see ``calls``).
REASONING_EFFORT_OVERRIDES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Provider endpoint URLs
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ZENMUX_URL = "https://zenmux.ai/api/v1"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
