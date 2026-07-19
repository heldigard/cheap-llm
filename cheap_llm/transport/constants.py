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

# Warm T1 budgets (model already in VRAM). Structured JSON is slower than free
# text on the same GPU; 2026-07 native-desktop smoke on RTX 5080 16GB measured
# ~2.9s free-text and ~12.5s schema JSON — so structured must not sit at 12s
# or valid local answers fall through to PAYG cloud. eff_timeout still clamps
# to the caller's timeout_total.
LOCAL_WARM_TIMEOUT_PRIMARY = float(os.environ.get("CHEAP_LLM_LOCAL_WARM_TIMEOUT", "8"))
LOCAL_WARM_TIMEOUT_STRUCTURED = float(os.environ.get("CHEAP_LLM_LOCAL_STRUCTURED_TIMEOUT", "18"))

# T1 budget when the local model is NOT loaded in VRAM yet (cold start).
# Warm budgets stay above; cold always wins when the model is unloaded.
LOCAL_COLD_TIMEOUT = float(os.environ.get("CHEAP_LLM_LOCAL_COLD_TIMEOUT", "25"))

# Keep the T1 model resident after a completion so the next preprocessor slot
# avoids a cold load. Ollama accepts duration strings ("15m") or seconds.
# Set CHEAP_LLM_KEEP_ALIVE=0/off/false to restore Ollama's own default unload.
_KEEP_ALIVE_RAW = os.environ.get("CHEAP_LLM_KEEP_ALIVE", "15m").strip()
if _KEEP_ALIVE_RAW.lower() in ("", "0", "false", "off", "no"):
    LOCAL_KEEP_ALIVE: str | int | None = None
elif _KEEP_ALIVE_RAW.lstrip("-").isdigit():
    LOCAL_KEEP_ALIVE = int(_KEEP_ALIVE_RAW)
else:
    LOCAL_KEEP_ALIVE = _KEEP_ALIVE_RAW

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
