"""Transport layer — providers, endpoints, and API call functions.

Vertical-slice split of the former ``transport.py`` monolith. Each submodule
owns one responsibility:

- :mod:`constants` — timeouts, endpoint URLs, local-model defaults
- :mod:`pricing`   — pricing tables + reported-vs-estimated cost resolver
- :mod:`httpio`    — bounded response reads + sanitized public errors
- :mod:`providers` — endpoint specs, slug mapping, cascade order
- :mod:`calls`     — per-provider call functions + dispatch

This facade re-exports the full surface so ``from cheap_llm.transport import X``
keeps working unchanged for the package internals (``cascade``/``cli``) and the
public re-export in :mod:`cheap_llm`. Adding a new provider = one
``_PROVIDERS`` entry (providers) + one ``_call_*`` function and one
``_PROVIDER_DISPATCH`` entry (calls).
"""

from __future__ import annotations

from .calls import (
    _PROVIDER_DISPATCH,
    _call_deepinfra,
    _call_deepseek,
    _call_ollama,
    _call_openrouter,
    _call_provider,
    _call_zenmux,
    _ollama_model_loaded,
    _openai_compat_call,
    _strip_reasoning,
)
from .constants import (
    DEEPINFRA_URL,
    DEEPSEEK_URL,
    DEFAULT_LOCAL_PRIMARY,
    DEFAULT_LOCAL_STRUCTURED,
    LOCAL_COLD_TIMEOUT,
    LOCAL_KEEP_ALIVE,
    LOCAL_WARM_TIMEOUT_PRIMARY,
    LOCAL_WARM_TIMEOUT_STRUCTURED,
    MAX_RESPONSE_BYTES,
    OLLAMA_URL,
    OPENROUTER_URL,
    REASONING_EFFORT_OVERRIDES,
    ZENMUX_URL,
)
from .httpio import (
    _normalize_model_name,
    _public_attempt_error,
    _read_json_response,
    _ReadableResponse,
)
from .pricing import (
    DEEPINFRA_PRICING,
    DEEPSEEK_PRICING,
    MODEL_PRICING,
    ZENMUX_DEFAULT_MULTIPLIER,
    ZENMUX_MODEL_MULTIPLIERS,
    ZENMUX_MODEL_PRICING,
    _resolve_cost,
)
from .providers import (
    _PROVIDERS,
    DEEPINFRA_ENDPOINT,
    LEGACY_CASCADE,
    OPENROUTER_ENDPOINT,
    TOP3_CASCADE,
    ZENMUX_ENDPOINT,
    _Endpoint,
    _normalize_deepinfra_model,
    _provider_billing,
    _provider_spec,
    _ProviderSpec,
)

__all__ = [
    # constants
    "DEFAULT_LOCAL_PRIMARY",
    "DEFAULT_LOCAL_STRUCTURED",
    "LOCAL_COLD_TIMEOUT",
    "LOCAL_KEEP_ALIVE",
    "LOCAL_WARM_TIMEOUT_PRIMARY",
    "LOCAL_WARM_TIMEOUT_STRUCTURED",
    "MAX_RESPONSE_BYTES",
    "REASONING_EFFORT_OVERRIDES",
    "OPENROUTER_URL",
    "ZENMUX_URL",
    "DEEPSEEK_URL",
    "DEEPINFRA_URL",
    "OLLAMA_URL",
    # pricing
    "MODEL_PRICING",
    "DEEPSEEK_PRICING",
    "DEEPINFRA_PRICING",
    "ZENMUX_MODEL_PRICING",
    "ZENMUX_MODEL_MULTIPLIERS",
    "ZENMUX_DEFAULT_MULTIPLIER",
    "_resolve_cost",
    # httpio
    "_ReadableResponse",
    "_read_json_response",
    "_public_attempt_error",
    "_normalize_model_name",
    # providers
    "_Endpoint",
    "_ProviderSpec",
    "_PROVIDERS",
    "_provider_spec",
    "_provider_billing",
    "_normalize_deepinfra_model",
    "OPENROUTER_ENDPOINT",
    "ZENMUX_ENDPOINT",
    "DEEPINFRA_ENDPOINT",
    "TOP3_CASCADE",
    "LEGACY_CASCADE",
    # calls
    "_strip_reasoning",
    "_ollama_model_loaded",
    "_call_ollama",
    "_openai_compat_call",
    "_call_openrouter",
    "_call_zenmux",
    "_call_deepseek",
    "_call_deepinfra",
    "_PROVIDER_DISPATCH",
    "_call_provider",
]
