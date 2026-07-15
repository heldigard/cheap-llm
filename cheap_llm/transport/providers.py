"""Provider registry — endpoint specs, slug mapping, and cascade order.

Adding a new provider = one new ``_PROVIDERS`` entry (plus one ``_call_*``
function and ``_PROVIDER_DISPATCH`` entry in ``calls``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    DEEPINFRA_URL,
    DEEPSEEK_URL,
    OPENROUTER_URL,
    ZENMUX_URL,
)


@dataclass(frozen=True)
class _Endpoint:
    """OpenAI-compatible chat-completions endpoint config.

    Bundles url + key_env + provider_label + headers so the call-site helper
    only sees one endpoint token instead of 4-5 positional params.
    """

    url: str
    key_env: str
    provider_label: str
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _ProviderSpec:
    """A unified provider spec — extends _Endpoint with slug_map and probe URL."""

    endpoint: _Endpoint
    slug_map: dict[str, str] = field(default_factory=dict)
    probe_url: str | None = None


_PROVIDERS: dict[str, _ProviderSpec] = {
    "openrouter": _ProviderSpec(
        endpoint=_Endpoint(
            url=OPENROUTER_URL,
            key_env="OPENROUTER_API_KEY",
            provider_label="openrouter",
            extra_headers={"X-OpenRouter-Title": "cheap-llm-cascade"},
        ),
        probe_url=f"{OPENROUTER_URL}/models",
    ),
    "zenmux": _ProviderSpec(
        endpoint=_Endpoint(
            url=ZENMUX_URL,
            key_env="ZENMUX_API_KEY",
            provider_label="zenmux",
        ),
        probe_url=f"{ZENMUX_URL}/models",
    ),
    "deepinfra": _ProviderSpec(
        endpoint=_Endpoint(
            url=DEEPINFRA_URL,
            key_env="DEEPINFRA_API_KEY",
            provider_label="deepinfra",
        ),
        slug_map={
            "deepseek/deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek/deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
            "qwen3.7-max": "Qwen/Qwen3.7-Max",
            "glm-5.2": "zai-org/GLM-5.2",
            "mimo-v2.5-pro": "XiaomiMiMo/MiMo-V2.5-Pro",
            "kimi-k2.7-code": "moonshotai/Kimi-K2.7-Code",
        },
        probe_url=f"{DEEPINFRA_URL}/models",
    ),
    "deepseek": _ProviderSpec(
        endpoint=_Endpoint(
            url=DEEPSEEK_URL,
            key_env="DEEPSEEK_API_KEY",
            provider_label="deepseek",
        ),
        probe_url=f"{DEEPSEEK_URL}/models",
    ),
}


def _provider_spec(name: str) -> _ProviderSpec:
    spec = _PROVIDERS.get(name)
    if spec is None:
        raise ValueError(f"unknown provider: {name}")
    return spec


def _normalize_deepinfra_model(model: str) -> str:
    """Map generic/OpenRouter model slugs to DeepInfra-specific slugs."""
    low = model.lower()
    for needle, slug in _PROVIDERS["deepinfra"].slug_map.items():
        if needle in low:
            return slug
    return model


def _provider_billing(provider: str) -> str:
    """Billing class for a provider slug, without exposing credential values.

    ``"local"`` for Ollama (free, private T1); ``"payg"`` for every cloud route
    in ``_PROVIDERS`` (openrouter/zenmux/deepseek/deepinfra are pay-as-you-go
    capacity, not CLI-seat subscriptions — those live in ``cli-orchestration``
    and ``fusion-local``, a separate authority lane). Unknown slugs return
    ``"unknown"`` rather than raising, since this feeds route-plan display and
    cascade telemetry where a crash is worse than an unspecific label.
    """
    if provider == "ollama":
        return "local"
    if provider in _PROVIDERS:
        return "payg"
    return "unknown"


# Derived endpoints (single source of truth = _PROVIDERS)
OPENROUTER_ENDPOINT = _provider_spec("openrouter").endpoint
ZENMUX_ENDPOINT = _provider_spec("zenmux").endpoint
DEEPINFRA_ENDPOINT = _provider_spec("deepinfra").endpoint


# Cascade as (model, provider) pairs. For each top model we try OpenRouter
# first, then ZenMux as backup.
TOP3_CASCADE: list[tuple[str, str]] = [
    ("inclusionai/ling-2.6-flash", "openrouter"),
    ("inclusionai/ling-2.6-flash", "zenmux"),
    ("inclusionai/ling-2.6-1t", "openrouter"),
    ("inclusionai/ling-2.6-1t", "zenmux"),
    ("google/gemini-3.1-flash-lite", "openrouter"),
    ("google/gemini-3.1-flash-lite", "zenmux"),
]

LEGACY_CASCADE: list[tuple[str, str]] = [
    ("openai/gpt-5.4-nano", "openrouter"),
    ("deepseek/deepseek-v4-flash", "openrouter"),
]
