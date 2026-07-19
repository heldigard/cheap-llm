"""Cost accounting — pricing tables and the reported-vs-estimated resolver.

Provider-specific prices avoid pretending the same model costs the same at
every endpoint. Values are USD per 1M tokens and are only fallbacks used when
a response carries no positive ``usage.cost``.
"""

from __future__ import annotations

# Public listing price per 1M tokens (input, output) in USD — used to estimate
# cost when a provider returns usage.cost=None (ZenMux always; OpenRouter for
# some promo/preview models). Source: OpenRouter catalog + tested-models.md.
# ZenMux has its own table below; this generic table is only a last-resort
# estimate for models without an exact provider listing.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "inclusionai/ling-2.6-flash": (0.01, 0.03),
    "inclusionai/ling-2.6-1t": (0.075, 0.625),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "openai/gpt-5.4-nano": (0.20, 1.25),
    "moonshotai/kimi-k2": (0.57, 2.30),  # kept for cost lookup only
    "deepseek/deepseek-v4-flash": (0.098, 0.196),
    "kwaipilot/kat-coder-air-v2.5": (0.15, 0.60),
    "kwaipilot/kat-coder-pro-v2.5": (0.74, 2.96),
}

# Direct DeepSeek prices include its unusually deep cache discount; DeepInfra
# slugs use the public Flex listing.
DEEPSEEK_PRICING: dict[str, tuple[float, float, float]] = {
    # model: (fresh input, cached input, output)
    "deepseek/deepseek-v4-flash": (0.14, 0.0028, 0.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.003625, 0.87),
}
DEEPINFRA_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-ai/DeepSeek-V4-Flash": (0.09, 0.18),
    "deepseek-ai/DeepSeek-V4-Pro": (1.30, 2.60),
    "Qwen/Qwen3.7-Max": (2.50, 7.50),
    "zai-org/GLM-5.2": (0.93, 3.00),
    "XiaomiMiMo/MiMo-V2.5-Pro": (1.00, 3.00),
    "moonshotai/Kimi-K2.7-Code": (0.74, 3.50),
}

# Exact ZenMux public catalog prices, when published. Refreshed 2026-07-19
# from GET /api/v1/models. For tiered models this table intentionally records
# the highest published text-token tier so fallback telemetry does not
# understate spend. The multiplier fallback remains deliberately conservative
# for models whose catalog omits pricing.
ZENMUX_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "inclusionai/ling-2.6-1t": (0.1318155, 1.0984625),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "deepseek/deepseek-v4-flash": (0.14, 0.28),
    "kuaishou/kat-coder-air-v2.5": (0.135, 0.54),
    "kuaishou/kat-coder-pro-v2.5": (0.444, 1.776),
    "bytedance/doubao-seed-2.1-pro": (0.422571632, 2.11285816),
    "bytedance/doubao-seed-2.1-turbo": (0.422571632, 2.11285816),
    "baidu/ernie-5.1": (0.636889554, 2.335983795),
    "x-ai/grok-build-0.1": (1.0, 2.0),
    "stepfun/step-3.7-flash": (0.1350354, 0.77645355),
    "qwen/qwen3.7-plus": (0.4119228, 1.6476912),
    "tencent/hy3": (0.1323, 0.5301),
    "qwen/qwen3.7-max": (0.4307775, 1.2923325),
    "z-ai/glm-5.2": (0.98, 3.08),
    "minimax/minimax-m3": (0.2746152, 1.0984608),
    "xiaomi/mimo-v2.5-pro": (0.43499984, 0.86999968),
}
ZENMUX_MODEL_MULTIPLIERS: dict[str, float] = {
    "inclusionai/ling-2.6-flash": 10.0,
}
ZENMUX_DEFAULT_MULTIPLIER = 5.0


def _resolve_cost(model: str, usage: dict, provider: str = "openrouter") -> float | None:
    """Reported API cost if present, else estimate from the listing price.

    ZenMux returns usage.cost=None and OpenRouter returns $0 for some promo
    models; without an estimate those calls show $0 in telemetry, which hides
    real spend. Provider prices vary in both directions, so exact ZenMux
    listings take precedence over the generic table. Returns None only when we
    have neither a reported cost nor a known price.
    """
    reported = usage.get("cost")
    # DeepInfra reports an estimate rather than a settled cost field.
    if provider == "deepinfra" and not reported:
        reported = usage.get("estimated_cost")
    if reported is not None and reported > 0 and provider != "zenmux":
        return reported
    price = DEEPINFRA_PRICING.get(model) if provider == "deepinfra" else None
    if provider == "zenmux":
        price = ZENMUX_MODEL_PRICING.get(model)
    if price is None:
        price = MODEL_PRICING.get(model)
    if not price:
        return reported  # may be None/0 — caller treats falsy as 0.0
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    in_per_m, out_per_m = price
    raw_cost = (in_tok * in_per_m + out_tok * out_per_m) / 1_000_000.0
    if provider == "zenmux" and model not in ZENMUX_MODEL_PRICING:
        multiplier = ZENMUX_DEFAULT_MULTIPLIER
        for needle, mult in ZENMUX_MODEL_MULTIPLIERS.items():
            if needle in model:
                multiplier = mult
                break
        return raw_cost * multiplier
    return raw_cost
