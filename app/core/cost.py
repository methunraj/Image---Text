from __future__ import annotations

from typing import Dict, Any, Optional


def calculate_cost(model: str, input_tokens: int, output_tokens: int, catalog: Dict[str, Any]) -> float:
    """Compute USD cost using a legacy catalog mapping.

    Expects pricing like {"pricing": {"input_per_1k": float, "output_per_1k": float}}.
    """
    m = catalog.get(model, {}) if catalog else {}
    pricing = (m.get("pricing") or {}) if isinstance(m, dict) else {}
    in_per_1k = float(pricing.get("input_per_1k", 0.0))
    out_per_1k = float(pricing.get("output_per_1k", 0.0))
    return (input_tokens / 1000.0) * in_per_1k + (output_tokens / 1000.0) * out_per_1k


def _unit_divisor(cost_block: Dict[str, Any]) -> float:
    """Determine token unit divisor from a models.dev 'cost' block.

    Supports 'unit' of '1K tokens', '1M tokens', 'token', defaults to 1K.
    """
    unit = str(cost_block.get("unit", "1K tokens")).strip().lower()
    if "1m" in unit:
        return 1_000_000.0
    if "1k" in unit:
        return 1_000.0
    if "token" in unit:
        return 1.0
    return 1_000.0


def cost_from_usage(usage: Optional[Dict[str, Any]], catalog_caps_json: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute per-run USD cost using provider usage and models.dev pricing.

    - usage keys: prompt_tokens, completion_tokens, cached_read_tokens?, cached_write_tokens?
    - pricing: catalog_caps_json['cost'] with keys input, output, cache_read?, cache_write?, and optional 'unit'.

    Returns a dict with input/output/cache_* and total_usd. If usage is missing,
    returns {"total_usd": None}.
    """
    if not usage or not isinstance(usage, dict):
        return {"total_usd": None}

    # Extract counts safely
    prompt = float(usage.get("prompt_tokens", 0) or 0)
    completion = float(usage.get("completion_tokens", 0) or 0)
    cache_read = float(usage.get("cached_read_tokens", 0) or 0)
    cache_write = float(usage.get("cached_write_tokens", 0) or 0)

    # Extract pricing block; support 'cost' preferred, fallback to 'pricing'
    cost_block = {}
    if isinstance(catalog_caps_json, dict):
        cost_block = (catalog_caps_json.get("cost") or catalog_caps_json.get("pricing") or {})
    unit_div = _unit_divisor(cost_block) if isinstance(cost_block, dict) else 1000.0

    input_price = float((cost_block.get("input") if isinstance(cost_block, dict) else 0) or 0)
    output_price = float((cost_block.get("output") if isinstance(cost_block, dict) else 0) or 0)
    cache_read_price = float((cost_block.get("cache_read") if isinstance(cost_block, dict) else 0) or 0)
    cache_write_price = float((cost_block.get("cache_write") if isinstance(cost_block, dict) else 0) or 0)

    input_usd = (prompt / unit_div) * input_price if input_price else 0.0
    output_usd = (completion / unit_div) * output_price if output_price else 0.0
    cache_read_usd = (cache_read / unit_div) * cache_read_price if cache_read_price else 0.0
    cache_write_usd = (cache_write / unit_div) * cache_write_price if cache_write_price else 0.0
    total = input_usd + output_usd + cache_read_usd + cache_write_usd

    return {
        "input_usd": round(input_usd, 10),
        "output_usd": round(output_usd, 10),
        "cache_read_usd": round(cache_read_usd, 10),
        "cache_write_usd": round(cache_write_usd, 10),
        "total_usd": round(total, 10),
    }
