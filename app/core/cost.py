from __future__ import annotations

from typing import Dict, Any, Optional


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
    """Compute per-run USD cost using provider usage and pricing.

    Supports two pricing shapes:
    - models.dev style: {'cost': {'input': <per-unit>, 'output': <per-unit>, 'unit': '1K tokens'|'1M tokens'|'token', ...}}
    - app style: {'pricing': {'input_per_1k': x, 'output_per_1k': y, 'cache_read_per_1k'?, 'cache_write_per_1k'?,
                              'input_per_million'?, 'output_per_million'?, 'cache_read_per_million'?, 'cache_write_per_million'?}}

    Usage keys handled (with fallbacks):
    - prompt:  prompt_tokens | input_tokens | input_token_count
    - completion: completion_tokens | output_tokens | output_token_count
    - cache: cached_read_tokens | cache_read_tokens ; cached_write_tokens | cache_write_tokens

    Returns detailed amounts and total_usd; total_usd is None if usage missing.
    """
    if not usage or not isinstance(usage, dict):
        return {"total_usd": None}

    def _get_first(d: Dict[str, Any], keys: list[str]) -> float:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    continue
        return 0.0

    prompt = _get_first(usage, ["prompt_tokens", "input_tokens", "input_token_count"]) or 0.0
    completion = _get_first(usage, ["completion_tokens", "output_tokens", "output_token_count"]) or 0.0
    cache_read = _get_first(usage, ["cached_read_tokens", "cache_read_tokens"]) or 0.0
    cache_write = _get_first(usage, ["cached_write_tokens", "cache_write_tokens"]) or 0.0

    # Extract pricing block
    block: Dict[str, Any] = {}
    if isinstance(catalog_caps_json, dict):
        block = (catalog_caps_json.get("cost") or catalog_caps_json.get("pricing") or {})  # type: ignore[assignment]
        if not isinstance(block, dict):
            block = {}

    # Case A: models.dev 'cost' with unit and 'input'/'output' keys
    if any(k in block for k in ("input", "output", "cache_read", "cache_write")):
        unit_div = _unit_divisor(block)
        input_price = float(block.get("input") or 0)
        output_price = float(block.get("output") or 0)
        cache_read_price = float(block.get("cache_read") or 0)
        cache_write_price = float(block.get("cache_write") or 0)
        input_usd = (prompt / unit_div) * input_price if input_price else 0.0
        output_usd = (completion / unit_div) * output_price if output_price else 0.0
        cache_read_usd = (cache_read / unit_div) * cache_read_price if cache_read_price else 0.0
        cache_write_usd = (cache_write / unit_div) * cache_write_price if cache_write_price else 0.0
    else:
        # Case B: app 'pricing' with per_1k and/or per_million keys
        in_per_1k = block.get("input_per_1k")
        out_per_1k = block.get("output_per_1k")
        cr_per_1k = block.get("cache_read_per_1k")
        cw_per_1k = block.get("cache_write_per_1k")
        # If per_1k missing, derive from per_million
        if in_per_1k is None and block.get("input_per_million") is not None:
            in_per_1k = float(block.get("input_per_million")) / 1000.0
        if out_per_1k is None and block.get("output_per_million") is not None:
            out_per_1k = float(block.get("output_per_million")) / 1000.0
        if cr_per_1k is None and block.get("cache_read_per_million") is not None:
            cr_per_1k = float(block.get("cache_read_per_million")) / 1000.0
        if cw_per_1k is None and block.get("cache_write_per_million") is not None:
            cw_per_1k = float(block.get("cache_write_per_million")) / 1000.0

        in_per_1k = float(in_per_1k or 0)
        out_per_1k = float(out_per_1k or 0)
        cr_per_1k = float(cr_per_1k or 0)
        cw_per_1k = float(cw_per_1k or 0)

        input_usd = (prompt / 1000.0) * in_per_1k if in_per_1k else 0.0
        output_usd = (completion / 1000.0) * out_per_1k if out_per_1k else 0.0
        cache_read_usd = (cache_read / 1000.0) * cr_per_1k if cr_per_1k else 0.0
        cache_write_usd = (cache_write / 1000.0) * cw_per_1k if cw_per_1k else 0.0

    total = input_usd + output_usd + cache_read_usd + cache_write_usd
    return {
        "input_usd": round(input_usd, 10),
        "output_usd": round(output_usd, 10),
        "cache_read_usd": round(cache_read_usd, 10),
        "cache_write_usd": round(cache_write_usd, 10),
        "total_usd": round(total, 10),
    }
