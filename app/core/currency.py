from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_DATA_PATH = Path("data/currency.json")


def _ensure_dir() -> None:
    _DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_rates() -> Dict[str, Any]:
    """Load stored currency rates. Returns empty dict if not present."""
    try:
        if _DATA_PATH.exists():
            return json.loads(_DATA_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {}


def save_rates(data: Dict[str, Any]) -> None:
    """Persist currency rates to data/currency.json."""
    _ensure_dir()
    try:
        _DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_usd_to_inr() -> Optional[float]:
    rates = load_rates()
    try:
        v = float(rates.get("usd_to_inr"))
        return v if v > 0 else None
    except Exception:
        return None


def convert_usd_to_inr(amount_usd: float, *, rate: Optional[float] = None) -> Optional[float]:
    if rate is None:
        rate = get_usd_to_inr()
    if rate is None or amount_usd is None:
        return None
    try:
        return float(amount_usd) * float(rate)
    except Exception:
        return None

