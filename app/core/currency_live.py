from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore


CACHE_FILE = Path("data/currency_cache.json")
CACHE_DURATION = 3600  # 1 hour in seconds
API_URL = "https://api.exchangerate-api.com/v4/latest/USD"


def fetch_live_rate(from_currency: str = "USD", to_currency: str = "INR") -> Optional[float]:
    """Fetch live exchange rate from API.
    
    Uses exchangerate-api.com free tier (no API key needed).
    Returns None if fetch fails.
    """
    if requests is None:
        return None
    
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        rates = data.get("rates", {})
        rate = rates.get(to_currency)
        
        if rate is not None:
            return float(rate)
        return None
    except Exception:
        return None


def get_cached_rate() -> Optional[float]:
    """Get cached exchange rate if available and fresh."""
    try:
        if not CACHE_FILE.exists():
            return None
        
        cache_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        cached_time = cache_data.get("timestamp", 0)
        cached_rate = cache_data.get("rate")
        
        # Check if cache is still fresh (within CACHE_DURATION)
        if time.time() - cached_time < CACHE_DURATION and cached_rate is not None:
            return float(cached_rate)
        
        return None
    except Exception:
        return None


def save_cache(rate: float) -> None:
    """Save rate to cache file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "rate": rate,
            "timestamp": time.time(),
            "currency_pair": "USD_to_INR"
        }
        CACHE_FILE.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    except Exception:
        pass  # Ignore cache save failures


def get_rate_with_fallback() -> Optional[float]:
    """Get exchange rate with fallback chain: cache → live → user-set.
    
    Returns:
        - Cached rate if fresh (< 1 hour old)
        - Live rate from API if cache stale or missing
        - None if all methods fail (caller should use user-set rate)
    """
    # Try cache first
    cached = get_cached_rate()
    if cached is not None:
        return cached
    
    # Cache miss or stale - fetch live
    live_rate = fetch_live_rate()
    if live_rate is not None:
        save_cache(live_rate)
        return live_rate
    
    # Live fetch failed - try to return stale cache as last resort
    try:
        if CACHE_FILE.exists():
            cache_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            stale_rate = cache_data.get("rate")
            if stale_rate is not None:
                return float(stale_rate)
    except Exception:
        pass
    
    return None


def force_refresh() -> Optional[float]:
    """Force refresh the exchange rate from API, bypassing cache."""
    rate = fetch_live_rate()
    if rate is not None:
        save_cache(rate)
    return rate

