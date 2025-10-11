from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple, List
import time

try:
    import httpx
except ImportError:
    httpx = None

CATALOG_URL = "https://models.dev/api.json"
CATALOG_CACHE = Path("data/models_dev_cache.json")
LOGO_DIR = Path("data/logos")
DEFAULT_LOGO_URL = "https://models.dev/logos/default.svg"

# Global indices for fast lookups
_idx_by_id: Dict[str, Dict[str, Any]] = {}
_idx_by_provider_model: Dict[str, Dict[str, Any]] = {}
_catalog_loaded = False


def _cache_fresh(path: Path, max_age_seconds: int) -> bool:
    """Check if cached file is fresh within max_age_seconds."""
    try:
        if not path.exists():
            return False
        mtime = path.stat().st_mtime
        return (time.time() - mtime) < max_age_seconds
    except Exception:
        return False


def _build_indices(catalog_data: Dict[str, Any]) -> None:
    """Build lookup indices from catalog data."""
    global _idx_by_id, _idx_by_provider_model, _catalog_loaded
    
    _idx_by_id.clear()
    _idx_by_provider_model.clear()
    
    for provider_id, provider_data in catalog_data.items():
        if not isinstance(provider_data, dict) or "models" not in provider_data:
            continue
            
        provider_name = provider_data.get("name", provider_id)
        models = provider_data.get("models", {})
        
        for model_id, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
                
            # Create normalized record
            record = {
                "provider_id": provider_id,
                "provider_name": provider_name,
                "model_id": model_id,
                "name": model_data.get("name", model_id),
                "modalities": model_data.get("modalities", {}),
                "limit": model_data.get("limit", {}),
                "cost": model_data.get("cost", {}),
                "dates": {
                    "release_date": model_data.get("release_date"),
                    "last_updated": model_data.get("last_updated"), 
                    "knowledge": model_data.get("knowledge")
                },
                "logo_url": f"https://models.dev/logos/{provider_id}.svg"
            }
            
            # Index by exact model ID
            _idx_by_id[model_id] = record
            
            # Index by provider/model combination (lowercase)
            provider_model_key = f"{provider_id.lower()}/{model_id.lower()}"
            _idx_by_provider_model[provider_model_key] = record
    
    _catalog_loaded = True


def _fetch_and_cache_catalog(max_age_seconds: int = 24 * 3600) -> Dict[str, Any]:
    """Fetch catalog from API and cache to disk."""
    if _cache_fresh(CATALOG_CACHE, max_age_seconds):
        try:
            return json.loads(CATALOG_CACHE.read_text())
        except Exception:
            pass
    
    data: Dict[str, Any] = {}
    
    # Try to fetch from API
    if httpx:
        try:
            resp = httpx.get(CATALOG_URL, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                data = {}
        except Exception:
            # If we have a stale cache, use it
            if CATALOG_CACHE.exists():
                try:
                    return json.loads(CATALOG_CACHE.read_text())
                except Exception:
                    pass
    
    # If no data and no httpx, try to use existing cache
    if not data and CATALOG_CACHE.exists():
        try:
            return json.loads(CATALOG_CACHE.read_text())
        except Exception:
            pass
    
    # If all else fails, return empty dict (caller should handle gracefully)
    if not data:
        data = {}
    
    # Cache to disk
    try:
        CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_CACHE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass
    
    return data


def _ensure_catalog_loaded() -> None:
    """Ensure catalog is loaded and indices are built."""
    global _catalog_loaded
    if not _catalog_loaded:
        catalog_data = _fetch_and_cache_catalog()
        _build_indices(catalog_data)


def resolve(model_id: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Resolve model ID to record with fallback strategies."""
    _ensure_catalog_loaded()
    
    notes = []
    
    # Strategy 1: Exact match (as-is)
    if model_id in _idx_by_id:
        return _idx_by_id[model_id], notes
    notes.append(f"exact match '{model_id}' not found")
    
    # Strategy 2: Strip suffix after ":" and retry
    if ":" in model_id:
        base_id = model_id.split(":")[0]
        if base_id in _idx_by_id:
            notes.append(f"found after stripping suffix: '{base_id}'")
            return _idx_by_id[base_id], notes
        notes.append(f"base ID '{base_id}' not found")
    
    # Strategy 3: Lowercase and retry both forms
    model_id_lower = model_id.lower()
    if model_id_lower in _idx_by_id:
        notes.append(f"found with lowercase: '{model_id_lower}'")
        return _idx_by_id[model_id_lower], notes
    
    if ":" in model_id:
        base_id_lower = model_id.split(":")[0].lower()
        if base_id_lower in _idx_by_id:
            notes.append(f"found with lowercase base: '{base_id_lower}'")
            return _idx_by_id[base_id_lower], notes
    
    # Strategy 4: If contains "/", try provider/model lookup
    if "/" in model_id:
        provider_model_key = model_id.lower()
        if provider_model_key in _idx_by_provider_model:
            notes.append(f"found via provider/model lookup: '{provider_model_key}'")
            return _idx_by_provider_model[provider_model_key], notes
        
        # Try without suffix if present
        if ":" in model_id:
            provider_model_key = model_id.split(":")[0].lower()
            if provider_model_key in _idx_by_provider_model:
                notes.append(f"found via provider/model lookup (no suffix): '{provider_model_key}'")
                return _idx_by_provider_model[provider_model_key], notes
    
    notes.append("no matches found with any strategy")
    return None, notes


def get_logo_path(provider_id: str) -> Optional[str]:
    """Download and cache provider logo, return local path."""
    if not provider_id:
        return None
        
    provider_safe = provider_id.strip().lower()
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGO_DIR / f"{provider_safe}.svg"
    
    # Return cached logo if fresh
    if _cache_fresh(path, 30 * 24 * 3600):  # 30 days
        return str(path)
    
    # Try to download logo
    if httpx:
        url = f"https://models.dev/logos/{provider_safe}.svg"
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return str(path)
        except Exception:
            pass
    
    # Try to download default logo as fallback
    if httpx:
        try:
            r = httpx.get(DEFAULT_LOGO_URL, timeout=5.0)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return str(path)
        except Exception:
            pass
    
    return None


def lookup_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Main API: Look up model by ID with robust fallback strategies."""
    if not model_id or not isinstance(model_id, str):
        return None
    
    record, notes = resolve(model_id.strip())
    
    if record:
        # Add logo path to record
        result = record.copy()
        logo_path = get_logo_path(record["provider_id"])
        if logo_path:
            result["logo_path"] = logo_path
        return result
    
    return None


def get_cached_catalog() -> Dict[str, Any]:
    """Return the full cached catalog data."""
    return _fetch_and_cache_catalog()
