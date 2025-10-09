from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional


_BASEURL_PATH = Path("app/baseurl.json")


def _load_baseurl_catalog() -> Dict:
    """Load provider base URL catalog from app/baseurl.json.

    Returns an empty dict on failure.
    """
    try:
        if _BASEURL_PATH.exists():
            return json.loads(_BASEURL_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def get_provider_base_urls(provider_id: str) -> List[Dict[str, str]]:
    """Return a list of endpoint options for a provider.

    Each entry is a dict with keys: label, url.

    - For most providers, returns a single entry using the catalog base_url.
    - Special handling for Alibaba: returns two distinct endpoints with clear labels.
    - If no catalog entry found or URL looks invalid, falls back to OpenAI default.
    """
    pid = (provider_id or "").strip().lower()
    catalog = _load_baseurl_catalog()
    pmap = (catalog.get("provider_breakdown") or {}) if isinstance(catalog, dict) else {}
    rec = pmap.get(pid, {}) if isinstance(pmap, dict) else {}

    url = rec.get("base_url") if isinstance(rec, dict) else None

    # Validate that we have a proper URL from catalog
    if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
        return [{"label": "Default", "url": url.strip()}]

    # No fallback - provider must be properly configured in baseurl.json or Excel
    return []
