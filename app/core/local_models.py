from __future__ import annotations

from typing import Dict, List, Any

import httpx

from .provider_endpoints import get_provider_base_urls


def is_local_provider(provider_id: str) -> bool:
    """Check if provider has local endpoint (localhost or private IP).
    
    This is now a heuristic check. Proper way is via Excel 'is_local' flag.
    """
    from .provider_endpoints import get_provider_base_urls
    pid = (provider_id or "").strip().lower()
    if not pid:
        return False
    
    endpoints = get_provider_base_urls(pid)
    if not endpoints:
        return False
    
    url = endpoints[0].get("url", "").lower()
    local_indicators = [
        "localhost", "127.0.0.1", "0.0.0.0",
        "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
        "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
        "172.30.", "172.31.",
        "10.",
        ".local", ".lan",
    ]
    return any(indicator in url for indicator in local_indicators)


def list_local_providers() -> List[tuple[str, str]]:
    """Return [(provider_id, provider_name), ...] for known local providers.
    
    Note: This should ideally come from Excel with is_local flag.
    """
    # Basic well-known local providers
    return [
        ("lmstudio", "LM Studio"),
        ("ollama", "Ollama"),
    ]


def _openai_models(base_url: str, timeout: float = 2.0) -> List[str]:
    """Query OpenAI-compatible /models endpoint and return list of model IDs."""
    url = base_url.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
            models = data.get("data") if isinstance(data, dict) else None
            ids: List[str] = []
            if isinstance(models, list):
                for m in models:
                    mid = m.get("id") if isinstance(m, dict) else None
                    if isinstance(mid, str) and mid:
                        ids.append(mid)
            return ids
    except Exception:
        return []


def _ollama_fallback_models(base_url: str, timeout: float = 2.0) -> List[str]:
    """Fallback: query Ollama native API for tags if /v1/models not available.
    
    Args:
        base_url: The base URL from provider config (e.g., http://localhost:11434/v1)
    """
    # Convert OpenAI-compatible endpoint to native Ollama API
    # e.g., http://localhost:11434/v1 -> http://localhost:11434/api/tags
    native_base = base_url.rstrip("/").replace("/v1", "")
    tags_url = f"{native_base}/api/tags"
    
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(tags_url)
            r.raise_for_status()
            data = r.json()
            models = data.get("models") if isinstance(data, dict) else None
            ids: List[str] = []
            if isinstance(models, list):
                for m in models:
                    name = m.get("name") if isinstance(m, dict) else None
                    if isinstance(name, str) and name:
                        ids.append(name)
            return ids
    except Exception:
        return []


def discover_provider_models(provider_id: str) -> Dict[str, Any]:
    """Return a catalog-like dict for a local provider with discovered models.

    Shape:
    { 'name': 'LM Studio', 'models': { 'id': { ... }, ... } }
    """
    pid = (provider_id or "").strip().lower()
    name = LOCAL_PROVIDER_NAMES.get(pid, pid)

    # Resolve base URL
    opts = get_provider_base_urls(pid)
    base_url = (opts[0]["url"] if opts else "")

    ids: List[str] = []
    # Try OpenAI-compatible endpoint first
    if base_url:
        ids = _openai_models(base_url)

    # Ollama fallback if needed
    if pid == "ollama" and not ids and base_url:
        ids = _ollama_fallback_models(base_url)

    models: Dict[str, Any] = {}
    for mid in ids:
        models[mid] = {
            "id": mid,
            "name": mid,
            "modalities": {"input": ["text"], "output": ["text"]},
            # Unknown without deep inspection; UI will allow override
            "limit": {"context": None, "output": None},
            "cost": {"input": 0, "output": 0},
        }

    return {"name": name, "models": models}

