from __future__ import annotations

from typing import Dict, List, Any

import httpx

from .provider_endpoints import get_provider_base_urls


LOCAL_PROVIDER_NAMES: Dict[str, str] = {
    "lmstudio": "LM Studio",
    "ollama": "Ollama",
}


def is_local_provider(provider_id: str) -> bool:
    return (provider_id or "").strip().lower() in LOCAL_PROVIDER_NAMES


def list_local_providers() -> List[tuple[str, str]]:
    """Return [(provider_id, provider_name), ...] for built-in local providers."""
    return [(pid, name) for pid, name in LOCAL_PROVIDER_NAMES.items()]


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


def _ollama_fallback_models(timeout: float = 2.0) -> List[str]:
    """Fallback: query Ollama native API for tags if /v1/models not available."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get("http://localhost:11434/api/tags")
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
    if pid == "ollama" and not ids:
        ids = _ollama_fallback_models()

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

