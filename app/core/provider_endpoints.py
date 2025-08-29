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

    # Alibaba: provide 2 explicit endpoints with neutral labels
    if pid == "alibaba":
        return [
            {
                "label": "International",
                "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            },
            {
                "label": "China (Mainland)",
                "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            },
        ]

    url = rec.get("base_url") if isinstance(rec, dict) else None

    # Some entries in catalog are descriptive and not actual URLs; validate prefix
    if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
        return [{"label": "Default", "url": url.strip()}]

    # Fallbacks for common providers if catalog missing or non-URL
    fallback = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "mistral": "https://api.mistral.ai/v1",
        "perplexity": "https://api.perplexity.ai",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "fireworks-ai": "https://api.fireworks.ai/inference/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "moonshotai-cn": "https://api.moonshot.cn/v1",
        "moonshotai": "https://api.moonshot.ai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "google-vertex": "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
        "amazon-bedrock": "https://{bedrock-runtime-endpoint}/openai/v1",
        "groq": "https://api.groq.com/openai/v1",
        # Local providers
        "lmstudio": "http://localhost:1234/v1",
        "ollama": "http://localhost:11434/v1",
    }

    if pid in fallback:
        return [{"label": "Default", "url": fallback[pid]}]

    # Final fallback to OpenAI
    return [{"label": "Default", "url": "https://api.openai.com/v1"}]
