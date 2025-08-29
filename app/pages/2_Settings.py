from __future__ import annotations

import base64
import io
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from PIL import Image
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import streamlit as st

# Ensure project root on path when Streamlit runs page modules directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core import ui as core_ui
from app.core import storage
from app.core.models_dev import lookup_model, get_cached_catalog, get_logo_path
from app.core.provider_openai import OpenAIProvider, OAIGateway, tiny_png_data_url
from app.core.templating import Example, RenderedMessages, render_user_prompt


def _get_kms() -> Optional[Fernet]:
    """Return a Fernet instance.

    - If APP_KMS_KEY is set, use it.
    - Otherwise, load or create a local key at data/kms.key for automatic persistence.
    """
    key = os.getenv("APP_KMS_KEY")
    if not key:
        try:
            key_path = Path("data/kms.key")
            if key_path.exists():
                key = key_path.read_text(encoding="utf-8").strip()
            else:
                Path("data").mkdir(parents=True, exist_ok=True)
                k = Fernet.generate_key().decode()
                key_path.write_text(k, encoding="utf-8")
                key = k
        except Exception:
            return None
    try:
        return Fernet(key)
    except Exception:
        return None


def _encrypt(kms: Optional[Fernet], plaintext: str) -> Optional[str]:
    if not kms:
        return None
    return kms.encrypt(plaintext.encode()).decode()


def _decrypt(kms: Optional[Fernet], token: Optional[str]) -> Optional[str]:
    if not kms or not token:
        return None


def _get_api_key_for_provider_code(provider_code: str) -> Optional[str]:
    """Resolve API key for a provider code from session or encrypted DB."""
    code = (provider_code or "").strip().lower()
    if not code:
        return None
    # Session-stored
    keys: Dict[str, str] = st.session_state.get("_provider_api_keys", {})
    if code in keys:
        return keys[code]
    # Persisted
    return storage.get_decrypted_api_key(code)


def _set_api_key_for_provider_code(provider_code: str, key_storage: str, new_key: str) -> tuple[str, Optional[str]]:
    """Set provider-scoped API key.

    Returns (storage_mode, enc_token) where enc_token is for persisted mode.
    Also updates DB via storage.set_provider_key.
    """
    code = (provider_code or "").strip().lower()
    if not code:
        return "session", None
    if key_storage == "session":
        keys: Dict[str, str] = st.session_state.setdefault("_provider_api_keys", {})
        if new_key:
            keys[code] = new_key
        storage.set_provider_key(code, "session", None)
        return "session", None
    # encrypted
    kms = _get_kms()
    enc = _encrypt(kms, new_key) if new_key else None
    storage.set_provider_key(code, "encrypted" if enc else "session", enc)
    return ("encrypted" if enc else "session"), enc
    try:
        return kms.decrypt(token.encode()).decode()
    except (InvalidToken, Exception):
        return None


def _get_api_key_for_provider(pid: int, key_storage: str, enc: Optional[str]) -> Optional[str]:
    if key_storage == "session":
        keys: Dict[int, str] = st.session_state.get("_api_keys", {})
        return keys.get(pid)
    kms = _get_kms()
    return _decrypt(kms, enc)


def _set_api_key_for_provider(pid: int, key_storage: str, new_key: str) -> tuple[str, Optional[str]]:
    if key_storage == "session":
        keys: Dict[int, str] = st.session_state.setdefault("_api_keys", {})
        if new_key:
            keys[pid] = new_key
        return "session", None
    kms = _get_kms()
    enc = _encrypt(kms, new_key) if new_key else None
    return ("encrypted" if enc else "session"), enc


def _json_input(label: str, value: Any, key: str, height: int = 140) -> tuple[Any, bool]:
    text = st.text_area(label, value=json.dumps(value, indent=2) if value else "{}", key=key, height=height)
    try:
        parsed = json.loads(text) if text.strip() else {}
        return parsed, True
    except Exception as e:
        st.error(f"Invalid JSON for {label}: {e}")
        return value, False


def _ping_chat(base_url: str, api_key: str, model_id: str, headers: Dict[str, str] | None, timeout_s: float) -> Dict[str, Any]:
    """Send a minimal ping to test basic chat functionality."""
    try:
        gateway = OAIGateway(
            base_url=base_url,
            api_key=api_key,
            headers=headers,
            timeout=int(timeout_s),
            prefer_json_mode=False,
            prefer_tools=False,
            detected_caps=None
        )
        
        result = gateway.chat_vision(
            model=model_id,
            system_text="",
            user_text="Say 'pong'.",
            image_paths=[],
            fewshot_messages=None,
            schema=None,
            gen_params={"temperature": 1.0, "max_tokens": 10}  # Small for test only
        )
        
        if result.get("error"):
            return {"status": "error", "message": result["error"], "details": result}
        elif result.get("text"):
            return {"status": "success", "message": f"✓ Response: {result['text'][:100]}", "details": result}
        else:
            return {"status": "error", "message": "No text response received", "details": result}
            
    except Exception as e:
        return {"status": "error", "message": f"Exception: {str(e)[:200]}", "details": None}


def _capability_probe(base_url: str, api_key: str, model_id: str, headers: Dict[str, str] | None, timeout_s: float) -> Dict[str, Any]:
    """Comprehensive capability probing with encoding fallbacks."""
    gateway = OAIGateway(
        base_url=base_url,
        api_key=api_key, 
        headers=headers,
        timeout=int(timeout_s),
        prefer_json_mode=True,
        prefer_tools=False,
        detected_caps=None  # Don't use caps during probing
    )
    
    results: Dict[str, Any] = {}
    
    # Vision probe with EncA/EncB fallbacks
    try:
        import tempfile
        import base64
        from pathlib import Path
        
        # Create tiny PNG file for testing
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(base64.b64decode(b64_data))
            tmp.flush()
            tiny_png_path = tmp.name
            
            # Try EncA first (this will use image_url format internally)
            result = gateway.chat_vision(
                model=model_id,
                system_text="",
                user_text="Describe this image briefly.",
                image_paths=[tiny_png_path],
                fewshot_messages=None,
                schema=None,
                gen_params={"temperature": 1.0, "max_tokens": 10}
            )
        
        # Check if EncA worked or if it fell back to EncB
        if result.get("error"):
            results["vision"] = False
            results["vision_encoding"] = "failed"
            error_msg = result.get("error", "")
            if "EncB" in error_msg:
                results["vision_encoding"] = "EncA_failed_EncB_failed"
            else:
                results["vision_encoding"] = "EncA_failed"
        else:
            results["vision"] = True
            # Extract encoding info from error message or raw data
            raw_data = result.get("raw", {})
            if "EncB" in str(raw_data):
                results["vision_encoding"] = "EncB"  # Fell back to EncB
            else:
                results["vision_encoding"] = "EncA"  # EncA worked
            
    except Exception as e:
        results["vision"] = False
        results["vision_encoding"] = "exception"
        results.setdefault("errors", {})["vision"] = str(e)[:200]
    
    # JSON mode probe
    try:
        simple_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"}
            },
            "required": ["response"]
        }
        
        result = gateway.chat_vision(
            model=model_id,
            system_text="",
            user_text="Respond with JSON containing a 'response' field with value 'ok'.",
            image_paths=[],
            fewshot_messages=None,
            schema=simple_schema,
            gen_params={"temperature": 1.0, "max_tokens": 20}
        )
        
        # Check if server respected JSON mode
        if result.get("error"):
            results["json_mode"] = False
        else:
            # Look for JSON response format being used
            raw_response = result.get("raw", {})
            resp_format_used = "response_format" in str(raw_response) or result.get("tool_call_json") is None
            results["json_mode"] = resp_format_used and result.get("text") is not None
            
    except Exception as e:
        results["json_mode"] = False
        results.setdefault("errors", {})["json_mode"] = str(e)[:200]
    
    # Tools probe
    try:
        gateway_tools = OAIGateway(
            base_url=base_url,
            api_key=api_key,
            headers=headers, 
            timeout=int(timeout_s),
            prefer_json_mode=False,
            prefer_tools=True,
            detected_caps=None
        )
        
        emit_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "A simple message"}
            },
            "required": ["message"]
        }
        
        result = gateway_tools.chat_vision(
            model=model_id,
            system_text="",
            user_text="Use the emit function to send a message saying 'tools work'.",
            image_paths=[],
            fewshot_messages=None,
            schema=emit_schema,
            gen_params={"temperature": 1.0, "max_tokens": 20}
        )
        
        if result.get("error"):
            results["tools"] = False
        else:
            tool_call = result.get("tool_call_json")
            results["tools"] = tool_call is not None and isinstance(tool_call, dict)
            
    except Exception as e:
        results["tools"] = False
        results.setdefault("errors", {})["tools"] = str(e)[:200]
    
    return results


def _model_browser(catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Display model browser and return selected model info."""
    # Build flat list of all models
    all_models = []
    for provider_id, provider_data in catalog.items():
        if not isinstance(provider_data, dict) or "models" not in provider_data:
            continue
        provider_name = provider_data.get("name", provider_id)
        models = provider_data.get("models", {})
        
        for model_id, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            
            # Check if model has vision
            modalities = model_data.get("modalities", {})
            input_mods = modalities.get("input", []) if isinstance(modalities, dict) else []
            has_vision = "image" in input_mods
            
            # Get pricing
            cost = model_data.get("cost", {})
            is_free = any(v == 0 for v in cost.values() if isinstance(v, (int, float)))
            
            all_models.append({
                "provider_id": provider_id,
                "provider_name": provider_name,
                "model_id": model_id,
                "name": model_data.get("name", model_id),
                "has_vision": has_vision,
                "is_free": is_free,
                "cost": cost,
                "limit": model_data.get("limit", {}),
                "modalities": modalities,
                "dates": model_data.get("dates", {}),
                "full_data": model_data
            })
    
    # Sort by provider name, then model name
    all_models.sort(key=lambda x: (x["provider_name"], x["name"]))
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        filter_vision = st.checkbox("🖼️ Vision capable models only")
    with col2:
        filter_free = st.checkbox("🆓 Free models only")
    
    # Apply filters
    filtered = all_models
    if filter_vision:
        filtered = [m for m in filtered if m["has_vision"]]
    if filter_free:
        filtered = [m for m in filtered if m["is_free"]]
    
    if not filtered:
        st.info("No models found matching your filters")
        return None
    
    # Create display labels for dropdown
    model_options = []
    model_map = {}
    
    for model in filtered:
        # Create a nice display label
        label_parts = [f"{model['provider_name']}: {model['name']}"]
        
        # Add badges
        badges = []
        if model["has_vision"]:
            badges.append("🖼️")
        if model["is_free"]:
            badges.append("🆓")
        context = model["limit"].get("context")
        if context and context >= 100000:
            badges.append("📏")  # Long context
        
        if badges:
            label_parts.append(f"[{' '.join(badges)}]")
        
        label = " ".join(label_parts)
        model_options.append(label)
        model_map[label] = model
    
    # Model selector dropdown
    selected_label = st.selectbox(
        "🔍 Select a model",
        options=["Choose a model..."] + model_options,
        help="Browse available models from the catalog"
    )
    
    if selected_label and selected_label != "Choose a model...":
        selected_model = model_map.get(selected_label)
        
        if selected_model:
            # Show selected model details
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Provider:** {selected_model['provider_name']}")
                    st.markdown(f"**Model:** {selected_model['name']}")
                
                with col2:
                    features = []
                    if selected_model["has_vision"]:
                        features.append("🖼️ Vision")
                    if selected_model["is_free"]:
                        features.append("🆓 Free")
                    
                    context = selected_model["limit"].get("context")
                    if context:
                        if context >= 1000000:
                            features.append(f"📏 {context//1000000}M tokens")
                        elif context >= 1000:
                            features.append(f"📏 {context//1000}K tokens")
                        else:
                            features.append(f"📏 {context} tokens")
                    
                    if features:
                        st.markdown("**Features:**")
                        for f in features:
                            st.write(f)
                
                with col3:
                    # Show pricing if available
                    cost = selected_model.get("cost", {})
                    if cost:
                        input_cost = cost.get("input", 0)
                        if input_cost == 0:
                            st.success("Free")
                        else:
                            st.caption(f"${input_cost}/1M tokens")
            
            return selected_model
    
    return None


def _grouped_model_selector(catalog: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Two-step model selector grouped by provider."""
    # Build provider list
    providers: List[tuple[str, str]] = []  # (provider_id, provider_name)
    for provider_id, provider_data in catalog.items():
        if not isinstance(provider_data, dict) or "models" not in provider_data:
            continue
        providers.append((provider_id, provider_data.get("name", provider_id)))
    if not providers:
        st.info("No providers available in catalog")
        return None

    providers.sort(key=lambda x: x[1].lower())
    provider_display = [f"{name} ({pid})" for pid, name in providers]
    choice = st.selectbox("Provider", options=["Choose a provider..."] + provider_display)
    if not choice or choice == "Choose a provider...":
        # Clear selected provider state if no provider chosen
        st.session_state["_selected_provider_code"] = ""
        return None
    sel_idx = provider_display.index(choice)
    provider_id, provider_name = providers[sel_idx]
    # Persist selection so other sections can react conditionally
    st.session_state["_selected_provider_code"] = provider_id

    # Require/confirm API key before listing models
    current_key = _get_api_key_for_provider_code(provider_id)
    if current_key:
        st.success("API key loaded for this provider")
        show_models = True
    else:
        st.warning("No API key saved for this provider.")
        ik1, ik2 = st.columns([3, 2])
        with ik1:
            inline_api = st.text_input(
                f"API Key ({provider_id})",
                value="",
                type="password",
                key=f"inline_api_top_{provider_id}"
            )
        with ik2:
            kms = _get_kms()
            default_mode = "encrypted" if kms else "session"
            inline_mode = st.radio(
                "Storage",
                options=["encrypted", "session"] if kms else ["session"],
                index=(0 if default_mode == "encrypted" else 0),
                horizontal=True,
                key=f"inline_mode_top_{provider_id}"
            )
        save_top = st.button("Save Provider Key", key=f"inline_save_top_{provider_id}", disabled=not inline_api)
        if save_top:
            _set_api_key_for_provider_code(provider_id, inline_mode, inline_api)
            st.success("Saved provider key")
            st.rerun()
        show_models = False

    if not show_models:
        return None

    # Filters (only after key exists)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        filter_vision = st.checkbox("🖼️ Vision only", value=False)
    with fcol2:
        filter_free = st.checkbox("🆓 Free only", value=False)

    # Models under selected provider
    models = catalog.get(provider_id, {}).get("models", {}) if isinstance(catalog.get(provider_id), dict) else {}
    items: List[Dict[str, Any]] = []
    for model_id, model_data in models.items():
        if not isinstance(model_data, dict):
            continue
        modalities = model_data.get("modalities", {})
        input_mods = modalities.get("input", []) if isinstance(modalities, dict) else []
        has_vision = "image" in input_mods
        cost = model_data.get("cost", {})
        is_free = any(v == 0 for v in cost.values() if isinstance(v, (int, float)))
        if filter_vision and not has_vision:
            continue
        if filter_free and not is_free:
            continue
        items.append({
            "provider_id": provider_id,
            "provider_name": provider_name,
            "model_id": model_id,
            "name": model_data.get("name", model_id),
            "has_vision": has_vision,
            "is_free": is_free,
            "cost": cost,
            "limit": model_data.get("limit", {}),
            "modalities": modalities,
            "dates": model_data.get("dates", {}),
            "full_data": model_data
        })

    items.sort(key=lambda x: x["name"].lower())
    labels = []
    mapping = {}
    for m in items:
        badges = []
        if m["has_vision"]:
            badges.append("🖼️")
        if m["is_free"]:
            badges.append("🆓")
        label = f"{m['name']} ({m['model_id']})" + (f"  [{' '.join(badges)}]" if badges else "")
        labels.append(label)
        mapping[label] = m

    selected = st.selectbox("Model", options=["Choose a model..."] + labels)
    if selected and selected != "Choose a model...":
        chosen = mapping.get(selected)
        if chosen:
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f"**Provider:** {provider_name}")
                    st.markdown(f"**Model:** {chosen['name']}")
                with c2:
                    feats = []
                    if chosen["has_vision"]:
                        feats.append("🖼️ Vision")
                    if chosen["is_free"]:
                        feats.append("🆓 Free")
                    ctx = chosen["limit"].get("context")
                    if ctx:
                        feats.append(f"📏 {ctx:,} tokens")
                    if feats:
                        st.markdown("**Features:** " + ", ".join(feats))
                with c3:
                    logo_path = get_logo_path(chosen.get("provider_id", ""))
                    if logo_path and Path(logo_path).exists():
                        st.image(logo_path, width=48)
            return chosen
    return None


def _default_base_url(provider_id: str) -> str:
    provider_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "mistral": "https://api.mistral.ai/v1",
        "perplexity": "https://api.perplexity.ai",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "moonshot": "https://api.moonshot.cn/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    return provider_urls.get((provider_id or "").lower(), "https://api.openai.com/v1")


def _api_keys_manager(catalog: Dict[str, Any]) -> None:
    """Compact provider key manager: add/edit one provider at a time."""
    st.markdown("#### 🔐 Provider API Keys")
    st.caption("Save one key per provider; used automatically for matching models")

    # Build provider options from catalog
    options = []  # list[tuple[pcode, pname]]
    for pid, pdata in catalog.items():
        if isinstance(pdata, dict) and "models" in pdata:
            options.append((pid, pdata.get("name", pid)))
    options.sort(key=lambda x: x[1].lower())
    display = [f"{name} ({pid})" for pid, name in options]

    # Show existing configured providers inline
    existing = storage.list_provider_keys()
    if existing:
        st.markdown("Configured:")
        st.write(" ".join([f"`{rec.provider_code}`" for rec in existing]))

    colA, colB = st.columns([3, 2])
    with colA:
        chosen = st.selectbox("Provider", options=["Choose a provider..."] + display, key="key_provider_select")
    with colB:
        # Determine selected code
        code = ""
        if chosen and chosen != "Choose a provider...":
            idx = display.index(chosen)
            code = options[idx][0]

        # Load current mode/value
        rec = storage.get_provider_key(code) if code else None
        current_val = _get_api_key_for_provider_code(code) if code else None
        default_mode = (rec.key_storage if rec else ("encrypted" if _get_kms() else "session"))
        mode = st.radio("Storage", options=["encrypted", "session"], index=(0 if default_mode == "encrypted" else 1), horizontal=True, key="key_storage_mode")

    # API key field
    api_val = st.text_input("API Key", value=(current_val or ""), type="password", key="key_api_value", help="Leave blank to keep existing")

    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        disabled = not code or (not api_val and not current_val)
        if st.button("Save", disabled=not code):
            new_val = api_val or (current_val or "")
            _set_api_key_for_provider_code(code, mode, new_val)
            st.success("Saved")
    with c2:
        if st.button("Delete", disabled=not rec):
            storage.delete_provider_key(code)
            # clear session value if exists
            keys: Dict[str, str] = st.session_state.get("_provider_api_keys", {})
            keys.pop(code, None)
            st.warning("Deleted")


def _streamlined_model_configuration(selected_model_info: Dict[str, Any]) -> None:
    """Streamlined configuration that sets an active model using provider keys."""
    storage.init_db()

    is_custom = bool(selected_model_info.get("is_custom"))
    if is_custom:
        provider_id = st.text_input("Provider Code", value=(selected_model_info.get("provider_name") or "openai")).strip().lower()
        default_base_url = selected_model_info.get("base_url", "")
        default_model_id = selected_model_info.get("model_id", "")
        default_headers = selected_model_info.get("headers", {})
        provider_name = selected_model_info.get("provider_name", provider_id or "Custom")
    else:
        provider_id = selected_model_info.get("provider_id", "").strip().lower()
        provider_name = selected_model_info.get("provider_name", provider_id)
        default_base_url = _default_base_url(provider_id)
        default_model_id = selected_model_info.get("model_id", "")
        default_headers = {}

    with st.container(border=True):
        col_info, col_logo = st.columns([4, 1])
        with col_info:
            st.markdown(f"**Selected Model:** {selected_model_info.get('name', selected_model_info.get('model_id'))}")
            st.caption(f"Provider: {provider_name} [{provider_id or 'custom'}]")
            if not is_custom:
                badges = []
                if selected_model_info.get("has_vision"):
                    badges.append("🖼️ Vision")
                if selected_model_info.get("is_free"):
                    badges.append("🆓 Free")
                ctx = selected_model_info.get("limit", {}).get("context")
                if ctx:
                    badges.append(f"📏 {ctx:,} tokens")
                if badges:
                    st.write(" · ".join(badges))
        with col_logo:
            if not is_custom:
                logo_path = get_logo_path(provider_id)
                if logo_path and Path(logo_path).exists():
                    st.image(logo_path, width=80)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            model_id = st.text_input("Model ID", value=default_model_id)
            base_url = st.text_input("Base URL", value=default_base_url)
            if base_url and not base_url.startswith(("http://", "https://")):
                st.error("Base URL must start with http:// or https://")

            url_lower = base_url.lower() if base_url else ""
            is_local = any(indicator in url_lower for indicator in [
                "localhost", "127.0.0.1", "0.0.0.0", "192.168.",
                "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
                "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
                "172.28.", "172.29.", "172.30.", "172.31.",
                "host.docker.internal", ".local", ".lan", ":1234", ":5000", ":5001",
                ":8000", ":8080", ":8888", ":9000", ":11434", ":7860", ":7861"
            ])
            if not is_local and "10." in url_lower:
                import re
                if re.search(r"(?:^|[^0-9])10\.\d{1,3}\.\d{1,3}\.\d{1,3}", url_lower):
                    is_local = True

            default_timeout = 300 if is_local else 120
            timeout_s = st.number_input("⏱️ Request Timeout (seconds)", min_value=30, max_value=600, value=int(default_timeout), step=30)

            # Adapt token limits to model's published limit if available
            model_limits = selected_model_info.get("limit", {}) if isinstance(selected_model_info.get("limit", {}), dict) else {}
            limit_output = model_limits.get("output")
            try:
                limit_output_int = int(limit_output) if limit_output is not None else None
            except Exception:
                limit_output_int = None

            ui_max_tokens = limit_output_int if (isinstance(limit_output_int, int) and limit_output_int > 0) else 32768
            default_tokens = int(limit_output_int or 4096)
            # Clamp default within min/max bounds to satisfy Streamlit API
            default_tokens = max(256, min(default_tokens, ui_max_tokens))

            max_output_tokens = st.number_input(
                "Max Output Tokens 📝",
                min_value=256,
                max_value=int(ui_max_tokens),
                step=256,
                value=int(default_tokens),
            )
            temperature = st.slider("Temperature 🌡️", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        with col2:
            headers_json, ok_headers = _json_input("Headers JSON (optional)", value=default_headers, key="hdrs", height=140)
            st.markdown("**Advanced Options**")
            force_json = st.checkbox("Force JSON Mode", value=False)
            prefer_tools = st.checkbox("Prefer Tools (function calling)", value=False)
            st.markdown("**Authentication**")
            if is_custom:
                # Custom flow: allow inline key entry here (since provider is user-defined)
                skip_key = st.checkbox("Skip API Key (local endpoints only)")
                current_key = _get_api_key_for_provider_code(provider_id) if provider_id else None
                if not skip_key and provider_id:
                    if is_local:
                        st.info("Local endpoint detected; API key not required.")
                    elif current_key:
                        st.success("API key saved for this provider")
                    else:
                        ik1, ik2 = st.columns([3, 2])
                        with ik1:
                            inline_api = st.text_input(
                                f"API Key ({provider_id or 'provider'})",
                                value="",
                                type="password",
                                key=f"inline_api_custom_{provider_id}"
                            )
                        with ik2:
                            kms = _get_kms()
                            default_mode = "encrypted" if kms else "session"
                            inline_mode = st.radio(
                                "Storage",
                                options=["encrypted", "session"] if kms else ["session"],
                                index=(0 if default_mode == "encrypted" else 0),
                                horizontal=True,
                                key=f"inline_mode_custom_{provider_id}"
                            )
                        if st.button("Save Provider Key", key=f"inline_save_custom_{provider_id}", disabled=not inline_api):
                            _set_api_key_for_provider_code(provider_id, inline_mode, inline_api)
                            st.success("Saved provider key")
                            st.rerun()
            else:
                # Catalog providers: key handled at provider selection step; just show status here
                current_key = _get_api_key_for_provider_code(provider_id) if provider_id else None
                st.info("API key loaded" if current_key else "No API key saved for this provider")
                skip_key = False

            st.divider()
            b1, b2 = st.columns([1, 1])
            with b1:
                test_clicked = st.button("🧪 Test Connection")
            with b2:
                apply_clicked = st.button("▶️ Set Active Model", type="primary")

        if test_clicked:
            api_key = current_key or ""
            if not skip_key and not is_local and not api_key:
                st.error("No API key available. Save a key first or enable 'Skip API Key'.")
            else:
                with st.spinner("Testing connection..."):
                    ping_result = _ping_chat(base_url, api_key, model_id, headers_json or {}, timeout_s)
                if ping_result["status"] == "success":
                    st.success(ping_result["message"])
                else:
                    st.error(ping_result["message"])
                    if ping_result.get("details"):
                        with st.expander("Error Details"):
                            st.json(ping_result["details"])

        if apply_clicked:
            if not model_id or not base_url:
                st.error("Model ID and Base URL are required")
            elif not skip_key and not is_local and not current_key:
                st.error("No API key saved for this provider. Add it in 'Provider API Keys' or enable 'Skip API Key'.")
            else:
                active_row = storage.get_active_provider()
                if not active_row:
                    active_row = storage.create_provider("Active Model", base_url)
                updated = storage.update_provider(
                    active_row.id,
                    name="Active Model",
                    base_url=base_url,
                    provider_code=provider_id or None,
                    model_id=model_id,
                    headers_json=headers_json if ok_headers else {},
                    timeout_s=float(timeout_s),
                    key_storage="session",
                    api_key_enc=None,
                    default_max_output_tokens=max_output_tokens,
                    default_temperature=float(temperature),
                    default_force_json_mode=bool(force_json),
                    default_prefer_tools=bool(prefer_tools),
                )
                if not is_custom:
                    catalog_info = lookup_model(model_id)
                    if catalog_info:
                        logo_path = get_logo_path(catalog_info.get("provider_id", ""))
                        storage.update_provider(updated.id, catalog_caps_json=catalog_info, logo_path=logo_path)
                storage.set_active_provider(updated.id)
                st.success(f"✅ Now using {model_id}")
                st.rerun()


def _custom_model_form() -> Optional[Dict[str, Any]]:
    """Display custom model form and return model info if submitted."""
    st.markdown("Enter details for a model not in the catalog:")
    
    # Add provider-specific guidance
    with st.info("ℹ️ Provider-Specific Notes"):
        st.markdown("""
        **OpenRouter:** 
        - Use model IDs like `openai/gpt-4-vision-preview` or `anthropic/claude-3-opus`
        - Base URL: `https://openrouter.ai/api/v1`
        - May need headers: `{"HTTP-Referer": "your-app-name"}`
        
        **LM Studio:**
        - Base URL: `http://localhost:1234/v1`
        - No API key needed (check "Skip API Key" option)
        - If output truncates at ~200 tokens, check model's n_predict setting
        - Some models have hardcoded 175-200 token limits
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        custom_model_id = st.text_input(
            "Model ID", 
            placeholder="e.g., my-local-llama, openai/gpt-4-vision-preview",
            help="For OpenRouter, include provider prefix (e.g., 'openai/gpt-4-vision-preview')"
        )
        custom_provider = st.text_input(
            "Provider Name (optional)", 
            placeholder="e.g., LM Studio, OpenRouter",
            help="Display name for the provider"
        )
    with col2:
        custom_base_url = st.text_input(
            "Base URL", 
            placeholder="e.g., http://localhost:1234/v1 or https://openrouter.ai/api/v1",
            help="API endpoint URL (without /chat/completions)"
        )
        custom_headers = st.text_area(
            "Headers JSON (optional)",
            placeholder='{"HTTP-Referer": "your-app", "X-Title": "MyApp"}',
            height=100,
            help="OpenRouter may require HTTP-Referer header"
        )
    
    if st.button("Use Custom Model", type="primary", disabled=not (custom_model_id and custom_base_url)):
        # Parse headers if provided
        headers = {}
        if custom_headers:
            try:
                headers = json.loads(custom_headers)
            except Exception as e:
                st.error(f"Invalid headers JSON: {e}")
                return None
        
        return {
            "model_id": custom_model_id,
            "provider_name": custom_provider or "Custom",
            "base_url": custom_base_url,
            "headers": headers,
            "is_custom": True
        }
    
    return None


def _model_configuration(selected_model_info: Dict[str, Any]) -> None:
    """Display model configuration section."""
    storage.init_db()
    providers = storage.list_providers()
    names = [p.name for p in providers]
    name_to_provider = {p.name: p for p in providers}
    
    # Use session state to check for existing profile
    selected = None
    if "selected_profile" in st.session_state:
        selected = name_to_provider.get(st.session_state["selected_profile"])
    
    # Determine base URL based on selection
    if selected_model_info.get("is_custom"):
        default_base_url = selected_model_info.get("base_url", "")
        default_model_id = selected_model_info.get("model_id", "")
        default_headers = selected_model_info.get("headers", {})
    else:
        # From catalog - need to determine base URL from provider
        provider_id = selected_model_info.get("provider_id", "")
        # Map common providers to their base URLs
        provider_urls = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "google": "https://generativelanguage.googleapis.com/v1beta",
            "mistral": "https://api.mistral.ai/v1",
            "perplexity": "https://api.perplexity.ai",
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "fireworks": "https://api.fireworks.ai/inference/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "moonshot": "https://api.moonshot.cn/v1",
            "openrouter": "https://openrouter.ai/api/v1"
        }
        default_base_url = provider_urls.get(provider_id, "https://api.openai.com/v1")
        default_model_id = selected_model_info.get("model_id", "")
        default_headers = {}
    
    with st.container(border=True):
        # Show selected model info
        col_info, col_logo = st.columns([4, 1])
        with col_info:
            st.markdown(f"**Selected Model:** {selected_model_info.get('name', selected_model_info.get('model_id'))}")
            st.caption(f"Provider: {selected_model_info.get('provider_name', 'Custom')}")
            
            # Show capabilities if from catalog
            if not selected_model_info.get("is_custom"):
                badges = []
                if selected_model_info.get("has_vision"):
                    badges.append("🖼️ Vision")
                if selected_model_info.get("is_free"):
                    badges.append("🆓 Free")
                context = selected_model_info.get("limit", {}).get("context")
                if context:
                    badges.append(f"📏 {context:,} tokens")
                if badges:
                    st.write(" · ".join(badges))
        
        with col_logo:
            # Try to show provider logo
            if not selected_model_info.get("is_custom"):
                logo_path = get_logo_path(selected_model_info.get("provider_id", ""))
                if logo_path and Path(logo_path).exists():
                    st.image(logo_path, width=80)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            model_id = st.text_input(
                "Model ID", 
                value=(selected.model_id if selected else default_model_id),
                help="The model identifier for API calls"
            )
            base_url = st.text_input(
                "Base URL", 
                value=(selected.base_url if selected else default_base_url),
                help="API endpoint URL"
            )
            # Validate URL
            if base_url and not base_url.startswith(("http://", "https://")):
                st.error("Base URL must start with http:// or https://")
            
            # Configurable timeout with smart defaults
            url_lower = base_url.lower() if base_url else ""
            is_local = any(indicator in url_lower for indicator in [
                "localhost", "127.0.0.1", "0.0.0.0", "192.168.",
                "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
                "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
                "172.28.", "172.29.", "172.30.", "172.31.",  # Private IP range B
                "host.docker.internal", ".local", ".lan", ":1234", ":5000", ":5001",
                ":8000", ":8080", ":8888", ":9000", ":11434", ":7860", ":7861"
            ])
            
            # Special check for 10.x.x.x range
            if not is_local and "10." in url_lower:
                import re
                if re.search(r'(?:^|[^0-9])10\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
                    is_local = True
            
            default_timeout = 300 if is_local else 120  # 5 min for local, 2 min for cloud
            
            timeout_s = st.number_input(
                "⏱️ Request Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=int(selected.timeout_s if (selected and selected.timeout_s) else default_timeout),
                step=30,
                help=f"How long to wait for model response. {'Local models may need more time (300s recommended).' if is_local else 'Cloud APIs typically respond within 120s.'}"
            )
            
            max_output_tokens = st.number_input(
                "Max Output Tokens 📝", 
                min_value=100, 
                max_value=32768, 
                value=int(selected.default_max_output_tokens if selected else 4096),
                step=512,
                help="Maximum tokens the model can generate. Increase for longer outputs (default: 4096)."
            )

            temperature = st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(selected.default_temperature if (selected and selected.default_temperature is not None) else 1.0),
                step=0.1,
                help="Sampling temperature. Use 1.0 for default; some models only allow 1.0.",
            )
            
            # Warning for LM Studio users
            if base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
                st.warning("""
                ⚠️ **LM Studio Configuration Required**:
                
                1. **In LM Studio Server**, set the model's max tokens to: **{max_output_tokens}**
                   - Look for: n_predict, max_tokens, or context_length in Model Settings
                   - Some models show this in the "Model Configuration" tab
                
                2. **Check the model's config file** (usually in `.gguf` metadata or `config.json`):
                   - Look for hardcoded limits like `max_position_embeddings` or `max_seq_len`
                   - Some models have built-in limits (e.g., 175-200 tokens) that cannot be overridden
                
                3. **The app automatically tries these parameters**: 
                   - n_predict, max_tokens, max_new_tokens, max_length, max_gen_tokens, num_predict
                
                4. **If output still cuts off**, try:
                   - Using a different model without hardcoded limits
                   - Checking LM Studio logs for actual parameters being used
                   - Disabling JSON mode (use Unstructured/Markdown option)
                """.format(max_output_tokens=max_output_tokens))
        
        with col2:
            # Local model toggle (no API key required)
            is_local_default = bool(base_url) and ("localhost" in base_url or "127.0.0.1" in base_url)
            local_no_key = st.checkbox(
                "Local model (no API key)",
                value=is_local_default,
                help="For local endpoints (e.g., LM Studio). Omits Authorization header.",
            )

            # API Key (hidden when local model)
            api_key = ""
            key_storage = "session"
            if not local_no_key:
                existing_key = _get_api_key_for_provider(selected.id, selected.key_storage, selected.api_key_enc) if selected else None
                api_key = st.text_input(
                    "API Key 🔑", 
                    value=(existing_key or ""), 
                    type="password",
                    help="Your API key for authentication"
                )
                
                kms = _get_kms()
                storage_mode_options = ["session", "encrypted"] if kms else ["session"]
                # Prefer encrypted by default when a KMS key is configured; keep existing choice for saved profiles
                if kms:
                    default_index = (
                        storage_mode_options.index(selected.key_storage)
                        if selected and selected.key_storage in storage_mode_options
                        else storage_mode_options.index("encrypted")
                    )
                else:
                    default_index = 0
                key_storage = st.radio(
                    "API Key Storage",
                    storage_mode_options,
                    index=default_index,
                    help=("Encrypted in DB requires APP_KMS_KEY"),
                )
            
            # Advanced settings (collapsible)
            with st.expander("Advanced Settings"):
                headers_default = default_headers if not selected else (selected.headers_json or {})
                headers_json, ok_headers = _json_input("Headers (JSON)", headers_default, key="headers_json", height=100)
        
        # Test and Use buttons
        st.divider()
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            test_clicked = st.button(
                "🧪 Test Connection", 
                use_container_width=True,
                type="primary",
                disabled=not (api_key or local_no_key)
            )
        with bcol2:
            use_clicked = st.button(
                "✅ Use Without Saving", 
                use_container_width=True,
                disabled=not (api_key or local_no_key)
            )
        with bcol3:
            save_profile_clicked = st.button(
                "💾 Save as Profile", 
                use_container_width=True
            )
            if save_profile_clicked:
                st.session_state["show_save_form"] = True
        
        if test_clicked and (api_key or local_no_key):
            with st.spinner("Testing connection..."):
                ping_result = _ping_chat(
                    base_url, 
                    api_key, 
                    model_id,
                    headers_json or {},
                    timeout_s
                )
            
            if ping_result["status"] == "success":
                st.success(ping_result["message"])
            else:
                st.error(ping_result["message"])
                if ping_result.get("details"):
                    with st.expander("Error Details"):
                        st.json(ping_result["details"])
        
        if use_clicked and (api_key or local_no_key):
            # Create temporary provider for this session
            temp_name = f"Quick: {model_id}"
            # Check if temp provider exists
            temp_provider = storage.get_provider_by_name(temp_name)
            if not temp_provider:
                temp_provider = storage.create_provider(temp_name, base_url)
            
            mode, enc_token = _set_api_key_for_provider(temp_provider.id, key_storage, api_key)
            temp_provider = storage.update_provider(
                temp_provider.id,
                base_url=base_url,
                model_id=model_id,
                headers_json=headers_json if ok_headers else {},
                timeout_s=float(timeout_s),
                key_storage=(mode if not local_no_key else "session"),
                api_key_enc=(enc_token if not local_no_key else None),
                default_max_output_tokens=max_output_tokens,
                default_temperature=float(temperature),
            )
            
            # Apply catalog info if available
            if not selected_model_info.get("is_custom"):
                catalog_info = lookup_model(model_id)
                if catalog_info:
                    logo_path = get_logo_path(catalog_info.get("provider_id", ""))
                    storage.update_provider(
                        temp_provider.id, 
                        catalog_caps_json=catalog_info, 
                        logo_path=logo_path
                    )
            
            storage.set_active_provider(temp_provider.id)
            st.success(f"✅ Now using {model_id}")
            st.rerun()
        
        if st.session_state.get("show_save_form", False):
            # Show profile saving form
            st.markdown("### Save as Profile")
            
            # Generate a sensible default name
            default_name = f"{model_id}"
            if hasattr(selected_model_info, 'get') and selected_model_info.get('provider_name'):
                default_name = f"{selected_model_info['provider_name']} - {model_id}"
            
            profile_name = st.text_input(
                "Profile Name", 
                value=default_name,
                key="profile_name_input"
            )
            
            bcol_save, bcol_cancel = st.columns(2)
            with bcol_save:
                confirm_save = st.button("Confirm Save", type="primary", disabled=not profile_name)
            with bcol_cancel:
                cancel_save = st.button("Cancel")
            
            if confirm_save and profile_name:
                try:
                    # Create new profile
                    p = storage.create_provider(profile_name.strip(), base_url)
                    mode, enc_token = _set_api_key_for_provider(p.id, key_storage, api_key)
                    p = storage.update_provider(
                        p.id,
                        base_url=base_url,
                        model_id=model_id,
                        headers_json=headers_json if ok_headers else {},
                        timeout_s=float(timeout_s),
                        key_storage=(mode if not local_no_key else "session"),
                        api_key_enc=(enc_token if not local_no_key else None),
                        default_max_output_tokens=max_output_tokens,
                        default_temperature=float(temperature),
                    )
                    
                    # Apply catalog info if available
                    if not selected_model_info.get("is_custom"):
                        catalog_info = lookup_model(model_id)
                        if catalog_info:
                            logo_path = get_logo_path(catalog_info.get("provider_id", ""))
                            storage.update_provider(
                                p.id, 
                                catalog_caps_json=catalog_info, 
                                logo_path=logo_path
                            )
                    
                    st.success(f"Profile '{profile_name}' saved!")
                    st.session_state["selected_profile"] = profile_name
                    st.session_state["show_save_form"] = False
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            
            if cancel_save:
                st.session_state["show_save_form"] = False
                st.rerun()


def _to_data_url(file) -> str:
    """Convert uploaded file to data URL."""
    data = file.read()
    mime = "image/png"
    try:
        img = Image.open(io.BytesIO(data))
        if img.format and img.format.lower() == "jpeg":
            mime = "image/jpeg"
        elif img.format and img.format.lower() == "webp":
            mime = "image/webp"
        elif img.format and img.format.lower() == "png":
            mime = "image/png"
    except Exception:
        pass
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _template_management() -> None:
    """Display template management interface."""
    storage.init_db()
    
    templates = storage.list_templates()
    name_to_tpl = {t.name: t for t in templates}
    names = [t.name for t in templates]
    
    # Add Starter Template button
    with st.container(border=True):
        st.markdown("#### Quick Start")
        col_starter, col_spacer = st.columns([1, 3])
        with col_starter:
            if st.button("🚀 Starter: Generic Doc", use_container_width=True, type="primary"):
                try:
                    # Create the starter template
                    starter_schema = {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "date": {"type": "string"},
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["title"]
                    }
                    
                    starter_system = "Return ONLY valid JSON matching the schema. No code fences. Use null for missing."
                    
                    starter_user = """You are a precise extractor. Use this schema:
{schema}

Examples:
{examples}

Respond ONLY with JSON."""
                    
                    # Check if starter already exists
                    starter_name = "Starter: Generic Doc"
                    existing = storage.get_template_by_name(starter_name)
                    
                    if existing:
                        st.warning("Starter template already exists!")
                    else:
                        new_tpl = storage.create_template(
                            starter_name,
                            content="",
                            schema_json=starter_schema,
                            examples_json=[]
                        )
                        storage.update_template(
                            new_tpl.id,
                            description="A basic template for extracting common document fields",
                            system_prompt=starter_system,
                            user_prompt=starter_user
                        )
                        st.success("✅ Starter template created!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Failed to create starter template: {e}")

    st.divider()
    
    # Template Selection
    with st.container(border=True):
        st.markdown("#### Template Editor")
        sel = st.selectbox("Select Template", options=["<New Template>"] + names)
        is_new = sel == "<New Template>"
        tpl = None if is_new else name_to_tpl.get(sel)

        name = st.text_input("Name", value=("" if is_new else tpl.name))
        desc = st.text_input("Description (optional)", value=("" if is_new else (tpl.description or "")))
        
        # Use tabs for better organization
        tab_prompts, tab_schema, tab_examples = st.tabs(["📝 Prompts", "🔧 Schema", "📚 Examples"])
        
        with tab_prompts:
            system_prompt = st.text_area(
                "System Prompt", 
                value=("" if is_new else (tpl.system_prompt or "")), 
                height=120,
                help="Instructions for the AI assistant"
            )
            user_prompt = st.text_area(
                "User Prompt",
                value=(
                    "You are a precise JSON generator. Use the schema to format results.\n{schema}\n\n{examples}\nRespond with only JSON."
                    if is_new
                    else (tpl.user_prompt or "")
                ),
                height=200,
                help="Supports {schema}, {examples}, {today}, {locale}, {doc_type}",
            )

        with tab_schema:
            schema_default = tpl.schema_json if (tpl and tpl.schema_json) else {"type": "object", "properties": {}}
            schema_text = st.text_area(
                "JSON Schema", 
                value=json.dumps(schema_default, indent=2), 
                height=220, 
                key="schema_text",
                help="Define the expected structure of the JSON response"
            )
            
            # Schema validation banner
            schema_valid = True
            schema_obj: Dict[str, Any] = {}
            validation_message = ""
            
            try:
                schema_obj = json.loads(schema_text) if schema_text.strip() else {}
                validation_message = "✅ Schema is valid JSON"
                schema_valid = True
            except Exception as e:
                schema_valid = False
                validation_message = f"❌ Schema JSON invalid: {str(e)[:100]}"
            
            # Show validation banner
            if schema_text.strip():
                if schema_valid:
                    st.success(validation_message, icon="✅")
                else:
                    st.error(validation_message, icon="❌")

        with tab_examples:
            st.markdown("Few-shot Examples (0–3)")
            examples: List[Example] = []
            existing_examples: List[Dict[str, Any]] = (tpl.examples_json if (tpl and tpl.examples_json) else [])
            count = st.slider("Number of examples", min_value=0, max_value=3, value=min(3, len(existing_examples)))
            
            for i in range(count):
                with st.container(border=True):
                    st.write(f"**Example {i+1}**")
                    cols = st.columns([3, 2])
                    with cols[0]:
                        up = st.file_uploader(
                            f"Images {i+1}", 
                            type=["png", "jpg", "jpeg", "webp"], 
                            accept_multiple_files=True, 
                            key=f"ex_up_{i}"
                        )
                        imgs: List[str] = []
                        if up and len(up) > 0:
                            for f in up[:3]:
                                imgs.append(_to_data_url(f))
                        else:
                            if i < len(existing_examples):
                                imgs = [str(u) for u in (existing_examples[i].get("images") or [])][:3]
                        if imgs:
                            st.image(imgs, width=96)
                    
                    with cols[1]:
                        expected_default = (
                            existing_examples[i].get("expected") if i < len(existing_examples) else {"ok": True}
                        )
                        exp_text = st.text_area(
                            f"Expected JSON {i+1}", 
                            value=json.dumps(expected_default, indent=2), 
                            height=140, 
                            key=f"ex_exp_{i}"
                        )
                        try:
                            expected_obj = json.loads(exp_text) if exp_text.strip() else {}
                            examples.append(Example(images=imgs, expected=expected_obj))
                        except Exception as e:
                            st.error(f"Example {i+1} expected JSON invalid: {e}")

    colA, colB, colC = st.columns(3)
    with colA:
        save_clicked = st.button("Save", use_container_width=True, disabled=not schema_valid or not name.strip())
    with colB:
        clone_clicked = st.button("Clone", use_container_width=True, disabled=is_new)
    with colC:
        del_clicked = st.button("Delete", use_container_width=True, disabled=is_new)

    if save_clicked:
        body = {
            "name": name.strip(),
            "description": desc or None,
            "system_prompt": system_prompt or None,
            "user_prompt": user_prompt or None,
            "schema": schema_obj,
            "examples": [asdict(e) for e in examples],
            "version_tag": (tpl.version_tag if tpl else None),
        }
        yaml_blob = yaml.safe_dump(body, sort_keys=False)
        if is_new:
            tnew = storage.create_template(
                name.strip(), content="", schema_json=schema_obj, examples_json=[asdict(e) for e in examples]
            )
            tnew = storage.update_template(
                tnew.id,
                description=desc or None,
                system_prompt=system_prompt or None,
                user_prompt=user_prompt or None,
                yaml_blob=yaml_blob,
            )
            st.success("Template created")
            st.rerun()
        else:
            storage.update_template(
                tpl.id,
                name=name.strip(),
                description=desc or None,
                system_prompt=system_prompt or None,
                user_prompt=user_prompt or None,
                schema_json=schema_obj,
                examples_json=[asdict(e) for e in examples],
                yaml_blob=yaml_blob,
            )
            st.success("Template saved")
            st.rerun()

    if clone_clicked and tpl:
        clone_name = st.text_input("Clone As Name", value=f"{tpl.name} (copy)")
        if st.button("Confirm Clone"):
            t2 = storage.create_template(
                clone_name.strip(), content="", schema_json=tpl.schema_json, examples_json=tpl.examples_json
            )
            storage.update_template(
                t2.id,
                description=tpl.description,
                system_prompt=tpl.system_prompt,
                user_prompt=tpl.user_prompt,
                yaml_blob=tpl.yaml_blob,
                version_tag=tpl.version_tag,
            )
            st.success("Template cloned")
            st.rerun()

    if del_clicked and tpl:
        storage.delete_template(tpl.id)
        st.warning("Template deleted")
        st.rerun()


def run() -> None:
    load_dotenv(override=False)
    st.title("Settings")
    storage.init_db()
    
    # Get active provider for sidebar
    active = storage.get_active_provider()
    
    with st.sidebar:
        prof_name = active.name if active else "None"
        logo = active.logo_path if active else None
        core_ui.status_chip("Active Profile", prof_name, logo_path=logo)
    
    # Main tabs: Model and Template
    tab_model, tab_template = st.tabs(["🤖 Model", "📝 Template"])
    
    with tab_model:
        st.markdown("### Configure AI Model")
        st.caption("Select and configure the model for image-to-JSON extraction")
        
        # Profile Management Section
        providers = storage.list_providers()
        if providers:
            st.markdown("#### 📂 Saved Profiles")
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                profile_names = ["Select a saved profile..."] + [p.name for p in providers]
                selected_profile_name = st.selectbox(
                    "Load Profile",
                    options=profile_names,
                    index=0,
                    help="Select a previously saved profile to load its settings"
                )
            
            with col2:
                if selected_profile_name != "Select a saved profile...":
                    if st.button("✅ Set Active", use_container_width=True):
                        selected_provider = next((p for p in providers if p.name == selected_profile_name), None)
                        if selected_provider:
                            storage.set_active_provider(selected_provider.id)
                            # Hydrate session with the stored API key (encrypted or session) for seamless usage
                            try:
                                key = _get_api_key_for_provider(
                                    selected_provider.id,
                                    selected_provider.key_storage,
                                    selected_provider.api_key_enc,
                                )
                                if key:
                                    keys: Dict[int, str] = st.session_state.setdefault("_api_keys", {})
                                    keys[selected_provider.id] = key
                            except Exception:
                                pass
                            st.success(f"'{selected_profile_name}' is now active!")
                            st.rerun()
            
            with col3:
                if selected_profile_name != "Select a saved profile...":
                    if st.button("🗑️ Delete", use_container_width=True):
                        selected_provider = next((p for p in providers if p.name == selected_profile_name), None)
                        if selected_provider:
                            storage.delete_provider(selected_provider.id)
                            st.warning(f"Deleted profile: {selected_profile_name}")
                            st.rerun()
            
            if selected_profile_name != "Select a saved profile...":
                # Load the selected profile
                selected_provider = next((p for p in providers if p.name == selected_profile_name), None)
                if selected_provider:
                    st.session_state["selected_profile"] = selected_profile_name
                    # Display loaded profile configuration
                    st.success(f"✅ Loaded profile: {selected_profile_name}")
                    st.divider()
                    st.markdown("### Loaded Profile Configuration")
                    _model_configuration({
                        "model_id": selected_provider.model_id,
                        "provider_name": selected_provider.name,
                        "is_loaded_profile": True
                    })
            else:
                st.divider()
                st.markdown("#### 🆕 Or Configure New Model")
                
                # Model selection sub-tabs - only show when no profile selected
                tab_catalog, tab_custom = st.tabs(["📚 Browse Catalog", "🔧 Custom Model"])
                
                selected_model_info = None
                
                with tab_catalog:
                    catalog = get_cached_catalog()
                    if catalog:
                        selected_model_info = _model_browser(catalog)
                        if selected_model_info:
                            st.success(f"✅ Selected: {selected_model_info['name']} from {selected_model_info['provider_name']}")
                            st.session_state["selected_model"] = selected_model_info
                    else:
                        st.error("Could not load models catalog")
                
                with tab_custom:
                    custom_info = _custom_model_form()
                    if custom_info:
                        selected_model_info = custom_info
                        st.success(f"✅ Using custom model: {custom_info['model_id']}")
                        st.session_state["selected_model"] = selected_model_info
                
                # Use session state to persist selection
                if not selected_model_info and "selected_model" in st.session_state:
                    selected_model_info = st.session_state["selected_model"]
                
                if selected_model_info:
                    st.divider()
                    st.markdown("### Connection Configuration")
                    _model_configuration(selected_model_info)
                else:
                    st.info("👆 Select a model from the catalog or add a custom model to configure connection settings")
        else:
            # No saved profiles - show new model configuration
            st.markdown("#### Configure New Model")
            
            # Model selection sub-tabs
            tab_catalog, tab_custom = st.tabs(["📚 Browse Catalog", "🔧 Custom Model"])
            
            selected_model_info = None
            
            with tab_catalog:
                catalog = get_cached_catalog()
                if catalog:
                    selected_model_info = _model_browser(catalog)
                    if selected_model_info:
                        st.success(f"✅ Selected: {selected_model_info['name']} from {selected_model_info['provider_name']}")
                        st.session_state["selected_model"] = selected_model_info
                else:
                    st.error("Could not load models catalog")
            
            with tab_custom:
                custom_info = _custom_model_form()
                if custom_info:
                    selected_model_info = custom_info
                    st.success(f"✅ Using custom model: {custom_info['model_id']}")
                    st.session_state["selected_model"] = selected_model_info
            
            # Use session state to persist selection
            if not selected_model_info and "selected_model" in st.session_state:
                selected_model_info = st.session_state["selected_model"]
            
            if selected_model_info:
                st.divider()
                st.markdown("### Connection Configuration")
                _model_configuration(selected_model_info)
            else:
                st.info("👆 Select a model from the catalog or add a custom model to configure connection settings")
    
    with tab_template:
        st.markdown("### Manage Templates")
        st.caption("Create and manage prompt templates for consistent extraction")
        _template_management()


# Override old run() with streamlined workflow
def run() -> None:  # type: ignore[no-redef]
    load_dotenv(override=False)
    st.title("Settings")
    storage.init_db()

    # Sidebar: show active model
    active = storage.get_active_provider()
    with st.sidebar:
        prof_name = (active.model_id or active.name) if active else "None"
        logo = active.logo_path if active else None
        core_ui.status_chip("Active Model", prof_name, logo_path=logo)

    # Tabs
    tab_model, tab_template = st.tabs(["🤖 Model", "📝 Template"])

    with tab_model:
        st.markdown("### Configure AI Model")
        st.caption("Pick a provider/model; if a key is missing you can add it inline.")

        catalog = get_cached_catalog()
        st.markdown("#### Select Model")
        tab_catalog, tab_custom = st.tabs(["📚 Browse Catalog", "🔧 Custom Model"])

        selected_model_info = None
        with tab_catalog:
            if catalog:
                selected_model_info = _grouped_model_selector(catalog)
                if selected_model_info:
                    st.success(f"✅ Selected: {selected_model_info['name']} from {selected_model_info['provider_name']}")
                    st.session_state["selected_model"] = selected_model_info
            else:
                st.error("Could not load models catalog")

        with tab_custom:
            custom_info = _custom_model_form()
            if custom_info:
                selected_model_info = custom_info
                st.success(f"✅ Using custom model: {custom_info['model_id']}")
                st.session_state["selected_model"] = selected_model_info

        if not selected_model_info and "selected_model" in st.session_state:
            # Rehydrate previous selection from session, but only if the provider key exists
            maybe_saved = st.session_state["selected_model"]
            pid = (maybe_saved or {}).get("provider_id")
            if pid and _get_api_key_for_provider_code(pid):
                selected_model_info = maybe_saved
            else:
                # Drop stale selection without key
                selected_model_info = None

        if selected_model_info:
            st.divider()
            st.markdown("### Connection Configuration")
            _streamlined_model_configuration(selected_model_info)
        else:
            st.info("👆 Select a provider and model, or add a custom model")

        # Advanced key management removed from default flow to avoid duplicate key entry points

    with tab_template:
        st.markdown("### Manage Templates")
        st.caption("Create and manage prompt templates for consistent extraction")
        _template_management()


run()
