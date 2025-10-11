from __future__ import annotations

import hashlib
from pathlib import Path
import streamlit as st
from textwrap import dedent


def _render_html(html: str) -> None:
    # Prefer st.html when available (Streamlit >= 1.37), fallback to markdown
    try:
        func = getattr(st, "html", None)
        if callable(func):
            func(dedent(html))
            return
    except Exception:
        pass
    st.markdown(dedent(html), unsafe_allow_html=True)


def generate_provider_icon(provider_name: str, size: int = 24) -> str:
    """Generate a unique SVG icon for a provider based on its name.
    
    Args:
        provider_name: Name or ID of the provider
        size: Size of the icon in pixels
    
    Returns:
        Data URL containing the SVG icon
    """
    # Generate consistent colors from provider name
    hash_obj = hashlib.md5(provider_name.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Extract RGB values from hash
    r = int(hash_hex[0:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    
    # Make colors more vibrant (adjust saturation)
    max_val = max(r, g, b)
    if max_val > 0:
        factor = 200 / max_val
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
    
    bg_color = f"rgb({r},{g},{b})"
    
    # Generate lighter shade for gradient
    r2 = min(255, r + 40)
    g2 = min(255, g + 40)
    b2 = min(255, b + 40)
    bg_color2 = f"rgb({r2},{g2},{b2})"
    
    # Get initials (max 2 characters)
    words = provider_name.replace("-", " ").replace("_", " ").split()
    if len(words) >= 2:
        initials = (words[0][0] + words[1][0]).upper()
    else:
        initials = provider_name[:2].upper()
    
    # Create SVG with gradient background and initials
    svg = f'''
    <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad_{hash_hex[:8]}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{bg_color};stop-opacity:1" />
                <stop offset="100%" style="stop-color:{bg_color2};stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="{size}" height="{size}" rx="4" fill="url(#grad_{hash_hex[:8]})"/>
        <text x="50%" y="50%" text-anchor="middle" dy=".35em" 
              fill="white" font-family="Arial, sans-serif" 
              font-size="{int(size * 0.45)}" font-weight="600">
            {initials}
        </text>
    </svg>
    '''
    
    # Convert to data URL
    import base64
    svg_bytes = svg.strip().encode('utf-8')
    svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
    return f"data:image/svg+xml;base64,{svg_b64}"


def _validate_svg(svg_text: str) -> bool:
    """Ensure LLM output is valid SVG markup.
    
    Args:
        svg_text: SVG content to validate
    
    Returns:
        True if valid SVG, False otherwise
    """
    if not svg_text or not isinstance(svg_text, str):
        return False
    
    svg_lower = svg_text.lower().strip()
    
    # Must contain <svg> opening and closing tags
    if '<svg' not in svg_lower or '</svg>' not in svg_lower:
        return False
    
    # Check for potentially unsafe content
    dangerous_tags = ['script', 'iframe', 'object', 'embed', 'link']
    for tag in dangerous_tags:
        if f'<{tag}' in svg_lower:
            return False
    
    # Must be reasonably sized (not too large)
    if len(svg_text) > 50000:  # 50KB limit
        return False
    
    return True


def _extract_svg_from_response(text: str) -> str | None:
    """Extract SVG content from LLM response.
    
    Args:
        text: LLM response text
    
    Returns:
        Extracted SVG content or None
    """
    if not text:
        return None
    
    # Try to find SVG in markdown code blocks
    import re
    
    # Pattern 1: ```svg ... ```
    svg_block = re.search(r'```svg\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if svg_block:
        return svg_block.group(1).strip()
    
    # Pattern 2: ``` ... ``` (generic code block)
    code_block = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_block:
        content = code_block.group(1).strip()
        if '<svg' in content.lower():
            return content
    
    # Pattern 3: Direct SVG content
    svg_match = re.search(r'(<svg[\s\S]*?</svg>)', text, re.IGNORECASE)
    if svg_match:
        return svg_match.group(1).strip()
    
    return None


def generate_llm_provider_icon(provider_name: str, size: int = 24) -> str | None:
    """Generate SVG icon using LLM, with caching in database.
    
    This function will:
    1. Check database cache first
    2. If not cached, use active LLM to generate a minimalist SVG icon
    3. Validate the SVG output
    4. Cache in database for future use
    5. Return as data URL
    
    Args:
        provider_name: Provider identifier (e.g., 'openai', 'anthropic')
        size: Icon size in pixels
    
    Returns:
        Data URL containing SVG icon, or None if generation fails
    """
    try:
        from app.core import storage
        
        # Check cache first
        cached_svg = storage.get_provider_icon(provider_name)
        if cached_svg and _validate_svg(cached_svg):
            # Convert cached SVG to data URL
            import base64
            svg_bytes = cached_svg.encode('utf-8')
            svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
            return f"data:image/svg+xml;base64,{svg_b64}"
        
        # Not cached or invalid - generate new icon using LLM
        try:
            from app.core.provider_openai import OAIGateway
            
            # Get active provider configuration
            active_provider = storage.get_active_provider()
            if not active_provider:
                return None
            
            # Get API key
            if active_provider.key_storage == "session":
                # Try to get from session state
                try:
                    api_key = st.session_state.get("_api_keys", {}).get(active_provider.id)
                except Exception:
                    api_key = None
            else:
                # Decrypt from database
                try:
                    from cryptography.fernet import Fernet
                    import os
                    key = os.getenv("APP_KMS_KEY")
                    if not key:
                        key_path = Path("data/kms.key")
                        if key_path.exists():
                            key = key_path.read_text(encoding="utf-8").strip()
                    if key and active_provider.api_key_enc:
                        fernet = Fernet(key)
                        api_key = fernet.decrypt(active_provider.api_key_enc.encode()).decode()
                    else:
                        api_key = None
                except Exception:
                    api_key = None
            
            if not api_key and not active_provider.base_url.startswith("http://127.0.0.1"):
                return None
            
            # Create gateway
            gateway = OAIGateway(
                base_url=active_provider.base_url,
                api_key=api_key or "",
                headers=active_provider.headers_json or {},
                timeout=30,  # Short timeout for icon generation
                prefer_json_mode=False,
                prefer_tools=False,
                detected_caps=None
            )
            
            # Generate prompt
            prompt = f"""Generate a minimalist SVG icon ({size}x{size} pixels) for the "{provider_name}" AI provider.

Requirements:
- Use simple geometric shapes (circles, rectangles, paths)
- Include a subtle gradient for depth
- Theme should relate to the provider name (e.g., OpenAI → brain/neural network, Anthropic → simple 'A', Google → colorful abstract)
- No text or letters in the icon itself
- Professional and modern aesthetic
- Single focal point
- Use vibrant, recognizable colors

Output ONLY the complete SVG code, nothing else. Start with <svg and end with </svg>."""

            # Call LLM
            result = gateway.chat_vision(
                model=active_provider.model_id or "gpt-4",
                system_text="You are an expert SVG icon designer. Generate clean, minimalist SVG code.",
                user_text=prompt,
                image_paths=[],
                fewshot_messages=None,
                schema=None,
                gen_params={"temperature": 0.7, "max_tokens": 1000}
            )
            
            if result.get("error") or not result.get("text"):
                return None
            
            # Extract SVG from response
            svg_content = _extract_svg_from_response(result["text"])
            
            if not svg_content or not _validate_svg(svg_content):
                return None
            
            # Clean up SVG - ensure it has proper size attributes
            import re
            if f'width="{size}"' not in svg_content and 'width=' not in svg_content:
                svg_content = svg_content.replace('<svg', f'<svg width="{size}" height="{size}"', 1)
            
            # Cache the validated SVG
            storage.save_provider_icon(provider_name, svg_content)
            
            # Convert to data URL
            import base64
            svg_bytes = svg_content.encode('utf-8')
            svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
            return f"data:image/svg+xml;base64,{svg_b64}"
            
        except Exception:
            # LLM generation failed - return None to use fallback
            return None
            
    except Exception:
        # Any error - return None to use fallback
        return None


def status_chip(label_or_active: str, value: str | None = None, *, color: str = "#10b981", logo_path: str | None = None, provider_id: str | None = None) -> None:
    """Render a small status chip.

    Backward-compatible flexible signature:
    - status_chip("Active Profile", "MyProfile", logo_path="...")
    - status_chip("MyProfile", ".../logo.svg")  # interpreted as (active_profile, provider_logo_path)
    - status_chip("Active Model", "MyModel", provider_id="openai")  # auto-generates icon
    """
    # Flexible args: if only two positionals and second looks like a path/url, treat as (active, logo)
    label = "Active Profile"
    active = label_or_active
    if value is not None and logo_path is None and ("/" in value or value.endswith(".svg") or value.endswith(".png")):
        logo_path = value
        value = None
    if value is not None:
        label = label_or_active
        active = value

    # Generate icon if provider_id is given and no logo_path
    if provider_id and not logo_path:
        # Try LLM-generated icon first (with cache)
        logo_path = generate_llm_provider_icon(provider_id, size=16)
        # Fallback to hash-based if LLM fails
        if not logo_path:
            logo_path = generate_provider_icon(provider_id, size=16)
    
    img_html = f"<img src='{logo_path}' style='height:16px;width:16px;vertical-align:middle;border-radius:50%;background:white;padding:2px;box-shadow:0 1px 3px rgba(0,0,0,0.2);'/>" if logo_path else ""
    html = f"""
<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>
  <span style='color:#6b7280;font-size:0.85rem;'>{label}</span>
  <span style='display:inline-flex;align-items:center;gap:6px;background:{color};color:white;border-radius:9999px;padding:2px 10px;font-size:0.75rem;'>
    {img_html}
    <span>{active}</span>
  </span>
</div>
"""
    _render_html(html)


def badge(label: str, state: bool | None = None) -> str:
    """Render a single badge with color coding.
    
    Args:
        label: Text to display in the badge
        state: True=green, False=gray, None=amber (unknown/mismatch)
    
    Returns:
        HTML string for the badge
    """
    if state is True:
        color = "#10b981"  # Green for true
        icon = "✅"
    elif state is False:
        color = "#6b7280"  # Gray for false
        icon = "❌"
    else:
        color = "#f59e0b"  # Amber for unknown/mismatch
        icon = "⚠️"
    
    return f"<span style='display:inline-block;margin-right:8px;background:{color};color:white;border-radius:9999px;padding:2px 8px;font-size:0.75rem;'>{icon} {label}</span>"


def capability_badges(catalog_caps: dict | None, detected_caps: dict | None) -> None:
    """Render capability badges comparing catalog vs detected caps.

    Shows badges for Vision, JSON Mode, Tools with green/amber indicators.
    """
    det = detected_caps or {}
    cat = catalog_caps or {}
    
    # Vision capability
    cat_mods = cat.get("modality") or cat.get("modalities")
    if isinstance(cat_mods, dict):
        cat_mods = cat_mods.get("input", [])
    cat_vision = ("image" in cat_mods) if isinstance(cat_mods, list) else None
    det_vision = det.get("vision") if isinstance(det, dict) else None
    vision_state = det_vision if det_vision is not None else cat_vision
    
    # JSON Mode capability
    det_json = det.get("json_mode") if isinstance(det, dict) else None
    
    # Tools capability
    det_tools = det.get("tools") if isinstance(det, dict) else None

    html = "<div style='margin:6px 0;'>" \
        + badge("Vision", vision_state) \
        + badge("JSON Mode", det_json) \
        + badge("Tools", det_tools) \
        + "</div>"
    _render_html(html)


def toast(msg: str, *, kind: str = "info") -> None:
    """Show a toast when available; fallback to message types."""
    try:
        st.toast(msg)
    except Exception:
        if kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.info(msg)


def info(msg: str) -> None:
    st.info(msg)


def success(msg: str) -> None:
    st.success(msg)


def warning(msg: str) -> None:
    st.warning(msg)


def error(msg: str) -> None:
    st.error(msg)
