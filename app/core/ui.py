from __future__ import annotations

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


def status_chip(label_or_active: str, value: str | None = None, *, color: str = "#10b981", logo_path: str | None = None) -> None:
    """Render a small status chip.

    Backward-compatible flexible signature:
    - status_chip("Active Profile", "MyProfile", logo_path="...")
    - status_chip("MyProfile", ".../logo.svg")  # interpreted as (active_profile, provider_logo_path)
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

    img_html = f"<img src='{logo_path}' style='height:14px;width:14px;vertical-align:middle;border-radius:2px;background:white;padding:1px;'/>" if logo_path else ""
    html = f"""
<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>
  <span style='color:#6b7280;font-size:0.85rem;'>{label}</span>
  <span style='display:inline-flex;align-items:center;gap:6px;background:{color};color:white;border-radius:9999px;padding:2px 8px;font-size:0.75rem;'>
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
