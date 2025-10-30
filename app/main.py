from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
import time
import traceback

# Ensure project root on sys.path so `app` package imports work when running `streamlit run app/main.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Local imports
from app.core import storage
from app.core import template_assets
from app.core import ui as core_ui
from app.core.model_registry import ModelRegistryError, active_model, ensure_registry
from app.core.models_dev import get_logo_path
from app.integrations.supabase_client import SupabaseMetaClient, SupabaseAPIError
from app.auth.session import SessionManager, NotAuthenticated
from app.sync.metadata_sync import MetadataSync

def _redirect_to_login() -> None:
    # Try to switch to the login page in multipage mode; fall back to rerun.
    for candidate in ("app/pages/0_Login.py", "pages/0_Login.py", "0_Login.py"):
        try:
            st.switch_page(candidate)
            return
        except Exception:
            continue
    st.experimental_set_query_params(auth="login", t=str(time.time()))
    st.rerun()


def _load_or_redirect() -> tuple[SessionManager, MetadataSync]:
    """
    Ensure we have a valid Supabase session and a fresh local metadata cache.
    This lets the rest of main.py run exactly as before.
    """
    # Build Supabase client (will raise if env missing)
    try:
        supa = SupabaseMetaClient()
    except SupabaseAPIError as e:
        st.error(f"Supabase not configured: {e}")
        st.stop()

    sm = SessionManager(supabase=supa)

    # Try load persisted tokens (set by 0_Login or a previous run)
    sess = sm.try_load_session_from_disk()
    if not sess:
        # If main.py is opened directly without login, redirect.
        _redirect_to_login()
        st.stop()

    # Prepare local cache and sync if needed
    db_path = os.getenv("APP_DB_PATH", "data/app.db")
    sync = MetadataSync(db_path=db_path)
    sync.ensure_schema()
    info = sync.get_last_sync_info()
    # Sync if cache is empty, belongs to another user, or is old (>10 min)
    need_sync = (
        not info.get("last_sync_user_id")
        or info.get("last_sync_user_id") != sess.user.user_id
        or _is_older_than_minutes(info.get("last_sync_ts"), 10)
    )
    if need_sync:
        with st.spinner("Syncing configuration..."):
            sm.sync_metadata_for_current_user(sync)

    # Expose minimal session info to st.session_state for the rest of your app
    sm.attach_to_streamlit_state()
    return sm, sync


def _is_older_than_minutes(ts_iso: str | None, minutes: int) -> bool:
    if not ts_iso:
        return True
    try:
        import datetime as _dt
        last = _dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        return (_dt.datetime.now(_dt.timezone.utc) - last).total_seconds() > minutes * 60
    except Exception:
        return True


def _sidebar_account_and_project(sm: SessionManager, sync: MetadataSync) -> None:
    """
    Safe, minimal sidebar: shows current user + project picker filtered to assignments.
    If your main.py already renders a project dropdown, you can remove this block.
    """
    with st.sidebar:
        # Account box
        user = sm.current_user()
        st.markdown(f"**{user.display_name or user.email}**")
        st.caption(f"Role: `{user.role}`")

        # Project select (assigned projects only)
        projects = sync.list_projects_for_user()
        if projects:
            name_to_id = {p["name"]: p["id"] for p in projects}
            # pick current
            current_id = st.session_state.get("active_project_id", sm.get_active_project_id())
            if current_id not in name_to_id.values():
                current_id = next(iter(name_to_id.values()))
                sm.set_active_project(current_id)

            names = list(name_to_id.keys())
            idx = list(name_to_id.values()).index(current_id) if current_id in name_to_id.values() else 0
            chosen = st.selectbox("Project", names, index=idx, key="__auth_bootstrap_project_select")
            new_id = name_to_id[chosen]
            if new_id != current_id:
                sm.set_active_project(new_id)

        # Optional actions
        col1, col2 = st.columns(2)
        if col1.button("🔄 Resync", use_container_width=True):
            with st.spinner("Refreshing configuration..."):
                sm.sync_metadata_for_current_user(sync)
                sm.attach_to_streamlit_state()
            st.experimental_set_query_params(t=str(time.time()))
            st.rerun()
        if col2.button("🚪 Logout", use_container_width=True):
            sm.logout()
            _redirect_to_login()


# ---- run the bootstrap unless explicitly disabled (for tests) ----
if os.getenv("DISABLE_AUTH_BOOTSTRAP", "0").lower() not in ("1", "true", "yes", "y", "on"):
    try:
        __SM__, __SYNC__ = _load_or_redirect()
        # Render a tiny sidebar section. If you already manage the sidebar elsewhere,
        # you can delete this line and keep using st.session_state["active_project_id"].
        _sidebar_account_and_project(__SM__, __SYNC__)
    except NotAuthenticated:
        _redirect_to_login()
    except Exception as e:
        st.error("Startup failed.")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        st.stop()

def _get_active_profile() -> tuple[str, str | None, str | None]:
    """Resolve the active model/profile, logo, and provider_id from DB or env.
    
    Returns:
        Tuple of (name, logo_path, provider_id)
    """
    # Prefer an explicitly active provider saved in DB (from Settings)
    try:
        db_active = storage.get_active_provider()
        if db_active is not None:
            # Compose a human-readable label
            label_parts = []
            if db_active.provider_code:
                label_parts.append(db_active.provider_code)
            if db_active.model_id:
                label_parts.append(db_active.model_id)
            composed = " • ".join(label_parts) if label_parts else (db_active.name or "Active Model")
            logo = db_active.logo_path
            provider_id = db_active.provider_code or "custom"
            return composed, logo, provider_id
    except Exception:
        pass
    try:
        descriptor = active_model()
        name = descriptor.label
        logo = get_logo_path(descriptor.provider_id)
        return name, logo, descriptor.provider_id
    except ModelRegistryError:
        pass
    
    # Fallback: Try to extract provider info from models.xlsx
    profile_name = None
    provider_id = None
    model_name = None
    
    # Get profile from secrets or env
    project_secrets = _ROOT / ".streamlit" / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if project_secrets.exists() or user_secrets.exists():
        try:
            profile_name = st.secrets.get("APP_PROFILE")  # type: ignore[attr-defined]
        except Exception:
            pass
    if not profile_name:
        profile_name = os.getenv("APP_PROFILE", "dev")
    
    # Try to read from models.xlsx to get provider info
    try:
        import pandas as pd
        excel_path = Path("config/models.xlsx")
        if excel_path.exists():
            df = pd.read_excel(excel_path)
            # Find row matching current profile
            profile_row = df[df['profile'] == profile_name]
            if not profile_row.empty:
                provider_id = profile_row.iloc[0].get('default_provider')
                default_model = profile_row.iloc[0].get('default_model')
                if default_model:
                    # Format model name nicely (gemini-2.5-flash -> Gemini 2.5 Flash)
                    model_name = default_model.replace('-', ' ').replace('_', ' ').title()
    except Exception:
        pass
    
    # Compose display name
    if model_name:
        display_name = model_name
    else:
        display_name = str(profile_name) if profile_name else "Default"
    
    return display_name, None, provider_id


def _ensure_runtime_dirs() -> None:
    for d in ("data", "export", "app/assets"):
        Path(d).mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Load environment from config/.env first (preferred), then fallback to project .env
    config_env = _ROOT / "config" / ".env"
    try:
        loaded = False
        if config_env.exists():
            loaded = load_dotenv(config_env, override=False)
        if not loaded:
            load_dotenv(_ROOT / ".env", override=False)
    except Exception:
        # Do not crash app if dotenv loading fails
        pass
    registry_error: str | None = None
    registry = None
    try:
        registry = ensure_registry()
    except Exception as exc:  # pragma: no cover - surface configuration issues
        registry_error = str(exc)
    st.set_page_config(
        page_title="Images -> JSON",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ensure folders exist and database is initialized
    _ensure_runtime_dirs()
    storage.init_db()
    template_assets.sync_from_assets()

    if registry_error:
        st.error(f"Failed to load model configuration: {registry_error}")

    # Sidebar: navigation is auto-generated by Streamlit's pages.
    with st.sidebar:
        st.markdown("#### Navigation")
        st.markdown("<hr>", unsafe_allow_html=True)
        prof_name, logo, provider_id = _get_active_profile()
        # Use auto-generated icon if no logo is available
        core_ui.status_chip("Active Model", prof_name, logo_path=logo, provider_id=provider_id)
        
        # Project selector and money tracker
        st.markdown("---")
        st.markdown("#### 📁 Active Project")
        
        try:
            projects = storage.list_projects()
            if not projects:
                # Initialize with default project if none exist
                storage.init_db()
                projects = storage.list_projects()
            
            if projects:
                active_project = storage.get_active_project()
                project_names = [p.name for p in projects]
                
                # If no active project, set the first one as active
                if not active_project and projects:
                    storage.set_active_project(projects[0].id)
                    active_project = projects[0]
                
                current_index = 0
                if active_project and active_project.name in project_names:
                    current_index = project_names.index(active_project.name)
                
                selected = st.selectbox(
                    "Select Project",
                    options=project_names,
                    index=current_index,
                    key="sidebar_project_selector",
                    label_visibility="collapsed"
                )
                
                # Switch project if selection changed
                if active_project and selected != active_project.name:
                    new_proj = next((p for p in projects if p.name == selected), None)
                    if new_proj:
                        storage.set_active_project(new_proj.id)
                        st.rerun()
                
                # Show project stats
                if active_project:
                    stats = storage.get_project_stats(active_project.id)
                    
                    # Get currency rate for INR display
                    from app.core.currency import get_usd_to_inr, convert_usd_to_inr
                    usd_to_inr = get_usd_to_inr()
                    
                    # Show currency rate badge
                    if usd_to_inr:
                        st.markdown(
                            f'<div style="background-color: #d1fae5; color: #065f46; padding: 6px 10px; '
                            f'border-radius: 6px; font-size: 0.8rem; margin: 12px 0; text-align: center; '
                            f'font-weight: 500; border: 1px solid #a7f3d0;">'
                            f'💱 1 USD = ₹{usd_to_inr:.2f}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Stats in clean columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div style="text-align: center; padding: 8px; background-color: #f9fafb; '
                            f'border-radius: 6px; margin-bottom: 8px;">'
                            f'<div style="font-size: 0.7rem; color: #6b7280; font-weight: 500;">IMAGES</div>'
                            f'<div style="font-size: 1.5rem; font-weight: 600; color: #111827; margin-top: 2px;">{stats["total_images"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div style="text-align: center; padding: 8px; background-color: #f9fafb; '
                            f'border-radius: 6px; margin-bottom: 8px;">'
                            f'<div style="font-size: 0.7rem; color: #6b7280; font-weight: 500;">RUNS</div>'
                            f'<div style="font-size: 1.5rem; font-weight: 600; color: #111827; margin-top: 2px;">{stats["total_runs"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Total cost section
                    total_usd = stats['total_cost_usd']
                    cost_html = '<div style="padding: 10px; background-color: #fef3c7; border-radius: 6px; border: 1px solid #fde68a; margin-top: 8px;">'
                    cost_html += '<div style="font-size: 0.7rem; color: #92400e; font-weight: 500; margin-bottom: 4px;">TOTAL SPENT</div>'
                    cost_html += f'<div style="font-size: 1.1rem; font-weight: 600; color: #78350f;">${total_usd:.4f}</div>'
                    
                    if usd_to_inr:
                        total_inr = convert_usd_to_inr(total_usd, rate=usd_to_inr)
                        if total_inr is not None:
                            cost_html += f'<div style="font-size: 0.85rem; color: #92400e; margin-top: 2px;">₹{total_inr:.2f}</div>'
                    
                    cost_html += '</div>'
                    st.markdown(cost_html, unsafe_allow_html=True)
                    
                    # Average per image - only show if there are images
                    if stats['total_images'] > 0:
                        avg_usd = total_usd / stats['total_images']
                        avg_text = f"${avg_usd:.4f}"
                        if usd_to_inr:
                            avg_inr = convert_usd_to_inr(avg_usd, rate=usd_to_inr)
                            if avg_inr is not None:
                                avg_text += f" • ₹{avg_inr:.4f}"
                        st.markdown(
                            f'<div style="text-align: center; font-size: 0.75rem; color: #6b7280; margin-top: 8px;">'
                            f'Avg per image: {avg_text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        except Exception as e:
            st.error(f"Error loading projects: {e}")

    # Clean title with muted subtitle
    st.title("🧩 Images -> JSON")
    st.markdown(
        '<p style="color: #6b7280; font-size: 1.1rem; margin-top: -10px; margin-bottom: 30px;">'
        'Transform images into structured data with AI-powered extraction'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Welcome section in container
    with st.container(border=True):
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **📤 Upload & Process** - Upload images and extract structured data
        2. **⚙️ Settings** - Configure AI models and manage templates
        
        **Quick Start:** Upload images -> Select template -> Process -> Export results
        """)
        
        # Quick stats if we have data
        try:
            templates = storage.list_templates()
            model_count = len(registry.models) if registry else 0
            st.markdown(f"**Status:** {len(templates)} templates • {model_count} models configured")
        except Exception:
            pass


if __name__ == "__main__":
    main()
