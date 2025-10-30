"""
Login page (email/password) for the Desktop OCR app.

- Minimal UI change: this page appears only for authentication.
- After successful login it:
    * syncs metadata from Supabase to local cache
    * attaches auth info to st.session_state
    * redirects to main app page

Relies on:
  - File 2: app/integrations/supabase_client.py
  - File 3: app/sync/metadata_sync.py
  - File 4: app/auth/session.py

Env:
  SUPABASE_URL, SUPABASE_ANON_KEY (required)
  SUPABASE_PERSIST_SESSION=1 (optional; remember session to disk)
  APP_DB_PATH=./data/app.db (optional)
  APP_SESSION_FILE=./data/session.json (optional)
"""

from __future__ import annotations

import os
import time
import traceback
import streamlit as st

from app.integrations.supabase_client import SupabaseMetaClient, SupabaseAPIError
from app.sync.metadata_sync import MetadataSync
from app.auth.session import SessionManager, NotAuthenticated, Forbidden


MAIN_ENTRY_CANDIDATES = [
    # prefer explicit path (most repos run: streamlit run app/main.py)
    "app/main.py",
    # fallback if main sits at project root
    "main.py",
]


def _goto_main():
    for target in MAIN_ENTRY_CANDIDATES:
        try:
            st.switch_page(target)
            return
        except Exception:
            continue
    # If switch_page isn't available or paths don't match, just rerun.
    st.experimental_set_query_params(t=str(time.time()))
    st.rerun()


def _render_sidebar_help():
    with st.sidebar:
        st.markdown("### 🔐 Sign in")
        st.caption(
            "Use your email and password. After signing in, your projects, templates, "
            "providers & models will sync locally. No files or prompts are uploaded."
        )
        supabase_url = os.getenv("SUPABASE_URL", "")
        if not supabase_url:
            st.error("Missing SUPABASE_URL / SUPABASE_ANON_KEY in environment.")
        else:
            st.caption(f"Supabase: `{supabase_url}`")


def _try_autologin(supa: SupabaseMetaClient, db_path: str) -> bool:
    """Attempt to load a previously persisted session and fast-path to main."""
    sm = SessionManager(supabase=supa)
    sess = sm.try_load_session_from_disk()
    if not sess:
        return False

    # If tokens OK, sync metadata and go to main
    sync = MetadataSync(db_path=db_path)
    with st.spinner("Restoring session & syncing metadata..."):
        sm.sync_metadata_for_current_user(sync)
        sm.attach_to_streamlit_state()
    _goto_main()
    return True


def _login_flow():
    st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")
    _render_sidebar_help()

    db_path = os.getenv("APP_DB_PATH", "data/app.db")

    # Build client (will raise if env missing)
    try:
        supa = SupabaseMetaClient()
    except SupabaseAPIError as e:
        st.error(str(e))
        st.stop()

    # Fast-path: if we have a valid persisted session, go straight to main
    if os.getenv("SUPABASE_PERSIST_SESSION", "1").lower() in ("1", "true", "yes", "y", "on"):
        if _try_autologin(supa, db_path):
            return

    st.title("Sign in")
    st.caption("Authenticate to continue. Your data stays on your device.")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", value="", autocomplete="username")
        password = st.text_input("Password", value="", type="password", autocomplete="current-password")
        remember = st.checkbox("Remember me on this device", value=True, help="Stores encrypted Supabase tokens locally for auto-login.")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if not email or not password:
            st.warning("Please enter both email and password.")
            st.stop()

        # Run login
        try:
            # Override persistence for this instance based on checkbox
            sm = SessionManager(supabase=supa, persist_tokens=remember)

            with st.spinner("Signing in..."):
                sm.login_email_password(email, password)

            # Sync metadata visible to this user
            sync = MetadataSync(db_path=db_path)
            with st.spinner("Syncing configuration & projects..."):
                sm.sync_metadata_for_current_user(sync)

            sm.attach_to_streamlit_state()
            st.success("Signed in successfully. Redirecting...")
            _goto_main()

        except (SupabaseAPIError, NotAuthenticated, Forbidden) as e:
            st.error(f"Authentication failed: {e}")
            with st.expander("Details"):
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    _login_flow()
else:
    _login_flow()
