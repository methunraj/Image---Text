"""
Session + role handling for the Desktop OCR app.

What this file does
-------------------
- Holds Supabase auth tokens (in-memory; optional disk persistence).
- Validates the current user with Supabase and loads the app role.
- Provides helpers to:
    * gate pages by role (admin/editor/user)
    * pick & remember an active project for the session
    * decrypt project-scoped API key ciphertexts from the local Supabase cache
- Integrates with:
    * File 2: app.integrations.supabase_client.SupabaseMetaClient
    * File 3: app.sync.metadata_sync.MetadataSync

Security notes
--------------
- Plaintext API keys are NEVER written to disk by this module.
- Project API keys are decrypted in-memory only, using a symmetric Fernet key.
- Provide the Fernet key via env var APP_KMS_KEY, or a file at data/kms.key.
- Supabase access/refresh tokens can be optionally persisted for auto-login
  (set SUPABASE_PERSIST_SESSION=1). If you disable it, tokens live in memory only.

Environment variables
---------------------
APP_DB_PATH=./data/app.db                    # used elsewhere, not directly here
APP_KMS_KEY=<base64url_fernet_key>           # preferred
APP_KMS_KEY_FILE=./data/kms.key              # alternative if APP_KMS_KEY unset
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
SUPABASE_PERSIST_SESSION=1                   # 1/true to save session tokens to disk
APP_SESSION_FILE=./data/session.json         # where tokens are saved if persistence enabled
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

try:
    # cryptography is already used elsewhere in the repo for Fernet
    from cryptography.fernet import Fernet, InvalidToken
except Exception as _e:  # pragma: no cover
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore

# Optional Streamlit import (only used if you call helpers with st)
try:  # pragma: no cover
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

from app.integrations.supabase_client import (
    SupabaseMetaClient,
    SessionTokens,
    UserProfile,
)
from app.sync.metadata_sync import MetadataSync


# ============================= Data models ============================= #

@dataclass
class AppUser:
    user_id: str
    email: str
    role: str  # 'admin' | 'editor' | 'user'
    display_name: Optional[str] = None


@dataclass
class AppSession:
    tokens: SessionTokens
    user: AppUser


# ============================= Exceptions ============================= #

class NotAuthenticated(RuntimeError):
    pass


class Forbidden(RuntimeError):
    pass


class KMSError(RuntimeError):
    pass


# ============================= SessionManager ============================= #

class SessionManager:
    """
    Orchestrates Supabase session lifecycle and role gating.
    """

    def __init__(
        self,
        *,
        supabase: SupabaseMetaClient,
        session_file: Optional[str] = None,
        persist_tokens: Optional[bool] = None,
    ) -> None:
        self.supa = supabase
        self.session_file = session_file or os.getenv("APP_SESSION_FILE", "data/session.json")
        self.persist_tokens = _env_truthy("SUPABASE_PERSIST_SESSION") if persist_tokens is None else bool(persist_tokens)

        self._app_session: Optional[AppSession] = None
        # ephemeral cache: (project_id, provider_id) -> plaintext API key
        self._runtime_keys: Dict[Tuple[str, str], str] = {}

        # active project id remembered here (and optionally in Streamlit session_state)
        self._active_project_id: Optional[str] = None

    # ----------------------------- Authentication ----------------------------- #

    def login_email_password(self, email: str, password: str) -> AppSession:
        tokens = self.supa.sign_in_email_password(email, password)
        user_o = self.supa.get_current_user()
        prof = self.supa.get_user_profile(user_o["id"])
        if prof is None:
            # Fallback to a safe default role if profile row doesn't exist yet.
            prof = UserProfile(user_id=user_o["id"], email=user_o.get("email") or email, display_name=None, role="user")

        app_user = AppUser(user_id=prof.user_id, email=prof.email, role=prof.role, display_name=prof.display_name)
        self._app_session = AppSession(tokens=tokens, user=app_user)

        if self.persist_tokens:
            self._save_session_to_disk(tokens)

        return self._app_session

    def set_session_from_tokens(self, access_token: str, refresh_token: str, expires_at: float) -> AppSession:
        tokens = SessionTokens(access_token=access_token, refresh_token=refresh_token, expires_at=float(expires_at))
        self.supa.set_session(access_token, refresh_token, expires_at)
        user_o = self.supa.get_current_user()
        prof = self.supa.get_user_profile(user_o["id"])
        if prof is None:
            prof = UserProfile(user_id=user_o["id"], email=user_o.get("email") or "", display_name=None, role="user")
        app_user = AppUser(user_id=prof.user_id, email=prof.email, role=prof.role, display_name=prof.display_name)
        self._app_session = AppSession(tokens=tokens, user=app_user)
        return self._app_session

    def try_load_session_from_disk(self) -> Optional[AppSession]:
        """
        Attempts to load tokens from disk and validate the session.
        Returns AppSession or None if not available/invalid.
        """
        if not self.persist_tokens:
            return None
        try:
            if not os.path.exists(self.session_file):
                return None
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            access = data.get("access_token")
            refresh = data.get("refresh_token")
            expires_at = float(data.get("expires_at", 0))
            if not (access and refresh and expires_at):
                return None
            # Allow small negative skew
            if time.time() > expires_at + 2 * 60 * 60:  # >2h past expiry, ignore
                return None
            return self.set_session_from_tokens(access, refresh, expires_at)
        except Exception:
            return None

    def ensure_authenticated(self) -> AppSession:
        if not self._app_session:
            raise NotAuthenticated("No active session. Please log in.")
        # also verify tokens are fresh
        self.supa.ensure_fresh_session()
        return self._app_session

    def logout(self) -> None:
        self._app_session = None
        self._active_project_id = None
        self._runtime_keys.clear()
        if os.path.exists(self.session_file):
            try:
                os.remove(self.session_file)
            except Exception:
                pass

    # ----------------------------- Role gating ----------------------------- #

    def current_user(self) -> AppUser:
        sess = self.ensure_authenticated()
        return sess.user

    def is_admin(self) -> bool:
        try:
            return self.current_user().role == "admin"
        except NotAuthenticated:
            return False

    def is_editor_or_admin(self) -> bool:
        try:
            return self.current_user().role in ("editor", "admin")
        except NotAuthenticated:
            return False

    def require_authenticated(self) -> None:
        self.ensure_authenticated()

    def require_admin(self) -> None:
        if not self.is_admin():
            raise Forbidden("Admin access required.")

    def require_editor_or_admin(self) -> None:
        if not self.is_editor_or_admin():
            raise Forbidden("Editor or Admin access required.")

    # ----------------------------- Streamlit helpers (optional) ----------------------------- #

    def attach_to_streamlit_state(self) -> None:  # pragma: no cover
        """
        Populate st.session_state with minimal, non-sensitive info for UI branching.
        """
        if st is None:
            return
        sess = self.ensure_authenticated()
        st.session_state.setdefault("auth", {})
        st.session_state["auth"].update(
            {
                "user_id": sess.user.user_id,
                "email": sess.user.email,
                "role": sess.user.role,
                "expires_at": sess.tokens.expires_at,
            }
        )
        # active project id mirrored to streamlit state for convenience
        if self._active_project_id is not None:
            st.session_state["active_project_id"] = self._active_project_id

    def read_active_project_from_streamlit(self) -> Optional[str]:  # pragma: no cover
        if st is None:
            return self._active_project_id
        pid = st.session_state.get("active_project_id")
        if isinstance(pid, str):
            self._active_project_id = pid
        return self._active_project_id

    # ----------------------------- Metadata sync + selection ----------------------------- #

    def sync_metadata_for_current_user(self, meta_sync: MetadataSync) -> Dict[str, int]:
        sess = self.ensure_authenticated()
        meta_sync.ensure_schema()
        counts = meta_sync.sync_for_user(self.supa, sess.user.user_id, wipe_before=True)
        # ensure we have an active project
        self._ensure_active_project(meta_sync)
        return counts

    def _ensure_active_project(self, meta_sync: MetadataSync) -> None:
        if self._active_project_id:
            return
        projects = meta_sync.list_projects_for_user()
        if projects:
            self._active_project_id = projects[0]["id"]
            if st is not None:  # pragma: no cover
                st.session_state["active_project_id"] = self._active_project_id

    def set_active_project(self, project_id: str) -> None:
        self._active_project_id = project_id
        if st is not None:  # pragma: no cover
            st.session_state["active_project_id"] = project_id

    def get_active_project_id(self) -> Optional[str]:
        return self._active_project_id

    # ----------------------------- Project API keys (decrypt in-memory) ----------------------------- #

    def get_decrypted_project_keys(
        self, meta_sync: MetadataSync, project_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Returns {provider_id: plaintext_api_key} for the given (or active) project.
        Keys are decrypted in-memory using APP_KMS_KEY / APP_KMS_KEY_FILE.
        """
        if project_id is None:
            project_id = self.get_active_project_id()
        if not project_id:
            return {}

        # Build from cache
        out: Dict[str, str] = {}
        for row in meta_sync.get_project_api_keys(project_id):
            cache_key = (row["project_id"], row["provider_id"])
            if cache_key in self._runtime_keys:
                out[row["provider_id"]] = self._runtime_keys[cache_key]
                continue

            plaintext = self._decrypt_fernet(row["api_key_ciphertext"])
            if plaintext:
                self._runtime_keys[cache_key] = plaintext
                out[row["provider_id"]] = plaintext

        return out

    def resolve_api_key_for_provider(
        self, meta_sync: MetadataSync, provider_id: str, project_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Convenience: return plaintext for a single provider if available.
        """
        keys = self.get_decrypted_project_keys(meta_sync, project_id=project_id)
        return keys.get(provider_id)

    # ----------------------------- KMS / Fernet ----------------------------- #

    def _load_kms_key(self) -> bytes:
        """
        Load Fernet key. Accepts:
          - APP_KMS_KEY env var (raw Fernet base64url bytes)
          - APP_KMS_KEY_FILE path containing the raw key
          - fallback: data/kms.key
        """
        key_env = os.getenv("APP_KMS_KEY")
        if key_env:
            try:
                # Accept either urlsafe base64 or plain
                try:
                    return base64.urlsafe_b64decode(key_env.encode("utf-8"))
                except Exception:
                    # If it's already the 32-byte raw key in base64url form, Fernet expects that string directly.
                    return key_env.encode("utf-8")
            except Exception as e:
                raise KMSError(f"Invalid APP_KMS_KEY: {e}")

        key_file = os.getenv("APP_KMS_KEY_FILE", "data/kms.key")
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                key = f.read().strip()
            if not key:
                raise KMSError("APP_KMS_KEY_FILE is empty.")
            return key

        raise KMSError(
            "No KMS key found. Set APP_KMS_KEY or provide APP_KMS_KEY_FILE (e.g., data/kms.key)."
        )

    def _decrypt_fernet(self, ciphertext: str) -> Optional[str]:
        """
        Decrypts a Fernet ciphertext string using the app KMS key.
        Returns plaintext or None if decryption fails.
        """
        if Fernet is None:
            raise KMSError("cryptography is not installed. Please add 'cryptography' to requirements.txt")
        try:
            key_bytes = self._load_kms_key()
            # Fernet() expects a base64url key (32 url-safe bytes)
            # If key_bytes isn't already url-safe, assume it's provided correctly (existing repo uses Fernet)
            f = Fernet(key_bytes)
            pt = f.decrypt(ciphertext.encode("utf-8"))
            return pt.decode("utf-8")
        except InvalidToken:
            # wrong key or corrupted ciphertext
            return None
        except Exception as e:
            raise KMSError(f"Fernet decrypt failed: {e}")

    # ----------------------------- Persistence ----------------------------- #

    def _save_session_to_disk(self, tokens: SessionTokens) -> None:
        os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
        data = {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "expires_at": tokens.expires_at,
        }
        tmp = self.session_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.session_file)


# ============================= Utilities ============================= #

def _env_truthy(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


# ============================= Minimal usage ============================= #
# (Example — call this from your 0_Login.py or main.py)
#
# from app.integrations.supabase_client import SupabaseMetaClient
# from app.sync.metadata_sync import MetadataSync
#
# supa = SupabaseMetaClient()
# sm = SessionManager(supabase=supa)
# sess = sm.try_load_session_from_disk() or sm.login_email_password(email, password)
# sync = MetadataSync(db_path=os.getenv("APP_DB_PATH", "data/app.db"))
# sm.sync_metadata_for_current_user(sync)
# sm.attach_to_streamlit_state()  # optional
#
# # In Upload & Process flow, when you need keys:
# keys = sm.get_decrypted_project_keys(sync)             # {provider_id: plaintext}
# key_for_provider = sm.resolve_api_key_for_provider(sync, provider_id)
#
# # Role-gated pages:
# sm.require_admin()  # raises Forbidden if not admin
#
