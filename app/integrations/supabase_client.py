"""
Supabase metadata client (no user files, no prompts) for the Desktop OCR app.

- Pure HTTP (httpx) so we don't depend on the supabase-py SDK.
- Works with Supabase Auth (email/password) + PostgREST + Edge Functions.
- Stores NO plaintext secrets; reads/writes ONLY configuration metadata.
- Session persistence will be handled by app/auth/session.py (File 4).
- Metadata syncing to local SQLite is done by app/sync/metadata_sync.py (File 3).

Environment variables expected (set via .env or Tauri):
  SUPABASE_URL=https://<project>.supabase.co
  SUPABASE_ANON_KEY=<anon-key>

Edge functions assumed (will be added later):
  - admin_add_user
  - save_provider_secret
  - test_model_connection
  - import_config_xlsx

RLS in DB ensures:
  - Everyone authenticated can read non-sensitive metadata.
  - Editors/Admins can write config.
  - Admins can manage users and project API keys.
  - Users only see projects they are assigned to for keys & analytics.

Usage (example):
    from app.integrations.supabase_client import SupabaseMetaClient

    supa = SupabaseMetaClient()
    supa.sign_in_email_password(email, password)
    user = supa.get_current_user()
    profile = supa.get_user_profile(user["id"])
    meta = supa.fetch_all_metadata_for_user(user["id"])
    supa.insert_usage_analytics(user_id=user["id"], project_id=..., template_id=..., model_id=..., meta={...})
    supa.close()
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import httpx


# ----------------------------- Data Models ----------------------------- #

@dataclass
class SessionTokens:
    access_token: str
    refresh_token: str
    expires_at: float  # epoch seconds


@dataclass
class UserProfile:
    user_id: str
    email: str
    display_name: Optional[str]
    role: str  # 'admin' | 'editor' | 'user'


@dataclass
class Provider:
    id: str
    name: str
    provider_key: str
    base_url: str
    status: str
    headers: Dict[str, Any]
    timeouts: Dict[str, Any]
    retry: Dict[str, Any]
    show_in_ui: Optional[bool] = True


@dataclass
class Model:
    id: str
    provider_id: str
    model_key: str
    display_name: str
    route: str  # 'chat_completions' | 'responses' | 'completions' | 'embeddings' | 'images'
    context_window: Optional[int]
    max_output_tokens: Optional[int]
    max_temperature: Optional[float]
    default_temperature: Optional[float]
    default_top_p: Optional[float]
    force_json_mode: bool
    prefer_tools: bool
    capabilities: Dict[str, Any]
    compatibility: Dict[str, Any]
    pricing: Dict[str, Any]
    reasoning: Dict[str, Any]
    show_in_ui: bool
    allow_frontend_override_temperature: bool
    allow_frontend_override_reasoning: bool
    status: str


@dataclass
class Template:
    id: str
    name: str
    purpose: str
    description: Optional[str]
    current_version_id: Optional[str]


@dataclass
class TemplateVersion:
    id: str
    template_id: str
    version: int
    system_prompt: str
    user_prompt: str
    schema_json: Dict[str, Any]
    variables: Dict[str, Any]
    is_active: bool


@dataclass
class Project:
    id: str
    name: str
    description: Optional[str]
    is_archived: bool


@dataclass
class ProjectTemplateLink:
    project_id: str
    template_id: str


@dataclass
class ProjectApiKey:
    project_id: str
    provider_id: str
    api_key_ciphertext: str
    key_storage: str  # 'session' | 'encrypted'


# ----------------------------- Exceptions ----------------------------- #

class SupabaseAPIError(RuntimeError):
    pass


# ----------------------------- Client ----------------------------- #

class SupabaseMetaClient:
    """
    Thin Supabase client for metadata + auth + edge functions.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_anon_key: Optional[str] = None,
        timeout_s: int = 30,
    ) -> None:
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.anon_key = supabase_anon_key or os.getenv("SUPABASE_ANON_KEY") or ""
        if not self.supabase_url or not self.anon_key:
            raise SupabaseAPIError("SUPABASE_URL and SUPABASE_ANON_KEY must be set.")

        self.auth_url = f"{self.supabase_url}/auth/v1"
        self.rest_url = f"{self.supabase_url}/rest/v1"
        self.func_url = f"{self.supabase_url}/functions/v1"

        self._http = httpx.Client(timeout=timeout_s)
        self._session: Optional[SessionTokens] = None

    # ------------------------- Session / Auth ------------------------- #

    def sign_in_email_password(self, email: str, password: str) -> SessionTokens:
        """
        Password sign-in. Stores tokens in memory.
        """
        url = f"{self.auth_url}/token?grant_type=password"
        resp = self._http.post(
            url,
            headers=self._base_headers(),
            json={"email": email, "password": password},
        )
        data = self._ok_json(resp)
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in")  # seconds
        if not (access_token and refresh_token and expires_in):
            raise SupabaseAPIError("Invalid auth response.")
        self._session = SessionTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + float(expires_in) - 30.0,  # 30s cushion
        )
        return self._session

    def set_session(self, access_token: str, refresh_token: str, expires_at: float) -> None:
        """
        Inject session tokens (when loaded by session.py).
        """
        self._session = SessionTokens(access_token, refresh_token, float(expires_at))

    def ensure_fresh_session(self) -> None:
        """
        Refresh if expired.
        """
        if not self._session:
            raise SupabaseAPIError("No session set.")
        if time.time() < self._session.expires_at:
            return
        # refresh
        url = f"{self.auth_url}/token?grant_type=refresh_token"
        resp = self._http.post(
            url,
            headers=self._base_headers(),
            json={"refresh_token": self._session.refresh_token},
        )
        data = self._ok_json(resp)
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token") or self._session.refresh_token
        expires_in = data.get("expires_in")
        if not (access_token and expires_in):
            raise SupabaseAPIError("Failed to refresh session.")
        self._session = SessionTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + float(expires_in) - 30.0,
        )

    def get_current_user(self) -> Dict[str, Any]:
        """
        Returns the GoTrue user object.
        """
        self.ensure_fresh_session()
        resp = self._http.get(
            f"{self.auth_url}/user",
            headers=self._auth_headers(),
        )
        return self._ok_json(resp)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Returns our app profile row for a user (role etc.).
        """
        rows = self._select("user_profiles", params={"user_id": f"eq.{user_id}"})
        if not rows:
            return None
        r = rows[0]
        return UserProfile(
            user_id=r["user_id"],
            email=r["email"],
            display_name=r.get("display_name"),
            role=r["role"],
        )

    # --------------------------- Fetchers --------------------------- #

    def fetch_all_metadata_for_user(self, user_id: str) -> Dict[str, Any]:
        """
        Pull all metadata this user can see (projects, templates, providers, models, keys).
        Returns a dict of lists of dataclasses.
        """
        self.ensure_fresh_session()

        # 1) projects for user
        user_projects = self._select(
            "user_project", params={"user_id": f"eq.{user_id}"}
        )
        project_ids = [p["project_id"] for p in user_projects]

        projects: List[Project] = []
        if project_ids:
            projects_raw = self._select_in("projects", "id", project_ids)
            projects = [
                Project(
                    id=r["id"],
                    name=r["name"],
                    description=r.get("description"),
                    is_archived=bool(r.get("is_archived", False)),
                )
                for r in projects_raw
            ]

        # 2) project_template links and templates
        ptmpl: List[ProjectTemplateLink] = []
        templates: Dict[str, Template] = {}
        template_versions: Dict[str, TemplateVersion] = {}

        if project_ids:
            links = self._select_in("project_template", "project_id", project_ids)
            ptmpl = [
                ProjectTemplateLink(project_id=r["project_id"], template_id=r["template_id"])
                for r in links
            ]
            template_ids = sorted({r["template_id"] for r in links})
            if template_ids:
                templ_rows = self._select_in("prompt_templates", "id", template_ids)
                for r in templ_rows:
                    t = Template(
                        id=r["id"],
                        name=r["name"],
                        purpose=r.get("purpose", "chat"),
                        description=r.get("description"),
                        current_version_id=r.get("current_version_id"),
                    )
                    templates[t.id] = t

                # Pull current versions (only those referenced)
                current_ids = [t.current_version_id for t in templates.values() if t.current_version_id]
                if current_ids:
                    vrows = self._select_in("prompt_template_versions", "id", current_ids)
                    for v in vrows:
                        tv = TemplateVersion(
                            id=v["id"],
                            template_id=v["template_id"],
                            version=v["version"],
                            system_prompt=v.get("system_prompt") or "",
                            user_prompt=v.get("user_prompt") or "",
                            schema_json=v.get("schema_json") or {},
                            variables=v.get("variables") or {},
                            is_active=bool(v.get("is_active", True)),
                        )
                        template_versions[tv.id] = tv

        # 3) providers & models (non-secret)
        prov_rows = self._select("model_providers")
        providers = [
            Provider(
                id=r["id"],
                name=r["name"],
                provider_key=r["provider_key"],
                base_url=r["base_url"],
                status=r["status"],
                headers=r.get("headers") or {},
                timeouts=r.get("timeouts") or {},
                retry=r.get("retry") or {},
                show_in_ui=bool(r.get("show_in_ui", True)),
            )
            for r in prov_rows
        ]

        model_rows = self._select("models")
        models = [
            Model(
                id=r["id"],
                provider_id=r["provider_id"],
                model_key=r["model_key"],
                display_name=r["display_name"],
                route=r["route"],
                context_window=r.get("context_window"),
                max_output_tokens=r.get("max_output_tokens"),
                max_temperature=float(r.get("max_temperature", 1.0)) if r.get("max_temperature") is not None else None,
                default_temperature=float(r.get("default_temperature", 0.7)) if r.get("default_temperature") is not None else None,
                default_top_p=float(r.get("default_top_p")) if r.get("default_top_p") is not None else None,
                force_json_mode=bool(r.get("force_json_mode", False)),
                prefer_tools=bool(r.get("prefer_tools", False)),
                capabilities=r.get("capabilities") or {},
                compatibility=r.get("compatibility") or {},
                pricing=r.get("pricing") or {},
                reasoning=r.get("reasoning") or {},
                show_in_ui=bool(r.get("show_in_ui", True)),
                allow_frontend_override_temperature=bool(r.get("allow_frontend_override_temperature", True)),
                allow_frontend_override_reasoning=bool(r.get("allow_frontend_override_reasoning", True)),
                status=r["status"],
            )
            for r in model_rows
        ]

        # 4) project API keys for these projects (ciphertexts only)
        proj_keys: List[ProjectApiKey] = []
        if project_ids:
            krows = self._select_in("project_api_keys", "project_id", project_ids)
            proj_keys = [
                ProjectApiKey(
                    project_id=r["project_id"],
                    provider_id=r["provider_id"],
                    api_key_ciphertext=r["api_key_ciphertext"],
                    key_storage=r.get("key_storage", "encrypted"),
                )
                for r in krows
            ]

        return {
            "projects": projects,
            "project_template": ptmpl,
            "templates": list(templates.values()),
            "template_versions": list(template_versions.values()),
            "providers": providers,
            "models": models,
            "project_api_keys": proj_keys,
        }

    # --------------------------- Admin Ops --------------------------- #

    # Users
    def admin_list_users(self) -> List[UserProfile]:
        rows = self._select("user_profiles")
        return [
            UserProfile(
                user_id=r["user_id"], email=r["email"],
                display_name=r.get("display_name"), role=r["role"]
            )
            for r in rows
        ]

    def admin_invite_user(self, email: str, role: str = "user", display_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Calls Edge Function 'admin_add_user' (server-side uses service role).
        """
        payload = {"email": email, "role": role, "displayName": display_name}
        return self._invoke_edge("admin_add_user", payload)

    # Providers
    def upsert_provider(
        self,
        name: str,
        provider_key: str,
        base_url: str,
        status: str = "active",
        headers: Optional[Dict[str, Any]] = None,
        timeouts: Optional[Dict[str, Any]] = None,
        retry: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rec = {
            "id": id,
            "name": name,
            "provider_key": provider_key,
            "base_url": base_url,
            "status": status,
            "headers": headers or {},
            "timeouts": timeouts or {},
            "retry": retry or {},
        }
        rows = self._upsert("model_providers", [rec], on_conflict="provider_key")
        return rows[0]

    def admin_save_provider_secret(self, provider_id: str, api_key: str) -> Dict[str, Any]:
        """
        Calls Edge Function 'save_provider_secret' to encrypt & store centrally (optional feature).
        """
        return self._invoke_edge("save_provider_secret", {"providerId": provider_id, "apiKey": api_key})

    # Models
    def upsert_model(
        self,
        provider_id: str,
        model_key: str,
        display_name: str,
        route: str,
        *,
        context_window: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        max_temperature: Optional[float] = None,
        default_temperature: Optional[float] = None,
        default_top_p: Optional[float] = None,
        force_json_mode: bool = False,
        prefer_tools: bool = False,
        capabilities: Optional[Dict[str, Any]] = None,
        compatibility: Optional[Dict[str, Any]] = None,
        pricing: Optional[Dict[str, Any]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        show_in_ui: bool = True,
        allow_frontend_override_temperature: bool = True,
        allow_frontend_override_reasoning: bool = True,
        status: str = "active",
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        rec = {
            "id": id,
            "provider_id": provider_id,
            "model_key": model_key,
            "display_name": display_name,
            "route": route,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "max_temperature": max_temperature,
            "default_temperature": default_temperature,
            "default_top_p": default_top_p,
            "force_json_mode": force_json_mode,
            "prefer_tools": prefer_tools,
            "capabilities": capabilities or {},
            "compatibility": compatibility or {},
            "pricing": pricing or {},
            "reasoning": reasoning or {},
            "show_in_ui": show_in_ui,
            "allow_frontend_override_temperature": allow_frontend_override_temperature,
            "allow_frontend_override_reasoning": allow_frontend_override_reasoning,
            "status": status,
        }
        rows = self._upsert("models", [rec], on_conflict="provider_id,model_key")
        return rows[0]

    # Templates
    def upsert_template(self, name: str, purpose: str = "chat", description: Optional[str] = None, id: Optional[str] = None) -> Dict[str, Any]:
        rec = {"id": id, "name": name, "purpose": purpose, "description": description}
        rows = self._upsert("prompt_templates", [rec], on_conflict="name")
        return rows[0]

    def add_template_version(
        self,
        template_id: str,
        version: int,
        system_prompt: str,
        user_prompt: str,
        schema_json: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> Dict[str, Any]:
        rec = {
            "template_id": template_id,
            "version": version,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema_json": schema_json or {},
            "variables": variables or {},
            "is_active": is_active,
        }
        rows = self._insert("prompt_template_versions", [rec])
        return rows[0]

    # Projects
    def upsert_project(self, name: str, description: Optional[str] = None, is_archived: bool = False, id: Optional[str] = None) -> Dict[str, Any]:
        rec = {"id": id, "name": name, "description": description, "is_archived": is_archived}
        rows = self._upsert("projects", [rec], on_conflict="name")
        return rows[0]

    def assign_user_to_project(self, user_id: str, project_id: str) -> Dict[str, Any]:
        rows = self._upsert("user_project", [{"user_id": user_id, "project_id": project_id}], on_conflict="user_id,project_id")
        return rows[0] if rows else {"user_id": user_id, "project_id": project_id}

    def link_template_to_project(self, project_id: str, template_id: str) -> Dict[str, Any]:
        rows = self._upsert("project_template", [{"project_id": project_id, "template_id": template_id}], on_conflict="project_id,template_id")
        return rows[0] if rows else {"project_id": project_id, "template_id": template_id}

    def upsert_project_api_key(self, project_id: str, provider_id: str, api_key_ciphertext: str, key_storage: str = "encrypted") -> Dict[str, Any]:
        rec = {
            "project_id": project_id,
            "provider_id": provider_id,
            "api_key_ciphertext": api_key_ciphertext,
            "key_storage": key_storage,
        }
        rows = self._upsert("project_api_keys", [rec], on_conflict="project_id,provider_id")
        return rows[0]

    # Analytics (tiny, no content)
    def insert_usage_analytics(
        self,
        *,
        user_id: str,
        project_id: str,
        template_id: Optional[str],
        model_id: Optional[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rec = {
            "user_id": user_id,
            "project_id": project_id,
            "template_id": template_id,
            "model_id": model_id,
            "meta": meta or {},
        }
        rows = self._insert("usage_analytics", [rec])
        return rows[0]

    # ------------------------- Low-level REST ------------------------- #

    def _base_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.anon_key,
            "Content-Type": "application/json",
        }

    def _auth_headers(self) -> Dict[str, str]:
        if not self._session:
            raise SupabaseAPIError("No session set.")
        return {
            **self._base_headers(),
            "Authorization": f"Bearer {self._session.access_token}",
        }

    def _ok_json(self, resp: httpx.Response) -> Any:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = e.response.text
            raise SupabaseAPIError(f"HTTP {e.response.status_code}: {detail}") from e
        try:
            return resp.json()
        except ValueError:
            return {}

    def _select(self, table: str, params: Optional[Dict[str, str]] = None, select: str = "*") -> List[Dict[str, Any]]:
        self.ensure_fresh_session()
        query = {"select": select}
        if params:
            query.update(params)
        url = f"{self.rest_url}/{table}"
        resp = self._http.get(url, headers=self._auth_headers(), params=query)
        return self._ok_json(resp)

    def _select_in(self, table: str, col: str, values: Iterable[str], select: str = "*") -> List[Dict[str, Any]]:
        vals = ",".join(values)
        params = {"select": select, col: f"in.({vals})"} if vals else {"select": select, col: "in.()"}
        url = f"{self.rest_url}/{table}"
        resp = self._http.get(url, headers=self._auth_headers(), params=params)
        return self._ok_json(resp)

    def _insert(self, table: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        self.ensure_fresh_session()
        url = f"{self.rest_url}/{table}"
        headers = {**self._auth_headers(), "Prefer": "return=representation"}
        resp = self._http.post(url, headers=headers, content=json.dumps(records))
        return self._ok_json(resp)

    def _upsert(self, table: str, records: List[Dict[str, Any]], on_conflict: Optional[str] = None) -> List[Dict[str, Any]]:
        if not records:
            return []
        self.ensure_fresh_session()
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        url = f"{self.rest_url}/{table}"
        headers = {**self._auth_headers(), "Prefer": "resolution=merge-duplicates,return=representation"}
        resp = self._http.post(url, headers=headers, params=params, content=json.dumps(records))
        return self._ok_json(resp)

    def _patch(self, table: str, match: Dict[str, str], update: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.ensure_fresh_session()
        url = f"{self.rest_url}/{table}"
        headers = {**self._auth_headers(), "Prefer": "return=representation"}
        resp = self._http.patch(url, headers=headers, params=match, content=json.dumps(update))
        return self._ok_json(resp)

    def _delete(self, table: str, match: Dict[str, str]) -> Dict[str, Any]:
        self.ensure_fresh_session()
        url = f"{self.rest_url}/{table}"
        headers = self._auth_headers()
        resp = self._http.delete(url, headers=headers, params=match)
        return self._ok_json(resp)

    # ----------------------- Edge Functions ----------------------- #

    def _invoke_edge(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke Supabase Edge Function with current JWT.
        """
        self.ensure_fresh_session()
        url = f"{self.func_url}/{function_name}"
        headers = self._auth_headers()
        resp = self._http.post(url, headers=headers, content=json.dumps(payload))
        return self._ok_json(resp)

    # ----------------------------- Misc ----------------------------- #

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass
