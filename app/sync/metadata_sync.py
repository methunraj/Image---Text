"""
Metadata sync from Supabase -> local SQLite cache.

- Uses File 2: app.integrations.supabase_client.SupabaseMetaClient
- Creates lightweight *cache tables* (prefix: supa_*) that mirror Supabase metadata
  visible to the current user (projects, templates, providers, models, project keys).
- Does NOT store user files, prompts, or LLM outputs.
- Idempotent: safe to call at every login.

Typical usage:
    from app.integrations.supabase_client import SupabaseMetaClient
    from app.sync.metadata_sync import MetadataSync

    supa = SupabaseMetaClient()
    supa.set_session(access_token, refresh_token, expires_at)  # or sign_in_email_password()

    sync = MetadataSync(db_path=os.getenv("APP_DB_PATH", "data/app.db"))
    sync.ensure_schema()
    summary = sync.sync_for_user(supa, user_id)
    # Then use helpers (list_projects_for_user, list_templates_for_project, etc.)
"""

from __future__ import annotations

import os
import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

# Types only (no runtime dependency on dataclass shapes)
try:
    from app.integrations.supabase_client import (
        SupabaseMetaClient,
    )
except Exception:  # pragma: no cover
    SupabaseMetaClient = object  # type: ignore


def _bool(x: Any) -> int:
    return 1 if bool(x) else 0


def _jdump(obj: Any) -> str:
    return json.dumps(obj or {}, ensure_ascii=False, separators=(",", ":"))


class MetadataSync:
    """
    Creates/maintains local cache tables and syncs metadata down from Supabase.

    Cache tables (scoped to what the current user can see):
      - supa_meta(key TEXT PRIMARY KEY, value TEXT)
      - supa_projects
      - supa_project_template
      - supa_templates
      - supa_template_versions
      - supa_providers
      - supa_models
      - supa_project_api_keys
    """

    def __init__(self, db_path: str = "data/app.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @contextmanager
    def _tx(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    def ensure_schema(self) -> None:
        with self._tx() as cur:
            # metadata
            cur.execute(
                """
                create table if not exists supa_meta (
                    key   text primary key,
                    value text not null
                )
                """
            )

            # projects visible to user
            cur.execute(
                """
                create table if not exists supa_projects (
                    id           text primary key,
                    name         text not null,
                    description  text,
                    is_archived  integer not null default 0
                )
                """
            )

            # project-template links
            cur.execute(
                """
                create table if not exists supa_project_template (
                    project_id  text not null,
                    template_id text not null,
                    primary key (project_id, template_id)
                )
                """
            )

            # templates
            cur.execute(
                """
                create table if not exists supa_templates (
                    id                  text primary key,
                    name                text not null,
                    purpose             text not null,
                    description         text,
                    current_version_id  text
                )
                """
            )

            # template versions (only current ones are required, but we allow more)
            cur.execute(
                """
                create table if not exists supa_template_versions (
                    id             text primary key,
                    template_id    text not null,
                    version        integer not null,
                    system_prompt  text,
                    user_prompt    text,
                    schema_json    text not null default '{}',
                    variables_json text not null default '{}',
                    is_active      integer not null default 1
                )
                """
            )

            # providers
            cur.execute(
                """
                create table if not exists supa_providers (
                    id           text primary key,
                    name         text not null,
                    provider_key text not null,
                    base_url     text not null,
                    status       text not null,
                    headers_json text not null default '{}',
                    timeouts_json text not null default '{}',
                    retry_json   text not null default '{}',
                    show_in_ui   integer not null default 1
                )
                """
            )

            # models
            cur.execute(
                """
                create table if not exists supa_models (
                    id                                   text primary key,
                    provider_id                          text not null,
                    model_key                             text not null,
                    display_name                          text not null,
                    route                                 text not null,
                    context_window                        integer,
                    max_output_tokens                     integer,
                    max_temperature                       real,
                    default_temperature                   real,
                    default_top_p                         real,
                    force_json_mode                       integer not null default 0,
                    prefer_tools                          integer not null default 0,
                    capabilities_json                     text not null default '{}',
                    compatibility_json                    text not null default '{}',
                    pricing_json                          text not null default '{}',
                    reasoning_json                        text not null default '{}',
                    show_in_ui                            integer not null default 1,
                    allow_frontend_override_temperature   integer not null default 1,
                    allow_frontend_override_reasoning     integer not null default 1,
                    status                                text not null
                )
                """
            )

            # project-scoped API keys (ciphertext only)
            cur.execute(
                """
                create table if not exists supa_project_api_keys (
                    project_id         text not null,
                    provider_id        text not null,
                    api_key_ciphertext text not null,
                    key_storage        text not null,
                    primary key (project_id, provider_id)
                )
                """
            )

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #

    def _wipe_cache(self) -> None:
        with self._tx() as cur:
            cur.execute("delete from supa_project_api_keys")
            cur.execute("delete from supa_models")
            cur.execute("delete from supa_providers")
            cur.execute("delete from supa_template_versions")
            cur.execute("delete from supa_templates")
            cur.execute("delete from supa_project_template")
            cur.execute("delete from supa_projects")

    def sync_for_user(
        self,
        supa: SupabaseMetaClient,
        user_id: str,
        *,
        wipe_before: bool = True,
    ) -> Dict[str, int]:
        """
        Pull all metadata visible to this user and store in cache tables.

        Returns a summary dict with row counts.
        """
        meta = supa.fetch_all_metadata_for_user(user_id)

        if wipe_before:
            self._wipe_cache()

        counts = {
            "projects": 0,
            "project_template": 0,
            "templates": 0,
            "template_versions": 0,
            "providers": 0,
            "models": 0,
            "project_api_keys": 0,
        }

        with self._tx() as cur:
            # projects
            for p in meta.get("projects", []):
                cur.execute(
                    """
                    insert or replace into supa_projects (id, name, description, is_archived)
                    values (?, ?, ?, ?)
                    """,
                    (p.id, p.name, p.description, _bool(p.is_archived)),
                )
                counts["projects"] += 1

            # project_template links
            for link in meta.get("project_template", []):
                cur.execute(
                    """
                    insert or replace into supa_project_template (project_id, template_id)
                    values (?, ?)
                    """,
                    (link.project_id, link.template_id),
                )
                counts["project_template"] += 1

            # templates
            templates_by_id: Dict[str, Any] = {}
            for t in meta.get("templates", []):
                cur.execute(
                    """
                    insert or replace into supa_templates (id, name, purpose, description, current_version_id)
                    values (?, ?, ?, ?, ?)
                    """,
                    (t.id, t.name, t.purpose, t.description, t.current_version_id),
                )
                templates_by_id[t.id] = t
                counts["templates"] += 1

            # template current versions
            for tv in meta.get("template_versions", []):
                cur.execute(
                    """
                    insert or replace into supa_template_versions
                    (id, template_id, version, system_prompt, user_prompt, schema_json, variables_json, is_active)
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tv.id,
                        tv.template_id,
                        tv.version,
                        tv.system_prompt,
                        tv.user_prompt,
                        _jdump(tv.schema_json),
                        _jdump(tv.variables),
                        _bool(tv.is_active),
                    ),
                )
                counts["template_versions"] += 1

            # providers
            for pr in meta.get("providers", []):
                cur.execute(
                    """
                    insert or replace into supa_providers
                    (id, name, provider_key, base_url, status, headers_json, timeouts_json, retry_json, show_in_ui)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pr.id,
                        pr.name,
                        pr.provider_key,
                        pr.base_url,
                        pr.status,
                        _jdump(pr.headers),
                        _jdump(pr.timeouts),
                        _jdump(pr.retry),
                        _bool(getattr(pr, "show_in_ui", True)),
                    ),
                )
                counts["providers"] += 1

            # models
            for m in meta.get("models", []):
                cur.execute(
                    """
                    insert or replace into supa_models
                    (id, provider_id, model_key, display_name, route,
                     context_window, max_output_tokens, max_temperature, default_temperature, default_top_p,
                     force_json_mode, prefer_tools, capabilities_json, compatibility_json, pricing_json, reasoning_json,
                     show_in_ui, allow_frontend_override_temperature, allow_frontend_override_reasoning, status)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.id,
                        m.provider_id,
                        m.model_key,
                        m.display_name,
                        m.route,
                        m.context_window,
                        m.max_output_tokens,
                        m.max_temperature,
                        m.default_temperature,
                        m.default_top_p,
                        _bool(m.force_json_mode),
                        _bool(m.prefer_tools),
                        _jdump(m.capabilities),
                        _jdump(m.compatibility),
                        _jdump(m.pricing),
                        _jdump(m.reasoning),
                        _bool(m.show_in_ui),
                        _bool(m.allow_frontend_override_temperature),
                        _bool(m.allow_frontend_override_reasoning),
                        m.status,
                    ),
                )
                counts["models"] += 1

            # project API key ciphertexts (no plaintext)
            for k in meta.get("project_api_keys", []):
                cur.execute(
                    """
                    insert or replace into supa_project_api_keys
                    (project_id, provider_id, api_key_ciphertext, key_storage)
                    values (?, ?, ?, ?)
                    """,
                    (k.project_id, k.provider_id, k.api_key_ciphertext, k.key_storage),
                )
                counts["project_api_keys"] += 1

            # meta
            cur.execute(
                "insert or replace into supa_meta (key, value) values (?, ?)",
                ("last_sync_ts", datetime.utcnow().isoformat() + "Z"),
            )
            cur.execute(
                "insert or replace into supa_meta (key, value) values (?, ?)",
                ("last_sync_user_id", user_id),
            )

        return counts

    # ------------------------------------------------------------------ #
    # Read helpers for the app (query the cache)
    # ------------------------------------------------------------------ #

    def list_projects_for_user(self) -> List[Dict[str, Any]]:
        q = "select id, name, coalesce(description,'') as description, is_archived from supa_projects where is_archived=0 order by name"
        return [dict(row) for row in self._conn.execute(q).fetchall()]

    def list_templates_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        q = """
            select t.id, t.name, t.purpose, t.description,
                   v.system_prompt, v.user_prompt, v.schema_json, v.variables_json
            from supa_project_template pt
            join supa_templates t on t.id = pt.template_id
            left join supa_template_versions v on v.id = t.current_version_id
            where pt.project_id = ?
            order by t.name
        """
        rows = self._conn.execute(q, (project_id,)).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "purpose": r["purpose"],
                    "description": r["description"],
                    "system_prompt": r["system_prompt"] or "",
                    "user_prompt": r["user_prompt"] or "",
                    "schema": json.loads(r["schema_json"] or "{}"),
                    "variables": json.loads(r["variables_json"] or "{}"),
                }
            )
        return out

    def list_providers(self) -> List[Dict[str, Any]]:
        q = """
            select id, name, provider_key, base_url, status,
                   headers_json, timeouts_json, retry_json, show_in_ui
            from supa_providers
            where show_in_ui = 1 and status = 'active'
            order by name
        """
        rows = self._conn.execute(q).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "provider_key": r["provider_key"],
                    "base_url": r["base_url"],
                    "status": r["status"],
                    "headers": json.loads(r["headers_json"] or "{}"),
                    "timeouts": json.loads(r["timeouts_json"] or "{}"),
                    "retry": json.loads(r["retry_json"] or "{}"),
                }
            )
        return out

    def list_models(self, *, provider_id: Optional[str] = None, route: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ["status = 'active'", "show_in_ui = 1"]
        args: List[Any] = []
        if provider_id:
            clauses.append("provider_id = ?")
            args.append(provider_id)
        if route:
            clauses.append("route = ?")
            args.append(route)
        q = f"""
            select *
            from supa_models
            where {' and '.join(clauses)}
            order by display_name
        """
        rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "provider_id": r["provider_id"],
                    "model_key": r["model_key"],
                    "display_name": r["display_name"],
                    "route": r["route"],
                    "context_window": r["context_window"],
                    "max_output_tokens": r["max_output_tokens"],
                    "default_temperature": r["default_temperature"],
                    "default_top_p": r["default_top_p"],
                    "force_json_mode": bool(r["force_json_mode"]),
                    "prefer_tools": bool(r["prefer_tools"]),
                    "capabilities": json.loads(r["capabilities_json"] or "{}"),
                    "compatibility": json.loads(r["compatibility_json"] or "{}"),
                    "pricing": json.loads(r["pricing_json"] or "{}"),
                    "reasoning": json.loads(r["reasoning_json"] or "{}"),
                }
            )
        return out

    def get_project_api_keys(self, project_id: str) -> List[Dict[str, Any]]:
        q = """
            select project_id, provider_id, api_key_ciphertext, key_storage
            from supa_project_api_keys
            where project_id = ?
        """
        rows = self._conn.execute(q, (project_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_last_sync_info(self) -> Dict[str, Optional[str]]:
        rows = dict(self._conn.execute("select key, value from supa_meta").fetchall())
        return {
            "last_sync_ts": rows.get("last_sync_ts"),
            "last_sync_user_id": rows.get("last_sync_user_id"),
        }
