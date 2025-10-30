"""
Admin Console (Streamlit) — replaces Excel config with in-app settings.

Tabs:
  • Users — invite users, change roles, assign projects, create projects
  • Model Configuration — Providers, Models, Project API Keys, Test Connection
  • Templates — create templates, add versions, assign to projects (versioned)
  • Projects — list/archive projects
  • Analytics — lightweight usage view

All writes go to Supabase metadata tables (no user files or prompts).
After each write, we resync local cache so standard user pages reflect changes.

Requires:
  - File 2: app/integrations/supabase_client.py
  - File 3: app/sync/metadata_sync.py
  - File 4: app/auth/session.py
  - File 5: app/pages/0_Login.py
  - File 6: auth bootstrap in app/main.py

Env:
  SUPABASE_URL, SUPABASE_ANON_KEY
  APP_DB_PATH=./data/app.db
  APP_KMS_KEY (Fernet) or APP_KMS_KEY_FILE=./data/kms.key
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import streamlit as st

from app.integrations.supabase_client import SupabaseMetaClient, SupabaseAPIError
from app.sync.metadata_sync import MetadataSync
from app.auth.session import SessionManager, Forbidden

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def _kms_key_bytes() -> bytes:
    """Load Fernet key from env or file."""
    key_env = os.getenv("APP_KMS_KEY")
    if key_env:
        try:
            # Accept raw fernet key string (urlsafe base64) as-is
            return key_env.encode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Invalid APP_KMS_KEY: {e}")

    key_file = os.getenv("APP_KMS_KEY_FILE", "data/kms.key")
    if os.path.exists(key_file):
        with open(key_file, "rb") as f:
            key = f.read().strip()
        if not key:
            raise RuntimeError("APP_KMS_KEY_FILE is empty.")
        return key

    raise RuntimeError(
        "No KMS key found. Set APP_KMS_KEY or provide APP_KMS_KEY_FILE (e.g., data/kms.key)."
    )


def _encrypt_fernet(plaintext: str) -> str:
    if Fernet is None:
        raise RuntimeError("cryptography not installed. Add 'cryptography' to requirements.txt")
    key = _kms_key_bytes()
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def _json_loads_safe(s: str, fallback: Any) -> Any:
    try:
        return json.loads(s) if s else fallback
    except Exception:
        return fallback


def _resync(sm: SessionManager, sync: MetadataSync):
    with st.spinner("Syncing configuration..."):
        sm.sync_metadata_for_current_user(sync)


def _build_clients() -> tuple[SessionManager, MetadataSync, SupabaseMetaClient]:
    try:
        supa = SupabaseMetaClient()
    except SupabaseAPIError as e:
        st.error(f"Supabase not configured: {e}")
        st.stop()

    sm = SessionManager(supabase=supa)
    sess = sm.try_load_session_from_disk()
    if not sess:
        # Opened directly — go to login page
        for cand in ("app/pages/0_Login.py", "pages/0_Login.py", "0_Login.py"):
            try:
                st.switch_page(cand)
                return sm, None, supa  # type: ignore
            except Exception:
                continue
        st.stop()

    # Ensure cache exists
    db_path = os.getenv("APP_DB_PATH", "data/app.db")
    sync = MetadataSync(db_path=db_path)
    sync.ensure_schema()
    return sm, sync, supa


# ──────────────────────────────────────────────────────────────────────
# Page start
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Admin Console", page_icon="🛠", layout="wide")

sm, sync, supa = _build_clients()
try:
    sm.require_admin()
except Forbidden:
    st.error("You don’t have permission to view this page.")
    st.stop()

st.title("Admin Console")
st.caption("Manage users, providers/models, templates, projects, and keys. Only configuration metadata is stored in Supabase; files and prompts stay local.")

# optional quick actions
col_a, col_b = st.columns([1, 1], gap="small")
with col_a:
    if st.button("🔄 Resync now"):
        _resync(sm, sync)
with col_b:
    if st.button("🚪 Logout"):
        sm.logout()
        for cand in ("app/pages/0_Login.py", "pages/0_Login.py", "0_Login.py"):
            try:
                st.switch_page(cand)
            except Exception:
                pass
        st.stop()

tabs = st.tabs(
    ["👥 Users", "🧩 Model Configuration", "📝 Templates", "📁 Projects", "📊 Analytics"]
)

# ──────────────────────────────────────────────────────────────────────
# Tab: Users
# ──────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Users & Access")

    # Invite user
    with st.expander("➕ Invite new user", expanded=True):
        with st.form("invite_user_form", clear_on_submit=False):
            email = st.text_input("Email")
            display_name = st.text_input("Display name", value="")
            role = st.selectbox("Role", ["user", "editor", "admin"], index=0)
            submitted = st.form_submit_button("Send invite")
        if submitted:
            try:
                res = supa.admin_invite_user(email=email, role=role, display_name=display_name or None)
                st.success(f"Invitation sent to {email}.")
            except Exception as e:
                st.error(f"Invite failed: {e}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # List users / change role
    st.markdown("### All users")
    try:
        users = supa.admin_list_users()
    except Exception as e:
        st.error(f"Could not load users: {e}")
        users = []

    if users:
        for u in users:
            c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
            c1.markdown(f"**{u.display_name or u.email}**  \n`{u.email}`  ")
            new_role = c2.selectbox(
                "Role",
                ["user", "editor", "admin"],
                index=["user", "editor", "admin"].index(u.role),
                key=f"role_{u.user_id}",
            )
            if c3.button("Update role", key=f"role_btn_{u.user_id}"):
                try:
                    # Patch user_profiles.role
                    supa._patch("user_profiles", {"user_id": f"eq.{u.user_id}"}, {"role": new_role})
                    st.success(f"Updated role for {u.email} → {new_role}")
                except Exception as e:
                    st.error(f"Failed to update role: {e}")
            c4.write("")

    # Assign projects to a user
    st.markdown("### Assign projects")
    try:
        all_projects = supa._select("projects")
    except Exception as e:
        st.error(f"Could not load projects: {e}")
        all_projects = []

    if users and all_projects:
        u_map = {f"{u.display_name or u.email} ({u.email})": u.user_id for u in users}
        p_map = {p["name"]: p["id"] for p in all_projects}
        colx, coly = st.columns([2, 3])
        with colx:
            u_choice = st.selectbox("User", list(u_map.keys()))
        with coly:
            p_choice = st.multiselect("Projects", list(p_map.keys()))
        if st.button("Assign selected"):
            try:
                for pname in p_choice:
                    supa.assign_user_to_project(u_map[u_choice], p_map[pname])
                st.success("Assignments saved.")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Assignment failed: {e}")

    # Create a project
    with st.expander("📁 Create new project"):
        with st.form("new_project_form", clear_on_submit=True):
            n = st.text_input("Project name")
            d = st.text_area("Description", value="")
            ok = st.form_submit_button("Create project")
        if ok:
            try:
                rec = supa.upsert_project(name=n, description=d or None, is_archived=False)
                st.success(f"Project created: {rec.get('name')}")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Create failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Tab: Model Configuration
# ──────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Model Configuration")

    sub1, sub2, sub3 = st.tabs(["Providers", "Models", "Project API Keys & Test"])

    # Providers
    with sub1:
        st.markdown("#### Providers")
        try:
            prov_rows = supa._select("model_providers")
        except Exception as e:
            st.error(f"Failed to load providers: {e}")
            prov_rows = []

        existing = ["<New provider>"] + [f"{p['name']} ({p['provider_key']})" for p in prov_rows]
        choice = st.selectbox("Edit existing or create new", existing)

        editing = None
        if choice != "<New provider>":
            idx = existing.index(choice) - 1
            editing = prov_rows[idx]

        with st.form("provider_form", clear_on_submit=False):
            name = st.text_input("Name", value=(editing["name"] if editing else ""))
            provider_key = st.text_input("Provider key (stable id)", value=(editing["provider_key"] if editing else ""))
            base_url = st.text_input("Base URL", value=(editing["base_url"] if editing else ""))
            status = st.selectbox("Status", ["active", "inactive"], index=0 if (not editing or editing["status"]=="active") else 1)
            headers = st.text_area("Static headers (JSON)", value=json.dumps(editing.get("headers", {}), indent=2) if editing else "{}")
            timeouts = st.text_area("Timeouts (JSON)", value=json.dumps(editing.get("timeouts", {}), indent=2) if editing else "{}")
            retry = st.text_area("Retry policy (JSON)", value=json.dumps(editing.get("retry", {}), indent=2) if editing else "{}")
            submitted = st.form_submit_button("Save provider")

        if submitted:
            try:
                rec = supa.upsert_provider(
                    name=name.strip(),
                    provider_key=provider_key.strip(),
                    base_url=base_url.strip(),
                    status=status,
                    headers=_json_loads_safe(headers, {}),
                    timeouts=_json_loads_safe(timeouts, {}),
                    retry=_json_loads_safe(retry, {}),
                    id=(editing["id"] if editing else None),
                )
                st.success(f"Saved provider: {rec.get('name')}")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Save failed: {e}")

        with st.expander("🔑 Provider-wide API key (optional)"):
            if not prov_rows:
                st.info("Create a provider first.")
            else:
                prov_map = {f"{p['name']} ({p['provider_key']})": p["id"] for p in prov_rows}
                pid = st.selectbox("Provider", list(prov_map.keys()), key="prov_secret_sel")
                api_key = st.text_input("API key", type="password", key="prov_secret_key")
                if st.button("Save encrypted key"):
                    try:
                        # Stored in private.provider_secrets via Edge Function
                        res = supa.admin_save_provider_secret(prov_map[pid], api_key)
                        st.success("Provider key saved.")
                    except Exception as e:
                        st.error(f"Saving key failed: {e}")

    # Models
    with sub2:
        st.markdown("#### Models")

        try:
            prov_rows = supa._select("model_providers")
            model_rows = supa._select("models")
        except Exception as e:
            st.error(f"Failed to load providers/models: {e}")
            prov_rows, model_rows = [], []

        prov_map = {f"{p['name']} ({p['provider_key']})": p["id"] for p in prov_rows}
        prov_inv = {v: k for k, v in prov_map.items()}

        existing = ["<New model>"] + [f"{m['display_name']} — {m['model_key']} ({prov_inv.get(m['provider_id'],'?')})" for m in model_rows]
        choice = st.selectbox("Edit existing or create new", existing, key="model_edit_choice")

        editing = None
        if choice != "<New model>":
            editing = model_rows[existing.index(choice) - 1]

        with st.form("model_form", clear_on_submit=False):
            provider_label = st.selectbox("Provider", list(prov_map.keys()), index=0 if not editing else list(prov_map.values()).index(editing["provider_id"]))
            provider_id = prov_map[provider_label]
            model_key = st.text_input("Model ID / key", value=(editing["model_key"] if editing else ""))
            display_name = st.text_input("Display name", value=(editing["display_name"] if editing else ""))
            route = st.selectbox("API route", ["responses", "chat_completions", "completions", "embeddings", "images"],
                                 index=(["responses","chat_completions","completions","embeddings","images"].index(editing["route"]) if editing else 0))
            c1, c2, c3 = st.columns(3)
            with c1:
                context_window = st.number_input("Context window", min_value=0, value=int(editing["context_window"]) if editing and editing.get("context_window") else 0)
            with c2:
                max_output_tokens = st.number_input("Max output tokens", min_value=0, value=int(editing["max_output_tokens"]) if editing and editing.get("max_output_tokens") else 0)
            with c3:
                default_temperature = st.number_input("Default temperature", min_value=0.0, max_value=2.0, step=0.1,
                                                      value=float(editing["default_temperature"]) if editing and editing.get("default_temperature") is not None else 0.7)
            default_top_p = st.number_input("Default top_p", min_value=0.0, max_value=1.0, step=0.05,
                                            value=float(editing["default_top_p"]) if editing and editing.get("default_top_p") is not None else 1.0)
            c4, c5 = st.columns(2)
            with c4:
                force_json_mode = st.checkbox("Force JSON mode", value=bool(editing["force_json_mode"]) if editing else False)
            with c5:
                prefer_tools = st.checkbox("Prefer Tools", value=bool(editing["prefer_tools"]) if editing else False)

            capabilities = st.text_area("Capabilities (JSON)", value=json.dumps(editing.get("capabilities", {}), indent=2) if editing else "{}")
            compatibility = st.text_area("Compatibility (JSON)", value=json.dumps(editing.get("compatibility", {}), indent=2) if editing else "{}")
            pricing = st.text_area("Pricing (JSON)", value=json.dumps(editing.get("pricing", {}), indent=2) if editing else "{}")
            reasoning = st.text_area("Reasoning (JSON)", value=json.dumps(editing.get("reasoning", {}), indent=2) if editing else "{}")

            show_in_ui = st.checkbox("Show in UI", value=bool(editing["show_in_ui"]) if editing else True)
            allow_temp = st.checkbox("Allow user to override temperature", value=bool(editing["allow_frontend_override_temperature"]) if editing else True)
            allow_reason = st.checkbox("Allow user to override reasoning", value=bool(editing["allow_frontend_override_reasoning"]) if editing else True)
            status = st.selectbox("Status", ["active", "inactive"], index=0 if (not editing or editing["status"]=="active") else 1)

            submitted = st.form_submit_button("Save model")

        if submitted:
            try:
                rec = supa.upsert_model(
                    id=(editing["id"] if editing else None),
                    provider_id=provider_id,
                    model_key=model_key.strip(),
                    display_name=display_name.strip(),
                    route=route,
                    context_window=(int(context_window) or None),
                    max_output_tokens=(int(max_output_tokens) or None),
                    max_temperature=None,  # you can expose if needed
                    default_temperature=float(default_temperature),
                    default_top_p=float(default_top_p),
                    force_json_mode=bool(force_json_mode),
                    prefer_tools=bool(prefer_tools),
                    capabilities=_json_loads_safe(capabilities, {}),
                    compatibility=_json_loads_safe(compatibility, {}),
                    pricing=_json_loads_safe(pricing, {}),
                    reasoning=_json_loads_safe(reasoning, {}),
                    show_in_ui=bool(show_in_ui),
                    allow_frontend_override_temperature=bool(allow_temp),
                    allow_frontend_override_reasoning=bool(allow_reason),
                    status=status,
                )
                st.success(f"Saved model: {rec.get('display_name')} ({rec.get('model_key')})")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Save failed: {e}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

    # Project API keys & Test
    with sub3:
        st.markdown("#### Project-scoped API keys (encrypted)")

        try:
            all_projects = supa._select("projects")
            prov_rows = supa._select("model_providers")
        except Exception as e:
            st.error(f"Failed to load projects/providers: {e}")
            all_projects, prov_rows = [], []

        if not all_projects or not prov_rows:
            st.info("Create a project and a provider first.")
        else:
            proj_map = {p["name"]: p["id"] for p in all_projects}
            prov_map = {f"{p['name']} ({p['provider_key']})": p["id"] for p in prov_rows}

            with st.form("project_key_form", clear_on_submit=True):
                proj = st.selectbox("Project", list(proj_map.keys()))
                prov = st.selectbox("Provider", list(prov_map.keys()))
                api_key = st.text_input("API key", type="password")
                st.caption("Key will be encrypted on this device with your Fernet KMS key and stored as ciphertext.")
                ok = st.form_submit_button("Save encrypted key")
            if ok:
                try:
                    ciphertext = _encrypt_fernet(api_key)
                    supa.upsert_project_api_key(
                        project_id=proj_map[proj],
                        provider_id=prov_map[prov],
                        api_key_ciphertext=ciphertext,
                        key_storage="encrypted",
                    )
                    st.success("Project API key saved.")
                    _resync(sm, sync)
                except Exception as e:
                    st.error(f"Failed to save key: {e}")

        st.markdown("#### Test connection (optional)")
        st.caption("This calls your Supabase Edge Function `test_model_connection` (implement server-side).")
        with st.form("test_conn_form", clear_on_submit=True):
            provider_id = st.text_input("Provider ID (UUID)")
            model_key = st.text_input("Model key (e.g., gpt-4o, claude-3-5-sonnet)")
            test_type = st.selectbox("Test type", ["ping", "sample_prompt"])
            sample_prompt = st.text_area("Sample prompt (optional)")
            go = st.form_submit_button("Run test")
        if go:
            try:
                payload = {"providerId": provider_id, "modelKey": model_key, "testType": test_type, "samplePrompt": sample_prompt or None}
                res = supa._invoke_edge("test_model_connection", payload)
                st.success("Test executed.")
                st.json(res)
            except Exception as e:
                st.error(f"Test failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Tab: Templates (versioned)
# ──────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Templates")

    # Create template
    with st.form("create_template_form", clear_on_submit=True):
        t_name = st.text_input("Template name")
        t_purpose = st.selectbox("Purpose", ["chat", "system", "tool"], index=0)
        t_desc = st.text_area("Description", value="")
        btn_ct = st.form_submit_button("Create / upsert template")
    if btn_ct:
        try:
            rec = supa.upsert_template(name=t_name.strip(), purpose=t_purpose, description=t_desc or None)
            st.success(f"Template saved: {rec.get('name')} (id: {rec.get('id')})")
            _resync(sm, sync)
        except Exception as e:
            st.error(f"Template save failed: {e}")

    # Add a version
    try:
        # pull latest after potential create
        trows = supa._select("prompt_templates")
    except Exception as e:
        st.error(f"Failed to load templates: {e}")
        trows = []

    if trows:
        t_map = {f"{t['name']}": t["id"] for t in trows}
        with st.form("add_version_form", clear_on_submit=True):
            which = st.selectbox("Template", list(t_map.keys()))
            # fetch current versions to compute next
            vers = supa._select("prompt_template_versions", params={"template_id": f"eq.{t_map[which]}"})
            next_v = 1 + max([int(v["version"]) for v in vers], default=0)
            st.caption(f"Next version: {next_v}")
            system_prompt = st.text_area("System prompt", height=120)
            user_prompt = st.text_area("User prompt", height=160)
            schema_json = st.text_area("JSON schema (optional)", value="{}", height=120)
            variables = st.text_area("Variables doc (JSON, optional)", value="{}", height=80)
            active = st.checkbox("Set as active", value=True)
            btn_ver = st.form_submit_button("Add version")
        if btn_ver:
            try:
                rec = supa.add_template_version(
                    template_id=t_map[which],
                    version=next_v,
                    system_prompt=system_prompt or "",
                    user_prompt=user_prompt or "",
                    schema_json=_json_loads_safe(schema_json, {}),
                    variables=_json_loads_safe(variables, {}),
                    is_active=bool(active),
                )
                st.success(f"Version {next_v} added.")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Add version failed: {e}")

    # Assign templates to project
    st.markdown("### Assign to project")
    try:
        all_projects = supa._select("projects")
        trows = supa._select("prompt_templates")
    except Exception as e:
        st.error(f"Load failed: {e}")
        all_projects, trows = [], []
    if all_projects and trows:
        p_map = {p["name"]: p["id"] for p in all_projects}
        t_map = {t["name"]: t["id"] for t in trows}
        col1, col2 = st.columns([1, 2])
        with col1:
            p_choice = st.selectbox("Project", list(p_map.keys()), key="assign_proj")
        with col2:
            t_choice = st.multiselect("Templates", list(t_map.keys()), key="assign_templates")
        if st.button("Link templates to project"):
            try:
                for tname in t_choice:
                    supa.link_template_to_project(p_map[p_choice], t_map[tname])
                st.success("Linked.")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Link failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Tab: Projects
# ──────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Projects")

    try:
        prows = supa._select("projects")
    except Exception as e:
        st.error(f"Failed to load projects: {e}")
        prows = []

    if prows:
        for p in prows:
            c1, c2, c3 = st.columns([4, 2, 1])
            c1.markdown(f"**{p['name']}**  \n{p.get('description') or ''}")
            new_arch = c2.checkbox("Archived", value=bool(p.get("is_archived", False)), key=f"arch_{p['id']}")
            if c3.button("Save", key=f"save_arch_{p['id']}"):
                try:
                    supa._patch("projects", {"id": f"eq.{p['id']}"}, {"is_archived": bool(new_arch)})
                    st.success("Saved.")
                    _resync(sm, sync)
                except Exception as e:
                    st.error(f"Update failed: {e}")

    with st.expander("Create project"):
        with st.form("proj_create_form", clear_on_submit=True):
            n = st.text_input("Name")
            d = st.text_area("Description", value="")
            ok = st.form_submit_button("Create")
        if ok:
            try:
                rec = supa.upsert_project(name=n, description=d or None, is_archived=False)
                st.success(f"Created: {rec.get('name')}")
                _resync(sm, sync)
            except Exception as e:
                st.error(f"Create failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Tab: Analytics
# ──────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Analytics (lightweight)")

    try:
        # limited to projects user can see per RLS
        rows = supa._select("usage_analytics")
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        rows = []

    if not rows:
        st.info("No analytics yet.")
    else:
        st.caption("Sample of recent usage rows (no prompts/content stored).")
        st.dataframe(rows, use_container_width=True)
