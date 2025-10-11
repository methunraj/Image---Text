from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

from sqlalchemy import JSON, Float, ForeignKey, String, Boolean, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

from app.core.model_registry import ModelDescriptor

DB_PATH = Path("data/app.db")


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    base_url: Mapped[str] = mapped_column(String(255), nullable=False)
    # Logical provider code (e.g., 'openai', 'openrouter') for API key lookup
    provider_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Security
    key_storage: Mapped[str] = mapped_column(String(16), default="session")  # 'session' | 'encrypted'
    api_key_enc: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # Provider/model
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    headers_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timeout_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    logo_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Catalog + detection
    catalog_caps_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    detected_caps_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Defaults
    default_temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    default_top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    default_max_output_tokens: Mapped[int | None] = mapped_column(nullable=True)
    default_force_json_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    default_prefer_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    # Cached successful parameter names
    cached_max_tokens_param: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Active flag
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="provider", cascade="all, delete-orphan")  # type: ignore[name-defined]


class ProviderKey(Base):
    __tablename__ = "provider_keys"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # e.g., 'openai', 'openrouter', 'groq'
    provider_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    key_storage: Mapped[str] = mapped_column(String(16), default="encrypted")  # 'session' | 'encrypted'
    api_key_enc: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())


class ProviderIcon(Base):
    __tablename__ = "provider_icons"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    svg_content: Mapped[str] = mapped_column(nullable=False)  # Full SVG markup
    created_at: Mapped[datetime] = mapped_column(default=func.now())


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="project", cascade="all, delete-orphan")  # type: ignore[name-defined]


class Template(Base):
    __tablename__ = "templates"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(nullable=True)
    user_prompt: Mapped[str | None] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(nullable=False, default="")  # legacy
    schema_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    examples_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    yaml_blob: Mapped[str | None] = mapped_column(nullable=True)
    version_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="template", cascade="all, delete-orphan")  # type: ignore[name-defined]


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    template_id: Mapped[int | None] = mapped_column(ForeignKey("templates.id"), nullable=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    input_images_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created")
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    provider: Mapped[Provider] = relationship(back_populates="runs")
    template: Mapped[Template | None] = relationship(back_populates="runs")
    project: Mapped[Project | None] = relationship(back_populates="runs")  # type: ignore[name-defined]
    tests: Mapped[list[Test]] = relationship(back_populates="run", cascade="all, delete-orphan")  # type: ignore[name-defined]


class Test(Base):
    __tablename__ = "tests"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    expected_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    diff_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    passed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    run: Mapped[Run] = relationship(back_populates="tests")


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db() -> None:
    """Initialize the SQLite database and create tables if not exist."""
    global _engine, _SessionLocal
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{DB_PATH}"
    _engine = create_engine(url, echo=False, future=True)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        _run_migrations()
    except Exception as e:
        # Best-effort migrations; log but continue
        import sys
        print(f"Warning: Database migration failed: {e}", file=sys.stderr)


def _run_migrations() -> None:
    """Lightweight, additive migrations for SQLite to add missing columns."""
    if _engine is None:
        return
    with _engine.begin() as conn:
        def table_cols(name: str) -> set[str]:
            # Use parameterized query to avoid SQL injection
            if name not in ("providers", "templates", "runs", "tests", "projects"):
                raise ValueError(f"Invalid table name: {name}")
            rows = conn.exec_driver_sql(f"PRAGMA table_info({name})").fetchall()
            return {r[1] for r in rows}  # type: ignore

        # Providers table new columns
        cols = table_cols("providers")
        add_map = {
            "key_storage": "TEXT",
            "api_key_enc": "TEXT",
            "provider_code": "TEXT",
            "model_id": "TEXT",
            "headers_json": "JSON",
            "timeout_s": "REAL",
            "logo_path": "TEXT",
            "catalog_caps_json": "JSON",
            "detected_caps_json": "JSON",
            "default_temperature": "REAL",
            "default_top_p": "REAL",
            "default_max_output_tokens": "INTEGER",
            "default_force_json_mode": "BOOLEAN",
            "default_prefer_tools": "BOOLEAN",
            "cached_max_tokens_param": "TEXT",
            "is_active": "BOOLEAN",
        }
        for col, typ in add_map.items():
            if col not in cols:
                # Column names and types are from hardcoded dict, safe from injection
                conn.exec_driver_sql(f"ALTER TABLE providers ADD COLUMN {col} {typ}")

        # Templates table new columns
        cols_t = table_cols("templates")
        add_map_t = {
            "description": "TEXT",
            "system_prompt": "TEXT",
            "user_prompt": "TEXT",
            "yaml_blob": "TEXT",
            "version_tag": "TEXT",
            "source_path": "TEXT",
        }
        for col, typ in add_map_t.items():
            if col not in cols_t:
                # Column names and types are from hardcoded dict, safe from injection
                conn.exec_driver_sql(f"ALTER TABLE templates ADD COLUMN {col} {typ}")

        # Runs table new columns
        cols_r = table_cols("runs")
        if "project_id" not in cols_r:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN project_id INTEGER")
    
    # Create Default Project if no projects exist and assign existing runs
    _ensure_default_project()


def _ensure_default_project() -> None:
    """Create Default Project if no projects exist and assign existing runs to it."""
    try:
        with get_db() as db:
            # Check if any projects exist
            existing = db.query(Project).first()
            if existing is None:
                # Create default project
                default_proj = Project(
                    name="Default Project",
                    description="Auto-created for existing runs",
                    is_active=True
                )
                db.add(default_proj)
                db.commit()
                db.refresh(default_proj)
                
                # Assign all existing runs to default project
                db.query(Run).filter(Run.project_id.is_(None)).update(
                    {Run.project_id: default_proj.id},
                    synchronize_session=False
                )
                db.commit()
    except Exception:
        # Ignore errors during migration (e.g., if tables don't exist yet)
        pass


def get_db() -> Session:
    """Return a SQLAlchemy Session; ensure init_db() has been called."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal()


# --- Optional encryption helpers for persistent API keys ---
def _get_kms_key() -> str | None:
    """Return APP_KMS_KEY from env or auto-managed data/kms.key contents."""
    key = os.getenv("APP_KMS_KEY")
    if key:
        return key
    try:
        path = Path("data/kms.key")
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        # Auto-create if none exists
        from cryptography.fernet import Fernet  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        k = Fernet.generate_key().decode()
        path.write_text(k, encoding="utf-8")
        return k
    except Exception:
        return None


def get_decrypted_api_key(provider_code: str) -> str | None:
    """Return decrypted API key for provider_code if stored encrypted.

    Session-stored secrets are handled in UI; this only returns persisted keys.
    """
    rec = get_provider_key(provider_code)
    if not rec or rec.key_storage != "encrypted" or not rec.api_key_enc:
        return None
    try:
        key = _get_kms_key()
        if not key:
            return None
        from cryptography.fernet import Fernet  # type: ignore
        return Fernet(key).decrypt(rec.api_key_enc.encode()).decode()
    except Exception:
        return None


# Simple DAO stubs
def create_provider(name: str, base_url: str) -> Provider:
    """Create a provider if name is not already taken.

    Raises ValueError if a provider with the same name exists.
    """
    with get_db() as db:
        existing = db.query(Provider).filter(Provider.name == name).first()
        if existing:
            raise ValueError("Provider name already exists")
        p = Provider(name=name, base_url=base_url)
        db.add(p)
        db.commit()
        db.refresh(p)
        return p


def list_providers() -> list[Provider]:
    with get_db() as db:
        return db.query(Provider).order_by(Provider.name.asc()).all()


def create_template(
    name: str,
    content: str = "",
    *,
    description: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    schema_json: dict | None = None,
    examples_json: list | None = None,
    yaml_blob: str | None = None,
    version_tag: str | None = None,
    source_path: str | None = None,
) -> Template:
    with get_db() as db:
        t = Template(
            name=name,
            content=content,
            description=description,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_json=schema_json or {},
            examples_json=examples_json or [],
            yaml_blob=yaml_blob,
            version_tag=version_tag,
            source_path=source_path,
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return t


def list_templates() -> list[Template]:
    with get_db() as db:
        return db.query(Template).order_by(Template.created_at.desc()).all()


def get_template_by_id(tid: int) -> Template | None:
    with get_db() as db:
        return db.get(Template, tid)


def get_template_by_name(name: str) -> Template | None:
    with get_db() as db:
        return db.query(Template).filter(Template.name == name).first()


def update_template(tid: int, **fields) -> Template:
    with get_db() as db:
        t = db.get(Template, tid)
        if not t:
            raise ValueError("Template not found")
        for k, v in fields.items():
            if hasattr(t, k):
                setattr(t, k, v)
        db.add(t)
        db.commit()
        db.refresh(t)
        return t


def delete_template(tid: int) -> None:
    with get_db() as db:
        t = db.get(Template, tid)
        if t:
            db.delete(t)
            db.commit()


def record_run(
    provider_id: int, 
    template_id: int | None, 
    input_images: list | None, 
    output: dict | None, 
    cost_usd: float | None, 
    status: str = "completed",
    project_id: int | None = None
) -> Run:
    with get_db() as db:
        # If no project_id provided, use active project
        if project_id is None:
            active_proj = get_active_project()
            if active_proj:
                project_id = active_proj.id
        
        r = Run(
            provider_id=provider_id,
            template_id=template_id,
            project_id=project_id,
            input_images_json=input_images or [],
            output_json=output or {},
            cost_usd=cost_usd,
            status=status,
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return r


def ensure_registry_provider(descriptor: ModelDescriptor) -> Provider:
    """Ensure a database provider row exists for registry-backed descriptors."""
    name = f"Registry::{descriptor.provider_id}:{descriptor.id}"
    caps_dump = descriptor.capabilities.model_dump()
    headers = descriptor.extra_headers()

    with get_db() as db:
        provider = (
            db.query(Provider)
            .filter(Provider.provider_code == descriptor.provider_id, Provider.model_id == descriptor.id)
            .first()
        )

        if not provider:
            provider = Provider(
                name=name,
                base_url=descriptor.base_url,
                provider_code=descriptor.provider_id,
                model_id=descriptor.id,
                key_storage="encrypted",
                api_key_enc=None,
                headers_json=headers or {},
                timeout_s=descriptor.timeouts.total_s,
                catalog_caps_json=caps_dump,
                detected_caps_json=caps_dump,
                default_temperature=descriptor.default_temperature,
                default_top_p=descriptor.default_top_p,
                default_max_output_tokens=descriptor.max_output_tokens,
                default_force_json_mode=descriptor.force_json_mode,
                default_prefer_tools=descriptor.prefer_tools,
                is_active=True,
            )
            db.add(provider)
        else:
            provider.name = name
            provider.base_url = descriptor.base_url
            provider.provider_code = descriptor.provider_id
            provider.model_id = descriptor.id
            provider.headers_json = headers or {}
            provider.timeout_s = descriptor.timeouts.total_s
            provider.catalog_caps_json = caps_dump
            provider.detected_caps_json = caps_dump
            provider.default_temperature = descriptor.default_temperature
            provider.default_top_p = descriptor.default_top_p
            provider.default_max_output_tokens = descriptor.max_output_tokens
            provider.default_force_json_mode = descriptor.force_json_mode
            provider.default_prefer_tools = descriptor.prefer_tools
            provider.is_active = True

        db.commit()
        db.refresh(provider)
        return provider


def list_runs(limit: int = 50) -> list[Run]:
    with get_db() as db:
        return db.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
def get_provider_by_id(pid: int) -> Provider | None:
    with get_db() as db:
        return db.get(Provider, pid)


def get_provider_by_name(name: str) -> Provider | None:
    with get_db() as db:
        return db.query(Provider).filter(Provider.name == name).first()


def update_provider(pid: int, **fields) -> Provider:
    with get_db() as db:
        p = db.get(Provider, pid)
        if not p:
            raise ValueError("Provider not found")
        # enforce unique name if changed
        new_name = fields.get("name")
        if new_name and new_name != p.name:
            dup = db.query(Provider).filter(Provider.name == new_name).first()
            if dup:
                raise ValueError("Provider name already exists")
        for k, v in fields.items():
            if hasattr(p, k):
                setattr(p, k, v)
        db.add(p)
        db.commit()
        db.refresh(p)
        return p


def delete_provider(pid: int) -> None:
    with get_db() as db:
        p = db.get(Provider, pid)
        if p:
            db.delete(p)
            db.commit()


def set_active_provider(pid: int) -> None:
    with get_db() as db:
        # clear existing
        db.query(Provider).update({Provider.is_active: False})
        p = db.get(Provider, pid)
        if not p:
            raise ValueError("Provider not found")
        p.is_active = True
        db.add(p)
        db.commit()


def get_active_provider() -> Provider | None:
    with get_db() as db:
        return db.query(Provider).filter(Provider.is_active == True).first()  # noqa: E712


# Provider API key management (by provider_code)
def list_provider_keys() -> list[ProviderKey]:
    with get_db() as db:
        return db.query(ProviderKey).order_by(ProviderKey.provider_code.asc()).all()


def get_provider_key(provider_code: str) -> ProviderKey | None:
    if not provider_code:
        return None
    with get_db() as db:
        return (
            db.query(ProviderKey)
            .filter(ProviderKey.provider_code == provider_code.strip().lower())
            .first()
        )


def set_provider_key(provider_code: str, key_storage: str, api_key_enc: str | None) -> ProviderKey:
    """Create or update the provider key record for a provider_code.

    key_storage: 'session' or 'encrypted'. If 'session', api_key_enc is ignored and set to None.
    """
    code = provider_code.strip().lower()
    storage_mode = key_storage if key_storage in ("session", "encrypted") else "encrypted"
    enc = (api_key_enc if storage_mode == "encrypted" else None)
    with get_db() as db:
        rec = db.query(ProviderKey).filter(ProviderKey.provider_code == code).first()
        if rec:
            rec.key_storage = storage_mode
            rec.api_key_enc = enc
            db.add(rec)
        else:
            rec = ProviderKey(provider_code=code, key_storage=storage_mode, api_key_enc=enc)
            db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec


def delete_provider_key(provider_code: str) -> None:
    if not provider_code:
        return
    with get_db() as db:
        rec = db.query(ProviderKey).filter(ProviderKey.provider_code == provider_code.strip().lower()).first()
        if rec:
            db.delete(rec)
            db.commit()


# --- Project Management ---
def create_project(name: str, description: str | None = None) -> Project:
    """Create a new project."""
    with get_db() as db:
        existing = db.query(Project).filter(Project.name == name).first()
        if existing:
            raise ValueError("Project name already exists")
        
        proj = Project(name=name, description=description, is_active=False)
        db.add(proj)
        db.commit()
        db.refresh(proj)
        return proj


def list_projects() -> list[Project]:
    """List all projects ordered by creation date."""
    with get_db() as db:
        return db.query(Project).order_by(Project.created_at.desc()).all()


def get_project_by_id(project_id: int) -> Project | None:
    """Get project by ID."""
    with get_db() as db:
        return db.get(Project, project_id)


def get_project_by_name(name: str) -> Project | None:
    """Get project by name."""
    with get_db() as db:
        return db.query(Project).filter(Project.name == name).first()


def get_active_project() -> Project | None:
    """Get the currently active project."""
    with get_db() as db:
        return db.query(Project).filter(Project.is_active == True).first()  # noqa: E712


def set_active_project(project_id: int) -> None:
    """Set a project as active (deactivates all others)."""
    with get_db() as db:
        # Deactivate all projects
        db.query(Project).update({Project.is_active: False})
        
        # Activate the selected project
        proj = db.get(Project, project_id)
        if not proj:
            raise ValueError("Project not found")
        proj.is_active = True
        db.add(proj)
        db.commit()


def update_project(project_id: int, **fields) -> Project:
    """Update project fields."""
    with get_db() as db:
        proj = db.get(Project, project_id)
        if not proj:
            raise ValueError("Project not found")
        
        # Enforce unique name if changed
        new_name = fields.get("name")
        if new_name and new_name != proj.name:
            dup = db.query(Project).filter(Project.name == new_name).first()
            if dup:
                raise ValueError("Project name already exists")
        
        for k, v in fields.items():
            if hasattr(proj, k):
                setattr(proj, k, v)
        
        db.add(proj)
        db.commit()
        db.refresh(proj)
        return proj


def delete_project(project_id: int) -> None:
    """Delete a project if it has no runs."""
    with get_db() as db:
        proj = db.get(Project, project_id)
        if not proj:
            return
        
        # Check if project has runs
        run_count = db.query(Run).filter(Run.project_id == project_id).count()
        if run_count > 0:
            raise ValueError(f"Cannot delete project with {run_count} run(s). Please reassign or delete the runs first.")
        
        # Don't delete if it's the only project
        total_projects = db.query(Project).count()
        if total_projects <= 1:
            raise ValueError("Cannot delete the last project. Create another project first.")
        
        db.delete(proj)
        db.commit()


def get_project_stats(project_id: int) -> Dict[str, Any]:
    """Get statistics for a project."""
    with get_db() as db:
        runs = db.query(Run).filter(Run.project_id == project_id).all()
        
        total_images = sum(len(r.input_images_json or []) for r in runs)
        total_cost_usd = sum(r.cost_usd or 0.0 for r in runs)
        
        # Count unique models used
        models_used = set()
        for r in runs:
            if r.provider and r.provider.model_id:
                models_used.add(r.provider.model_id)
        
        return {
            "total_runs": len(runs),
            "total_images": total_images,
            "total_cost_usd": total_cost_usd,
            "avg_cost_per_image": (total_cost_usd / total_images if total_images > 0 else 0.0),
            "models_used": list(models_used),
        }


# ============================================================================
# Provider Icons (LLM-generated SVG icons)
# ============================================================================

def get_provider_icon(provider_code: str) -> str | None:
    """Retrieve cached SVG icon for a provider.
    
    Args:
        provider_code: Provider identifier (e.g., 'openai', 'anthropic')
    
    Returns:
        SVG content string or None if not found
    """
    with get_db() as db:
        icon = db.query(ProviderIcon).filter(ProviderIcon.provider_code == provider_code).first()
        return icon.svg_content if icon else None


def save_provider_icon(provider_code: str, svg_content: str) -> ProviderIcon:
    """Store generated SVG icon for a provider.
    
    Args:
        provider_code: Provider identifier
        svg_content: Full SVG markup
    
    Returns:
        Created or updated ProviderIcon instance
    """
    with get_db() as db:
        # Check if already exists
        icon = db.query(ProviderIcon).filter(ProviderIcon.provider_code == provider_code).first()
        
        if icon:
            # Update existing
            icon.svg_content = svg_content
        else:
            # Create new
            icon = ProviderIcon(provider_code=provider_code, svg_content=svg_content)
            db.add(icon)
        
        db.commit()
        db.refresh(icon)
        return icon


def delete_provider_icon(provider_code: str) -> bool:
    """Delete cached icon for a provider (to trigger regeneration).
    
    Args:
        provider_code: Provider identifier
    
    Returns:
        True if deleted, False if not found
    """
    with get_db() as db:
        icon = db.query(ProviderIcon).filter(ProviderIcon.provider_code == provider_code).first()
        if icon:
            db.delete(icon)
            db.commit()
            return True
        return False


def list_provider_icons() -> list[ProviderIcon]:
    """List all cached provider icons.
    
    Returns:
        List of ProviderIcon instances
    """
    with get_db() as db:
        return db.query(ProviderIcon).order_by(ProviderIcon.created_at.desc()).all()
