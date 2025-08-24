from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, Optional

from sqlalchemy import JSON, Float, ForeignKey, String, Boolean, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

DB_PATH = Path("data/app.db")


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    base_url: Mapped[str] = mapped_column(String(255), nullable=False)
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
    # Active flag
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="provider", cascade="all, delete-orphan")  # type: ignore[name-defined]


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
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="template", cascade="all, delete-orphan")  # type: ignore[name-defined]


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    template_id: Mapped[int | None] = mapped_column(ForeignKey("templates.id"), nullable=True)
    input_images_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created")
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    provider: Mapped[Provider] = relationship(back_populates="runs")
    template: Mapped[Template | None] = relationship(back_populates="runs")
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
    except Exception:
        # Best-effort migrations; continue even if no-op fails
        pass


def _run_migrations() -> None:
    """Lightweight, additive migrations for SQLite to add missing columns."""
    if _engine is None:
        return
    with _engine.begin() as conn:
        def table_cols(name: str) -> set[str]:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({name})").fetchall()
            return {r[1] for r in rows}  # type: ignore

        # Providers table new columns
        cols = table_cols("providers")
        add_map = {
            "key_storage": "TEXT",
            "api_key_enc": "TEXT",
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
            "is_active": "BOOLEAN",
        }
        for col, typ in add_map.items():
            if col not in cols:
                conn.exec_driver_sql(f"ALTER TABLE providers ADD COLUMN {col} {typ}")

        # Templates table new columns
        cols_t = table_cols("templates")
        add_map_t = {
            "description": "TEXT",
            "system_prompt": "TEXT",
            "user_prompt": "TEXT",
            "yaml_blob": "TEXT",
            "version_tag": "TEXT",
        }
        for col, typ in add_map_t.items():
            if col not in cols_t:
                conn.exec_driver_sql(f"ALTER TABLE templates ADD COLUMN {col} {typ}")


def get_db() -> Session:
    """Return a SQLAlchemy Session; ensure init_db() has been called."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal()


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


def create_template(name: str, content: str, schema_json: dict | None = None, examples_json: list | None = None) -> Template:
    with get_db() as db:
        t = Template(name=name, content=content, schema_json=schema_json or {}, examples_json=examples_json or [])
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


def record_run(provider_id: int, template_id: int | None, input_images: list | None, output: dict | None, cost_usd: float | None, status: str = "completed") -> Run:
    with get_db() as db:
        r = Run(
            provider_id=provider_id,
            template_id=template_id,
            input_images_json=input_images or [],
            output_json=output or {},
            cost_usd=cost_usd,
            status=status,
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return r


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
