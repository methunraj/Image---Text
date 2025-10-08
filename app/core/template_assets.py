from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from app.core import storage


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATE_DIR = _PROJECT_ROOT / "app" / "assets" / "templates"
_SLUG_REGEX = re.compile(r"[^a-z0-9]+")


def _ensure_template_dir(directory: Path | None) -> Path:
    path = directory or _DEFAULT_TEMPLATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    slug = _SLUG_REGEX.sub("_", name.lower()).strip("_")
    return slug or "template"


def _normalize_examples(examples: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        raw_images = item.get("images")
        if isinstance(raw_images, str):
            images = [raw_images]
        elif isinstance(raw_images, list):
            images = [str(i) for i in raw_images if isinstance(i, (str, bytes, Path))][:3]
        else:
            images = []

        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        normalized.append({"images": images, "expected": expected})
    return normalized


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    return str(value)


def _template_payload_from_yaml(raw: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    name = _coerce_optional_text(raw.get("name")) or fallback_name
    schema = raw.get("schema") if isinstance(raw.get("schema"), dict) else {}
    examples_field = raw.get("examples") if isinstance(raw.get("examples"), list) else []
    payload: Dict[str, Any] = {
        "name": name,
        "description": _coerce_optional_text(raw.get("description")),
        "system_prompt": _coerce_optional_text(raw.get("system_prompt")),
        "user_prompt": _coerce_optional_text(raw.get("user_prompt")),
        "schema_json": schema,
        "examples_json": _normalize_examples(examples_field),
        "version_tag": _coerce_optional_text(raw.get("version_tag")),
    }
    return payload


def _template_to_dict(template: storage.Template) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "name": template.name,
        "description": template.description,
        "system_prompt": template.system_prompt,
        "user_prompt": template.user_prompt,
        "schema": template.schema_json or {},
        "examples": template.examples_json or [],
    }
    if template.version_tag:
        data["version_tag"] = template.version_tag
    return data


def _canonical_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _relativize(path: Path) -> str:
    try:
        return path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sync_from_assets(template_dir: Path | None = None) -> List[str]:
    directory = _ensure_template_dir(template_dir)
    imported: List[str] = []

    if not directory.exists():
        return imported

    for path in sorted(directory.glob("*.y*ml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        try:
            raw = yaml.safe_load(text) or {}
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        if not raw:
            continue

        payload = _template_payload_from_yaml(raw, fallback_name=path.stem.replace("_", " ").title())
        name = payload["name"]
        payload["yaml_blob"] = text
        payload["source_path"] = _relativize(path)

        existing = storage.get_template_by_name(name)
        if existing is None:
            storage.create_template(
                name,
                description=payload["description"],
                system_prompt=payload["system_prompt"],
                user_prompt=payload["user_prompt"],
                schema_json=payload["schema_json"],
                examples_json=payload["examples_json"],
                yaml_blob=payload.get("yaml_blob"),
                version_tag=payload.get("version_tag"),
                source_path=payload.get("source_path"),
            )
        else:
            updates: Dict[str, Any] = {}
            if (existing.description or None) != payload["description"]:
                updates["description"] = payload["description"]
            if (existing.system_prompt or None) != payload["system_prompt"]:
                updates["system_prompt"] = payload["system_prompt"]
            if (existing.user_prompt or None) != payload["user_prompt"]:
                updates["user_prompt"] = payload["user_prompt"]
            if (existing.schema_json or {}) != (payload["schema_json"] or {}):
                updates["schema_json"] = payload["schema_json"] or {}
            if (existing.examples_json or []) != (payload["examples_json"] or []):
                updates["examples_json"] = payload["examples_json"] or []
            if (existing.version_tag or None) != payload.get("version_tag"):
                updates["version_tag"] = payload.get("version_tag")
            if (existing.yaml_blob or None) != payload.get("yaml_blob"):
                updates["yaml_blob"] = payload.get("yaml_blob")
            if (existing.source_path or None) != payload.get("source_path"):
                updates["source_path"] = payload.get("source_path")

            if updates:
                storage.update_template(existing.id, **updates)

        imported.append(name)

    return imported


def export_to_assets(template: storage.Template, template_dir: Path | None = None) -> Path:
    directory = _ensure_template_dir(template_dir)

    target_path: Path | None = None
    if template.source_path:
        candidate = (_PROJECT_ROOT / template.source_path).resolve()
        if candidate.exists() or candidate.parent.exists():
            target_path = candidate

    if target_path is None:
        target_path = directory / f"{_slugify(template.name)}.yaml"

    payload = _template_to_dict(template)
    yaml_text = _canonical_yaml(payload)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml_text, encoding="utf-8")

    rel = _relativize(target_path)
    updates: Dict[str, Any] = {}
    if template.source_path != rel:
        updates["source_path"] = rel
    if (template.yaml_blob or None) != yaml_text:
        updates["yaml_blob"] = yaml_text
    if updates:
        storage.update_template(template.id, **updates)

    return target_path


def delete_from_assets(template: storage.Template) -> None:
    if not template.source_path:
        return

    path = (_PROJECT_ROOT / template.source_path).resolve()
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
