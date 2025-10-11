from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional

import yaml
from cryptography.fernet import Fernet, InvalidToken

from app.core.config_excel import ExcelConfigError, parse_excel_config
from app.core.config_schema import AppModelConfig


@dataclass(frozen=True)
class LoadedConfig:
    config: AppModelConfig
    secrets: Mapping[str, str]
    profile: str
    source_path: Path


class ConfigLoaderError(RuntimeError):
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_profile(profile: Optional[str]) -> str:
    if profile:
        return profile
    env_profile = os.getenv("APP_PROFILE") or "dev"
    return env_profile.strip() or "dev"


def _resolve_config_path(profile: str) -> Path:
    if override := os.getenv("APP_MODELS_CONFIG"):
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise ConfigLoaderError(f"APP_MODELS_CONFIG points to missing file: {path}")
        return path

    root = _project_root()
    excel_candidate = root / "config" / "models.xlsx"
    if excel_candidate.exists():
        return excel_candidate

    candidate = root / "config" / f"models.{profile}.yaml"
    if candidate.exists():
        return candidate

    raise ConfigLoaderError(f"No configuration file found for profile '{profile}'")


def _load_yaml(path: Path) -> Dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - pass through message
        raise ConfigLoaderError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigLoaderError(f"Unable to read configuration file {path}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigLoaderError(f"Configuration root in {path} must be a mapping")
    return raw


def _fernet_from_key(raw_key: str) -> Fernet:
    token = raw_key.strip().encode("utf-8")
    try:
        return Fernet(token)
    except (ValueError, TypeError):
        digest = hashlib.sha256(token).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


def _load_encrypted_secrets(path: Path, kms_key: str) -> Dict[str, str]:
    fernet = _fernet_from_key(kms_key)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ConfigLoaderError(f"Unable to read encrypted secrets file: {path}") from exc

    try:
        decrypted = fernet.decrypt(payload)
    except InvalidToken as exc:
        raise ConfigLoaderError("Failed to decrypt secrets file; check APP_KMS_KEY") from exc

    try:
        raw = yaml.safe_load(decrypted.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoaderError("Decrypted secrets file is not valid YAML") from exc

    secrets = raw.get("secrets") if isinstance(raw, dict) else None
    if not isinstance(secrets, dict):
        raise ConfigLoaderError("Encrypted secrets file must contain a 'secrets' mapping")

    return {str(k): str(v) for k, v in secrets.items()}


def _load_plain_secrets(path: Path) -> Dict[str, str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoaderError(f"Invalid YAML in secrets file {path}") from exc
    except OSError as exc:
        raise ConfigLoaderError(f"Unable to read secrets file {path}") from exc

    secrets = raw.get("secrets") if isinstance(raw, dict) else None
    if not isinstance(secrets, dict):
        raise ConfigLoaderError("Secrets file must contain a 'secrets' mapping")

    return {str(k): str(v) for k, v in secrets.items()}


def _load_secrets(profile: str) -> Dict[str, str]:
    root = _project_root()
    enc_path = root / "config" / "secrets.enc.yaml"
    plain_path = root / "config" / "secrets.yaml"

    if enc_path.exists():
        kms_key = os.getenv("APP_KMS_KEY")
        if not kms_key:
            raise ConfigLoaderError("APP_KMS_KEY must be set to decrypt config/secrets.enc.yaml")
        return _load_encrypted_secrets(enc_path, kms_key)

    if plain_path.exists():
        return _load_plain_secrets(plain_path)

    # Support profile-specific secret overrides via environment variables
    prefix = f"APP_SECRET_{profile.upper()}_"
    collected: Dict[str, str] = {}
    for env_key, env_value in os.environ.items():
        if env_key.startswith(prefix):
            collected[env_key.removeprefix(prefix)] = env_value
    return collected


def load_config(profile: Optional[str] = None) -> LoadedConfig:
    resolved_profile = _resolve_profile(profile)
    path = _resolve_config_path(resolved_profile)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        try:
            app_config = parse_excel_config(path, resolved_profile)
        except ExcelConfigError as exc:
            raise ConfigLoaderError(str(exc)) from exc
    else:
        data = _load_yaml(path)
        app_config = AppModelConfig.model_validate(data)

    secrets = _load_secrets(resolved_profile)
    return LoadedConfig(
        config=app_config,
        secrets=MappingProxyType(dict(secrets)),
        profile=resolved_profile,
        source_path=path,
    )
