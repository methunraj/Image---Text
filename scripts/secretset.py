from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

import yaml
from cryptography.fernet import Fernet, InvalidToken

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "secrets.enc.yaml"
KMS_PATH = ROOT / "data" / "kms.key"


def _ensure_key() -> str:
    key = os.getenv("APP_KMS_KEY")
    if key:
        return key.strip()

    KMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if KMS_PATH.exists():
        return KMS_PATH.read_text(encoding="utf-8").strip()

    generated = Fernet.generate_key().decode("utf-8")
    KMS_PATH.write_text(generated, encoding="utf-8")
    return generated


def _fernet_from_key(raw_key: str) -> Fernet:
    token = raw_key.strip().encode("utf-8")
    try:
        return Fernet(token)
    except (ValueError, TypeError):
        digest = hashlib.sha256(token).digest()
        return Fernet(base64.urlsafe_b64encode(digest))


def _load_secrets(fernet: Fernet) -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        payload = CONFIG_PATH.read_bytes()
        decrypted = fernet.decrypt(payload)
    except (OSError, InvalidToken) as exc:
        raise SystemExit(f"Failed to decrypt {CONFIG_PATH}: {exc}")

    data = yaml.safe_load(decrypted.decode("utf-8")) or {}
    secrets = data.get("secrets", {})
    if not isinstance(secrets, dict):
        raise SystemExit("Malformed secrets file; expected mapping under 'secrets'")
    return {str(k): str(v) for k, v in secrets.items()}


def _write_secrets(fernet: Fernet, secrets: dict[str, str]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.safe_dump({"secrets": secrets}, sort_keys=True).encode("utf-8")
    encrypted = fernet.encrypt(dumped)
    CONFIG_PATH.write_bytes(encrypted)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set or delete encrypted provider secrets.")
    parser.add_argument("name", help="Secret key name, e.g. OPENAI_API_KEY")
    parser.add_argument("value", nargs="?", help="Secret value. Omit when using --delete.")
    parser.add_argument("--delete", action="store_true", help="Delete the specified secret.")

    args = parser.parse_args(argv)

    if args.delete and args.value is not None:
        parser.error("--delete does not accept a value argument")
    if not args.delete and args.value is None:
        parser.error("value is required unless --delete is provided")

    raw_key = _ensure_key()
    fernet = _fernet_from_key(raw_key)
    secrets = _load_secrets(fernet)

    if args.delete:
        removed = secrets.pop(args.name, None)
        if removed is None:
            print(f"Secret {args.name} not found; nothing to delete")
        else:
            _write_secrets(fernet, secrets)
            print(f"Deleted secret {args.name}")
        return 0

    secrets[args.name] = args.value or ""
    _write_secrets(fernet, secrets)
    print(f"Stored secret {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
