#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(_project_root() / ".env")
    except Exception:
        pass


def _ensure_registry():
    from app.core.model_registry import ensure_registry
    return ensure_registry()


def _resolve_model(registry, model_id: Optional[str]):
    if model_id:
        return registry.resolve(model_id)
    # Try to pick a Google model first, fallback to default
    for m in registry.models.values():
        if "google" in (m.provider_id.lower()):
            return m
    return registry.resolve(None)


def _gateway_from_descriptor(descriptor):
    from app.core.provider_openai import gateway_from_descriptor
    return gateway_from_descriptor(descriptor)


def _run(
    image: Path,
    model_id: Optional[str],
    mode: str,
    template_name: Optional[str],
    max_tokens: Optional[int],
) -> int:
    _add_sys_path()
    _load_env()

    registry = _ensure_registry()
    descriptor = _resolve_model(registry, model_id)
    gateway = _gateway_from_descriptor(descriptor)

    # Mode selection
    if mode == "unstructured":
        gateway.prefer_json_mode = False
        gateway.prefer_tools = False
        system_text = ""
        user_text = "Describe the image and extract key information in Markdown."
        schema = None
    elif mode == "auto-json":
        gateway.prefer_tools = False
        system_text = (
            "You are a precise JSON generator. Analyze the image and output ONLY valid JSON."
        )
        user_text = (
            "Extract information from the image. Respond with a single JSON object or array. Return JSON ONLY."
        )
        schema = None
    else:
        # Lightweight schema template
        system_text = "Return ONLY valid JSON matching the schema."
        user_text = "Use the schema to extract fields. Return JSON only."
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "total": {"type": "number"},
            },
        }

    gen_params: Dict[str, Any] = {}
    if max_tokens is not None and max_tokens > 0:
        gen_params["max_tokens"] = int(max_tokens)

    print("\n=== Smoke Test ===")
    print(f"Model: {descriptor.provider_label} • {descriptor.label}")
    print(f"Endpoint: {descriptor.base_url}")
    print(f"Mode: {mode}")
    print(f"Image: {image}")
    if gen_params:
        print(f"Params: {gen_params}")

    result = gateway.chat_vision(
        model=descriptor.id,
        system_text=system_text,
        user_text=user_text,
        image_paths=[str(image)],
        fewshot_messages=None,
        schema=schema,
        gen_params=gen_params or None,
    )

    print("\n--- Result (summary) ---")
    print(f"status: {result.get('status')}")
    if result.get("error"):
        print(f"error: {result.get('error')}")
    text = result.get("text")
    tool = result.get("tool_call_json")
    print(f"text: {(text[:200] + '...') if isinstance(text, str) and len(text) > 200 else text}")
    print(f"tool_call_json: {tool}")
    print(f"usage: {result.get('usage')}")
    try:
        cached = getattr(gateway, "cached_max_tokens_param", None)
        print(f"cached_max_tokens_param: {cached}")
    except Exception:
        pass

    # Non-zero exit on clear failure
    if result.get("error"):
        return 2
    if not (result.get("text") or result.get("tool_call_json")):
        return 3
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Smoke test a model with a single image.")
    p.add_argument("image", help="Path to a local image (png/jpg/webp)")
    p.add_argument("--model", help="Model id (defaults to Google if present, else default)")
    p.add_argument("--mode", choices=["auto-json", "unstructured", "schema"], default="auto-json")
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args(argv)

    img = Path(args.image)
    if not img.exists():
        print(f"Image not found: {img}")
        return 1
    return _run(img, args.model, args.mode, None, args.max_tokens)


if __name__ == "__main__":
    raise SystemExit(main())

