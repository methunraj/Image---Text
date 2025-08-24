from __future__ import annotations

import json
from typing import Any, Dict, List

from dataclasses import dataclass
from typing import TypedDict, Optional

from jinja2 import Environment


_env = Environment(autoescape=False)


def render_prompt(template_text: str, *, schema: Dict[str, Any], examples: List[Dict[str, Any]] | List[Any]) -> str:
    """Render a prompt template with schema and examples.

    - `template_text` is treated as a Jinja2 template.
    - `schema` is injected as a JSON string via `schema`.
    - `examples` is available as a Python object; use `| tojson` in the template for JSON.
    """
    tmpl = _env.from_string(template_text)
    rendered = tmpl.render(schema=json.dumps(schema, indent=2), examples=examples)
    return str(rendered)


class ChatPart(TypedDict, total=False):
    type: str
    text: str
    image_url: Dict[str, Any]


class ChatMessage(TypedDict):
    role: str
    content: List[ChatPart]


@dataclass
class Example:
    images: List[str]  # data URLs or external URLs
    expected: Dict[str, Any]


@dataclass
class RenderedMessages:
    messages: List[ChatMessage]


def _minify_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def render_user_prompt(user_text: str, schema_json: Dict[str, Any], examples: List[Example], vars: Dict[str, Any]) -> RenderedMessages:
    """Render a user prompt and expand examples into alternating chat messages.

    - {schema} replaced by minified JSON of `schema_json`.
    - {examples} is removed from the text; examples are expanded as:
        user: "Example image follows." + image attachments
        assistant: expected JSON as text
    - {today}, {locale}, {doc_type} replaced from vars (empty if missing).
    Returns messages with only example/user messages; a system message should be prepended by the caller if needed.
    """
    text = user_text or ""
    text = text.replace("{schema}", _minify_json(schema_json))
    text = text.replace("{today}", str(vars.get("today", "")))
    text = text.replace("{locale}", str(vars.get("locale", "")))
    text = text.replace("{doc_type}", str(vars.get("doc_type", "")))
    # Remove {examples} token from the user text; examples are emitted as messages
    text = text.replace("{examples}", "").strip()

    messages: List[ChatMessage] = []
    for ex in examples or []:
        # user with images
        parts: List[ChatPart] = [{"type": "text", "text": "Example image follows."}]
        for url in ex.images[:3]:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": "user", "content": parts})
        # assistant expected
        messages.append({"role": "assistant", "content": [{"type": "text", "text": _minify_json(ex.expected)}]})

    # Final user request
    final_parts: List[ChatPart] = [{"type": "text", "text": text}]
    messages.append({"role": "user", "content": final_parts})
    return RenderedMessages(messages=messages)
