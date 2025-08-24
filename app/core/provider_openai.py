from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx


def encode_image_to_data_url(path: str) -> str:
    """Encode a local image file into a data URL (PNG/JPG supported)."""
    mime = "image/png"
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        mime = "image/jpeg"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    try:
        with open(path, "rb") as f:
            data = f.read()
            # Check file size - warn if over 5MB
            if len(data) > 5 * 1024 * 1024:
                print(f"Warning: Large image file {path} ({len(data)/(1024*1024):.1f}MB) may cause issues")
            b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except FileNotFoundError:
        print(f"Warning: Image file not found: {path}")
        return tiny_png_data_url()
    except MemoryError:
        print(f"Error: Image file too large to encode: {path}")
        return tiny_png_data_url()
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        # As a last resort, return a 1x1 PNG
        return tiny_png_data_url()


def tiny_png_data_url() -> str:
    """Return a 1x1 transparent PNG data URL, useful for probes."""
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="
    return f"data:image/png;base64,{b64}"


def encode_image_to_b64(path: str) -> Tuple[str, str]:
    """Return (base64_str, mime) for a local image path.

    Falls back to 1x1 PNG if reading fails.
    """
    mime = "image/png"
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        mime = "image/jpeg"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            return b64, mime
    except Exception:
        # transparent 1x1 png
        tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="
        return tiny_b64, "image/png"


def tiny_png_b64() -> str:
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="


class OpenAIProvider:
    """Backward-compatible minimal client used by Settings probe."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0, max_tokens: int | None = None, **extra: Any) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def image_message(role: str, text: str | None, image_urls: List[str]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": role, "content": content}


@dataclass
class OAIGateway:
    """OpenAI-compatible gateway with robust fallbacks for vision + JSON.

    - Adds OpenRouter header handling.
    - Supports two vision encodings (image_url data URIs, and input_image objects) with fallback.
    - Selects Tools or JSON mode based on detected capabilities and preferences.
    - Retries on timeouts and 5xx.
    """

    base_url: str
    api_key: Optional[str]
    headers: Optional[Dict[str, str]]
    timeout: int
    prefer_json_mode: bool
    prefer_tools: bool
    detected_caps: Optional[Dict[str, Any]] = None

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        # Merge user headers first, then ensure OpenRouter-required headers exist with sensible defaults.
        if self.headers:
            h.update(self.headers)
        if "openrouter.ai" in (self.base_url or ""):
            # OpenRouter requires HTTP-Referer and X-Title; keep user overrides if provided.
            h.setdefault("HTTP-Referer", (self.headers or {}).get("HTTP-Referer", "http://localhost"))
            h.setdefault("X-Title", (self.headers or {}).get("X-Title", "Images-JSON App"))
        return h

    def _client(self) -> httpx.Client:
        t = httpx.Timeout(self.timeout, connect=self.timeout, read=self.timeout, write=self.timeout)
        return httpx.Client(timeout=t)

    def _post_with_retries(self, url: str, json_payload: Dict[str, Any], max_retries: int = 3) -> httpx.Response:
        # Debug the outgoing request
        print(f"Debug [_post_with_retries]: Making request to {url}")
        print(f"Debug [_post_with_retries]: Headers: {self._headers()}")
        print(f"Debug [_post_with_retries]: Request payload keys: {list(json_payload.keys())}")
        print(f"Debug [_post_with_retries]: Model: {json_payload.get('model', 'N/A')}")
        print(f"Debug [_post_with_retries]: Messages count: {len(json_payload.get('messages', []))}")
        if json_payload.get('messages'):
            print(f"Debug [_post_with_retries]: First message role: {json_payload['messages'][0].get('role', 'N/A')}")
        
        # Show payload structure (truncated for readability)
        import json
        payload_str = json.dumps(json_payload, indent=2, default=str)
        if len(payload_str) > 1000:
            print(f"Debug [_post_with_retries]: Request payload (truncated):")
            print(payload_str[:1000] + "...")
        else:
            print(f"Debug [_post_with_retries]: Request payload:")
            print(payload_str)
        
        delay = 0.5
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            print(f"Debug [_post_with_retries]: Attempt {attempt}/{max_retries}")
            try:
                with self._client() as client:
                    resp = client.post(url, headers=self._headers(), json=json_payload)
                    print(f"Debug [_post_with_retries]: Response status: {resp.status_code}")
                    print(f"Debug [_post_with_retries]: Response headers: {dict(resp.headers)}")
                    
                    # Retry on 5xx and 429
                    if resp.status_code in (429, 500, 502, 503, 504):
                        error_body = resp.text if hasattr(resp, 'text') else 'No body'
                        print(f"Debug [_post_with_retries]: Retryable error {resp.status_code}, body: {error_body[:200]}")
                        last_exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                        raise last_exc
                    resp.raise_for_status()
                    print(f"Debug [_post_with_retries]: Request successful on attempt {attempt}")
                    return resp
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                print(f"Debug [_post_with_retries]: Exception on attempt {attempt}: {type(e).__name__}: {e}")
                last_exc = e
                if attempt >= max_retries:
                    break
                print(f"Debug [_post_with_retries]: Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
        assert last_exc is not None
        print(f"Debug [_post_with_retries]: All attempts failed, raising: {last_exc}")
        raise last_exc

    @staticmethod
    def _build_user_with_images_enc_a(user_text: str, image_paths: Iterable[str]) -> Dict[str, Any]:
        # EncA: image_url with data URI
        parts: List[Dict[str, Any]] = []
        if user_text:
            parts.append({"type": "text", "text": user_text})
        for p in image_paths:
            parts.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(p)}})
        return {"role": "user", "content": parts}

    @staticmethod
    def _build_user_with_images_enc_b(user_text: str, image_paths: Iterable[str]) -> Dict[str, Any]:
        # EncB: input_image with base64 data + mime
        parts: List[Dict[str, Any]] = []
        if user_text:
            parts.append({"type": "text", "text": user_text})
        for p in image_paths:
            b64, mime = encode_image_to_b64(p)
            parts.append({"type": "input_image", "input_image": {"data": b64, "mime_type": mime}})
        return {"role": "user", "content": parts}

    def _extract_text_and_tool(self, data: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        text: Optional[str] = None
        tool_call: Optional[Dict[str, Any]] = None
        
        print(f"Debug [_extract_text_and_tool]: Starting extraction with data keys: {list(data.keys())}")
        
        try:
            # Handle different response formats
            choices = data.get("choices", [])
            print(f"Debug [_extract_text_and_tool]: Choices: {type(choices)}, length: {len(choices) if isinstance(choices, list) else 'N/A'}")
            
            if not choices:
                print(f"Debug [_extract_text_and_tool]: No choices found, checking top-level keys")
                # Some providers might return completion directly
                for key in ["completion", "text", "content", "output", "result"]:
                    if key in data:
                        text = data[key]
                        print(f"Debug [_extract_text_and_tool]: Found text in top-level key '{key}': {repr(text)[:100]}")
                        return text, tool_call
                print(f"Debug [_extract_text_and_tool]: No text found in top-level keys")
                return text, tool_call
            
            choice = choices[0]
            print(f"Debug [_extract_text_and_tool]: First choice type: {type(choice)}")
            print(f"Debug [_extract_text_and_tool]: First choice keys: {list(choice.keys()) if isinstance(choice, dict) else 'Not a dict'}")
            
            # Check all possible locations for text in the choice
            if isinstance(choice, dict):
                # Direct text in choice
                if "text" in choice:
                    text = choice["text"]
                    print(f"Debug [_extract_text_and_tool]: Found text directly in choice: {repr(text)[:100]}")
                
                # Message-based extraction
                msg = choice.get("message") or {}
                print(f"Debug [_extract_text_and_tool]: Message type: {type(msg)}")
                print(f"Debug [_extract_text_and_tool]: Message keys: {list(msg.keys()) if isinstance(msg, dict) else 'Not a dict'}")
                
                if isinstance(msg, dict):
                    # Text extraction from message content
                    content = msg.get("content")
                    print(f"Debug [_extract_text_and_tool]: Content type: {type(content)}")
                    print(f"Debug [_extract_text_and_tool]: Content value: {repr(content)}")
                    
                    if isinstance(content, str):
                        text = content
                        print(f"Debug [_extract_text_and_tool]: Extracted string content: {repr(text)[:100]}")
                    elif isinstance(content, list):
                        print(f"Debug [_extract_text_and_tool]: Content is list with {len(content)} items")
                        # OpenAI may return content as list of parts
                        texts = []
                        for i, part in enumerate(content):
                            print(f"Debug [_extract_text_and_tool]: Part {i}: {type(part)}, keys: {list(part.keys()) if isinstance(part, dict) else 'Not a dict'}")
                            if isinstance(part, dict):
                                part_type = part.get("type")
                                part_text = part.get("text", "")
                                print(f"Debug [_extract_text_and_tool]: Part {i} type: {part_type}, text: {repr(part_text)[:50]}")
                                if part_type == "text" and part_text:
                                    texts.append(part_text)
                        text = "\n".join(texts) if texts else None
                        print(f"Debug [_extract_text_and_tool]: Assembled text from list parts: {repr(text)[:100] if text else 'None'}")
                    elif content is None:
                        print(f"Debug [_extract_text_and_tool]: Content is None")
                    else:
                        print(f"Debug [_extract_text_and_tool]: Content is unexpected type: {type(content)}")
                    
                    # Tools extraction
                    tcs = msg.get("tool_calls")
                    print(f"Debug [_extract_text_and_tool]: Tool calls: {type(tcs)}, value: {tcs}")
                    if isinstance(tcs, list) and tcs:
                        print(f"Debug [_extract_text_and_tool]: Processing {len(tcs)} tool calls")
                        fun = tcs[0].get("function", {})
                        args = fun.get("arguments")
                        print(f"Debug [_extract_text_and_tool]: Tool args type: {type(args)}, value: {args}")
                        if isinstance(args, str):
                            try:
                                import json as _json
                                tool_call = _json.loads(args)
                                print(f"Debug [_extract_text_and_tool]: Parsed tool call from string: {tool_call}")
                            except Exception as parse_err:
                                print(f"Debug [_extract_text_and_tool]: Failed to parse tool args: {parse_err}")
                                tool_call = None
                        elif isinstance(args, dict):
                            tool_call = args
                            print(f"Debug [_extract_text_and_tool]: Using tool args as dict: {tool_call}")
                
                # Additional fallback checks
                if text is None:
                    print(f"Debug [_extract_text_and_tool]: No text found yet, checking additional choice keys")
                    for key in ["delta", "completion", "output"]:
                        if key in choice:
                            fallback_content = choice[key]
                            print(f"Debug [_extract_text_and_tool]: Found fallback key '{key}': {type(fallback_content)}")
                            if isinstance(fallback_content, dict) and "content" in fallback_content:
                                text = fallback_content["content"]
                                print(f"Debug [_extract_text_and_tool]: Extracted text from {key}.content: {repr(text)[:100]}")
                                break
                            elif isinstance(fallback_content, str):
                                text = fallback_content
                                print(f"Debug [_extract_text_and_tool]: Used {key} directly as text: {repr(text)[:100]}")
                                break
            
            print(f"Debug [_extract_text_and_tool]: Final extraction results:")
            print(f"  - Text: {repr(text) if text else 'None'}")
            print(f"  - Text length: {len(text) if text else 0} characters")
            # Estimate token count (rough approximation: ~4 chars per token)
            if text:
                estimated_tokens = len(text) // 4
                print(f"  - Estimated tokens: ~{estimated_tokens} tokens")
                if estimated_tokens < 250:
                    print(f"  - WARNING: Output appears truncated at ~{estimated_tokens} tokens!")
            print(f"  - Tool call: {tool_call}")
            
        except Exception as e:
            print(f"Debug [_extract_text_and_tool]: Exception during extraction: {e}")
            import traceback
            traceback.print_exc()
            
        # Final OpenRouter-specific fallback
        if text is None and "openrouter.ai" in getattr(self, 'base_url', ''):
            print(f"Debug [_extract_text_and_tool]: Applying OpenRouter-specific extraction fallbacks")
            text = self._openrouter_extract_text(data)
            if text:
                print(f"Debug [_extract_text_and_tool]: OpenRouter fallback extracted: {repr(text)[:100]}")
            
        return text, tool_call

    def _openrouter_extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """OpenRouter-specific text extraction fallbacks"""
        print(f"Debug [_openrouter_extract_text]: Checking OpenRouter-specific patterns")
        
        # Check for non-standard response structures that OpenRouter might use
        patterns_to_check = [
            # Direct response patterns
            ("response", "text"),
            ("response", "content"),
            ("data", "response"),
            ("output", "text"),
            ("result", "content"),
            
            # Nested patterns
            ("choices", 0, "text"),
            ("choices", 0, "content"),
            ("choices", 0, "message", "text"),
            ("choices", 0, "delta", "text"),
            ("data", "choices", 0, "message", "content"),
            ("data", "choices", 0, "text"),
            
            # Generation patterns (some APIs use these)
            ("generations", 0, "text"),
            ("completions", 0, "text"),
        ]
        
        for pattern in patterns_to_check:
            try:
                current = data
                for key in pattern:
                    if isinstance(key, int):
                        if isinstance(current, list) and len(current) > key:
                            current = current[key]
                        else:
                            break
                    elif isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        break
                else:
                    # We successfully navigated the pattern
                    if isinstance(current, str) and current.strip():
                        print(f"Debug [_openrouter_extract_text]: Found text via pattern {pattern}: {repr(current)[:50]}")
                        return current
            except Exception as e:
                print(f"Debug [_openrouter_extract_text]: Error checking pattern {pattern}: {e}")
                continue
        
        # Check for any string values in the response that might be the content
        def find_likely_content(obj, path=""):
            if isinstance(obj, str) and len(obj.strip()) > 10:  # Likely content if > 10 chars
                if not any(keyword in obj.lower() for keyword in ['error', 'invalid', 'failed', 'exception']):
                    print(f"Debug [_openrouter_extract_text]: Found potential content at {path}: {repr(obj)[:50]}")
                    return obj
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if key in ['content', 'text', 'response', 'output', 'result', 'completion']:
                        result = find_likely_content(value, f"{path}.{key}")
                        if result:
                            return result
                # Check other keys too
                for key, value in obj.items():
                    if key not in ['id', 'object', 'created', 'model', 'usage', 'error', 'status']:
                        result = find_likely_content(value, f"{path}.{key}")
                        if result:
                            return result
            elif isinstance(obj, list) and obj:
                for i, item in enumerate(obj[:3]):  # Check first 3 items only
                    result = find_likely_content(item, f"{path}[{i}]")
                    if result:
                        return result
            return None
        
        potential_content = find_likely_content(data)
        if potential_content:
            print(f"Debug [_openrouter_extract_text]: Found potential content: {repr(potential_content)[:100]}")
            return potential_content
            
        print(f"Debug [_openrouter_extract_text]: No content found via OpenRouter fallbacks")
        return None

    def chat_vision(
        self,
        model: str,
        system_text: str,
        user_text: str,
        image_paths: List[str],
        fewshot_messages: List[Dict[str, Any]] | Dict[str, List[Dict[str, Any]]] | None,
        schema: Optional[Dict[str, Any]] = None,
        gen_params: Optional[Dict[str, Any]] = None,
        # Back-compat: allow older callers using params={...}
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a vision chat with robust fallbacks and clear error reporting.

        - Builds messages: system -> system_text; user -> text + images.
        - EncA uses data URIs via image_url; on certain 4xx errors, retries once with EncB (input_image).
        - Tool/JSON selection based on detected capabilities and preferences.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        # Unified params handling
        effective_params = gen_params or params or {}
        # Keep only universal params here; handle max token key via compatibility layer below
        base_payload: Dict[str, Any] = {k: v for k, v in effective_params.items() if k in {"temperature", "top_p", "seed"}}
        max_tokens_val = effective_params.get("max_tokens")

        # Messages assembly
        payload_messages: List[Dict[str, Any]] = []
        if (system_text or "").strip():
            payload_messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        # Fewshot messages: accept either direct list or {messages: [...]}
        if isinstance(fewshot_messages, dict):
            maybe = fewshot_messages.get("messages")
            if isinstance(maybe, list):
                for m in maybe:
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                        payload_messages.append(m)
        elif isinstance(fewshot_messages, list):
            for m in fewshot_messages:
                if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                    payload_messages.append(m)

        # Build user message later per encoding variant

        # Decide modes
        caps = self.detected_caps or {}
        tools_ok = True if (self.detected_caps is None) else bool(caps.get("tools"))
        json_ok = True if (self.detected_caps is None) else bool(caps.get("json_mode"))
        use_tools = bool(self.prefer_tools and tools_ok and schema)
        use_json_mode = bool((not use_tools) and self.prefer_json_mode and json_ok)

        def attach_modes(payload: Dict[str, Any]) -> Dict[str, Any]:
            if use_tools and schema:
                payload["tools"] = [
                    {"type": "function", "function": {"name": "emit", "parameters": schema}}
                ]
                payload["tool_choice"] = {"type": "function", "function": {"name": "emit"}}
            elif use_json_mode:
                payload["response_format"] = {"type": "json_object"}
            return payload

        # Helper: select a compatible max-tokens key and send
        def _keys_order() -> List[str]:
            local = bool(self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url))
            if local:
                # Extended list for better local model compatibility
                return ["n_predict", "max_tokens", "max_new_tokens", "max_length", "max_gen_tokens", "num_predict", "max_tokens_to_sample"]
            # Default order covers common variants across providers
            return ["max_tokens", "max_output_tokens", "max_completion_tokens", "max_tokens_to_sample", "n_predict"]

        def _send_with_max_variants(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            tried: List[str] = []
            last_exc: Optional[httpx.HTTPStatusError] = None
            # Build ordered candidates; always include a "no max tokens param" attempt last
            if max_tokens_val is not None:
                order = _keys_order() + [None]  # type: ignore[operator]
            else:
                order = [None]  # type: ignore[list-item]
            for key in order:
                # Build payload with only one max token key (or none)
                payload: Dict[str, Any] = {"model": model, "messages": messages, **base_payload}
                if key:  # type: ignore[truthy-bool]
                    payload[key] = max_tokens_val
                    tried.append(key)
                    is_local = bool(self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url))
                    print(f"Debug [Compat]: {'[LOCAL MODEL]' if is_local else ''} Trying token key '{key}'={max_tokens_val}")
                payload = attach_modes(payload)
                try:
                    resp = self._post_with_retries(url, payload)
                    return result_ok(resp)
                except httpx.HTTPStatusError as e_send:
                    last_exc = e_send
                    status = e_send.response.status_code if hasattr(e_send, "response") else None
                    body = ""
                    try:
                        body = e_send.response.text  # type: ignore[assignment]
                    except Exception:
                        pass
                    low = (body or "").lower()
                    # Retry only on invalid/unsupported parameter errors; otherwise re-raise
                    if status in (400, 422) and ("unsupported parameter" in low or "unknown parameter" in low or "is not supported" in low or "invalid_request_error" in low):
                        print(f"Debug [Compat]: Token key '{key}' rejected with {status}; body starts: {low[:120]}")
                        continue
                    raise
            # If we exhausted all variants, re-raise the last HTTP error if available
            if last_exc is not None:
                raise last_exc
            # Otherwise, raise a generic HTTP error indicating no variants were tried
            raise httpx.HTTPError("No compatible max token parameter could be applied")

        # Try EncA first
        enc_used = "EncA"
        messages_a = payload_messages + [self._build_user_with_images_enc_a(user_text, image_paths)]

        def result_ok(resp: httpx.Response) -> Dict[str, Any]:
            # First, capture the raw response body before JSON parsing
            raw_body = resp.text
            print(f"Debug [OAIGateway]: RAW HTTP Response Body:")
            print(f"=== START RAW RESPONSE ===")
            print(raw_body)
            print(f"=== END RAW RESPONSE ===")
            
            # Try to parse JSON
            try:
                data = resp.json()
            except Exception as json_err:
                print(f"Debug [OAIGateway]: JSON parsing failed: {json_err}")
                return {
                    "text": None,
                    "tool_call_json": None,
                    "raw": {"raw_body": raw_body, "json_error": str(json_err)},
                    "usage": None,
                    "status": resp.status_code,
                    "error": f"Failed to parse JSON response: {json_err}",
                }
            
            # Debug logging for OpenRouter responses - now with complete structure
            print(f"Debug [OAIGateway]: Response status: {resp.status_code}")
            print(f"Debug [OAIGateway]: Complete response structure:")
            import json
            print(json.dumps(data, indent=2, default=str)[:2000] + ("..." if len(str(data)) > 2000 else ""))
            
            # Detailed analysis of the response structure
            print(f"Debug [OAIGateway]: Response top-level keys: {list(data.keys())}")
            
            if "choices" in data:
                choices = data["choices"]
                print(f"Debug [OAIGateway]: Choices type: {type(choices)}, length: {len(choices) if isinstance(choices, list) else 'N/A'}")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    print(f"Debug [OAIGateway]: First choice complete structure:")
                    print(json.dumps(first_choice, indent=2, default=str))
                    print(f"Debug [OAIGateway]: First choice keys: {list(first_choice.keys()) if isinstance(first_choice, dict) else 'Not a dict'}")
                    
                    if isinstance(first_choice, dict) and "message" in first_choice:
                        message = first_choice["message"]
                        print(f"Debug [OAIGateway]: Message structure:")
                        print(json.dumps(message, indent=2, default=str))
                        print(f"Debug [OAIGateway]: Message keys: {list(message.keys()) if isinstance(message, dict) else 'Not a dict'}")
                        
                        if isinstance(message, dict):
                            content = message.get("content")
                            print(f"Debug [OAIGateway]: Content type: {type(content)}")
                            print(f"Debug [OAIGateway]: Content value: {repr(content)}")
                            print(f"Debug [OAIGateway]: Content length: {len(str(content)) if content is not None else 'None'}")
            else:
                print(f"Debug [OAIGateway]: No 'choices' key found in response")
            
            # Check for other possible response formats
            for key in ["completion", "text", "content", "output", "result", "response"]:
                if key in data:
                    print(f"Debug [OAIGateway]: Found alternate key '{key}': {type(data[key])}, value: {repr(data[key])[:200]}")
            
            # Special handling for OpenRouter-specific response formats
            if "openrouter.ai" in (self.base_url or ""):
                print(f"Debug [OAIGateway]: Detected OpenRouter, checking for specific response patterns...")
                
                # OpenRouter might wrap responses differently
                if "data" in data and isinstance(data["data"], dict):
                    print(f"Debug [OAIGateway]: Found OpenRouter 'data' wrapper: {list(data['data'].keys())}")
                    # Try extracting from data wrapper
                    inner_data = data["data"]
                    if "choices" in inner_data:
                        print(f"Debug [OAIGateway]: Using choices from data wrapper")
                        data = inner_data  # Use the inner data for extraction
                
                # Check for streaming-style response that got concatenated
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    choice = data["choices"][0]
                    if isinstance(choice, dict) and "delta" in choice:
                        print(f"Debug [OAIGateway]: Found streaming-style delta response")
                        delta = choice["delta"]
                        if isinstance(delta, dict) and "content" in delta:
                            print(f"Debug [OAIGateway]: Extracting content from delta: {repr(delta['content'])[:100]}")
            
            text, tool_obj = self._extract_text_and_tool(data)
            resolved_text = text if (tool_obj is None) else None
            
            print(f"Debug [OAIGateway]: Final extracted text: {repr(text)}")
            print(f"Debug [OAIGateway]: Final extracted text length: {len(text) if text else 0}")
            print(f"Debug [OAIGateway]: Final extracted tool_obj: {tool_obj}")
            print(f"Debug [OAIGateway]: Final resolved_text: {repr(resolved_text)}")
            
            return {
                "text": resolved_text,
                "tool_call_json": tool_obj,
                "raw": data,
                "usage": data.get("usage"),
                "status": resp.status_code,
                "error": None,
            }

        def error_out(status: int, body: str, which: str) -> Dict[str, Any]:
            snippet = body[:400] if isinstance(body, str) else str(body)[:400]
            return {
                "text": None,
                "tool_call_json": None,
                "raw": {"status": status, "body": body, "encoding": which},
                "usage": None,
                "status": status,
                "error": f"HTTP {status} during {which}: {snippet}",
            }

        try:
            return _send_with_max_variants(messages_a)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_text = None
            try:
                body_text = e.response.text
            except Exception:
                body_text = ""
            # Check if we should fallback to EncB
            if status in (400, 415, 422):
                bt = (body_text or "").lower()
                # Retry once with EncB (input_image) and re-run token key variants
                if ("image_url" in bt) or ("data:" in bt) or True:
                    enc_used = "EncB"
                    messages_b = payload_messages + [self._build_user_with_images_enc_b(user_text, image_paths)]
                    try:
                        return _send_with_max_variants(messages_b)
                    except httpx.HTTPStatusError as e2:
                        status2 = e2.response.status_code if hasattr(e2, "response") else 500
                        body2 = e2.response.text if hasattr(e2, "response") and hasattr(e2.response, "text") else ""
                        return error_out(status2, body2, f"{enc_used} with {'tools' if use_tools else ('json' if use_json_mode else 'prompt')}")
                    except httpx.TimeoutException:
                        return error_out(408, "timeout", f"{enc_used} with {'tools' if use_tools else ('json' if use_json_mode else 'prompt')}")
            # Not a fallback case; surface error
            return error_out(status, body_text or "", f"{enc_used} with {'tools' if use_tools else ('json' if use_json_mode else 'prompt')}")
        except httpx.TimeoutException:
            return error_out(408, "timeout", f"{enc_used} with {'tools' if use_tools else ('json' if use_json_mode else 'prompt')}")
