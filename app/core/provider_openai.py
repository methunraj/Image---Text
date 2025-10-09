from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from urllib.parse import urlencode

import httpx

from app.core.config_schema import ReasoningConfig, RetryConfig, TimeoutConfig
from app.core.model_registry import ModelDescriptor


@lru_cache(maxsize=32)
def _get_file_hash(path: str) -> str:
    """Get a hash of file path and modification time for cache invalidation."""
    try:
        stat = os.stat(path)
        # Use a separator that won't appear in file paths
        return f"{path}|||{stat.st_mtime}|||{stat.st_size}"
    except:
        # Still use separator even on error for consistency
        return f"{path}|||0|||0"

@lru_cache(maxsize=32)
def encode_image_to_data_url_cached(cache_key: str) -> str:
    """Cached version of encode_image_to_data_url."""
    # Extract actual path from cache key (split on ||| separator)
    path = cache_key.split('|||')[0]
    return _encode_image_to_data_url_uncached(path)

def _encode_image_to_data_url_uncached(path: str) -> str:
    """Internal uncached image encoding."""
    mime = "image/png"
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg", ".jfif")):
        mime = "image/jpeg"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    elif lower.endswith(".bmp"):
        mime = "image/bmp"
    elif lower.endswith((".tif", ".tiff")):
        mime = "image/tiff"
    elif lower.endswith(".heic"):
        mime = "image/heic"
    elif lower.endswith(".heif"):
        mime = "image/heif"
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
        return tiny_png_data_url()

def encode_image_to_data_url(path: str) -> str:
    """Encode a local image file into a data URL with caching."""
    cache_key = _get_file_hash(path)
    return encode_image_to_data_url_cached(cache_key)


def tiny_png_data_url() -> str:
    """Return a 1x1 transparent PNG data URL, useful for probes."""
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="
    return f"data:image/png;base64,{b64}"


@lru_cache(maxsize=32)
def encode_image_to_b64_cached(cache_key: str) -> Tuple[str, str]:
    """Cached version of encode_image_to_b64."""
    # Extract actual path from cache key (split on ||| separator)
    path = cache_key.split('|||')[0]
    return _encode_image_to_b64_uncached(path)

def _encode_image_to_b64_uncached(path: str) -> Tuple[str, str]:
    """Internal uncached b64 encoding."""
    mime = "image/png"
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg", ".jfif")):
        mime = "image/jpeg"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    elif lower.endswith(".bmp"):
        mime = "image/bmp"
    elif lower.endswith((".tif", ".tiff")):
        mime = "image/tiff"
    elif lower.endswith(".heic"):
        mime = "image/heic"
    elif lower.endswith(".heif"):
        mime = "image/heif"
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            return b64, mime
    except Exception:
        # transparent 1x1 png
        tiny_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAukB9jQ9a6EAAAAASUVORK5CYII="
        return tiny_b64, "image/png"

def encode_image_to_b64(path: str) -> Tuple[str, str]:
    """Return (base64_str, mime) for a local image path with caching."""
    cache_key = _get_file_hash(path)
    return encode_image_to_b64_cached(cache_key)


@dataclass
class OAIGateway:
    """OpenAI-compatible gateway that works with registry-provided descriptors."""

    base_url: str
    timeouts: TimeoutConfig
    retry_config: RetryConfig
    prefer_json_mode: bool
    prefer_tools: bool
    default_temperature: float
    default_top_p: Optional[float]
    max_output_tokens: Optional[int]
    max_temperature: float
    headers: Optional[Dict[str, str]] = None
    capabilities: Optional[Dict[str, Any]] = None
    max_tokens_param_override: Optional[str] = None
    cached_max_tokens_param: Optional[str] = None
    allow_input_image_fallback: bool = True
    provider_id: Optional[str] = None
    auth_mode: str = "bearer_token"
    auth_token: Optional[str] = None
    auth_header_name: Optional[str] = None
    auth_query_param: Optional[str] = None
    reasoning: Optional[ReasoningConfig] = None
    def _is_local_model(self) -> bool:
        """Check if this is a local model endpoint."""
        if not self.base_url:
            return False
        
        url_lower = self.base_url.lower()
        
        # Check for local indicators - be more precise with IP ranges
        local_indicators = [
            "localhost", "127.0.0.1", "0.0.0.0",
            "192.168.",  # Private range C
            "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", 
            "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", 
            "172.30.", "172.31.",  # Private range B (172.16.0.0 - 172.31.255.255)
            "host.docker.internal",  # Docker
            ".local",  # mDNS
            ".lan",  # Local network
            ":1234", ":5000", ":5001", ":8000", ":8080", ":8888", ":9000",  # Common local ports
            ":11434",  # Ollama default
            ":7860", ":7861",  # Gradio defaults
            "lm-studio", "lmstudio", "ollama", "localai", "kobold"  # Known local servers
        ]
        
        # Special handling for 10.x.x.x range (must be more precise)
        if "10." in url_lower:
            # Check if it's really 10.x.x.x and not 100.x or 210.x etc
            import re
            # Match IP addresses starting with 10.
            if re.search(r'(?:^|[^0-9])10\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
                return True
        
        return any(indicator in url_lower for indicator in local_indicators)

    def _is_google_openai(self) -> bool:
        """Detect Google's OpenAI‑compatible endpoint (Generative Language)."""
        base = (self.base_url or "").lower()
        # Check for both /openai and /openai/ to handle URLs with or without trailing slash
        return "generativelanguage.googleapis.com" in base and "/openai" in base

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.headers:
            h.update(self.headers)

        if self.auth_mode == "bearer_token" and self.auth_token:
            header_name = self.auth_header_name or "Authorization"
            value = self.auth_token
            if header_name.lower() == "authorization" and not value.lower().startswith("bearer "):
                value = f"Bearer {value}"
            h[header_name] = value
        elif self.auth_mode == "header" and self.auth_token and self.auth_header_name:
            h[self.auth_header_name] = self.auth_token

        # Note: OpenRouter requires HTTP-Referer and X-Title headers.
        # These should be configured in Excel via extra_headers or passed in self.headers.
        # No hardcoded defaults to enforce explicit configuration.
        return h
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive information in debug output."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key.lower() in ('authorization', 'api_key', 'api-key', 'x-api-key', 'apikey'):
                    if isinstance(value, str) and len(value) > 8:
                        masked[key] = value[:4] + '***' + value[-4:]
                    else:
                        masked[key] = '***'
                elif isinstance(value, (dict, list)):
                    masked[key] = self._mask_sensitive_data(value)
                else:
                    masked[key] = value
            return masked
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        return data

    def _client(self) -> httpx.Client:
        timeout = httpx.Timeout(
            self.timeouts.total_s,
            connect=self.timeouts.connect_s,
            read=self.timeouts.read_s,
            write=self.timeouts.read_s,
        )
        return httpx.Client(timeout=timeout)

    def _post_with_retries(self, url: str, json_payload: Dict[str, Any], max_retries: Optional[int] = None) -> httpx.Response:
        # Only show verbose debug on first attempt or if VERBOSE_DEBUG is set
        verbose = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes")

        if verbose:
            print(f"Debug [API Request]: Making request to {url}")
            print(f"Debug [API Request]: Model: {json_payload.get('model', 'N/A')}")
            print(f"Debug [API Request]: Messages count: {len(json_payload.get('messages', []))}")
            
            # Show payload structure with sensitive data masked
            import json
            masked_headers = self._mask_sensitive_data(self._headers())
            print(f"Debug [API Request]: Headers (masked): {masked_headers}")
            
            # Don't show full payload to avoid leaking prompts with potential sensitive data
            payload_keys = list(json_payload.keys())
            print(f"Debug [API Request]: Payload keys: {payload_keys}")

        retries = max_retries if max_retries is not None else max(self.retry_config.max_retries, 0)
        retries = max(1, retries)
        delay = self.retry_config.backoff_s or 0.5
        last_exc: Optional[Exception] = None
        retryable = set(self.retry_config.retry_on or [429, 500, 502, 503, 504])

        for attempt in range(1, retries + 1):
            # Only show attempt info if it's a retry (attempt > 1) or there's an error
            try:
                with self._client() as client:
                    request_url = url
                    if self.auth_mode == "query" and self.auth_token and self.auth_query_param:
                        query = urlencode({self.auth_query_param: self.auth_token})
                        request_url = f"{url}{'&' if '?' in url else '?'}{query}"

                    resp = client.post(request_url, headers=self._headers(), json=json_payload)

                    # Check for retryable errors
                    if resp.status_code in retryable:
                        error_body = resp.text if hasattr(resp, 'text') else 'No body'
                        print(f"⚠️ Retryable error {resp.status_code} on attempt {attempt}/{retries}")
                        if verbose:
                            print(f"   Error body: {error_body[:200]}")
                        last_exc = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
                        raise last_exc

                    resp.raise_for_status()
                    
                    # Success - only show if it was a retry
                    if attempt > 1:
                        print(f"✅ Request successful after {attempt} attempts")
                    elif verbose:
                        print(f"Debug [API Request]: Response status: {resp.status_code}")
                    
                    return resp

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                # Always show errors
                if isinstance(e, httpx.TimeoutException):
                    print(f"⏱️ Timeout on attempt {attempt}/{retries}")
                elif not (hasattr(e, 'response') and e.response.status_code in retryable):
                    # Non-retryable HTTP error
                    print(f"❌ HTTP Error on attempt {attempt}: {e}")
                    if verbose and hasattr(e, 'response'):
                        print(f"   Response: {e.response.text[:200] if hasattr(e.response, 'text') else 'No body'}")

                last_exc = e
                if attempt >= retries:
                    break

                print(f"   Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 4.0)

        assert last_exc is not None
        print(f"❌ All {retries} attempts failed")
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
        """Extract text and tool call from OpenAI-compatible response.
        
        Supports standard OpenAI format: choices[0].message.content + tool_calls
        """
        text: Optional[str] = None
        tool_call: Optional[Dict[str, Any]] = None
        
        try:
            # Standard OpenAI format: choices[0].message
            choices = data.get("choices", [])
            if not choices:
                # Fallback: check top-level keys for non-standard providers
                for key in ["text", "content", "completion"]:
                    if key in data and isinstance(data[key], str):
                        return data[key], None
                return None, None
            
            choice = choices[0]
            if not isinstance(choice, dict):
                return None, None
            
            # Extract from message
            msg = choice.get("message", {})
            if isinstance(msg, dict):
                # Text content
                content = msg.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    # Content as list of parts (OpenAI format)
                    texts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    text = "\n".join(texts) if texts else None
                
                # Tool calls
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    func = tool_calls[0].get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            tool_call = json.loads(args)
                        except:
                            pass
                    elif isinstance(args, dict):
                        tool_call = args
            
            # Fallback: check for text directly in choice
            if text is None and "text" in choice:
                text = choice["text"]
                
        except Exception:
            pass
        
        return text, tool_call

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
        effective_params = dict(gen_params or params or {})

        # Temperature defaults and clamping
        temp = effective_params.get("temperature", self.default_temperature)
        try:
            temp_val = float(temp)
        except (TypeError, ValueError):
            temp_val = self.default_temperature
        temp_val = max(0.0, min(temp_val, self.max_temperature))
        effective_params["temperature"] = temp_val

        # Top-p defaults and clamping
        if "top_p" not in effective_params and self.default_top_p is not None:
            effective_params["top_p"] = self.default_top_p
        if "top_p" in effective_params:
            try:
                top_p_val = float(effective_params["top_p"])
            except (TypeError, ValueError):
                top_p_val = 1.0
            top_p_val = max(0.0, min(top_p_val, 1.0))
            effective_params["top_p"] = top_p_val

        # Max tokens defaults and clamping
        max_tokens_val = effective_params.get("max_tokens")
        limit = self.max_output_tokens if self.max_output_tokens not in (None, 0) else None
        if max_tokens_val is None and limit is not None:
            effective_params["max_tokens"] = limit
            max_tokens_val = limit
        elif max_tokens_val is not None:
            try:
                max_tokens_val = int(max_tokens_val)
            except (TypeError, ValueError):
                max_tokens_val = limit
            if limit is not None and max_tokens_val is not None:
                max_tokens_val = min(max_tokens_val, limit)
            if max_tokens_val is not None:
                effective_params["max_tokens"] = max_tokens_val

        extra_body_payload: Dict[str, Any] = {}
        raw_extra_body = effective_params.pop("extra_body", None)
        if isinstance(raw_extra_body, dict):
            extra_body_payload = dict(raw_extra_body)

        reasoning_payload: Optional[Dict[str, Any]] = None
        if self.reasoning and self.reasoning.provider and self.reasoning.provider != "none":
            provider_key = self.reasoning.provider

            # Respect user-provided overrides when allowed
            user_reasoning_effort = effective_params.get("reasoning_effort")

            if provider_key == "openai":
                if (
                    "reasoning" not in effective_params
                    and not reasoning_payload
                    and self.reasoning.effort_default
                    and not user_reasoning_effort
                ):
                    reasoning_payload = {"effort": self.reasoning.effort_default}
            elif provider_key == "google":
                if self.reasoning.effort_default and not user_reasoning_effort:
                    effective_params.setdefault("reasoning_effort", self.reasoning.effort_default)
                google_section = extra_body_payload.get("google")
                if isinstance(google_section, dict):
                    google_section.pop("thinking_config", None)
            # allow_override False -> enforce defaults
            if self.reasoning.allow_override is False:
                if provider_key == "openai" and self.reasoning.effort_default:
                    reasoning_payload = {"effort": self.reasoning.effort_default}
                    effective_params.pop("reasoning_effort", None)
                if provider_key == "google":
                    if self.reasoning.effort_default:
                        effective_params["reasoning_effort"] = self.reasoning.effort_default
                    google_section = extra_body_payload.get("google")
                    if isinstance(google_section, dict):
                        google_section.pop("thinking_config", None)

        # Keep only universal params here; handle max token key via compatibility layer below
        base_payload: Dict[str, Any] = {
            k: v for k, v in effective_params.items() if k in {"temperature", "top_p", "seed"}
        }
        additional_payload: Dict[str, Any] = {
            k: v
            for k, v in effective_params.items()
            if k not in {"temperature", "top_p", "seed", "max_tokens"}
        }

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

        # Decide modes - be more conservative with local models
        caps = self.capabilities or {}
        is_local = self._is_local_model()

        if is_local and not caps:
            tools_ok = False
            json_ok = False
        else:
            tools_ok = bool(caps.get("tools"))
            json_ok = bool(caps.get("json_mode")) or bool(caps.get("structured_output"))

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
            # Check if this is a local model (more comprehensive detection)
            local = self._is_local_model()
            
            if local:
                # Comprehensive list for ALL local model servers (deduplicated)
                # Using list to preserve order, but convert to dict first to remove duplicates
                params = []
                seen = set()
                
                candidates = [
                    # Most common first
                    "n_predict",  # LM Studio, llama.cpp
                    "max_tokens",  # OpenAI compatible
                    "max_new_tokens",  # Text Generation WebUI, TGI
                    "max_length",  # Multiple servers
                    # Ollama specific
                    "num_predict", "num_ctx",
                    # Other variants
                    "truncation_length",  # Oobabooga
                    "max_tokens_to_sample",  # LocalAI, Anthropic style
                    "max_gen_tokens",  # LocalAI
                    "prediction_length",  # Alternative llama.cpp
                    "max_completion_tokens",  # vLLM
                    "max_context_length",  # KoboldAI
                    "max_gen_len", "max_seq_len",  # Alpaca
                    # Generic fallbacks
                    "maximum_tokens", "response_length", "output_length",
                    "completion_tokens", "generate_length"
                ]
                
                for param in candidates:
                    if param not in seen:
                        params.append(param)
                        seen.add(param)
                
                return params
            
            # Cloud providers - ordered by likelihood
            return [
                # OpenAI / OpenAI-compatible
                "max_tokens", "max_completion_tokens",
                # Anthropic Claude
                "max_tokens_to_sample", "max_tokens",
                # Google (Gemini, PaLM)
                # When using OpenAI compatibility, prefer OpenAI-style params
                "max_tokens", "max_completion_tokens",  # Try OpenAI params first
                "max_output_tokens", "maxOutputTokens",  # Then Google native params
                # Cohere
                "max_tokens", "max_output_tokens",
                # Azure OpenAI
                "max_tokens", "maxTokens",
                # AWS Bedrock
                "maxTokenCount", "max_tokens_to_sample", "max_gen_len",
                # Replicate
                "max_tokens", "max_new_tokens", "max_length",
                # Together AI
                "max_tokens", "max_new_tokens",
                # Perplexity
                "max_tokens", "max_completion_tokens",
                # DeepSeek
                "max_tokens", "max_new_tokens",
                # Mistral
                "max_tokens", "maxTokens",
                # Generic fallbacks
                "n_predict", "response_max_tokens"
            ]


        def _send_with_max_variants(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Get verbose flag for debug output
            verbose = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes")

            # Use persisted cached parameter if available
            if self.max_tokens_param_override and not self.cached_max_tokens_param:
                self.cached_max_tokens_param = self.max_tokens_param_override
            cached_param = self.cached_max_tokens_param

            # Provider-specific correction: Google's OpenAI-compatible API expects
            # OpenAI-style params when in compatibility mode. Clear native Google params.
            if self._is_google_openai() and cached_param in ("maxOutputTokens", "max_output_tokens"):
                print("⚙️ Google OpenAI-compatible endpoint detected; clearing native parameter '" + str(cached_param) + "' to probe OpenAI-style params.")
                cached_param = None
                self.cached_max_tokens_param = None

            tried: List[str] = []
            last_exc: Optional[httpx.HTTPStatusError] = None
            failed_with_none = False

            # Build ordered candidates
            if cached_param and max_tokens_val is not None:
                # Use cached successful parameter first, then try None if it fails
                order = [cached_param, None]  # type: ignore[list-item]
                print(f"🔄 Using cached max_tokens parameter: '{cached_param}'")
            elif max_tokens_val is not None:
                # Get comprehensive parameter list and add None as final fallback
                keys = _keys_order()
                # Limit attempts: 8 for local (they have more variants), 5 for cloud
                is_local = self._is_local_model()
                limit = 8 if is_local else 5
                order = keys[:limit] + [None]  # Add None as final fallback
                if not cached_param:
                    print(f"🔍 Testing max_tokens parameter compatibility for {'LOCAL MODEL' if is_local else 'API endpoint'}...")
            else:
                order = [None]  # type: ignore[list-item]
            
            # Prevent infinite loops - max attempts = actual list length
            max_attempts = len(order)
            attempts = 0
            
            for i, key in enumerate(order):
                if attempts >= max_attempts:
                    print(f"⚠️ Reached maximum attempts ({max_attempts}), stopping parameter testing")
                    break
                attempts += 1
                
                # Build payload with only one max token key (or none)
                payload: Dict[str, Any] = {"model": model, "messages": messages, **base_payload}
                if key:  # type: ignore[truthy-bool]
                    payload[key] = max_tokens_val
                    tried.append(key)
                    # Only show parameter testing for uncached attempts
                    if not cached_param and len(tried) <= 3:  # Only show first 3 attempts
                        print(f"   Testing parameter {len(tried)}: '{key}'")
                else:
                    if verbose:
                        print(f"   Testing without max_tokens parameter")
                
                payload.update(additional_payload)
                payload = attach_modes(payload)
                if reasoning_payload and "reasoning" not in payload:
                    payload["reasoning"] = reasoning_payload
                if extra_body_payload and "extra_body" not in payload:
                    payload["extra_body"] = extra_body_payload

                try:
                    # Use reduced retries for parameter testing
                    test_retries = 1 if len(tried) > 1 else 2
                    resp = self._post_with_retries(url, payload, max_retries=test_retries)
                    
                    # Cache successful parameter for future use
                    if key and max_tokens_val is not None and not cached_param:
                        self.cached_max_tokens_param = key
                        print(f"✅ Parameter '{key}' works - caching for future requests")
                    elif key is None and max_tokens_val is not None:
                        print(f"ℹ️ Model works without max_tokens parameter")
                    
                    return result_ok(resp)
                    
                except httpx.HTTPStatusError as e_send:
                    last_exc = e_send
                    status = e_send.response.status_code if hasattr(e_send, "response") else None
                    body = ""
                    try:
                        body = e_send.response.text[:500]  # Limit body size
                    except Exception:
                        pass
                    low = (body or "").lower()
                    
                    # Check if this is a parameter compatibility issue
                    param_error_keywords = [
                        "unsupported parameter", "unknown parameter", "is not supported",
                        "invalid_request_error", "unrecognized parameter", "invalid parameter",
                        "not a valid parameter", "unexpected parameter", "extra parameter",
                        # Google's generic error messages
                        "invalid argument", "invalid value", "bad request",
                        "request contains an invalid argument"
                    ]
                    
                    is_param_error = status in (400, 422) and any(kw in low for kw in param_error_keywords)
                    # Treat Google OpenAI 400s with any max_tokens variant as parameter errors even if message is generic
                    if status in (400, 422) and self._is_google_openai() and key:
                        # Google often returns generic errors for parameter issues
                        is_param_error = True
                    
                    if is_param_error and key:
                        if cached_param == key:
                            # Cached param no longer works, clear cache
                            print(f"⚠️ Cached parameter '{key}' no longer works, clearing cache")
                            self.cached_max_tokens_param = None
                            cached_param = None
                        # Don't print for every failed attempt
                        if len(tried) <= 3 or verbose:
                            print(f"   ❌ Parameter '{key}' not supported")
                        continue
                    elif is_param_error and key is None:
                        # Even None (no parameter) failed with parameter error
                        failed_with_none = True
                        print(f"❌ API rejected request even without max_tokens parameter")
                        break
                    
                    # For non-parameter errors, don't continue testing
                    if status and status >= 500:
                        print(f"❌ Server error ({status}), stopping parameter testing")
                        raise
                    elif status in (401, 403):
                        print(f"❌ Authentication error ({status})")
                        raise
                    elif status == 429:
                        print(f"⚠️ Rate limited, stopping parameter testing")
                        raise
                    
                    # Unknown error, try next parameter but limit attempts
                    if len(tried) >= 5 and not key:  # If we've tried 5+ params and now trying None
                        print(f"❌ Too many failed attempts, stopping")
                        raise
            
            # If we exhausted all variants, provide helpful error message
            if last_exc is not None:
                if failed_with_none:
                    raise httpx.HTTPError(f"API incompatible - rejected all {len(tried)} parameter variants and no-parameter request")
                elif tried:
                    print(f"❌ None of {len(tried)} max_tokens parameters worked. Tried: {', '.join(tried[:5])}{'...' if len(tried) > 5 else ''}")
                raise last_exc
            
            # This should never happen, but handle it gracefully
            raise httpx.HTTPError("No request could be sent (no variants to try)")

        # Try EncA first
        enc_used = "EncA"
        messages_a = payload_messages + [self._build_user_with_images_enc_a(user_text, image_paths)]

        def result_ok(resp: httpx.Response) -> Dict[str, Any]:
            # First, capture the raw response body before JSON parsing
            raw_body = resp.text
            verbose = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes")
            if not verbose:
                def _noop(*args, **kwargs):
                    return None
                # Shadow print locally for non-verbose mode
                print = _noop  # type: ignore
            
            if verbose:
                print(f"Debug [Response]: RAW HTTP Response Body:")
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
            if status in (400, 415, 422) and self.allow_input_image_fallback:
                # Always retry with EncB on these status codes to re-probe parameters
                # This helps with both image encoding issues AND parameter compatibility
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


def gateway_from_descriptor(descriptor: ModelDescriptor) -> OAIGateway:
    capabilities = descriptor.capabilities.model_dump()
    compatibility = descriptor.compatibility
    max_tokens_param = compatibility.max_tokens_param
    allow_input_image = compatibility.allow_input_image_fallback
    return OAIGateway(
        base_url=descriptor.base_url,
        timeouts=descriptor.timeouts,
        retry_config=descriptor.retry,
        prefer_json_mode=descriptor.force_json_mode,
        prefer_tools=descriptor.prefer_tools,
        default_temperature=descriptor.default_temperature,
        default_top_p=descriptor.default_top_p,
        max_output_tokens=descriptor.max_output_tokens,
        max_temperature=descriptor.max_temperature,
        headers=descriptor.extra_headers(),
        capabilities=capabilities,
        max_tokens_param_override=max_tokens_param,
        cached_max_tokens_param=max_tokens_param,
        allow_input_image_fallback=allow_input_image,
        provider_id=descriptor.provider_id,
        auth_mode=descriptor.provider_auth.auth_mode,
        auth_token=descriptor.provider_auth.auth_token,
        auth_header_name=descriptor.provider_auth.auth_header_name,
        auth_query_param=descriptor.provider_auth.auth_query_param,
        reasoning=descriptor.reasoning,
    )
