from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, model_validator

ProviderType = Literal["openai_compatible", "ollama", "groq", "together", "custom"]
AuthMode = Literal["bearer_token", "header", "query", "none"]
TokenSource = Literal["env", "secret_ref", "plain"]
RouteType = Literal["chat_completions", "responses", "completions", "embeddings", "images"]


class PricingConfig(BaseModel):
    currency: str = Field(default="USD", min_length=1)
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    cache_read_per_1k: float | None = None
    cache_write_per_1k: float | None = None
    input_per_million: float | None = None
    output_per_million: float | None = None
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None

    @model_validator(mode="after")
    def _ensure_non_negative(self) -> "PricingConfig":
        for value in (
            self.input_per_1k,
            self.output_per_1k,
            self.cache_read_per_1k,
            self.cache_write_per_1k,
            self.input_per_million,
            self.output_per_million,
            self.cache_read_per_million,
            self.cache_write_per_million,
        ):
            if value is not None and value < 0:
                raise ValueError("pricing values must be non-negative")
        return self


class CapabilitiesConfig(BaseModel):
    json_mode: bool = False
    vision: bool = False
    audio_in: bool = False
    audio_out: bool = False
    tools: bool = False
    parallel_tool_calls: bool = False
    structured_output: bool = False
    streaming: bool = False
    embeddings: bool = False


class CompatibilityConfig(BaseModel):
    image_part_key: Optional[str] = None
    tool_schema_style: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    max_tokens_param: Optional[str] = None
    allow_input_image_fallback: bool = True


class RetryConfig(BaseModel):
    max_retries: int = Field(default=2, ge=0)
    backoff_s: float = Field(default=0.5, ge=0.0)
    retry_on: List[int] = Field(default_factory=list)


class TimeoutConfig(BaseModel):
    connect_s: float = Field(default=5.0, gt=0.0)
    read_s: float = Field(default=60.0, gt=0.0)
    total_s: float = Field(default=120.0, gt=0.0)


class AuthConfig(BaseModel):
    mode: AuthMode = "bearer_token"
    token_source: Optional[TokenSource] = "env"
    token_ref: Optional[str] = None
    header_name: Optional[str] = None
    query_param_name: Optional[str] = None

    @model_validator(mode="after")
    def _validate_with_mode(self) -> "AuthConfig":
        if self.mode == "none":
            return self
        if self.token_source is None:
            raise ValueError("token_source required when auth mode != none")
        if not self.token_ref:
            raise ValueError("token_ref is required when auth token source is set")
        if self.mode == "header" and not self.header_name:
            raise ValueError("header_name required when auth mode is header")
        if self.mode == "query" and not self.query_param_name:
            raise ValueError("query_param_name required when auth mode is query")
        return self


class ReasoningConfig(BaseModel):
    provider: Optional[str] = None
    effort_default: Optional[str] = None
    include_thoughts_default: bool = False
    allow_override: bool = True

    @model_validator(mode="after")
    def _normalize(self) -> "ReasoningConfig":
        if self.provider:
            provider_norm = self.provider.strip().lower()
            if provider_norm not in {"google", "openai", "custom", "none"}:
                raise ValueError("reasoning.provider must be one of google|openai|custom|none")
            object.__setattr__(self, "provider", provider_norm)
        if self.effort_default:
            object.__setattr__(self, "effort_default", self.effort_default.strip())
        return self


class ModelConfig(BaseModel):
    id: str = Field(min_length=1)
    label: Optional[str] = None
    route: RouteType
    context_window: PositiveInt
    max_output_tokens: Optional[int] = Field(default=None, ge=0)
    max_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    force_json_mode: bool = False
    prefer_tools: bool = False
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    compatibility: CompatibilityConfig = Field(default_factory=CompatibilityConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    show_in_ui: bool = True
    allow_frontend_override_temperature: bool = True
    allow_frontend_override_reasoning: bool = True

    @model_validator(mode="after")
    def _temp_within_limits(self) -> "ModelConfig":
        if self.default_temperature > self.max_temperature:
            raise ValueError("default_temperature cannot exceed max_temperature")
        return self


class ProviderConfig(BaseModel):
    id: str = Field(min_length=1)
    label: Optional[str] = None
    type: ProviderType
    base_url: str = Field(min_length=1)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    models: List[ModelConfig]
    enabled: bool = True

    @model_validator(mode="after")
    def _require_models(self) -> "ProviderConfig":
        if not self.models:
            raise ValueError("provider must define at least one model")
        seen: set[str] = set()
        for mdl in self.models:
            if mdl.id in seen:
                raise ValueError(f"duplicate model id '{mdl.id}' for provider '{self.id}'")
            seen.add(mdl.id)
        return self


class RedactionConfig(BaseModel):
    enabled: bool = False
    fields: List[str] = Field(default_factory=list)


class PoliciesConfig(BaseModel):
    max_concurrent_requests: int = Field(default=4, ge=1)
    allow_frontend_model_selection: bool = False
    allow_frontend_temperature: bool = False
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)


class DefaultsConfig(BaseModel):
    provider: str
    model: str


class AppModelConfig(BaseModel):
    version: int = 1
    profile: Optional[str] = None
    defaults: DefaultsConfig
    policies: PoliciesConfig = Field(default_factory=PoliciesConfig)
    providers: List[ProviderConfig]

    @model_validator(mode="after")
    def _validate_defaults(self) -> "AppModelConfig":
        if not self.providers:
            raise ValueError("configuration must define at least one provider")

        provider_ids = {p.id for p in self.providers}
        if self.defaults.provider not in provider_ids:
            raise ValueError("defaults.provider must reference a defined provider")

        target_models: Iterable[str] = (
            mdl.id
            for provider in self.providers
            if provider.id == self.defaults.provider
            for mdl in provider.models
        )
        if self.defaults.model not in set(target_models):
            raise ValueError("defaults.model must exist within the default provider models")

        if len(provider_ids) != len(self.providers):
            raise ValueError("provider ids must be unique")

        global_model_ids: set[str] = set()
        for provider in self.providers:
            for mdl in provider.models:
                if mdl.id in global_model_ids:
                    raise ValueError(f"model id '{mdl.id}' reused across providers; use unique ids")
                global_model_ids.add(mdl.id)

        return self
