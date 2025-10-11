from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

from app.core.config_loader import ConfigLoaderError, LoadedConfig, load_config
from app.core.config_schema import (
    AppModelConfig,
    AuthMode,
    CapabilitiesConfig,
    CompatibilityConfig,
    ModelConfig,
    PoliciesConfig,
    PricingConfig,
    ProviderConfig,
    ProviderType,
    ReasoningConfig,
    RouteType,
    TimeoutConfig,
    RetryConfig,
)


class ModelRegistryError(RuntimeError):
    pass


class MissingAuthTokenError(ModelRegistryError):
    """Raised when a provider is configured but no credential can be resolved."""

    def __init__(self, provider_id: str, source: str, ref: str | None) -> None:
        detail = f"{source}"
        if ref:
            detail = f"{source} reference '{ref}'"
        super().__init__(
            f"Missing credential for provider '{provider_id}' (expected {detail})."
        )
        self.provider_id = provider_id
        self.source = source
        self.ref = ref


@dataclass(frozen=True)
class ProviderDescriptor:
    id: str
    label: str
    type: ProviderType
    base_url: str
    timeouts: TimeoutConfig
    retry: RetryConfig
    auth_mode: AuthMode
    auth_header_name: Optional[str]
    auth_query_param: Optional[str]
    auth_token: Optional[str] = field(default=None, repr=False)
    auth_token_source: Optional[str] = None
    auth_token_ref: Optional[str] = None
    enabled: bool = True

    def auth_headers(self) -> Dict[str, str]:
        if self.auth_mode == "none" or not self.auth_token:
            return {}
        if self.auth_mode == "bearer_token":
            header = self.auth_header_name or "Authorization"
            return {header: f"Bearer {self.auth_token}"}
        if self.auth_mode == "header" and self.auth_header_name:
            return {self.auth_header_name: self.auth_token}
        return {}


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    provider_id: str
    provider_label: str
    label: str
    route: RouteType
    base_url: str
    provider_type: ProviderType
    context_window: int
    max_output_tokens: Optional[int]
    max_temperature: float
    default_temperature: float
    default_top_p: Optional[float]
    force_json_mode: bool
    prefer_tools: bool
    pricing: PricingConfig
    capabilities: CapabilitiesConfig
    compatibility: CompatibilityConfig
    timeouts: TimeoutConfig
    retry: RetryConfig
    provider_auth: ProviderDescriptor
    reasoning: ReasoningConfig
    show_in_ui: bool
    allow_frontend_override_temperature: bool
    allow_frontend_override_reasoning: bool

    def auth_headers(self) -> Dict[str, str]:
        return dict(self.provider_auth.auth_headers())

    def extra_headers(self) -> Dict[str, str]:
        return dict(self.compatibility.extra_headers or {})

    def requires_query_token(self) -> bool:
        return self.provider_auth.auth_mode == "query" and bool(self.provider_auth.auth_query_param)


@dataclass(frozen=True)
class ModelRegistry:
    profile: str
    source_path: str
    defaults_provider: str
    defaults_model: str
    policies: PoliciesConfig
    providers: Mapping[str, ProviderDescriptor]
    models: Mapping[str, ModelDescriptor]

    def resolve(self, model_id: Optional[str]) -> ModelDescriptor:
        if not model_id:
            return self.models[self.defaults_model]
        key = model_id
        if model_id not in self.models and ":" in model_id:
            key = model_id.split(":", 1)[-1]
        descriptor = self.models.get(key)
        if not descriptor:
            raise ModelRegistryError(f"Unknown model id '{model_id}'")
        return descriptor

    def get_provider(self, provider_id: str) -> ProviderDescriptor:
        provider = self.providers.get(provider_id)
        if not provider:
            raise ModelRegistryError(f"Unknown provider id '{provider_id}'")
        return provider

    def iter_models(self, capability: Optional[str] = None) -> Iterator[ModelDescriptor]:
        if capability is None:
            yield from self.models.values()
            return
        for descriptor in self.models.values():
            if getattr(descriptor.capabilities, capability, False):
                yield descriptor


@dataclass(frozen=True)
class _RegistryState:
    loaded: LoadedConfig
    registry: ModelRegistry


_STATE: Optional[_RegistryState] = None
_STATE_LOCK = threading.RLock()


def _resolve_token(
    provider: ProviderConfig,
    profile: str,
    secrets: Mapping[str, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    auth = provider.auth
    if auth.mode == "none":
        return None, None, None

    source = auth.token_source or "env"
    ref = auth.token_ref

    if source == "env":
        if not ref:
            raise ModelRegistryError(f"Provider '{provider.id}' auth requires token_ref for env lookup")
        token = os.getenv(ref)
        if not token:
            raise MissingAuthTokenError(provider.id, "environment variable", ref)
        return token, source, ref

    if source == "secret_ref":
        if not ref:
            raise ModelRegistryError(f"Provider '{provider.id}' auth requires token_ref for secret lookup")
        if ref not in secrets:
            raise MissingAuthTokenError(provider.id, "secret", ref)
        return secrets[ref], source, ref

    if source == "plain":
        if not ref:
            raise ModelRegistryError(f"Provider '{provider.id}' auth plain token requires token_ref value")
        if profile == "prod":
            warnings.warn(
                f"Provider '{provider.id}' uses plain tokens in production profile; consider secret_ref",
                RuntimeWarning,
                stacklevel=2,
            )
        return ref, source, None

    raise ModelRegistryError(f"Unsupported token source '{source}' for provider '{provider.id}'")


def _build_provider_descriptor(provider: ProviderConfig, profile: str, secrets: Mapping[str, str]) -> ProviderDescriptor:
    token, source, ref = _resolve_token(provider, profile, secrets)
    return ProviderDescriptor(
        id=provider.id,
        label=provider.label or provider.id,
        type=provider.type,
        base_url=provider.base_url.rstrip("/"),
        timeouts=provider.timeouts,
        retry=provider.retry,
        auth_mode=provider.auth.mode,
        auth_header_name=provider.auth.header_name,
        auth_query_param=provider.auth.query_param_name,
        auth_token=token,
        auth_token_source=source,
        auth_token_ref=ref,
        enabled=provider.enabled,
    )


def _build_model_descriptor(provider: ProviderDescriptor, provider_cfg: ProviderConfig, model_cfg: ModelConfig) -> ModelDescriptor:
    label = model_cfg.label or model_cfg.id
    return ModelDescriptor(
        id=model_cfg.id,
        provider_id=provider_cfg.id,
        provider_label=provider_cfg.label or provider_cfg.id,
        label=label,
        route=model_cfg.route,
        base_url=provider.base_url,
        provider_type=provider.type,
        context_window=model_cfg.context_window,
        max_output_tokens=model_cfg.max_output_tokens,
        max_temperature=model_cfg.max_temperature,
        default_temperature=model_cfg.default_temperature,
        default_top_p=model_cfg.default_top_p,
        force_json_mode=model_cfg.force_json_mode,
        prefer_tools=model_cfg.prefer_tools,
        pricing=model_cfg.pricing,
        capabilities=model_cfg.capabilities,
        compatibility=model_cfg.compatibility,
        timeouts=provider.timeouts,
        retry=provider.retry,
        provider_auth=provider,
        reasoning=model_cfg.reasoning,
        show_in_ui=model_cfg.show_in_ui,
        allow_frontend_override_temperature=model_cfg.allow_frontend_override_temperature,
        allow_frontend_override_reasoning=model_cfg.allow_frontend_override_reasoning,
    )


def build_registry(loaded: LoadedConfig) -> ModelRegistry:
    providers: Dict[str, ProviderDescriptor] = {}
    models: Dict[str, ModelDescriptor] = {}
    provider_models: Dict[str, List[ModelDescriptor]] = {}

    app_config: AppModelConfig = loaded.config

    missing_default_provider = False
    missing_default_model = False

    for provider_cfg in app_config.providers:
        if not provider_cfg.enabled:
            continue
        try:
            descriptor = _build_provider_descriptor(provider_cfg, loaded.profile, loaded.secrets)
        except MissingAuthTokenError as exc:
            warnings.warn(
                f"Skipping provider '{provider_cfg.id}' due to missing credentials: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if provider_cfg.id == app_config.defaults.provider:
                missing_default_provider = True
            continue
        providers[descriptor.id] = descriptor
        for model_cfg in provider_cfg.models:
            model_desc = _build_model_descriptor(descriptor, provider_cfg, model_cfg)
            models[model_desc.id] = model_desc
            provider_models.setdefault(descriptor.id, []).append(model_desc)
            if (
                provider_cfg.id == app_config.defaults.provider
                and model_cfg.id == app_config.defaults.model
            ):
                # Default model present once provider loads; mark as satisfied
                missing_default_model = False
        if (
            provider_cfg.id == app_config.defaults.provider
            and app_config.defaults.model not in models
        ):
            missing_default_model = True

    if not providers:
        raise ModelRegistryError(
            "No providers are available. Ensure at least one provider has valid credentials."
        )

    default_provider = app_config.defaults.provider
    default_model = app_config.defaults.model

    if default_provider not in providers or missing_default_provider:
        fallback_provider = next(iter(providers))
        fallback_models = provider_models.get(fallback_provider)
        if not fallback_models:
            raise ModelRegistryError(
                f"Provider '{fallback_provider}' has no models configured"
            )
        warnings.warn(
            (
                f"Default provider '{default_provider}' unavailable; "
                f"falling back to '{fallback_provider}:{fallback_models[0].id}'."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        default_provider = fallback_provider
        default_model = fallback_models[0].id
    elif missing_default_model or default_model not in models:
        fallback_models = provider_models.get(default_provider)
        if not fallback_models:
            raise ModelRegistryError(
                f"Provider '{default_provider}' has no models available for fallback"
            )
        warnings.warn(
            (
                f"Default model '{default_model}' unavailable; "
                f"falling back to '{default_provider}:{fallback_models[0].id}'."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        default_model = fallback_models[0].id

    return ModelRegistry(
        profile=loaded.profile,
        source_path=str(loaded.source_path),
        defaults_provider=default_provider,
        defaults_model=default_model,
        policies=app_config.policies,
        providers=providers,
        models=models,
    )


def set_registry(loaded: LoadedConfig) -> ModelRegistry:
    registry = build_registry(loaded)
    global _STATE
    with _STATE_LOCK:
        _STATE = _RegistryState(loaded=loaded, registry=registry)
    return registry


def ensure_registry() -> ModelRegistry:
    with _STATE_LOCK:
        if _STATE is not None:
            return _STATE.registry
    loaded = load_config()
    return set_registry(loaded)


def reload_registry(profile: Optional[str] = None) -> ModelRegistry:
    loaded = load_config(profile)
    return set_registry(loaded)


def current_registry() -> ModelRegistry:
    with _STATE_LOCK:
        if _STATE is None:
            raise ModelRegistryError("Registry has not been initialised")
        return _STATE.registry


def active_model() -> ModelDescriptor:
    return ensure_registry().resolve(None)


def config_metadata() -> Dict[str, str]:
    with _STATE_LOCK:
        if _STATE is None:
            raise ModelRegistryError("Registry has not been initialised")
        return {
            "profile": _STATE.loaded.profile,
            "source_path": str(_STATE.loaded.source_path),
        }
