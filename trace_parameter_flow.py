#!/usr/bin/env python3
"""
Trace the exact flow of parameter loading from Excel to execution.
This will show if parameters are loaded directly or if there's any transformation.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("STEP 1: Loading from Excel File")
print("=" * 80)

from app.core.config_excel import parse_excel_config

profile = os.getenv("APP_PROFILE", "dev")
excel_config = parse_excel_config(Path("config/models.xlsx"), profile)

print(f"Profile: {profile}")
print(f"Loaded {len(excel_config.providers)} providers from Excel\n")

# Find Google provider
google_provider = None
for provider in excel_config.providers:
    if 'google' in provider.id.lower():
        google_provider = provider
        print("Google Provider from Excel:")
        print(f"  ID: {provider.id}")
        print(f"  Base URL: {provider.base_url}")
        print(f"  Auth mode: {provider.auth.mode}")
        print(f"  Token source: {provider.auth.token_source}")
        print(f"  Timeouts: connect={provider.timeouts.connect_s}s, read={provider.timeouts.read_s}s, total={provider.timeouts.total_s}s")
        print(f"  Retry: max={provider.retry.max_retries}, backoff={provider.retry.backoff_s}s")

        # Show models
        print(f"\n  Models ({len(provider.models)}):")
        for model in provider.models:
            if 'flash' in model.id.lower():
                print(f"\n    Model: {model.id}")
                print(f"      Max output tokens: {model.max_output_tokens}")
                print(f"      Default temperature: {model.default_temperature}")
                print(f"      Max temperature: {model.max_temperature}")
                print(f"      Default top_p: {model.default_top_p}")
                print(f"      Context window: {model.context_window}")
                print(f"      Compatibility:")
                print(f"        - max_tokens_param: {model.compatibility.max_tokens_param}")
                print(f"        - allow_input_image_fallback: {model.compatibility.allow_input_image_fallback}")
                print(f"      Capabilities:")
                print(f"        - JSON mode: {model.capabilities.json_mode}")
                print(f"        - Tools: {model.capabilities.tools}")
                print(f"        - Vision: {model.capabilities.vision}")
                print(f"      Pricing:")
                print(f"        - Input per million: ${model.pricing.input_per_million}")
                print(f"        - Output per million: ${model.pricing.output_per_million}")
                break

print("\n" + "=" * 80)
print("STEP 2: Building Model Registry")
print("=" * 80)

from app.core.model_registry import build_registry
from app.core.config_loader import LoadedConfig

# Wrap excel_config in LoadedConfig for build_registry
loaded_config = LoadedConfig(
    config=excel_config,
    secrets=os.environ,
    profile=profile,
    source_path=Path("config/models.xlsx")
)

registry = build_registry(loaded_config)
print(f"Registry built with {len(registry.models)} models")

# Get a Google model descriptor
gemini_descriptor = None
for model_id, descriptor in registry.models.items():
    if 'gemini' in model_id.lower() and 'flash' in model_id.lower():
        gemini_descriptor = descriptor
        print(f"\nModel Descriptor for: {model_id}")
        print(f"  Provider ID: {descriptor.provider_id}")
        print(f"  Base URL: {descriptor.base_url}")
        print(f"  Max output tokens: {descriptor.max_output_tokens}")
        print(f"  Default temperature: {descriptor.default_temperature}")
        print(f"  Max temperature: {descriptor.max_temperature}")
        print(f"  Default top_p: {descriptor.default_top_p}")
        print(f"  Force JSON mode: {descriptor.force_json_mode}")
        print(f"  Prefer tools: {descriptor.prefer_tools}")
        print(f"  Compatibility max_tokens_param: {descriptor.compatibility.max_tokens_param}")
        print(f"  Provider auth token (masked): {'***' if descriptor.provider_auth.auth_token else 'None'}")
        break

print("\n" + "=" * 80)
print("STEP 3: Creating Gateway from Descriptor")
print("=" * 80)

from app.core.provider_openai import gateway_from_descriptor

gateway = gateway_from_descriptor(gemini_descriptor)

print(f"OAIGateway created:")
print(f"  Base URL: {gateway.base_url}")
print(f"  Timeout total: {gateway.timeouts.total_s}s")
print(f"  Max retries: {gateway.retry_config.max_retries}")
print(f"  Prefer JSON mode: {gateway.prefer_json_mode}")
print(f"  Prefer tools: {gateway.prefer_tools}")
print(f"  Default temperature: {gateway.default_temperature}")
print(f"  Default top_p: {gateway.default_top_p}")
print(f"  Max output tokens: {gateway.max_output_tokens}")
print(f"  Max temperature: {gateway.max_temperature}")
print(f"  Max tokens param override: {gateway.max_tokens_param_override}")
print(f"  Cached max tokens param: {gateway.cached_max_tokens_param}")
print(f"  Allow input image fallback: {gateway.allow_input_image_fallback}")
print(f"  Auth mode: {gateway.auth_mode}")
print(f"  Auth token (masked): {'***' if gateway.auth_token else 'None'}")

print("\n" + "=" * 80)
print("VERIFICATION: Direct Excel → Gateway Mapping")
print("=" * 80)

if google_provider:
    for model in google_provider.models:
        if 'flash' in model.id.lower():
            print("\nComparing Excel values to Gateway values:")

            checks = [
                ("Base URL", provider.base_url, gateway.base_url),
                ("Max output tokens", model.max_output_tokens, gateway.max_output_tokens),
                ("Default temperature", model.default_temperature, gateway.default_temperature),
                ("Default top_p", model.default_top_p, gateway.default_top_p),
                ("Max temperature", model.max_temperature, gateway.max_temperature),
                ("Timeout total", provider.timeouts.total_s, gateway.timeouts.total_s),
                ("Max retries", provider.retry.max_retries, gateway.retry_config.max_retries),
                ("Backoff", provider.retry.backoff_s, gateway.retry_config.backoff_s),
                ("Compat max_tokens_param", model.compatibility.max_tokens_param, gateway.max_tokens_param_override),
                ("Allow input image fallback", model.compatibility.allow_input_image_fallback, gateway.allow_input_image_fallback),
                ("Force JSON mode", model.force_json_mode, gateway.prefer_json_mode),
                ("Prefer tools", model.prefer_tools, gateway.prefer_tools),
            ]

            all_match = True
            for name, excel_val, gateway_val in checks:
                match = "✅" if excel_val == gateway_val else "❌"
                if excel_val != gateway_val:
                    all_match = False
                print(f"  {match} {name}: Excel={excel_val} | Gateway={gateway_val}")

            print(f"\n{'✅ ALL PARAMETERS MATCH!' if all_match else '❌ SOME PARAMETERS DIFFER!'}")
            print("\n📋 Conclusion:")
            if all_match:
                print("  ALL model parameters are loaded DIRECTLY from Excel with no modifications.")
                print("  The Excel file has complete control over the configuration.")
            else:
                print("  Some parameters may have transformations or defaults applied.")

            break
