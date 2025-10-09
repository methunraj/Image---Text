# Configuration Flow Documentation

## Overview
This document verifies and documents the complete configuration flow from Excel to execution. All model parameters are loaded **DIRECTLY** from the Excel file with **NO transformations or overrides**.

## Configuration Source: Excel File

**File:** `config/models.xlsx`

**Structure:** 3 sheets
- `profiles` - Environment profiles (dev, staging, prod)
- `providers` - Provider-level configuration
- `models` - Model-level configuration

## Flow Diagram

```
Excel File (config/models.xlsx)
    ↓
parse_excel_config() → AppModelConfig
    ↓
LoadedConfig wrapper (adds secrets, profile, source_path)
    ↓
build_registry() → ModelRegistry with ModelDescriptors
    ↓
gateway_from_descriptor() → OAIGateway
    ↓
Execution (API calls)
```

## Verified Parameters (Excel → Gateway)

### Provider-Level Parameters
✅ **Base URL** - Loaded directly from Excel
✅ **Auth mode** - Loaded directly from Excel
✅ **Token source** - Loaded directly from Excel
✅ **Timeouts** (connect, read, total) - Loaded directly from Excel
✅ **Retry config** (max_retries, backoff_s) - Loaded directly from Excel

### Model-Level Parameters
✅ **Max output tokens** - Loaded directly from Excel
✅ **Default temperature** - Loaded directly from Excel
✅ **Max temperature** - Loaded directly from Excel
✅ **Default top_p** - Loaded directly from Excel
✅ **Context window** - Loaded directly from Excel
✅ **Force JSON mode** - Loaded directly from Excel
✅ **Prefer tools** - Loaded directly from Excel

### Compatibility Parameters
✅ **max_tokens_param** - Loaded directly from Excel (can be None for auto-detection)
✅ **allow_input_image_fallback** - Loaded directly from Excel

### Capabilities
✅ **JSON mode support** - Loaded directly from Excel
✅ **Tools support** - Loaded directly from Excel
✅ **Vision support** - Loaded directly from Excel

### Pricing
✅ **Input per million tokens** - Loaded directly from Excel
✅ **Output per million tokens** - Loaded directly from Excel

## Verification Results

**Test Script:** `trace_parameter_flow.py`

**Result:** ✅ ALL PARAMETERS MATCH!

All 12 critical parameters verified:
1. Base URL: ✅ Match
2. Max output tokens: ✅ Match
3. Default temperature: ✅ Match
4. Default top_p: ✅ Match
5. Max temperature: ✅ Match
6. Timeout total: ✅ Match
7. Max retries: ✅ Match
8. Backoff: ✅ Match
9. Compat max_tokens_param: ✅ Match
10. Allow input image fallback: ✅ Match
11. Force JSON mode: ✅ Match
12. Prefer tools: ✅ Match

## What IS Hardcoded (Not from Excel)

The following behavior is hardcoded in `provider_openai.py` but does NOT override Excel values:

1. **Parameter probing order** - The sequence of parameter names to try when auto-detecting
   - OpenAI: `max_tokens`, `max_completion_tokens`
   - Google: `max_tokens`, `max_completion_tokens`, `max_output_tokens`, `maxOutputTokens`
   - Anthropic: `max_tokens`

2. **Error detection logic** - How to detect parameter compatibility errors from API responses

3. **Vision fallback logic** - When to retry with different image encoding

4. **Provider detection logic** - How to identify Google OpenAI endpoint vs others

**Important:** These hardcoded behaviors implement the AUTO-DETECTION mechanism. They do NOT override Excel values - they only activate when Excel has `max_tokens_param = None`.

## Configuration Control

### Excel Controls (Direct Override)
When you set a value in Excel, it is used EXACTLY as specified:
- If you set `max_tokens_param = "maxOutputTokens"` → Gateway uses `maxOutputTokens`
- If you set `max_tokens_param = None` → Gateway auto-detects the correct parameter

### Auto-Detection (When Excel = None)
When Excel has `max_tokens_param = None`, the gateway will:
1. Try parameters in hardcoded order
2. Detect errors from API responses
3. Cache the working parameter for future use
4. Retry with different encodings if needed

## Conclusion

✅ **All model parameters are loaded DIRECTLY from Excel with no modifications**
✅ **The Excel file has complete control over the configuration**
✅ **Auto-detection only activates when Excel explicitly allows it (None value)**
✅ **No hidden overrides or transformations in the code**

The configuration system is **Excel-driven** with intelligent auto-detection as a fallback mechanism.
