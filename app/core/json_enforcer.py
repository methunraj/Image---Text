from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator, ValidationError


@dataclass
class JSONEnforceResult:
    valid: bool
    repaired: bool
    data: Optional[Dict[str, Any]]
    errors: List[str]


def _try_parse_json(s: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def strip_code_fences(s: str) -> str:
    """Remove surrounding triple backtick fences or inline backticks from a string."""
    if not isinstance(s, str):
        return s
    txt = s.strip()
    if txt.startswith("```"):
        parts = txt.splitlines()
        if parts:
            parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        return "\n".join(parts)
    # strip single leading/trailing backticks
    if txt.startswith("`") and txt.endswith("`") and len(txt) >= 2:
        return txt[1:-1]
    return txt


def _error_report(errors: List[ValidationError]) -> List[str]:
    lines: List[str] = []
    for e in errors:
        # Build JSON Pointer-like path
        path = "/" + "/".join([str(p) for p in e.path]) if list(e.path) else "/"
        lines.append(f"{path}: {e.message}")
    return lines


def validate_and_repair(
    raw_text_or_tool_json: str | Dict[str, Any],
    schema: Dict[str, Any],
    retry_fn: Callable[[str], str | Dict[str, Any]],
    max_retries: int = 2,
    circuit_breaker: bool = True,
) -> Tuple[Optional[Dict[str, Any]], List[str], int]:
    """Validate JSON against a schema with a repair loop and circuit breaker.

    Returns: (json or None, error_messages, attempts_used)
    attempts_used is the number of retry_fn calls performed (0..max_retries).
    
    Circuit breaker: If the same errors repeat twice, stop trying to avoid loops.
    """
    validator = Draft202012Validator(schema)

    # Normalize initial candidate
    if isinstance(raw_text_or_tool_json, dict):
        current_obj: Optional[Dict[str, Any]] = raw_text_or_tool_json
        current_text: Optional[str] = None
    else:
        current_obj = None
        current_text = strip_code_fences(raw_text_or_tool_json)

    attempts = 0
    last_errors: List[str] = []
    previous_errors: List[str] = []
    repeated_error_count = 0

    def parse_if_needed() -> Optional[Dict[str, Any]]:
        nonlocal current_obj, current_text
        if current_obj is not None:
            return current_obj
        if current_text is None:
            return None
        return _try_parse_json(current_text)

    while True:
        candidate = parse_if_needed()
        if candidate is not None:
            try:
                validator.validate(candidate)
                return candidate, [], attempts
            except ValidationError:
                errs = list(validator.iter_errors(candidate))
                last_errors = _error_report(errs)
        else:
            # Not parseable
            last_errors = ["/ : Invalid JSON or could not parse text."]

        # Circuit breaker: Check if errors are repeating
        if circuit_breaker and last_errors == previous_errors:
            repeated_error_count += 1
            if repeated_error_count >= 2:
                last_errors.append("/ : Circuit breaker triggered - same errors repeating")
                break
        else:
            repeated_error_count = 0
            previous_errors = last_errors.copy()

        if attempts >= max_retries:
            break

        # Build repair instruction text
        instruct = (
            "Please return ONLY a JSON object that fixes these validation errors. "
            "Do not include explanations or code fences. "
            "Do not add fields not defined in the schema. "
            "Error report:\n- "
            + "\n- ".join(last_errors)
        )

        attempts += 1
        try:
            result = retry_fn(instruct)
        except Exception as e:
            last_errors = last_errors + [f"/ : retry_fn failed: {e}"]
            break

        if isinstance(result, dict):
            current_obj = result
            current_text = None
        else:
            current_text = strip_code_fences(str(result))
            current_obj = None

    # Final best-effort parse
    final = parse_if_needed()
    return final, last_errors, attempts


# Backward-compatible helper used in Run & Test page
def ensure_valid_json(raw: str, schema: Dict[str, Any], max_attempts: int = 3) -> JSONEnforceResult:
    attempts = 0
    repaired = False
    errors: List[str] = []
    validator = Draft202012Validator(schema)
    current = raw

    def _extract_braced(s: str) -> str:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return s

    while attempts < max_attempts:
        attempts += 1
        parsed = _try_parse_json(current)
        if parsed is None:
            if attempts == 1:
                current = strip_code_fences(current)
                repaired = True
                continue
            if attempts == 2:
                current = _extract_braced(current)
                repaired = True
                continue
            parsed = _try_parse_json(current)

        if parsed is not None:
            try:
                validator.validate(parsed)
                return JSONEnforceResult(valid=True, repaired=repaired, data=parsed, errors=[])
            except ValidationError as ve:
                errors.append(str(ve))
                break

    return JSONEnforceResult(valid=False, repaired=repaired, data=_try_parse_json(current), errors=errors)
