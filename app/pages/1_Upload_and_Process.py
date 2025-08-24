from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import streamlit as st
import io

# Ensure project root on path when Streamlit runs page modules directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core import storage
from app.core.cost import cost_from_usage
from app.core.provider_openai import OAIGateway
from app.core.templating import RenderedMessages, render_user_prompt
from scripts.export_records import ensure_records, all_columns, to_json_bytes, to_markdown_bytes, to_xlsx_bytes


UPLOAD_DIR = Path("data/uploads")
EXPORT_DIR = Path("export")


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "image"


def _save_uploaded_files(files: List["UploadedFile"]) -> List[str]:
    """Save uploaded files with validation and size limits."""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    saved: List[str] = []
    errors: List[str] = []
    
    for f in files:
        try:
            data = f.read()
            if not data:
                errors.append(f"{f.name}: Empty file")
                continue
            
            # Check file size
            file_size_mb = len(data) / (1024 * 1024)
            if len(data) > MAX_FILE_SIZE:
                errors.append(f"{f.name}: Exceeds 10MB limit ({file_size_mb:.1f}MB)")
                continue
            
            # Validate it's actually an image
            try:
                img = Image.open(io.BytesIO(data))
                # Also check if image dimensions are reasonable
                if img.width > 10000 or img.height > 10000:
                    errors.append(f"{f.name}: Image dimensions too large ({img.width}x{img.height})")
                    continue
            except Exception as img_error:
                errors.append(f"{f.name}: Invalid image format")
                continue
            
            h = hashlib.sha256(data).hexdigest()[:12]
            base = _sanitize_filename(f.name)
            path = UPLOAD_DIR / f"{h}_{base}"
            if not path.exists():
                path.write_bytes(data)
            saved.append(str(path))
        except MemoryError:
            errors.append(f"{f.name}: Out of memory - file too large")
            continue
        except Exception as e:
            errors.append(f"{f.name}: {str(e)[:50]}")
            continue
    
    # Show consolidated error message if any
    if errors:
        with st.expander(f"⚠️ {len(errors)} file(s) skipped", expanded=False):
            for error in errors:
                st.warning(error)
    
    return saved


def _image_capable() -> bool:
    """Check if active provider supports images."""
    p = storage.get_active_provider()
    if not p:
        return True  # unknown; don't block
    if isinstance(p.detected_caps_json, dict):
        v = p.detected_caps_json.get("vision")
        if isinstance(v, bool):
            return v
    mods = None
    if isinstance(p.catalog_caps_json, dict):
        mods = p.catalog_caps_json.get("modality") or p.catalog_caps_json.get("modalities")
    if isinstance(mods, list):
        return "image" in mods
    return False


def _get_active_provider_and_key() -> tuple[Optional[storage.Provider], Optional[str]]:
    """Get active provider and API key."""
    p = storage.get_active_provider()
    if not p:
        return None, None
    api_key: Optional[str] = None
    if p.key_storage == "session":
        keys: Dict[int, str] = st.session_state.get("_api_keys", {})
        api_key = keys.get(p.id)
    else:
        try:
            from cryptography.fernet import Fernet
            # Prefer env, else use local persisted key (data/kms.key) for automatic setup
            kms_key = os.getenv("APP_KMS_KEY")
            if not kms_key:
                try:
                    kms_path = Path("data/kms.key")
                    if kms_path.exists():
                        kms_key = kms_path.read_text(encoding="utf-8").strip()
                except Exception:
                    kms_key = None
            if kms_key and p.api_key_enc:
                api_key = Fernet(kms_key).decrypt(p.api_key_enc.encode()).decode()
        except Exception:
            api_key = None
    return p, api_key


def _derive_request_plan(p: Optional[storage.Provider], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Determine how to send request based on capabilities."""
    plan = {"use_tools": False, "use_json_mode": False, "fallback": "prompt"}
    if not p:
        return plan
    supports = p.detected_caps_json or {}
    tools_ok = bool(supports.get("tools")) if isinstance(supports, dict) else False
    json_ok = bool(supports.get("json_mode")) if isinstance(supports, dict) else False
    if p.default_prefer_tools and tools_ok and bool(schema):
        plan.update({"use_tools": True, "fallback": "json_mode" if (p.default_force_json_mode and json_ok) else "prompt"})
        return plan
    if p.default_force_json_mode and json_ok:
        plan.update({"use_json_mode": True, "fallback": "prompt"})
        return plan
    return plan


def _process_images(
    provider: storage.Provider,
    api_key: str,
    template: storage.Template,
    image_paths: List[str],
    tags: Dict[str, Dict[str, str]],
    unstructured: bool = False,
) -> Dict[str, Any]:
    """Process images with template and return results."""
    
    # Prepare template variables
    first_img = image_paths[0] if image_paths else ""
    first_tags = tags.get(first_img, {})
    template_vars = {
        "today": time.strftime("%Y-%m-%d"),
        "locale": first_tags.get("locale", "en-US"),
        "doc_type": first_tags.get("doc_type", "")
    }
    
    # Render messages
    rendered = render_user_prompt(
        template.user_prompt or "",
        template.schema_json or {},
        [],  # No examples for now
        template_vars
    )
    
    # Extract the user text from rendered messages
    user_text = ""
    if rendered.messages:
        # Get the last message (which should be the user message)
        last_message = rendered.messages[-1]
        if last_message.get("role") == "user" and last_message.get("content"):
            # Extract text from content parts
            for part in last_message["content"]:
                if part.get("type") == "text":
                    user_text = part.get("text", "")
                    break
    
    # Create gateway with proper timeout
    gateway = OAIGateway(
        base_url=provider.base_url,
        api_key=api_key,
        headers=provider.headers_json or {},
        timeout=int(provider.timeout_s or 120),  # Use 120s default, not 30s
        prefer_json_mode=False if unstructured else bool(provider.default_force_json_mode),
        prefer_tools=False if unstructured else bool(provider.default_prefer_tools),
        detected_caps=provider.detected_caps_json
    )
    
    # Execute request with proper max_tokens
    start_time = time.time()
    
    # Ensure max_output_tokens is properly set (default to 4096 if not set)
    max_tokens = int(provider.default_max_output_tokens) if provider.default_max_output_tokens else 4096
    
    # Debug info for users to see what's being sent
    print(f"Debug: Sending request with max_tokens={max_tokens}, model={provider.model_id}, timeout={provider.timeout_s or 120}s")
    
    result = gateway.chat_vision(
        model=provider.model_id or "gpt-4o-mini",
        system_text=template.system_prompt or "",
        user_text=user_text,
        image_paths=image_paths,
        fewshot_messages=None,
        schema=None if unstructured else template.schema_json,
        gen_params={
            "temperature": float(provider.default_temperature if provider.default_temperature is not None else 1.0),
            "top_p": float(provider.default_top_p or 1.0),
            "max_tokens": max_tokens  # Use the properly set max_tokens
        }
    )
    elapsed = time.time() - start_time
    
    # Process result with debugging
    output = None
    print(f"Debug: Raw API result keys: {result.keys()}")
    print(f"Debug: tool_call_json: {result.get('tool_call_json')}")
    print(f"Debug: text: {result.get('text')[:200] if result.get('text') else None}")
    
    if unstructured:
        # Always treat as freeform text (Markdown). Do not parse as JSON.
        output = {"raw_text": result.get("text") or "", "format": "markdown"}
        print("Debug: Unstructured mode - capturing raw text as markdown")
    else:
        if result.get("tool_call_json"):
            output = result["tool_call_json"]
            print(f"Debug: Using tool_call_json output")
        elif result.get("text"):
            try:
                output = json.loads(result["text"])
                print(f"Debug: Successfully parsed JSON from text")
            except json.JSONDecodeError as e:
                print(f"Debug: Failed to parse JSON: {e}")
                # Try to extract JSON from the text if it's embedded
                text = result["text"]
                if "{" in text and "}" in text:
                    try:
                        # Find JSON content between first { and last }
                        start = text.index("{")
                        end = text.rindex("}") + 1
                        json_str = text[start:end]
                        output = json.loads(json_str)
                        print(f"Debug: Extracted and parsed embedded JSON")
                    except:
                        output = {"raw_text": result["text"]}
                        print(f"Debug: Could not extract JSON, returning raw text")
                else:
                    output = {"raw_text": result["text"]}
                    print(f"Debug: No JSON markers found, returning raw text")
    
    print(f"Debug: Final output: {output}")
    
    return {
        "output": output,
        "error": result.get("error"),
        "usage": result.get("usage"),
        "elapsed_s": elapsed,
        "status": result.get("status", 200)
    }


def run() -> None:
    st.title("📤 Upload & Process")
    st.caption("Upload images and extract structured data using AI")
    
    storage.init_db()
    _ensure_dirs()
    
    # Check capabilities
    if not _image_capable():
        st.warning("⚠️ Active profile may not support image inputs. Please check Settings.")
        if st.button("Go to Settings"):
            st.switch_page("pages/2_Settings.py")
    
    # Upload Section
    st.markdown("## 1️⃣ Upload Images")
    
    uploaded = st.file_uploader(
        "Select images to upload",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Maximum 10MB per file"
    )
    
    if uploaded:
        saved_paths = _save_uploaded_files(uploaded)
        lst: List[str] = st.session_state.setdefault("uploaded_images", [])
        for pth in saved_paths:
            if pth not in lst:
                lst.append(pth)
        if saved_paths:
            st.success(f"✅ Uploaded {len(saved_paths)} file(s)")
    
    # Image Management (Simple List)
    imgs: List[str] = st.session_state.get("uploaded_images", [])
    
    if not imgs:
        st.info("No images uploaded yet. Upload some images to get started.")
        return
    
    st.markdown(f"### Uploaded Files ({len(imgs)})")
    
    # Simple list view without previews
    selected: List[str] = st.session_state.setdefault("selected_images", [])
    tags: Dict[str, Dict[str, str]] = st.session_state.setdefault("image_tags", {})
    
    # Select all/none buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Select All"):
            selected = imgs.copy()
            st.session_state["selected_images"] = selected
            st.rerun()
    with col2:
        if st.button("Select None"):
            selected = []
            st.session_state["selected_images"] = selected
            st.rerun()
    with col3:
        st.write(f"Selected: {len(selected)} of {len(imgs)}")
    
    # File list
    with st.container(border=True):
        for idx, pth in enumerate(imgs):
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                checked = st.checkbox("", value=(pth in selected), key=f"sel_{idx}")
                if checked and pth not in selected:
                    selected.append(pth)
                elif not checked and pth in selected:
                    selected.remove(pth)
            
            with col2:
                st.text(Path(pth).name)
            
            with col3:
                # File size
                try:
                    size_kb = Path(pth).stat().st_size / 1024
                    st.text(f"{size_kb:.1f} KB")
                except:
                    st.text("-")
            
            with col4:
                # Optional tags (collapsible)
                with st.expander("Tags"):
                    t = tags.get(pth, {"doc_type": "", "locale": "en-US"})
                    t["doc_type"] = st.text_input(
                        "Type", 
                        value=t.get("doc_type", ""), 
                        key=f"doc_{idx}",
                        placeholder="receipt, invoice, etc."
                    )
                    t["locale"] = st.text_input(
                        "Locale", 
                        value=t.get("locale", "en-US"), 
                        key=f"loc_{idx}"
                    )
                    tags[pth] = t
    
    st.session_state["selected_images"] = selected
    st.session_state["image_tags"] = tags
    
    if not selected:
        st.info("Select images to process")
        return
    
    st.divider()
    
    # Process Section
    st.markdown("## 2️⃣ Process Images")
    
    # Get provider and template
    provider, api_key = _get_active_provider_and_key()
    if not provider:
        st.error("❌ No active provider configured")
        if st.button("Configure Provider"):
            st.switch_page("pages/2_Settings.py")
        return
    
    # Allow local endpoints without API key (e.g., LM Studio)
    is_local = "localhost" in (provider.base_url or "") or "127.0.0.1" in (provider.base_url or "")
    if not api_key and not is_local:
        st.error("❌ No API key configured for active provider")
        if st.button("Add API Key"):
            st.switch_page("pages/2_Settings.py")
        return
    
    # Template selection
    templates = storage.list_templates()
    if not templates:
        st.warning("⚠️ No templates configured")
        if st.button("Create Template"):
            st.switch_page("pages/2_Settings.py")
        return
    
    template_names = [t.name for t in templates]
    selected_template_name = st.selectbox(
        "Select Template",
        options=template_names,
        help="Choose a template to extract data from images"
    )
    
    selected_template = next((t for t in templates if t.name == selected_template_name), None)
    
    if selected_template:
        # Show template info
        with st.expander("Template Details"):
            st.write(f"**Description:** {selected_template.description or 'No description'}")
            if selected_template.schema_json:
                st.json(selected_template.schema_json)
    
    # Show current settings
    with st.expander("Current Settings", expanded=False):
        st.write(f"**Model:** {provider.model_id}")
        st.write(f"**Max Output Tokens:** {provider.default_max_output_tokens or 4096}")
        st.write(f"**Temperature:** {provider.default_temperature if provider.default_temperature is not None else 1.0}")
        st.write(f"**API Endpoint:** {provider.base_url}")
    
    # Output mode toggle and Process button
    col1, col2 = st.columns([1, 5])
    with col1:
        # Minimal, per-run toggle to avoid DB/schema changes
        unstructured = st.checkbox("Unstructured (Markdown)", help="Skip schema; return plain text/Markdown.")
    with col2:
        process_clicked = st.button(
            f"▶️ Process {len(selected)} Image(s)",
            type="primary",
            use_container_width=True
        )
    
    if process_clicked and selected_template:
        st.markdown("### Processing Results")
        
        # Show estimated time based on number of images
        estimated_time = len(selected) * 5  # Estimate 5 seconds per image
        progress_text = f"Processing {len(selected)} image(s)... Estimated time: {estimated_time} seconds"
        
        with st.spinner(progress_text):
            result = _process_images(
                provider,
                api_key,
                selected_template,
                selected,
                tags,
                unstructured=unstructured,
            )
        
        # Display results
        if result.get("error"):
            st.error(f"❌ Error: {result['error']}")
            
            # Provide helpful suggestions based on error type
            error_msg = str(result.get("error", "")).lower()
            if "502" in error_msg or "500" in error_msg or "503" in error_msg:
                st.warning("🔧 **Server Error Detected**")
                st.write("The provider's server is experiencing issues. Try:")
                st.write("• Waiting a few minutes and trying again")
                st.write("• Using a different model from the same provider")
                st.write("• Switching to a different provider in Settings")
                
                # Show alternative providers if available
                other_providers = [p for p in storage.list_providers() if p.id != provider.id]
                if other_providers:
                    st.info(f"💡 Alternative providers available: {', '.join([p.name for p in other_providers])}")
            elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg:
                st.warning("🔑 **Authentication Issue**")
                st.write("There may be an issue with your API key. Check:")
                st.write("• API key is valid and active")
                st.write("• API key has sufficient credits/quota")
                st.write("• API key has permissions for the selected model")
            elif "429" in error_msg or "rate" in error_msg:
                st.warning("⏱️ **Rate Limit**")
                st.write("You've hit the rate limit. Try:")
                st.write("• Waiting a few minutes before retrying")
                st.write("• Processing fewer images at once")
                st.write("• Upgrading your API plan for higher limits")
            elif "timeout" in error_msg or "408" in error_msg:
                st.warning("⏰ **Timeout Error**")
                st.write("The request took too long. Try:")
                st.write("• Processing fewer images at once")
                st.write("• Using smaller images")
                st.write("• Simplifying your template prompt")
                
            # Show debug info
            with st.expander("Debug Information"):
                st.write("**Full error details:**")
                st.json(result)
                if result.get("status"):
                    st.write(f"**HTTP Status:** {result['status']}")
        else:
            st.success(f"✅ Processing complete in {result['elapsed_s']:.2f}s")
            
            # Show output (always show, even if empty)
            st.markdown("#### Extracted Data")

            output = result.get("output")
            if output is not None:
                if unstructured and isinstance(output, dict) and "raw_text" in output:
                    md_text = output.get("raw_text") or ""
                    st.markdown(md_text)

                    # Prepare exports (treat raw_text as a single-row dataset for JSON/XLSX)
                    recs = ensure_records(output)
                    cols = all_columns(recs)
                    md_bytes = (md_text or "").encode("utf-8")
                    json_bytes = to_json_bytes(recs)
                    xlsx_bytes = to_xlsx_bytes(recs, cols)

                    # Export options
                    st.markdown("#### Export")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            "📥 Markdown (.md)",
                            data=md_bytes,
                            file_name=f"extraction_{int(time.time())}.md",
                            mime="text/markdown",
                        )
                    with col2:
                        st.download_button(
                            "📥 JSON (.json)",
                            data=json_bytes,
                            file_name=f"extraction_{int(time.time())}.json",
                            mime="application/json",
                        )
                    with col3:
                        st.download_button(
                            "📥 Excel (.xlsx)",
                            data=xlsx_bytes,
                            file_name=f"extraction_{int(time.time())}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                else:
                    # Structured JSON (existing behavior)
                    st.json(output)

                    # Normalize and export in three formats
                    recs = ensure_records(output)
                    cols = all_columns(recs)
                    json_bytes = to_json_bytes(recs)
                    md_bytes = to_markdown_bytes(recs, cols)
                    xlsx_bytes = to_xlsx_bytes(recs, cols)

                    st.markdown("#### Export")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            "📥 JSON (.json)",
                            data=json_bytes,
                            file_name=f"extraction_{int(time.time())}.json",
                            mime="application/json",
                        )
                    with col2:
                        st.download_button(
                            "📥 Markdown (.md)",
                            data=md_bytes,
                            file_name=f"extraction_{int(time.time())}.md",
                            mime="text/markdown",
                        )
                    with col3:
                        st.download_button(
                            "📥 Excel (.xlsx)",
                            data=xlsx_bytes,
                            file_name=f"extraction_{int(time.time())}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
            else:
                st.warning("⚠️ No output received from the model. This could mean:")
                st.write("- The model returned an empty response")
                st.write("- The response couldn't be parsed as JSON")
                st.write("- The model timed out or encountered an error")
                
                # Show raw result for debugging
                with st.expander("Debug Information"):
                    st.write("**Raw result:**")
                    st.json(result)
            
            # Show usage/cost if available
            if result.get("usage"):
                usage = result["usage"]
                with st.expander("Usage Details"):
                    st.write(f"**Input tokens:** {usage.get('prompt_tokens', 0):,}")
                    output_tokens = usage.get('completion_tokens', 0)
                    st.write(f"**Output tokens:** {output_tokens:,}")
                    st.write(f"**Total tokens:** {usage.get('total_tokens', 0):,}")
                    
                    # Warning if output seems truncated
                    max_requested = int(provider.default_max_output_tokens) if provider.default_max_output_tokens else 4096
                    if output_tokens < 250 and max_requested > 1000:
                        st.warning(f"⚠️ Output may be truncated: only {output_tokens} tokens generated out of {max_requested} requested. Check your model's configuration.")
                    
                    # Calculate cost if pricing available
                    if provider.catalog_caps_json:
                        from app.core.cost import cost_from_usage
                        cost_info = cost_from_usage(usage, provider.catalog_caps_json)
                        if cost_info and cost_info.get("total", 0) > 0:
                            st.write(f"**Estimated cost:** ${cost_info['total']:.4f}")
            
            # Store run in database
            if result.get("output"):
                try:
                    # Calculate cost if pricing info available
                    cost_usd = None
                    if result.get("usage") and provider.catalog_caps_json:
                        cost_info = cost_from_usage(result["usage"], provider.catalog_caps_json)
                        if cost_info:
                            cost_usd = cost_info.get("total")
                    
                    run = storage.record_run(
                        provider_id=provider.id,
                        template_id=selected_template.id,
                        input_images=selected,
                        output=result["output"],
                        cost_usd=cost_usd,
                        status="completed" if not result.get("error") else "error"
                    )
                    st.caption(f"Run saved with ID: {run.id}")
                except Exception as e:
                    st.warning(f"Could not save run: {e}")


if __name__ == "__main__":
    run()
