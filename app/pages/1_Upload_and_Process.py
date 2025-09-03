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
from app.core.json_enforcer import strip_code_fences
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


def _guess_original_stem(path: Path) -> str:
    """Return the original file stem, stripping our upload hash prefix when present.

    Uploaded files are stored as '<hash12>_<original>'. For other files, use stem as-is.
    """
    try:
        if path.parent.resolve() == UPLOAD_DIR.resolve():
            parts = path.name.split("_", 1)
            if len(parts) == 2 and len(parts[0]) == 12 and all(c in "0123456789abcdef" for c in parts[0].lower()):
                return Path(parts[1]).stem
    except Exception:
        pass
    return path.stem


def _find_images_in_folder(folder: Path, recursive: bool = True) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    results: List[str] = []
    try:
        it = folder.rglob("*") if recursive else folder.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                results.append(str(p))
    except Exception:
        return []
    return sorted(results)


def _save_uploaded_files(files: List["UploadedFile"]) -> List[str]:
    """Save uploaded files with validation and size limits."""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for hashing
    saved: List[str] = []
    errors: List[str] = []
    
    for f in files:
        try:
            # First check file size without loading entire file
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Reset to beginning
            
            if file_size == 0:
                errors.append(f"{f.name}: Empty file")
                continue
            
            file_size_mb = file_size / (1024 * 1024)
            if file_size > MAX_FILE_SIZE:
                errors.append(f"{f.name}: Exceeds 10MB limit ({file_size_mb:.1f}MB)")
                continue
            
            # Since we limit to 10MB, just read once for simplicity
            # The chunked reading was over-optimization for small files
            f.seek(0)
            data = f.read()
            
            # Hash the data
            h = hashlib.sha256(data).hexdigest()[:12]
            
            # Validate it's actually an image
            try:
                img = Image.open(io.BytesIO(data))
                # Also check if image dimensions are reasonable
                if img.width > 10000 or img.height > 10000:
                    errors.append(f"{f.name}: Image dimensions too large ({img.width}x{img.height})")
                    continue
                img.close()  # Free memory
            except Exception as img_error:
                errors.append(f"{f.name}: Invalid image format")
                continue
            
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
    """Get active provider and corresponding API key (by provider_code)."""
    p = storage.get_active_provider()
    if not p:
        return None, None
    # Preferred path: provider-scoped key
    api_key: Optional[str] = None
    provider_code = (p.provider_code or "").strip().lower()
    if provider_code:
        # Check session-scoped key bucket by provider_code
        sess_keys: Dict[str, str] = st.session_state.get("_provider_api_keys", {})
        api_key = sess_keys.get(provider_code)
        if not api_key:
            # Fallback to persisted, encrypted key
            api_key = storage.get_decrypted_api_key(provider_code)
    else:
        # Backwards compatibility: use legacy per-profile storage
        if p.key_storage == "session":
            keys: Dict[int, str] = st.session_state.get("_api_keys", {})
            api_key = keys.get(p.id)
        else:
            try:
                from cryptography.fernet import Fernet
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
    # Smart timeout defaults: longer for local models
    url_lower = (provider.base_url or "").lower()
    is_local_model = any(indicator in url_lower for indicator in [
        "localhost", "127.0.0.1", "0.0.0.0", "192.168.",
        "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
        "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
        "172.28.", "172.29.", "172.30.", "172.31.",  # Private IP range B
        "host.docker.internal", ".local", ".lan", ":1234", ":5000", ":5001",
        ":8000", ":8080", ":8888", ":9000", ":11434", ":7860", ":7861"
    ])
    
    # Special check for 10.x.x.x range
    if not is_local_model and "10." in url_lower:
        import re
        if re.search(r'(?:^|[^0-9])10\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
            is_local_model = True
    
    # Use 1 hour timeout to prevent premature timeouts
    default_timeout = 3600
    
    gateway = OAIGateway(
        base_url=provider.base_url,
        api_key=api_key,
        headers=provider.headers_json or {},
        timeout=int(provider.timeout_s or default_timeout),
        prefer_json_mode=False if unstructured else bool(provider.default_force_json_mode),
        prefer_tools=False if unstructured else bool(provider.default_prefer_tools),
        detected_caps=provider.detected_caps_json,
        cached_max_tokens_param=provider.cached_max_tokens_param,
        provider_id=provider.id
    )
    
    # Execute request with proper max_tokens
    start_time = time.time()
    
    # Ensure max_output_tokens is properly set (default to 4096 if not set)
    max_tokens = int(provider.default_max_output_tokens) if provider.default_max_output_tokens else 4096
    
    # Show request configuration
    print(f"📤 Processing {len(image_paths)} image(s)")
    print(f"   Model: {provider.model_id}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Timeout: {provider.timeout_s or 120}s")
    
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
    
    # Process result
    output = None
    verbose = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes")
    
    if verbose:
        print(f"Debug: Raw API result keys: {result.keys()}")
        print(f"Debug: tool_call_json: {result.get('tool_call_json')}")
        print(f"Debug: text preview: {result.get('text')[:200] if result.get('text') else None}")
    
    if unstructured:
        # Always treat as freeform text (Markdown). Do not parse as JSON.
        output = {"raw_text": result.get("text") or "", "format": "markdown"}
        if verbose:
            print("Debug: Unstructured mode - capturing raw text as markdown")
    else:
        if result.get("tool_call_json"):
            output = result["tool_call_json"]
            print(f"✅ Extracted structured data using tools")
        elif result.get("text"):
            try:
                # Strip markdown code fences before parsing JSON
                cleaned_text = strip_code_fences(result["text"])
                output = json.loads(cleaned_text)
                print(f"✅ Extracted structured data from JSON response")
            except json.JSONDecodeError as e:
                # Try to extract JSON from the text if it's embedded
                text = result["text"]
                if "{" in text and "}" in text:
                    try:
                        # Find JSON content between first { and last }
                        start = text.index("{")
                        end = text.rindex("}") + 1
                        json_str = text[start:end]
                        output = json.loads(json_str)
                        print(f"✅ Extracted embedded JSON from response")
                    except Exception as extract_error:
                        output = {"raw_text": result["text"]}
                        if verbose:
                            print(f"⚠️ Could not parse JSON ({extract_error}), returning raw text")
                        else:
                            print(f"⚠️ Could not parse JSON, returning raw text")
                else:
                    output = {"raw_text": result["text"]}
                    print(f"⚠️ No JSON found in response, returning raw text")
    
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
        st.warning("⚠️ Active model may not support image inputs. Please check Settings.")
        if st.button("Go to Settings"):
            st.switch_page("pages/2_Settings.py")
    
    # Upload or Folder Section
    st.markdown("## 1️⃣ Add Input Files")

    src_choice = st.radio(
        "Source",
        options=["Upload files", "Select folder"],
        horizontal=True,
        help="Upload individual images or add an entire folder"
    )

    if src_choice == "Upload files":
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
    else:
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            default_dir = str((Path.home() / "Documents")) if (Path.home() / "Documents").exists() else str(Path.cwd())
            folder_path = st.text_input(
                "Folder path",
                value=st.session_state.get("last_folder_path", default_dir),
                placeholder="/path/to/folder",
                help="Enter a local folder path"
            )
        with col_f2:
            recursive = st.checkbox("Include subfolders", value=True)
        folder_ok = False
        if folder_path:
            fp = Path(folder_path).expanduser()
            folder_ok = fp.exists() and fp.is_dir()
        col_b1, col_b2, _ = st.columns([1, 1, 4])
        with col_b1:
            add_all = st.button("Add Folder")
        with col_b2:
            preview = st.button("Preview")
        if preview and folder_ok:
            imgs_found = _find_images_in_folder(Path(folder_path).expanduser(), recursive)
            if imgs_found:
                st.info(f"Found {len(imgs_found)} image(s). Click 'Add Folder' to add them.")
            else:
                st.warning("No images found in this folder.")
        if add_all:
            if not folder_ok:
                st.error("Folder does not exist or is not accessible.")
            else:
                st.session_state["last_folder_path"] = folder_path
                imgs_found = _find_images_in_folder(Path(folder_path).expanduser(), recursive)
                if not imgs_found:
                    st.warning("No images found in this folder.")
                else:
                    lst: List[str] = st.session_state.setdefault("uploaded_images", [])
                    added = 0
                    for p in imgs_found:
                        if p not in lst:
                            lst.append(p)
                            added += 1
                    st.success(f"✅ Added {added} image(s) from folder")
    
    # Image Management (Simple List)
    imgs: List[str] = st.session_state.get("uploaded_images", [])
    
    if not imgs:
        st.info("No images uploaded yet. Upload some images to get started.")
        return
    
    st.markdown(f"### Input Files ({len(imgs)})")
    
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
                stable_key = f"sel_{hashlib.sha1(pth.encode('utf-8')).hexdigest()[:10]}"
                checked = st.checkbox("", value=(pth in selected), key=stable_key)
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
                except Exception:
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
    col1, col2, col3, col4 = st.columns([1.2, 1.4, 3.4, 1])
    with col1:
        # Minimal, per-run toggle to avoid DB/schema changes
        unstructured = st.checkbox("Unstructured", help="Skip schema; return plain text/Markdown.")
    with col2:
        per_file_mode = st.checkbox("Per-file save", help="Process each file separately and auto-save next to inputs")
    with col3:
        process_clicked = st.button(
            f"▶️ Process {len(selected)} Image(s)",
            type="primary",
            use_container_width=True
        )
    with col4:
        # Clear results button
        if 'last_result' in st.session_state:
            if st.button("🗑️ Clear", help="Clear cached results"):
                # Clear all cached data
                keys_to_clear = ['last_result', 'last_template_name', 'last_images', 
                               'processing_timestamp', 'last_unstructured', 'export_data',
                               'last_debug_info', 'last_usage', 'cumulative_cost']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Check for cached results first
    show_cached = False
    if 'last_result' in st.session_state and not process_clicked and not st.session_state.get('per_file_mode_active', False):
        # Check if the cached result is for the same template and images
        cached_template = st.session_state.get('last_template_name', '')
        cached_images = st.session_state.get('last_images', [])
        
        if cached_template == selected_template_name and set(cached_images) == set(selected):
            show_cached = True
            result = st.session_state['last_result']
            cached_time = st.session_state.get('processing_timestamp', 0)
            time_ago = time.time() - cached_time
            
            # Show info about cached results
            st.info(f"📋 Showing cached results from {int(time_ago/60)} minute(s) ago. Click 'Process' to refresh.")
    
    # Per-file mode: run one-by-one and save
    if process_clicked and selected_template and per_file_mode:
        st.markdown("### Processing & Saving Per File")
        from scripts.export_records import ensure_records, all_columns, to_json_bytes, to_markdown_bytes
        from scripts.export_records import to_docx_bytes, to_docx_from_text_bytes
        total = len(selected)
        progress = st.progress(0.0, text=f"Processing 0/{total}")
        saved_counts = {"json": 0, "md": 0, "docx": 0}
        errors: List[str] = []
        st.session_state['per_file_mode_active'] = True
        for idx, img_path in enumerate(selected, start=1):
            try:
                result = _process_images(
                    provider,
                    api_key or "",
                    selected_template,
                    [img_path],
                    tags,
                    unstructured=unstructured,
                )
                out = result.get("output")
                in_path = Path(img_path)
                out_dir = in_path.parent
                base_name = _guess_original_stem(in_path)
                if not out:
                    errors.append(f"No output for {in_path.name}")
                else:
                    if unstructured and isinstance(out, dict) and 'raw_text' in out:
                        text = str(out.get('raw_text') or "")
                        (out_dir / f"{base_name}.md").write_bytes((text or "").encode("utf-8"))
                        docx_bytes = to_docx_from_text_bytes(text, title=base_name)
                        (out_dir / f"{base_name}.docx").write_bytes(docx_bytes)
                        recs = ensure_records({"raw_text": text})
                        (out_dir / f"{base_name}.json").write_bytes(to_json_bytes(recs))
                        saved_counts['md'] += 1
                        saved_counts['docx'] += 1
                        saved_counts['json'] += 1
                    else:
                        recs = ensure_records(out)
                        cols = all_columns(recs)
                        (out_dir / f"{base_name}.json").write_bytes(to_json_bytes(recs))
                        (out_dir / f"{base_name}.md").write_bytes(to_markdown_bytes(recs, cols))
                        (out_dir / f"{base_name}.docx").write_bytes(to_docx_bytes(recs, cols))
                        saved_counts['json'] += 1
                        saved_counts['md'] += 1
                        saved_counts['docx'] += 1
            except Exception as e:
                errors.append(f"Save failed for {Path(img_path).name}: {e}")
            finally:
                progress.progress(idx/total, text=f"Processing {idx}/{total}")
        progress.empty()
        st.success(f"Saved: {saved_counts['json']} JSON, {saved_counts['md']} MD, {saved_counts['docx']} DOCX next to inputs")
        if errors:
            with st.expander("Errors encountered"):
                for e in errors:
                    st.error(e)
        # End per-file mode early to avoid aggregated rendering
        st.session_state['per_file_mode_active'] = False
        return

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
        
        # Store result in session state for persistence
        st.session_state['last_result'] = result
        st.session_state['last_template_name'] = selected_template_name
        st.session_state['last_images'] = selected.copy()
        st.session_state['processing_timestamp'] = time.time()
        st.session_state['last_unstructured'] = unstructured
        show_cached = False
        
        # Display results
    
    # Display results (either new or cached)
    if (process_clicked or show_cached) and 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        # Show results header
        if not process_clicked and show_cached:
            st.markdown("### 📋 Cached Processing Results")
        elif process_clicked:
            st.markdown("### ✅ New Processing Results")
        
        if result.get("error"):
            # User-friendly error message
            error_raw = str(result.get("error", ""))
            error_msg_lower = error_raw.lower()
            
            # Provide user-friendly error messages
            if "timeout" in error_msg_lower:
                st.error("❌ Request timed out - the model took too long to respond")
            elif "401" in error_msg_lower or "403" in error_msg_lower or "unauthorized" in error_msg_lower:
                st.error("❌ Authentication failed - please check your API key")
            elif "429" in error_msg_lower or "rate limit" in error_msg_lower:
                st.error("❌ Rate limit exceeded - please wait a moment and try again")
            elif "404" in error_msg_lower:
                st.error("❌ Model or endpoint not found - please check your configuration")
            elif "500" in error_msg_lower or "502" in error_msg_lower or "503" in error_msg_lower:
                st.error("❌ Server error - the AI service is having issues")
            elif "connection" in error_msg_lower or "network" in error_msg_lower:
                st.error("❌ Connection failed - please check your network and endpoint URL")
            else:
                # Show simplified error
                if "httpx." in error_raw or "Exception" in error_raw:
                    st.error("❌ Processing failed - please try again or check your settings")
                else:
                    st.error(f"❌ Error: {error_raw[:200]}")
            
            # Provide helpful suggestions based on error type
            error_msg = error_msg_lower
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
                # Re-check if local model (is_local was defined earlier but might be out of scope)
                is_local_error_check = "localhost" in (provider.base_url or "") or "127.0.0.1" in (provider.base_url or "")
                if is_local_error_check:
                    st.write("Local models can be slow. Try:")
                    st.write("• Increasing the timeout in Settings (current: " + str(int(provider.timeout_s or 120)) + "s)")
                    st.write("• Using a smaller model")
                    st.write("• Reducing max_output_tokens")
                    st.write("• Processing one image at a time")
                    if provider.timeout_s and provider.timeout_s < 300:
                        st.info("💡 Consider increasing timeout to 300+ seconds for local models")
                else:
                    st.write("The request took too long. Try:")
                    st.write("• Processing fewer images at once")
                    st.write("• Using smaller images")
                    st.write("• Simplifying your template prompt")
                    st.write("• Switching to a faster model")
                
            # Show debug info
            with st.expander("Debug Information", expanded=False):
                st.write("**Full error details:**")
                # Store debug info in session state for persistence
                if process_clicked:
                    st.session_state['last_debug_info'] = {
                        'result': result,
                        'status': result.get("status"),
                        'timestamp': time.time()
                    }
                
                # Display debug info from current or cached
                debug_info = st.session_state.get('last_debug_info', {'result': result, 'status': result.get("status")})
                st.json(debug_info['result'])
                if debug_info.get('status'):
                    st.write(f"**HTTP Status:** {debug_info['status']}")
        else:
            st.success(f"✅ Processing complete in {result['elapsed_s']:.2f}s")
            
            # Show output (always show, even if empty)
            st.markdown("#### Extracted Data")

            output = result.get("output")
            if output is not None:
                # Check if this is unstructured mode (either current or cached)
                is_unstructured = unstructured if process_clicked else st.session_state.get('last_unstructured', False)
                
                if is_unstructured and isinstance(output, dict) and "raw_text" in output:
                    md_text = output.get("raw_text") or ""
                    st.markdown(md_text)

                    # Prepare exports (treat raw_text as a single-row dataset for JSON/XLSX)
                    recs = ensure_records(output)
                    cols = all_columns(recs)
                    
                    # Check if we have cached export data
                    if not process_clicked and 'export_data' in st.session_state:
                        # Use cached export data
                        export_data = st.session_state['export_data']
                        md_bytes = export_data.get('markdown', (md_text or "").encode("utf-8"))
                        json_bytes = export_data.get('json', to_json_bytes(recs))
                        xlsx_bytes = export_data.get('excel', to_xlsx_bytes(recs, cols))
                        docx_bytes = export_data.get('docx', None)
                    else:
                        # Generate new export data
                        md_bytes = (md_text or "").encode("utf-8")
                        json_bytes = to_json_bytes(recs)
                        xlsx_bytes = to_xlsx_bytes(recs, cols)
                        from scripts.export_records import to_docx_from_text_bytes
                        docx_bytes = to_docx_from_text_bytes(md_text, title="Extraction")
                        
                        # Cache export data
                        st.session_state['export_data'] = {
                            'markdown': md_bytes,
                            'json': json_bytes,
                            'excel': xlsx_bytes,
                            'docx': docx_bytes,
                            'timestamp': time.time()
                        }

                    # Export options
                    st.markdown("#### Export")
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        if docx_bytes is not None:
                            st.download_button(
                                "📥 Word (.docx)",
                                data=docx_bytes,
                                file_name=f"extraction_{int(time.time())}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            )
                else:
                    # Structured JSON (existing behavior)
                    st.json(output)

                    # Normalize and export in three formats
                    recs = ensure_records(output)
                    cols = all_columns(recs)
                    
                    # Check if we have cached export data
                    if not process_clicked and 'export_data' in st.session_state:
                        # Use cached export data
                        export_data = st.session_state['export_data']
                        json_bytes = export_data.get('json', to_json_bytes(recs))
                        md_bytes = export_data.get('markdown', to_markdown_bytes(recs, cols))
                        xlsx_bytes = export_data.get('excel', to_xlsx_bytes(recs, cols))
                        docx_bytes = export_data.get('docx', None)
                    else:
                        # Generate new export data
                        json_bytes = to_json_bytes(recs)
                        md_bytes = to_markdown_bytes(recs, cols)
                        xlsx_bytes = to_xlsx_bytes(recs, cols)
                        from scripts.export_records import to_docx_bytes
                        docx_bytes = to_docx_bytes(recs, cols)
                        
                        # Cache export data
                        st.session_state['export_data'] = {
                            'json': json_bytes,
                            'markdown': md_bytes,
                            'excel': xlsx_bytes,
                            'docx': docx_bytes,
                            'timestamp': time.time()
                        }

                    st.markdown("#### Export")
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col4:
                        if docx_bytes is not None:
                            st.download_button(
                                "📥 Word (.docx)",
                                data=docx_bytes,
                                file_name=f"extraction_{int(time.time())}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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
                
                # Store usage info in session state
                if process_clicked:
                    st.session_state['last_usage'] = {
                        'usage': usage,
                        'provider_info': {
                            'model_id': provider.model_id,
                            'max_output_tokens': provider.default_max_output_tokens,
                            'catalog_caps_json': provider.catalog_caps_json
                        }
                    }
                
                # Get usage from current or cached
                usage_info = st.session_state.get('last_usage', {'usage': usage})
                usage_data = usage_info['usage']
                
                with st.expander("Usage Details"):
                    st.write(f"**Input tokens:** {usage_data.get('prompt_tokens', 0):,}")
                    output_tokens = usage_data.get('completion_tokens', 0)
                    st.write(f"**Output tokens:** {output_tokens:,}")
                    st.write(f"**Total tokens:** {usage_data.get('total_tokens', 0):,}")
                    
                    # Warning if output seems truncated
                    max_requested = int(provider.default_max_output_tokens) if provider.default_max_output_tokens else 4096
                    if output_tokens < 250 and max_requested > 1000:
                        st.warning(f"⚠️ Output may be truncated: only {output_tokens} tokens generated out of {max_requested} requested. Check your model's configuration.")
                    
                    # Calculate cost if pricing available
                    if provider.catalog_caps_json:
                        from app.core.cost import cost_from_usage
                        cost_info = cost_from_usage(usage_data, provider.catalog_caps_json)
                        if cost_info and cost_info.get("total", 0) > 0:
                            st.write(f"**Estimated cost:** ${cost_info['total']:.4f}")
                            
                            # Track cumulative cost in session
                            if process_clicked:
                                if 'cumulative_cost' not in st.session_state:
                                    st.session_state['cumulative_cost'] = 0.0
                                st.session_state['cumulative_cost'] += cost_info['total']
                            
                            # Show cumulative cost if multiple runs
                            if 'cumulative_cost' in st.session_state:
                                st.caption(f"Session total: ${st.session_state['cumulative_cost']:.4f}")
            
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
