from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from PIL import Image
import streamlit as st

# Ensure project root on path when Streamlit runs page modules directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core import storage
from app.core import template_assets
from app.core.cost import cost_from_usage
from app.core.currency import get_usd_to_inr, convert_usd_to_inr
from app.core.json_enforcer import strip_code_fences
from app.core.model_registry import ModelRegistryError, ModelDescriptor, ensure_registry, active_model
from app.core.provider_openai import gateway_from_descriptor
from app.core.templating import RenderedMessages, render_user_prompt
from scripts.export_records import ensure_records, all_columns, to_json_bytes, to_markdown_bytes, to_xlsx_bytes
from app.core.checkpoints import FolderCheckpoint
from app.integrations.supabase_client import SupabaseMetaClient, SupabaseAPIError
from app.auth.session import SessionManager
from app.sync.metadata_sync import MetadataSync


UPLOAD_DIR = Path("data/uploads")
EXPORT_DIR = Path("export")

# ================== AUTH & CACHE BOOTSTRAP (File 17) ==================

def _redirect_to_login():
    for cand in ("app/pages/0_Login.py", "pages/0_Login.py", "0_Login.py"):
        try:
            st.switch_page(cand)
            return
        except Exception:
            continue
    st.experimental_set_query_params(auth="login", t=str(time.time()))
    st.rerun()

@st.cache_resource(show_spinner=False)
def _build_sync(db_path: str = None):
    db_path = db_path or os.getenv("APP_DB_PATH", "data/app.db")
    sync = MetadataSync(db_path=db_path)
    sync.ensure_schema()
    return sync

def _ensure_session_and_cache():
    try:
        supa = SupabaseMetaClient()
    except SupabaseAPIError as e:
        st.error(f"Supabase not configured: {e}")
        st.stop()

    sm = SessionManager(supabase=supa)
    sess = sm.try_load_session_from_disk()
    if not sess:
        _redirect_to_login()
        st.stop()

    # ensure cache exists (we don't force resync here; main.py already does periodic sync)
    sync = _build_sync()
    sm.attach_to_streamlit_state()  # expose role/email to the UI if needed

    # adopt project chosen in sidebar (File 6)
    pid = st.session_state.get("active_project_id")
    if isinstance(pid, str):
        sm.set_active_project(pid)

    active_pid = sm.get_active_project_id()
    if not active_pid:
        st.warning("No project assigned. Please ask an admin to assign you to a project.")
        st.stop()

    # Make project keys available to downstream processing (plaintext kept in memory only)
    st.session_state["__project_keys__"] = sm.get_decrypted_project_keys(sync, project_id=active_pid)

    return sm, sync, active_pid

__SM__, __SYNC__, __ACTIVE_PROJECT_ID__ = _ensure_session_and_cache()
# ================= END AUTH & CACHE BOOTSTRAP (File 17) ================

CURRENT_SYSTEM_PROMPT: str = ""
CURRENT_USER_PROMPT: str = ""
CURRENT_SCHEMA: Dict[str, Any] = {}


def _track_usage_analytics(template_id: Optional[str]) -> None:
    """Best-effort usage analytics hook; never breaks the flow."""
    if "__active_template__" not in st.session_state:
        return
    if __SM__ is None or not hasattr(__SM__, "supa"):
        return
    try:
        user = __SM__.current_user()
        chosen_model_id = st.session_state.get("__chosen_model_id__")
        template_state = st.session_state.get("__active_template__", {})
        effective_template_id = template_id or template_state.get("id")
        __SM__.supa.ensure_fresh_session()
        __SM__.supa.insert_usage_analytics(
            user_id=user.user_id,
            project_id=__ACTIVE_PROJECT_ID__,
            template_id=effective_template_id,
            model_id=chosen_model_id,
            meta={"page": "upload_and_process"},
        )
    except Exception:
        # swallow analytics errors; they should never break the user flow
        pass


def _apply_project_api_key(gateway, provider_id: str) -> None:
    """Inject project-scoped API key into the gateway when available."""
    try:
        project_keys = st.session_state.get("__project_keys__", {}) or {}
        token = project_keys.get(provider_id)
        if token:
            gateway.auth_token = token
    except Exception:
        # Non-fatal: fall back to default credentials
        pass


@dataclass
class ModelContext:
    descriptor: ModelDescriptor
    provider_record: storage.Provider
    caps: Dict[str, Any]
    pricing: Dict[str, Any]
    reasoning: Dict[str, Any]


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _build_model_context(descriptor: ModelDescriptor) -> ModelContext:
    provider_record = storage.ensure_registry_provider(descriptor)
    caps = descriptor.capabilities.model_dump()
    pricing_cfg = descriptor.pricing
    pricing: Dict[str, Any] = {
        "pricing": {
            "input_per_1k": pricing_cfg.input_per_1k,
            "output_per_1k": pricing_cfg.output_per_1k,
        }
    }
    if pricing_cfg.cache_read_per_1k is not None:
        pricing["pricing"]["cache_read_per_1k"] = pricing_cfg.cache_read_per_1k
    if pricing_cfg.cache_write_per_1k is not None:
        pricing["pricing"]["cache_write_per_1k"] = pricing_cfg.cache_write_per_1k
    pricing["pricing"]["input_per_million"] = pricing_cfg.input_per_million
    pricing["pricing"]["output_per_million"] = pricing_cfg.output_per_million
    pricing["pricing"]["cache_read_per_million"] = pricing_cfg.cache_read_per_million
    pricing["pricing"]["cache_write_per_million"] = pricing_cfg.cache_write_per_million
    reasoning_cfg = descriptor.reasoning
    reasoning = {
        "provider": reasoning_cfg.provider if reasoning_cfg else None,
        "effort_default": reasoning_cfg.effort_default if reasoning_cfg else None,
        "include_thoughts_default": reasoning_cfg.include_thoughts_default if reasoning_cfg else False,
        "allow_override": reasoning_cfg.allow_override if reasoning_cfg else True,
    }
    return ModelContext(
        descriptor=descriptor,
        provider_record=provider_record,
        caps=caps,
        pricing=pricing,
        reasoning=reasoning,
    )


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
    # Support a broader set of common image extensions
    exts = {
        ".png", ".jpg", ".jpeg", ".webp",
        ".bmp", ".gif", ".tif", ".tiff",
        ".heic", ".heif", ".jfif",
    }
    results: List[str] = []
    try:
        it = folder.rglob("*") if recursive else folder.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                results.append(str(p))
    except Exception:
        return []
    return sorted(results)


def _normalize_folder_input(raw: str) -> str:
    """Normalize a folder path string from user input.

    - Strips whitespace and surrounding quotes
    - Handles file:// URLs by converting to local paths
    - Expands ~ to home
    """
    s = (raw or "").strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip()
    # Convert file:// URLs (simple heuristic)
    if s.lower().startswith("file://"):
        # On macOS/Linux, file:///Users/name/… → /Users/name/…
        # On Windows, file:///C:/path → C:/path
        s = s[7:]
        # Remove leading slash for Windows drive letters like /C:/...
        if len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]
    # Expand ~ and environment vars
    s = os.path.expandvars(os.path.expanduser(s))
    return s


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


def _cleanup_uploaded_files() -> None:
    """Remove all uploaded image files from data/uploads/ directory.
    Preserves the uploads directory and pdf_tmp subdirectory structure.
    """
    try:
        if UPLOAD_DIR.exists():
            for item in UPLOAD_DIR.iterdir():
                if item.is_file() and item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    try:
                        item.unlink()
                    except Exception:
                        pass  # Skip files that can't be deleted
    except Exception:
        pass  # Fail silently if directory doesn't exist or can't be accessed


def _image_capable() -> bool:
    """Check if active provider supports images."""
    try:
        descriptor = active_model()
    except ModelRegistryError:
        return True
    return bool(descriptor.capabilities.vision)


def _process_images(
    model_ctx: ModelContext,
    gateway,
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
    
    descriptor = model_ctx.descriptor

    # Execute request with proper max_tokens
    start_time = time.time()

    max_tokens_limit = descriptor.max_output_tokens if descriptor.max_output_tokens not in (None, 0) else None
    if max_tokens_limit is None:
        fallback_limit = 4096
    else:
        fallback_limit = max_tokens_limit

    # Show request configuration
    print(f"📤 Processing {len(image_paths)} image(s)")
    print(f"   Model: {descriptor.id}")
    print(f"   Max tokens: {fallback_limit}")
    print(f"   Timeout: {descriptor.timeouts.total_s}s")

    gen_params: Dict[str, Any] = {"temperature": descriptor.default_temperature}
    if descriptor.default_top_p is not None:
        gen_params["top_p"] = descriptor.default_top_p
    if max_tokens_limit is not None:
        gen_params["max_tokens"] = max_tokens_limit

    result = gateway.chat_vision(
        model=descriptor.id,
        system_text=template.system_prompt or "",
        user_text=user_text,
        image_paths=image_paths,
        fewshot_messages=None,
        schema=None if unstructured else template.schema_json,
        gen_params=gen_params,
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
    template_assets.sync_from_assets()
    _ensure_dirs()
    
    # Clean up uploaded files from previous sessions
    _cleanup_uploaded_files()
    
    # Sidebar: Project stats and cost tracking
    with st.sidebar:
        st.markdown("#### 📁 Active Project")
        
        try:
            projects = storage.list_projects()
            if not projects:
                # Initialize with default project if none exist
                storage.init_db()
                projects = storage.list_projects()
            
            if projects:
                active_project = storage.get_active_project()
                project_names = [p.name for p in projects]
                
                # If no active project, set the first one as active
                if not active_project and projects:
                    storage.set_active_project(projects[0].id)
                    active_project = projects[0]
                
                current_index = 0
                if active_project and active_project.name in project_names:
                    current_index = project_names.index(active_project.name)
                
                selected = st.selectbox(
                    "Select Project",
                    options=project_names,
                    index=current_index,
                    key="upload_page_project_selector",
                    label_visibility="collapsed"
                )
                
                # Switch project if selection changed
                if active_project and selected != active_project.name:
                    new_proj = next((p for p in projects if p.name == selected), None)
                    if new_proj:
                        storage.set_active_project(new_proj.id)
                        st.rerun()
                
                # Show project stats
                if active_project:
                    stats = storage.get_project_stats(active_project.id)
                    
                    # Get currency rate for INR display
                    usd_to_inr = get_usd_to_inr()
                    
                    # Show currency rate badge
                    if usd_to_inr:
                        st.markdown(
                            f'<div style="background-color: #d1fae5; color: #065f46; padding: 6px 10px; '
                            f'border-radius: 6px; font-size: 0.8rem; margin: 12px 0; text-align: center; '
                            f'font-weight: 500; border: 1px solid #a7f3d0;">'
                            f'💱 1 USD = ₹{usd_to_inr:.2f}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Stats in clean columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div style="text-align: center; padding: 8px; background-color: #f9fafb; '
                            f'border-radius: 6px; margin-bottom: 8px;">'
                            f'<div style="font-size: 0.7rem; color: #6b7280; font-weight: 500;">IMAGES</div>'
                            f'<div style="font-size: 1.5rem; font-weight: 600; color: #111827; margin-top: 2px;">{stats["total_images"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f'<div style="text-align: center; padding: 8px; background-color: #f9fafb; '
                            f'border-radius: 6px; margin-bottom: 8px;">'
                            f'<div style="font-size: 0.7rem; color: #6b7280; font-weight: 500;">RUNS</div>'
                            f'<div style="font-size: 1.5rem; font-weight: 600; color: #111827; margin-top: 2px;">{stats["total_runs"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Total cost section
                    total_usd = stats['total_cost_usd']
                    cost_html = '<div style="padding: 10px; background-color: #fef3c7; border-radius: 6px; border: 1px solid #fde68a; margin-top: 8px;">'
                    cost_html += '<div style="font-size: 0.7rem; color: #92400e; font-weight: 500; margin-bottom: 4px;">TOTAL SPENT</div>'
                    cost_html += f'<div style="font-size: 1.1rem; font-weight: 600; color: #78350f;">${total_usd:.4f}</div>'
                    
                    if usd_to_inr:
                        total_inr = convert_usd_to_inr(total_usd, rate=usd_to_inr)
                        if total_inr is not None:
                            cost_html += f'<div style="font-size: 0.85rem; color: #92400e; margin-top: 2px;">₹{total_inr:.2f}</div>'
                    
                    cost_html += '</div>'
                    st.markdown(cost_html, unsafe_allow_html=True)
                    
                    # Average per image - only show if there are images
                    if stats['total_images'] > 0:
                        avg_usd = total_usd / stats['total_images']
                        avg_text = f"${avg_usd:.4f}"
                        if usd_to_inr:
                            avg_inr = convert_usd_to_inr(avg_usd, rate=usd_to_inr)
                            if avg_inr is not None:
                                avg_text += f" • ₹{avg_inr:.4f}"
                        st.markdown(
                            f'<div style="text-align: center; font-size: 0.75rem; color: #6b7280; margin-top: 8px;">'
                            f'Avg per image: {avg_text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        except Exception as e:
            st.error(f"Error loading projects: {e}")
        
        st.markdown("---")
    
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
    
    # Info about PDF conversion
    st.info("💡 Need to convert PDFs to images? Use **Settings → PDF Tools** to convert PDF pages to images first.")

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
            selected_lst: List[str] = st.session_state.setdefault("selected_images", [])
            for pth in saved_paths:
                if pth not in lst:
                    lst.append(pth)
                if pth not in selected_lst:
                    selected_lst.append(pth)
            if saved_paths:
                st.success(f"✅ Uploaded {len(saved_paths)} file(s)")
    elif src_choice == "Select folder":
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
            normalized = _normalize_folder_input(folder_path)
            fp = Path(normalized)
            folder_ok = fp.exists() and fp.is_dir()
            # Small hint for users to see how input was interpreted
            st.caption(f"Resolved path: {str(fp)}")
        col_b1, col_b2, _ = st.columns([1, 1, 4])
        with col_b1:
            add_all = st.button("Add Folder")
        with col_b2:
            preview = st.button("Preview")

        # Compute images and checkpoint info when folder is OK
        imgs_found: List[str] = []
        cp: FolderCheckpoint | None = None
        if folder_ok:
            folder_abs = Path(_normalize_folder_input(folder_path)).resolve()
            imgs_found = _find_images_in_folder(folder_abs, recursive)
            # Load or initialize checkpoint
            cp = FolderCheckpoint(folder_abs)
            cp.load()
            if imgs_found:
                cp.ensure_entries(imgs_found)
                cp.prune_missing()
                try:
                    cp.save()
                except Exception:
                    pass

        if preview and folder_ok:
            if imgs_found:
                if cp:
                    stats = cp.get_stats_for(imgs_found)
                    st.info(f"Found {stats['total']} image(s) • {stats['processed']} processed • {stats['failed']} failed • {stats['pending']} pending")
                else:
                    st.info(f"Found {len(imgs_found)} image(s). Click 'Add Folder' to add them.")
            else:
                st.warning("No images found in this folder. Supported types: PNG, JPG, JPEG, WEBP, BMP, GIF, TIF, TIFF, HEIC, HEIF, JFIF.")

        # Resume/Retry/Reset controls when a checkpoint is available
        if folder_ok and imgs_found and cp is not None:
            st.markdown("#### Checkpoint")
            r1, r2, r3 = st.columns([1.2, 1.2, 1])
            with r1:
                resume = st.button("Resume pending", key="resume_pending")
            with r2:
                retry_failed = st.button("Retry failed only", key="retry_failed")
            with r3:
                reset_cp = st.button("Reset checkpoint", key="reset_checkpoint")

            if reset_cp:
                cp.reset()
                cp.save()
                st.success("Checkpoint reset: all files marked pending")

            if resume:
                pending = cp.pending_files(imgs_found)
                lst: List[str] = st.session_state.setdefault("uploaded_images", [])
                selected_lst: List[str] = st.session_state.setdefault("selected_images", [])
                added = 0
                for p in pending:
                    if p not in lst:
                        lst.append(p)
                        added += 1
                    if p not in selected_lst:
                        selected_lst.append(p)
                st.session_state["last_folder_path"] = folder_path
                st.success(f"✅ Added {added} pending image(s)")

            if retry_failed:
                failed = cp.failed_files(imgs_found)
                lst: List[str] = st.session_state.setdefault("uploaded_images", [])
                selected_lst: List[str] = st.session_state.setdefault("selected_images", [])
                added = 0
                for p in failed:
                    if p not in lst:
                        lst.append(p)
                        added += 1
                    if p not in selected_lst:
                        selected_lst.append(p)
                st.session_state["last_folder_path"] = folder_path
                st.success(f"✅ Added {added} failed image(s)")

        if add_all:
            if not folder_ok:
                st.error("Folder does not exist or is not accessible.")
            else:
                st.session_state["last_folder_path"] = folder_path
                if not imgs_found:
                    st.warning("No images found in this folder. Supported types: PNG, JPG, JPEG, WEBP, BMP, GIF, TIF, TIFF, HEIC, HEIF, JFIF.")
                else:
                    lst: List[str] = st.session_state.setdefault("uploaded_images", [])
                    selected_lst: List[str] = st.session_state.setdefault("selected_images", [])
                    added = 0
                    for p in imgs_found:
                        if p not in lst:
                            lst.append(p)
                            added += 1
                        if p not in selected_lst:
                            selected_lst.append(p)
                    st.success(f"✅ Added {added} image(s) from folder")
    
    # Image Management (Simple List)
    imgs: List[str] = st.session_state.get("uploaded_images", [])
    
    if not imgs:
        st.info("No images uploaded yet. Upload some images to get started.")
        return
    
    # Original simple list (no wrapper), keep compact controls
    selected: List[str] = st.session_state.setdefault("selected_images", [])
    tags: Dict[str, Dict[str, str]] = st.session_state.setdefault("image_tags", {})

    st.markdown(f"### Input Files ({len(imgs)})")

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

    with st.container(border=True):
        for idx, pth in enumerate(imgs):
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            
            with col1:
                stable_key = f"sel_{hashlib.sha1(pth.encode('utf-8')).hexdigest()[:10]}"
                checked = st.checkbox(
                    "Select",
                    value=(pth in selected),
                    key=stable_key,
                    label_visibility="collapsed",
                )
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
                # Optional tags (collapsible per file)
                with st.expander("Tags"):
                    t = tags.get(pth, {"doc_type": "", "locale": "en-US"})
                    t["doc_type"] = st.text_input(
                        "Type",
                        value=t.get("doc_type", ""),
                        key=f"doc_{idx}",
                        placeholder="receipt, invoice, etc.",
                    )
                    t["locale"] = st.text_input(
                        "Locale",
                        value=t.get("locale", "en-US"),
                        key=f"loc_{idx}",
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

    try:
        registry = ensure_registry()
    except ModelRegistryError as exc:
        st.error(f"❌ Model registry error: {exc}")
        return

    selectable_models = [
        descriptor
        for descriptor in registry.models.values()
        if descriptor.show_in_ui
    ]

    selected_model_id = st.session_state.get("_selected_model_id")
    try:
        current_descriptor = registry.resolve(selected_model_id)
    except ModelRegistryError:
        current_descriptor = registry.resolve(None)
        st.session_state["_selected_model_id"] = current_descriptor.id
        st.session_state["__chosen_model_id__"] = current_descriptor.id
    else:
        st.session_state["_selected_model_id"] = current_descriptor.id
        st.session_state["__chosen_model_id__"] = current_descriptor.id

    if registry.policies.allow_frontend_model_selection and len(selectable_models) > 1:
        options = sorted(
            selectable_models,
            key=lambda m: (m.provider_label.lower(), m.label.lower()),
        )
        default_index = next(
            (idx for idx, mdl in enumerate(options) if mdl.id == current_descriptor.id),
            0,
        )
        choice = st.selectbox(
            "Select Model",
            options=options,
            index=default_index,
            format_func=lambda mdl: f"{mdl.provider_label} • {mdl.label}",
            help="Choose which configured model to use for this run.",
            key="model_selection",
        )
        if choice.id != current_descriptor.id:
            current_descriptor = choice
            st.session_state["_selected_model_id"] = choice.id
            st.session_state["__chosen_model_id__"] = choice.id
    else:
        st.caption(f"Using model: {current_descriptor.provider_label} • {current_descriptor.label}")
        st.session_state["__chosen_model_id__"] = current_descriptor.id

    # Load model context from registry
    try:
        model_ctx = _build_model_context(current_descriptor)
    except Exception as exc:
        st.error(f"❌ Failed to load model configuration: {exc}")
        return

    descriptor = model_ctx.descriptor

    # Output format selection (replaces single 'Unstructured' toggle)
    st.markdown("#### Output Format")
    output_mode = st.radio(
        "Choose output format",
        options=["Structured (Template)", "Structured (Auto JSON)", "Unstructured (Markdown)"],
        help="Use a saved template (schema + prompts), or let the model infer JSON without a schema, or get raw Markdown.",
        horizontal=True,
        key="output_mode_radio",
    )
    auto_json = output_mode == "Structured (Auto JSON)"
    unstructured = output_mode == "Unstructured (Markdown)"

    # Template selection
    selected_template = None
    selected_template_name = None
    global CURRENT_SYSTEM_PROMPT, CURRENT_USER_PROMPT, CURRENT_SCHEMA
    if not auto_json and not unstructured:
        project_templates = __SYNC__.list_templates_for_project(__ACTIVE_PROJECT_ID__)

        if not project_templates:
            st.info(
                "No templates are assigned to your current project. "
                "Ask an admin to assign templates in Admin → Templates."
            )
            st.stop()

        tpl_names = [tpl["name"] for tpl in project_templates]
        if "selected_template_name" in st.session_state:
            prev_selection = st.session_state.get("selected_template_name")
            default_idx = next(
                (idx for idx, name in enumerate(tpl_names) if name == prev_selection),
                0,
            )
        else:
            default_idx = 0

        selected_template_name = st.selectbox(
            "Template",
            tpl_names,
            index=default_idx,
            key="selected_template_name",
            help="Choose a template to extract data from images",
        )
        tpl = next(t for t in project_templates if t["name"] == selected_template_name)

        CURRENT_SYSTEM_PROMPT = tpl.get("system_prompt", "") or ""
        CURRENT_USER_PROMPT = tpl.get("user_prompt", "") or ""
        CURRENT_SCHEMA = tpl.get("schema", {}) or {}

        st.session_state["__active_template__"] = {
            "id": tpl.get("id"),
            "name": tpl.get("name"),
            "system_prompt": CURRENT_SYSTEM_PROMPT,
            "user_prompt": CURRENT_USER_PROMPT,
            "schema": CURRENT_SCHEMA,
        }

        selected_template = SimpleNamespace(
            id=tpl.get("id"),
            name=tpl.get("name"),
            description=tpl.get("description"),
            system_prompt=CURRENT_SYSTEM_PROMPT,
            user_prompt=CURRENT_USER_PROMPT,
            schema_json=CURRENT_SCHEMA,
        )

    # Synthetic template for Auto JSON / Unstructured
    if auto_json:
        selected_template_name = "Auto JSON"
        selected_template = SimpleNamespace(
            id=None,
            name=selected_template_name,
            system_prompt=(
                "You are a precise JSON generator. Analyze the image and output ONLY valid JSON. "
                "Capture fields present in the document with reasonable keys. Use numbers for numeric values, "
                "use arrays where multiple items are present, and nest objects as needed. Do not include code fences or explanations."
            ),
            user_prompt=(
                "Extract information from the image. Document type: {doc_type}. Locale: {locale}.\n"
                "Today's date is {today}. Respond with a single JSON object (or array if multiple entries). "
                "Return JSON ONLY."
            ),
            schema_json=None,
        )
        CURRENT_SYSTEM_PROMPT = selected_template.system_prompt
        CURRENT_USER_PROMPT = selected_template.user_prompt
        CURRENT_SCHEMA = selected_template.schema_json or {}
        st.session_state["__active_template__"] = {
            "id": selected_template.id,
            "name": selected_template.name,
            "system_prompt": CURRENT_SYSTEM_PROMPT,
            "user_prompt": CURRENT_USER_PROMPT,
            "schema": CURRENT_SCHEMA,
        }
    elif unstructured and selected_template is None:
        selected_template_name = "Unstructured"
        selected_template = SimpleNamespace(
            id=None,
            name=selected_template_name,
            system_prompt="",
            user_prompt="",
            schema_json=None,
        )
        CURRENT_SYSTEM_PROMPT = selected_template.system_prompt
        CURRENT_USER_PROMPT = selected_template.user_prompt
        CURRENT_SCHEMA = selected_template.schema_json or {}
        st.session_state["__active_template__"] = {
            "id": selected_template.id,
            "name": selected_template.name,
            "system_prompt": CURRENT_SYSTEM_PROMPT,
            "user_prompt": CURRENT_USER_PROMPT,
            "schema": CURRENT_SCHEMA,
        }
    
    if selected_template and not auto_json and not unstructured:
        # Show template info
        with st.expander("Template Details"):
            st.write(f"**Description:** {selected_template.description or 'No description'}")
            if selected_template.schema_json:
                st.json(selected_template.schema_json)
    
    # Show current settings
    with st.expander("Current Settings", expanded=False):
        st.write(f"**Provider:** {descriptor.provider_label}")
        st.write(f"**Model:** {descriptor.label}")
        max_tokens_display = descriptor.max_output_tokens if descriptor.max_output_tokens not in (None, 0) else "Unlimited"
        st.write(f"**Max Output Tokens:** {max_tokens_display}")
        st.write(f"**Temperature:** {descriptor.default_temperature}")
        st.write(f"**Endpoint:** {descriptor.base_url}")
        per_million_in = model_ctx.pricing["pricing"].get("input_per_million")
        per_million_out = model_ctx.pricing["pricing"].get("output_per_million")
        if per_million_in is not None or per_million_out is not None:
            in_display = f"${per_million_in:.4f}" if per_million_in is not None else "—"
            out_display = f"${per_million_out:.4f}" if per_million_out is not None else "—"
            st.write(f"**Price (input/output per 1M):** {in_display} / {out_display}")
        # Show INR per 1M if exchange rate available
        try:
            rate = get_usd_to_inr()
            if rate and (per_million_in is not None or per_million_out is not None):
                in_inr = f"₹{(per_million_in * rate):.2f}" if per_million_in is not None else "—"
                out_inr = f"₹{(per_million_out * rate):.2f}" if per_million_out is not None else "—"
                st.write(f"**Price (INR per 1M):** {in_inr} / {out_inr}  ")
        except Exception:
            pass
        reasoning_provider = model_ctx.reasoning.get("provider")
        if reasoning_provider:
            reason_parts = [reasoning_provider]
            effort = model_ctx.reasoning.get("effort_default")
            if effort:
                reason_parts.append(str(effort))
            st.write(f"**Reasoning defaults:** {', '.join(reason_parts)}")
    
    # Output mode toggle and Process button
    col1, col2, col3, col4 = st.columns([1.2, 1.4, 3.4, 1])
    with col1:
        # Reflect selected output format (disabled, driven by radio above)
        st.checkbox("Unstructured", value=unstructured, help="Skip schema; return plain text/Markdown.", disabled=True)
    with col2:
        per_file_mode = st.checkbox(
            "Per-file save",
            value=st.session_state.get('per_file_mode_default', False),
            key='per_file_mode_checkbox',
            help="Process each file separately and auto-save next to inputs",
        )

    # Format selection when per-file save is enabled
    save_formats = {'json': True, 'md': True, 'docx': True, 'xlsx': True}  # Default: all formats
    if per_file_mode:
        st.markdown("**Select formats to save:**")
        fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 1])
        with fcol1:
            save_formats['json'] = st.checkbox("JSON", value=True, key='save_json')
        with fcol2:
            save_formats['md'] = st.checkbox("Markdown", value=True, key='save_md')
        with fcol3:
            save_formats['docx'] = st.checkbox("Word", value=True, key='save_docx')
        with fcol4:
            save_formats['xlsx'] = st.checkbox("Excel", value=True, key='save_xlsx')

        # Ensure at least one format is selected
        if not any(save_formats.values()):
            st.warning("⚠️ Please select at least one format to save")
    # Allow auto-processing (e.g., after PDF conversion)
    auto_process_flag = bool(st.session_state.get('_auto_process_request', False))
    if auto_process_flag:
        st.session_state['per_file_mode_default'] = True
        per_file_mode = True
    with col3:
        process_clicked = st.button(
            f"▶️ Process {len(selected)} Image(s)",
            type="primary",
            use_container_width=True,
        )
    process_clicked = bool(process_clicked or auto_process_flag)
    if auto_process_flag:
        st.session_state.pop('_auto_process_request', None)
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
    
    # Handle automatic refresh after processing to update sidebar
    show_processing_results = False
    if st.session_state.get('_just_processed_refresh', False):
        st.session_state['_just_processed_refresh'] = False
        show_processing_results = True  # Show results even without button click
    
    # Per-file mode: run one-by-one and save
    if process_clicked and selected_template and per_file_mode and not show_processing_results:
        # Check if at least one format is selected
        if not any(save_formats.values()):
            st.error("⚠️ Please select at least one format to save before processing")
            return

        st.markdown("### Processing & Saving Per File")
        from scripts.export_records import to_docx_bytes, to_docx_from_text_bytes
        total = len(selected)
        progress = st.progress(0.0, text=f"Processing 0/{total}")
        saved_counts = {"json": 0, "md": 0, "docx": 0, "xlsx": 0}
        errors: List[str] = []
        st.session_state['per_file_mode_active'] = True
        # Prepare checkpoint (works for both folder-based and uploaded files)
        checkpoint: FolderCheckpoint | None = None
        base_folder_raw = st.session_state.get("last_folder_path")
        
        try:
            if base_folder_raw:
                # Folder-based processing
                base_folder = Path(_normalize_folder_input(base_folder_raw)).resolve()
            else:
                # Uploaded files - use upload directory
                base_folder = UPLOAD_DIR.resolve()
            
            checkpoint = FolderCheckpoint(base_folder)
            checkpoint.load()
            # record run context
            try:
                checkpoint.set_run_context(selected_template_name or None, descriptor.id, unstructured)
                # Set project context
                active_project = storage.get_active_project()
                if active_project:
                    checkpoint.set_project_context(active_project.id, active_project.name)
                checkpoint.save()
            except Exception:
                pass
        except Exception:
            checkpoint = None

        # Detect processing mode: folder-based or uploaded files
        is_folder_mode = bool(base_folder_raw)
        
        # For folder mode, create output directory
        output_dir = None
        if is_folder_mode:
            try:
                source_folder = Path(_normalize_folder_input(base_folder_raw)).resolve()
                output_dir = source_folder / "output"
                output_dir.mkdir(exist_ok=True)
                st.caption(f"📁 Output directory: {output_dir}")
            except Exception as e:
                st.error(f"Failed to create output directory: {e}")
                return
        
        # Store download data for uploaded files
        download_data = []  # List of (filename, format, bytes)
        db_records_created = 0  # Track successful database recordings
        
        for idx, img_path in enumerate(selected, start=1):
            try:
                per_file_gateway = gateway_from_descriptor(descriptor)
                _apply_project_api_key(per_file_gateway, descriptor.provider_id)
                if unstructured:
                    per_file_gateway.prefer_json_mode = False
                    per_file_gateway.prefer_tools = False
                result = _process_images(
                    model_ctx,
                    per_file_gateway,
                    selected_template,
                    [img_path],
                    tags,
                    unstructured=unstructured,
                )
                out = result.get("output")
                usage = result.get("usage")
                in_path = Path(img_path)
                base_name = _guess_original_stem(in_path)
                
                # Update checkpoint stats with tokens and cost
                if checkpoint is not None and usage:
                    tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
                    tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
                    cost_info = cost_from_usage(usage, model_ctx.pricing)
                    cost_usd = cost_info.get("total_usd", 0.0) if cost_info else 0.0
                    checkpoint.update_processing_stats(tokens_in, tokens_out, cost_usd)
                
                if not out:
                    errors.append(f"No output for {in_path.name}")
                    if checkpoint is not None:
                        checkpoint.mark_failed(img_path, "No output")
                else:
                    saved_files = {}
                    
                    if unstructured and isinstance(out, dict) and 'raw_text' in out:
                        text = str(out.get('raw_text') or "")
                        
                        # Process formats based on mode
                        if is_folder_mode:
                            # FOLDER MODE: Save to output/ directory
                            if save_formats.get('md', False):
                                md_bytes = (text or "").encode("utf-8")
                                (output_dir / f"{base_name}.md").write_bytes(md_bytes)
                                saved_files['md'] = str(output_dir / f"{base_name}.md")
                                saved_counts['md'] += 1
                            if save_formats.get('docx', False):
                                docx_bytes = to_docx_from_text_bytes(text, title=base_name)
                                (output_dir / f"{base_name}.docx").write_bytes(docx_bytes)
                                saved_files['docx'] = str(output_dir / f"{base_name}.docx")
                                saved_counts['docx'] += 1
                            if save_formats.get('json', False):
                                recs = ensure_records({"raw_text": text})
                                json_bytes = to_json_bytes(recs)
                                (output_dir / f"{base_name}.json").write_bytes(json_bytes)
                                saved_files['json'] = str(output_dir / f"{base_name}.json")
                                saved_counts['json'] += 1
                            if save_formats.get('xlsx', False):
                                recs = ensure_records({"raw_text": text})
                                cols = all_columns(recs)
                                xlsx_bytes = to_xlsx_bytes(recs, cols)
                                (output_dir / f"{base_name}.xlsx").write_bytes(xlsx_bytes)
                                saved_files['xlsx'] = str(output_dir / f"{base_name}.xlsx")
                                saved_counts['xlsx'] += 1
                        else:
                            # UPLOAD MODE: Prepare for download
                            if save_formats.get('md', False):
                                md_bytes = (text or "").encode("utf-8")
                                download_data.append((base_name, 'md', md_bytes))
                                saved_counts['md'] += 1
                            if save_formats.get('docx', False):
                                docx_bytes = to_docx_from_text_bytes(text, title=base_name)
                                download_data.append((base_name, 'docx', docx_bytes))
                                saved_counts['docx'] += 1
                            if save_formats.get('json', False):
                                recs = ensure_records({"raw_text": text})
                                json_bytes = to_json_bytes(recs)
                                download_data.append((base_name, 'json', json_bytes))
                                saved_counts['json'] += 1
                            if save_formats.get('xlsx', False):
                                recs = ensure_records({"raw_text": text})
                                cols = all_columns(recs)
                                xlsx_bytes = to_xlsx_bytes(recs, cols)
                                download_data.append((base_name, 'xlsx', xlsx_bytes))
                                saved_counts['xlsx'] += 1
                        
                        if checkpoint is not None:
                            checkpoint.mark_processed(img_path, saved_files)
                        
                        # Record run in database for project stats
                        try:
                            cost_to_record = cost_usd if 'cost_usd' in locals() and cost_usd else None
                            storage.record_run(
                                provider_id=model_ctx.provider_record.id,
                                template_id=(getattr(selected_template, 'id', None) or None),
                                input_images=[img_path],
                                output={"raw_text": text},
                                cost_usd=cost_to_record,
                                status="completed"
                            )
                            db_records_created += 1
                            _track_usage_analytics(getattr(selected_template, "id", None))
                        except Exception as e:
                            # Silent but log to errors for debugging
                            errors.append(f"DB recording failed for {Path(img_path).name}: {str(e)[:100]}")
                    else:
                        recs = ensure_records(out)
                        cols = all_columns(recs)
                        
                        # Process formats based on mode
                        if is_folder_mode:
                            # FOLDER MODE: Save to output/ directory
                            if save_formats.get('json', False):
                                json_bytes = to_json_bytes(recs)
                                (output_dir / f"{base_name}.json").write_bytes(json_bytes)
                                saved_files['json'] = str(output_dir / f"{base_name}.json")
                                saved_counts['json'] += 1
                            if save_formats.get('md', False):
                                md_bytes = to_markdown_bytes(recs, cols)
                                (output_dir / f"{base_name}.md").write_bytes(md_bytes)
                                saved_files['md'] = str(output_dir / f"{base_name}.md")
                                saved_counts['md'] += 1
                            if save_formats.get('docx', False):
                                docx_bytes = to_docx_bytes(recs, cols)
                                (output_dir / f"{base_name}.docx").write_bytes(docx_bytes)
                                saved_files['docx'] = str(output_dir / f"{base_name}.docx")
                                saved_counts['docx'] += 1
                            if save_formats.get('xlsx', False):
                                xlsx_bytes = to_xlsx_bytes(recs, cols)
                                (output_dir / f"{base_name}.xlsx").write_bytes(xlsx_bytes)
                                saved_files['xlsx'] = str(output_dir / f"{base_name}.xlsx")
                                saved_counts['xlsx'] += 1
                        else:
                            # UPLOAD MODE: Prepare for download
                            if save_formats.get('json', False):
                                json_bytes = to_json_bytes(recs)
                                download_data.append((base_name, 'json', json_bytes))
                                saved_counts['json'] += 1
                            if save_formats.get('md', False):
                                md_bytes = to_markdown_bytes(recs, cols)
                                download_data.append((base_name, 'md', md_bytes))
                                saved_counts['md'] += 1
                            if save_formats.get('docx', False):
                                docx_bytes = to_docx_bytes(recs, cols)
                                download_data.append((base_name, 'docx', docx_bytes))
                                saved_counts['docx'] += 1
                            if save_formats.get('xlsx', False):
                                xlsx_bytes = to_xlsx_bytes(recs, cols)
                                download_data.append((base_name, 'xlsx', xlsx_bytes))
                                saved_counts['xlsx'] += 1
                        
                        if checkpoint is not None:
                            checkpoint.mark_processed(img_path, saved_files)
                        
                        # Record run in database for project stats
                        try:
                            cost_to_record = cost_usd if 'cost_usd' in locals() and cost_usd else None
                            storage.record_run(
                                provider_id=model_ctx.provider_record.id,
                                template_id=(getattr(selected_template, 'id', None) or None),
                                input_images=[img_path],
                                output=out,
                                cost_usd=cost_to_record,
                                status="completed"
                            )
                            db_records_created += 1
                            _track_usage_analytics(getattr(selected_template, "id", None))
                        except Exception as e:
                            # Silent but log to errors for debugging
                            errors.append(f"DB recording failed for {Path(img_path).name}: {str(e)[:100]}")
            except Exception as e:
                errors.append(f"Save failed for {Path(img_path).name}: {e}")
                if checkpoint is not None:
                    checkpoint.mark_failed(img_path, str(e))
            finally:
                progress.progress(idx/total, text=f"Processing {idx}/{total}")
                if checkpoint is not None:
                    try:
                        checkpoint.save()
                    except Exception:
                        pass
        progress.empty()

        # Build success message and download buttons based on mode
        saved_msgs = []
        if save_formats.get('json', False):
            saved_msgs.append(f"{saved_counts['json']} JSON")
        if save_formats.get('md', False):
            saved_msgs.append(f"{saved_counts['md']} Markdown")
        if save_formats.get('docx', False):
            saved_msgs.append(f"{saved_counts['docx']} Word")
        if save_formats.get('xlsx', False):
            saved_msgs.append(f"{saved_counts['xlsx']} Excel")

        if is_folder_mode:
            # FOLDER MODE: Show save location
            if saved_msgs:
                st.success(f"✅ Saved: {', '.join(saved_msgs)} files to `{output_dir}` directory")
            else:
                st.warning("⚠️ No files saved (no formats selected)")
        else:
            # UPLOAD MODE: Show download buttons
            if download_data:
                st.success(f"✅ Processed {len(selected)} files. Download buttons below:")
                st.markdown("### 📥 Download Processed Files")
                
                # Group downloads by filename
                files_by_name = {}
                for base_name, fmt, data in download_data:
                    if base_name not in files_by_name:
                        files_by_name[base_name] = []
                    files_by_name[base_name].append((fmt, data))
                
                # Create download buttons for each file
                for base_name, formats in files_by_name.items():
                    st.markdown(f"**{base_name}**")
                    cols = st.columns(len(formats))
                    for idx, (fmt, data) in enumerate(formats):
                        with cols[idx]:
                            # Map format to extension and MIME type
                            ext_map = {
                                'json': ('.json', 'application/json'),
                                'md': ('.md', 'text/markdown'),
                                'docx': ('.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
                                'xlsx': ('.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                            }
                            ext, mime = ext_map[fmt]
                            label_map = {'json': '📄 JSON', 'md': '📝 Markdown', 'docx': '📘 Word', 'xlsx': '📊 Excel'}
                            st.download_button(
                                label_map[fmt],
                                data=data,
                                file_name=f"{base_name}{ext}",
                                mime=mime,
                                key=f"download_{base_name}_{fmt}",
                                use_container_width=True
                            )
                    st.divider()
            else:
                st.warning("⚠️ No files processed")
        
        if errors:
            with st.expander("Errors encountered"):
                for e in errors:
                    st.error(e)
        
        # Show checkpoint and database information
        st.success(f"✅ **Processing complete:** {len(selected) - len(errors)}/{len(selected)} files successful")
        
        if db_records_created > 0:
            st.info(f"📊 **Database updated:** {db_records_created} runs recorded for project stats")
        else:
            st.warning("⚠️ **Database not updated:** No runs were recorded. Check errors below.")
        
        if checkpoint is not None:
            checkpoint_stats = checkpoint.get_processing_stats()
            st.info(
                f"📋 **Checkpoint saved:** ${checkpoint_stats.get('total_cost_usd', 0.0):.4f} total cost"
            )
            if is_folder_mode:
                checkpoint_path = Path(_normalize_folder_input(base_folder_raw)).resolve() / ".img2json.checkpoint.json"
                st.caption(f"📁 Checkpoint location: `{checkpoint_path}`")
        
        # Cleanup uploaded files after successful processing in upload mode
        if not is_folder_mode:
            # Store list of uploaded files to clean
            uploaded_files_to_clean = [img for img in selected if str(UPLOAD_DIR.resolve()) in img]
            if uploaded_files_to_clean:
                try:
                    for img_path in uploaded_files_to_clean:
                        Path(img_path).unlink(missing_ok=True)
                    st.caption(f"🗑️ Cleaned up {len(uploaded_files_to_clean)} uploaded file(s)")
                except Exception:
                    pass  # Silent failure
            
            # Clear session state after cleanup
            if 'uploaded_images' in st.session_state:
                # Remove only the files that were processed
                remaining = [img for img in st.session_state['uploaded_images'] 
                            if img not in uploaded_files_to_clean]
                st.session_state['uploaded_images'] = remaining
                st.session_state['selected_images'] = []
        
        # Generate project report
        active_project = storage.get_active_project()
        if active_project and checkpoint:
            st.markdown("### 📊 Project Report")
            with st.spinner("Generating project report..."):
                try:
                    from app.core.report_generator import save_project_report
                    export_dir = Path("export")
                    # Use source folder for folder mode, None for upload mode
                    checkpoint_dir = Path(_normalize_folder_input(base_folder_raw)).resolve() if base_folder_raw else None
                    report_path = save_project_report(active_project.id, export_dir, checkpoint_dir=checkpoint_dir)
                    
                    st.success(f"✅ Project report generated: `{report_path}`")
                    
                    # Offer download
                    report_content = report_path.read_text(encoding="utf-8")
                    st.download_button(
                        "📥 Download Report",
                        data=report_content,
                        file_name=report_path.name,
                        mime="text/markdown",
                        key="download_project_report"
                    )
                except Exception as e:
                    st.warning(f"Could not generate report: {e}")
        
        # Store results in session state for display after refresh
        st.session_state['_last_processing_results'] = {
            'saved_msgs': saved_msgs,
            'download_data': download_data,
            'is_folder_mode': is_folder_mode,
            'output_dir': str(output_dir) if output_dir else None,
            'errors': errors,
            'db_records_created': db_records_created,
            'checkpoint_stats': checkpoint.get_processing_stats() if checkpoint else None,
            'checkpoint_path': str(Path(_normalize_folder_input(base_folder_raw)).resolve() / ".img2json.checkpoint.json") if is_folder_mode and base_folder_raw else None,
            'selected_count': len(selected),
            'uploaded_files_to_clean': [img for img in selected if str(UPLOAD_DIR.resolve()) in img] if not is_folder_mode else []
        }
        
        # End per-file mode early to avoid aggregated rendering
        st.session_state['per_file_mode_active'] = False
        
        # Trigger automatic page refresh to update sidebar stats
        st.session_state['_just_processed_refresh'] = True
        st.rerun()
        
        return
    
    # Display stored results after refresh
    if show_processing_results and '_last_processing_results' in st.session_state:
        results = st.session_state['_last_processing_results']
        
        st.markdown("### Processing & Saving Per File")
        st.markdown("---")
        
        # Build success message and download buttons based on mode
        if results['is_folder_mode']:
            # FOLDER MODE: Show save location
            if results['saved_msgs']:
                st.success(f"✅ Saved: {', '.join(results['saved_msgs'])} files to `{results['output_dir']}` directory")
            else:
                st.warning("⚠️ No files saved (no formats selected)")
        else:
            # UPLOAD MODE: Show download buttons
            if results['download_data']:
                st.success(f"✅ Processed {results['selected_count']} files. Download buttons below:")
                st.markdown("### 📥 Download Processed Files")
                
                # Group downloads by filename
                files_by_name = {}
                for base_name, fmt, data in results['download_data']:
                    if base_name not in files_by_name:
                        files_by_name[base_name] = []
                    files_by_name[base_name].append((fmt, data))
                
                # Create download buttons for each file
                for base_name, formats in files_by_name.items():
                    st.markdown(f"**{base_name}**")
                    cols = st.columns(len(formats))
                    for idx, (fmt, data) in enumerate(formats):
                        with cols[idx]:
                            ext_map = {
                                'json': ('.json', 'application/json'),
                                'md': ('.md', 'text/markdown'),
                                'docx': ('.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
                                'xlsx': ('.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                            }
                            ext, mime = ext_map[fmt]
                            label_map = {'json': '📄 JSON', 'md': '📝 Markdown', 'docx': '📘 Word', 'xlsx': '📊 Excel'}
                            st.download_button(
                                label_map[fmt],
                                data=data,
                                file_name=f"{base_name}{ext}",
                                mime=mime,
                                key=f"download_refresh_{base_name}_{fmt}",
                                use_container_width=True
                            )
                    st.divider()
            else:
                st.warning("⚠️ No files processed")
        
        if results['errors']:
            with st.expander("Errors encountered"):
                for e in results['errors']:
                    st.error(e)
        
        # Show status messages
        st.success(f"✅ **Processing complete:** {results['selected_count'] - len(results['errors'])}/{results['selected_count']} files successful")
        
        if results['db_records_created'] > 0:
            st.info(f"📊 **Database updated:** {results['db_records_created']} runs recorded (sidebar stats refreshed!)")
        else:
            st.warning("⚠️ **Database not updated:** No runs were recorded. Check errors above.")
        
        if results['checkpoint_stats']:
            st.info(f"📋 **Checkpoint saved:** ${results['checkpoint_stats'].get('total_cost_usd', 0.0):.4f} total cost")
            if results['checkpoint_path']:
                st.caption(f"📁 Checkpoint location: `{results['checkpoint_path']}`")
        
        # Cleanup uploaded files
        if results['uploaded_files_to_clean']:
            try:
                for img_path in results['uploaded_files_to_clean']:
                    Path(img_path).unlink(missing_ok=True)
                st.caption(f"🗑️ Cleaned up {len(results['uploaded_files_to_clean'])} uploaded file(s)")
            except Exception:
                pass
            
            # Clear session state
            if 'uploaded_images' in st.session_state:
                remaining = [img for img in st.session_state['uploaded_images'] 
                            if img not in results['uploaded_files_to_clean']]
                st.session_state['uploaded_images'] = remaining
                st.session_state['selected_images'] = []
        
        # Clear the stored results
        del st.session_state['_last_processing_results']
        
        return

    if process_clicked and selected_template:
        st.markdown("### Processing Results")
        
        # Show estimated time based on number of images
        estimated_time = len(selected) * 5  # Estimate 5 seconds per image
        progress_text = f"Processing {len(selected)} image(s)... Estimated time: {estimated_time} seconds"
        
        with st.spinner(progress_text):
            gateway = gateway_from_descriptor(descriptor)
            _apply_project_api_key(gateway, descriptor.provider_id)
            if unstructured:
                gateway.prefer_json_mode = False
                gateway.prefer_tools = False
            # Update checkpoint run context if folder-based
            base_folder_raw2 = st.session_state.get("last_folder_path")
            if base_folder_raw2:
                try:
                    base_folder = Path(_normalize_folder_input(base_folder_raw2)).resolve()
                    checkpoint2 = FolderCheckpoint(base_folder)
                    checkpoint2.load()
                    checkpoint2.set_run_context(selected_template_name or None, descriptor.id, unstructured)
                    # Set project context
                    active_project = storage.get_active_project()
                    if active_project:
                        checkpoint2.set_project_context(active_project.id, active_project.name)
                    checkpoint2.save()
                except Exception:
                    pass
            result = _process_images(
                model_ctx,
                gateway,
                selected_template,
                selected,
                tags,
                unstructured=unstructured,
            )
        
        # Store result in session state for persistence
        st.session_state['last_result'] = result
        st.session_state['last_template_name'] = selected_template_name or ""
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
                st.write("• Asking an administrator to verify the server configuration")
            elif "401" in error_msg or "403" in error_msg or "authentication" in error_msg:
                st.warning("🔑 **Authentication Issue**")
                st.write("The configured server credentials may be invalid. Contact an administrator to refresh secrets or tokens.")
            elif "429" in error_msg or "rate" in error_msg:
                st.warning("⏱️ **Rate Limit**")
                st.write("You've hit the rate limit. Try:")
                st.write("• Waiting a few minutes before retrying")
                st.write("• Processing fewer images at once")
                st.write("• Upgrading your API plan for higher limits")
            elif "timeout" in error_msg or "408" in error_msg:
                st.warning("⏰ **Timeout Error**")
                base_url_lower = (descriptor.base_url or "").lower()
                is_local_error_check = any(token in base_url_lower for token in ("localhost", "127.0.0.1", "0.0.0.0"))
                if is_local_error_check:
                    st.write("Local models can be slow. Try:")
                    st.write("• Increasing the server timeout (current: " + str(int(descriptor.timeouts.total_s)) + "s)")
                    st.write("• Using a smaller model")
                    st.write("• Reducing max_output_tokens")
                    st.write("• Processing one image at a time")
                    if descriptor.timeouts.total_s < 300:
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
                            'model_id': descriptor.id,
                            'max_output_tokens': descriptor.max_output_tokens,
                            'capabilities': model_ctx.caps,
                            'pricing': model_ctx.pricing,
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
                    limit_tokens = descriptor.max_output_tokens if descriptor.max_output_tokens not in (None, 0) else 4096
                    if output_tokens < 250 and limit_tokens > 1000:
                        st.warning(f"⚠️ Output may be truncated: only {output_tokens} tokens generated out of {limit_tokens} requested. Check your model's configuration.")
                    
                    # Calculate cost if pricing available
                    if model_ctx.pricing:
                        cost_info = cost_from_usage(usage_data, model_ctx.pricing)
                        total_cost = cost_info.get("total_usd") if cost_info else None
                        if total_cost:
                            # USD + INR (if available)
                            try:
                                rate = get_usd_to_inr()
                                if rate:
                                    inr = convert_usd_to_inr(float(total_cost), rate=rate)
                                    if inr is not None:
                                        st.write(f"**Estimated cost:** ${float(total_cost):.4f} • ₹{inr:.2f} (USD→INR {rate})")
                                    else:
                                        st.write(f"**Estimated cost:** ${float(total_cost):.4f}")
                                else:
                                    st.write(f"**Estimated cost:** ${float(total_cost):.4f}")
                            except Exception:
                                st.write(f"**Estimated cost:** ${float(total_cost):.4f}")
                            
                            # Track cumulative cost in session
                            if process_clicked:
                                if 'cumulative_cost' not in st.session_state:
                                    st.session_state['cumulative_cost'] = 0.0
                                st.session_state['cumulative_cost'] += float(total_cost)
                            
                            # Show cumulative cost if multiple runs
                            if 'cumulative_cost' in st.session_state:
                                st.caption(f"Session total: ${st.session_state['cumulative_cost']:.4f}")
            
            # Store run in database
            if result.get("output"):
                try:
                    # Calculate cost if pricing info available
                    cost_usd = None
                    if result.get("usage") and model_ctx.pricing:
                        cost_info = cost_from_usage(result["usage"], model_ctx.pricing)
                        if cost_info:
                            cost_usd = cost_info.get("total_usd")
                    
                    run = storage.record_run(
                        provider_id=model_ctx.provider_record.id,
                        template_id=(getattr(selected_template, 'id', None) or None),
                        input_images=selected,
                        output=result["output"],
                        cost_usd=cost_usd,
                        status="completed" if not result.get("error") else "error"
                    )
                    st.caption(f"Run saved with ID: {run.id}")
                    _track_usage_analytics(getattr(selected_template, "id", None))
                except Exception as e:
                    st.warning(f"Could not save run: {e}")


if __name__ == "__main__":
    run()
