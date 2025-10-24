from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKPOINT_FILENAME = ".img2json.checkpoint.json"


def _now_iso() -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception:
        return str(int(time.time()))


def _rel_under(base: Path, child: Path) -> Optional[str]:
    try:
        return str(child.resolve().relative_to(base.resolve()))
    except Exception:
        return None


@dataclass
class FileEntry:
    rel_path: str
    size: int
    mtime: float
    status: str = "pending"  # pending | processed | failed
    attempts: int = 0
    last_error: Optional[str] = None
    outputs: Optional[Dict[str, str]] = None


class FolderCheckpoint:
    """A lightweight JSON checkpoint stored inside the selected folder.

    File format (v1):
    {
      "version": 1,
      "folder_abs": "/abs/path",
      "created_at": "...",
      "updated_at": "...",
      "template_name": "...",    # optional context
      "model_id": "...",          # optional context
      "unstructured": false,       # optional context
      "files": {
        "relative/path.jpg": {
          "size": 12345,
          "mtime": 1712345678.0,
          "status": "pending|processed|failed",
          "attempts": 0,
          "last_error": null,
          "outputs": {"json": "...", "md": "...", "docx": "..."}
        }, ...
      }
    }
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.path = folder / CHECKPOINT_FILENAME
        self.data: Dict[str, Any] = {}

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8")) or {}
            except Exception:
                self.data = {}
        if not self.data:
            self.data = {
                "version": 1,
                "folder_abs": str(self.folder.resolve()),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "files": {},
            }

    def save(self) -> None:
        self.data["updated_at"] = _now_iso()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        text = json.dumps(self.data, ensure_ascii=False, indent=2)
        tmp.write_text(text, encoding="utf-8")
        try:
            os.replace(tmp, self.path)
        except Exception as e:
            # Fallback to direct write, but raise if that fails too
            try:
                self.path.write_text(text, encoding="utf-8")
            except Exception as write_err:
                # Ensure temporary file is removed before surfacing the error
                try:
                    if tmp.exists():
                        tmp.unlink()
                finally:
                    raise IOError(f"Failed to save checkpoint: {e}; fallback write failed: {write_err}")
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    # Context helpers
    def set_run_context(self, template_name: Optional[str], model_id: Optional[str], unstructured: Optional[bool]) -> None:
        if template_name is not None:
            self.data["template_name"] = template_name
        if model_id is not None:
            self.data["model_id"] = model_id
        if unstructured is not None:
            self.data["unstructured"] = bool(unstructured)
    
    def set_project_context(self, project_id: Optional[int], project_name: Optional[str]) -> None:
        """Set project context for this checkpoint."""
        if project_id is not None:
            self.data["project_id"] = project_id
        if project_name is not None:
            self.data["project_name"] = project_name
    
    def update_processing_stats(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        """Update processing statistics incrementally."""
        stats = self.data.setdefault("processing_stats", {
            "total_images": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_cost_usd": 0.0,
            "images_processed": 0,
            "images_failed": 0,
            "avg_cost_per_image": 0.0
        })
        
        MAX_TOKEN_COUNT = 2**53 - 1  # JavaScript safe integer max
        cur_in = int(stats.get("total_tokens_input", 0) or 0)
        cur_out = int(stats.get("total_tokens_output", 0) or 0)
        if cur_in + int(tokens_in) > MAX_TOKEN_COUNT:
            raise ValueError(f"Token count overflow: {cur_in} + {tokens_in} exceeds max")
        if cur_out + int(tokens_out) > MAX_TOKEN_COUNT:
            raise ValueError(f"Token count overflow: {cur_out} + {tokens_out} exceeds max")
        stats["total_tokens_input"] = cur_in + int(tokens_in)
        stats["total_tokens_output"] = cur_out + int(tokens_out)
        stats["total_cost_usd"] = stats.get("total_cost_usd", 0.0) + cost_usd
        stats["images_processed"] = stats.get("images_processed", 0) + 1
        stats["total_images"] = stats.get("images_processed", 0)
        
        # Update average
        if stats["images_processed"] > 0 and math.isfinite(stats.get("total_cost_usd", 0.0)):
            stats["avg_cost_per_image"] = stats["total_cost_usd"] / stats["images_processed"]
        else:
            stats["avg_cost_per_image"] = 0.0
        
        self.data["processing_stats"] = stats
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.data.get("processing_stats", {
            "total_images": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_cost_usd": 0.0,
            "images_processed": 0,
            "images_failed": 0,
            "avg_cost_per_image": 0.0
        })

    # File mapping helpers
    def _files(self) -> Dict[str, Any]:
        files = self.data.get("files")
        if not isinstance(files, dict):
            files = {}
            self.data["files"] = files
        return files

    def ensure_entries(self, abs_paths: List[str]) -> None:
        files = self._files()
        for pth in abs_paths:
            rel = _rel_under(self.folder, Path(pth))
            if not rel:
                continue
            try:
                st = Path(pth).stat()
                size = int(st.st_size)
                mtime = float(st.st_mtime)
            except Exception:
                size = 0
                mtime = 0.0
            if rel not in files:
                files[rel] = {"size": size, "mtime": mtime, "status": "pending", "attempts": 0}
            else:
                # Update size/mtime to reflect current file state
                entry = files[rel]
                if isinstance(entry, dict):
                    entry["size"] = size
                    entry["mtime"] = mtime

    def mark_processed(self, abs_path: str, outputs: Optional[Dict[str, str]] = None) -> None:
        files = self._files()
        rel = _rel_under(self.folder, Path(abs_path))
        if not rel:
            return
        entry = files.get(rel) or {}
        entry.update({"status": "processed", "last_error": None})
        if outputs:
            entry["outputs"] = outputs
        files[rel] = entry

    def mark_failed(self, abs_path: str, error_message: str) -> None:
        files = self._files()
        rel = _rel_under(self.folder, Path(abs_path))
        if not rel:
            return
        entry = files.get(rel) or {}
        attempts = int(entry.get("attempts", 0) or 0) + 1
        entry.update({"status": "failed", "attempts": attempts, "last_error": str(error_message)[:500]})
        files[rel] = entry

    def reset(self) -> None:
        files = self._files()
        for rel, entry in list(files.items()):
            if isinstance(entry, dict):
                entry["status"] = "pending"
                entry["attempts"] = 0
                entry["last_error"] = None

    def get_stats_for(self, abs_paths: List[str]) -> Dict[str, int]:
        files = self._files()
        counts = {"total": 0, "processed": 0, "failed": 0, "pending": 0}
        for pth in abs_paths:
            rel = _rel_under(self.folder, Path(pth))
            if not rel:
                continue
            counts["total"] += 1
            status = (files.get(rel) or {}).get("status")
            if status == "processed":
                counts["processed"] += 1
            elif status == "failed":
                counts["failed"] += 1
            else:
                counts["pending"] += 1
        return counts

    def pending_files(self, abs_paths: List[str]) -> List[str]:
        files = self._files()
        out: List[str] = []
        for pth in abs_paths:
            rel = _rel_under(self.folder, Path(pth))
            if not rel:
                continue
            status = (files.get(rel) or {}).get("status")
            if status not in ("processed",):
                out.append(pth)
        return out

    def failed_files(self, abs_paths: List[str]) -> List[str]:
        files = self._files()
        out: List[str] = []
        for pth in abs_paths:
            rel = _rel_under(self.folder, Path(pth))
            if not rel:
                continue
            status = (files.get(rel) or {}).get("status")
            if status == "failed":
                out.append(pth)
        return out

    def prune_missing(self) -> None:
        files = self._files()
        to_delete: List[str] = []
        for rel in files.keys():
            abs_p = self.folder / rel
            try:
                if not abs_p.exists():
                    to_delete.append(rel)
            except Exception:
                to_delete.append(rel)
        for rel in to_delete:
            files.pop(rel, None)
