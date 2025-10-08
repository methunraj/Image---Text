from __future__ import annotations

import json
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
        except Exception:
            # best effort: fall back to direct write
            self.path.write_text(text, encoding="utf-8")
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

