"""Hugging Face model management — download, list, delete.

Downloads use ``huggingface_hub.snapshot_download`` with a native
``tqdm`` progress callback instead of shelling out to
``huggingface-cli``.  This is faster, more reliable, and works for
every model format (GGUF, safetensors, pytorch).
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.utils import (
    tqdm as hf_tqdm,            # HF's wrapped tqdm
)


class DownloadCancelled(Exception):
    """Raised when a download is stopped by the user."""


class ModelManager:
    """Manages local Hugging Face model cache."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(
            cache_dir or os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        self.hub_cache = self.cache_dir / "hub"
        self._cancel_event = threading.Event()

    # ── List downloaded models ─────────────────────────────────────────
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        try:
            cache_info = scan_cache_dir(self.hub_cache)
            models: list[dict] = []
            for repo in cache_info.repos:
                if repo.repo_type == "model":
                    # Detect model format from file extensions
                    file_exts = set()
                    for rev in repo.revisions:
                        for f in rev.files:
                            file_exts.add(Path(f.file_path).suffix)
                    if ".gguf" in file_exts:
                        fmt = "gguf"
                    elif ".safetensors" in file_exts:
                        fmt = "safetensors"
                    elif ".bin" in file_exts:
                        fmt = "pytorch"
                    else:
                        fmt = "unknown"
                    models.append(
                        {
                            "repo_id": repo.repo_id,
                            "size": repo.size_on_disk,
                            "size_str": self._human_size(repo.size_on_disk),
                            "nb_files": repo.nb_files,
                            "last_modified": repo.last_modified,
                            "format": fmt,
                            "revisions": [
                                r.commit_hash[:10] for r in repo.revisions
                            ],
                        }
                    )
            return sorted(models, key=lambda m: m["repo_id"])
        except Exception:
            return []

    # ── Search Hugging Face Hub ────────────────────────────────────────
    def search_models(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        api = HfApi()
        results = api.list_models(
            search=query,
            sort="downloads",
            direction=-1,
            limit=limit,
        )
        return [
            {
                "id": m.id,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "pipeline_tag": getattr(m, "pipeline_tag", "—"),
            }
            for m in results
        ]

    # ── Download ───────────────────────────────────────────────────────
    def download_model(self, model_id: str) -> str:
        """Download *model_id* into the local cache. Returns local path."""
        path = snapshot_download(
            model_id,
            cache_dir=str(self.hub_cache),
        )
        return path

    def list_repo_files(
        self, model_id: str
    ) -> List[Dict[str, Any]]:
        """Return the file manifest for *model_id* (name, size, size_str)."""
        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)
        result: List[Dict[str, Any]] = []
        for s in (info.siblings or []):
            fsize = getattr(s, "size", 0) or 0
            result.append({
                "name": s.rfilename,
                "size": fsize,
                "size_str": self._human_size(fsize),
            })
        return result

    def download_model_with_progress(
        self,
        model_id: str,
        on_progress: Optional[Callable] = None,
        on_file_list: Optional[Callable] = None,
        filenames: Optional[List[str]] = None,
    ) -> str:
        """Download *model_id* using ``huggingface_hub.snapshot_download``.

        Progress is reported through the native ``tqdm`` callback —
        no subprocess, no stderr parsing.

        Parameters
        ----------
        filenames:
            If given, only these files are downloaded (e.g. a single
            GGUF variant).  ``None`` means download everything.
        """
        self._cancel_event.clear()

        # ── File manifest (for UI) ─────────────────────────────────
        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)
        siblings = info.siblings or []
        if filenames:
            selected = set(filenames)
            siblings = [s for s in siblings if s.rfilename in selected]

        files_info: List[Dict[str, Any]] = []
        for s in siblings:
            fsize = getattr(s, "size", 0) or 0
            files_info.append({
                "name": s.rfilename,
                "size": fsize,
                "size_str": self._human_size(fsize),
            })

        total_bytes = sum(f["size"] for f in files_info)
        total_files = len(files_info)

        if on_file_list:
            on_file_list(files_info)

        # ── Progress tracking state ────────────────────────────────
        _state: Dict[str, Any] = {
            "bytes_done": 0,
            "current_file": files_info[0]["name"] if files_info else "",
            "start_time": time.monotonic(),
            "last_update": 0.0,
        }

        # Monkey-patch tqdm to intercept download progress
        _orig_tqdm_init = hf_tqdm.__init__
        _orig_tqdm_update = hf_tqdm.update
        _outer_self = self   # reference for cancel check

        def _patched_init(tqdm_self: Any, *args: Any, **kwargs: Any) -> None:
            _orig_tqdm_init(tqdm_self, *args, **kwargs)
            desc = getattr(tqdm_self, "desc", "") or ""
            # tqdm description often contains the filename
            for f in files_info:
                if f["name"] in desc or desc.endswith(f["name"]):
                    _state["current_file"] = f["name"]
                    break

        def _patched_update(tqdm_self: Any, n: int = 1) -> None:
            # Cancel check
            if _outer_self._cancel_event.is_set():
                # Close the tqdm bar and raise to abort
                tqdm_self.close()
                raise DownloadCancelled(
                    f"Download of {model_id} cancelled"
                )

            _orig_tqdm_update(tqdm_self, n)

            _state["bytes_done"] += n

            # Throttle UI updates to ~4×/sec
            now = time.monotonic()
            if on_progress and (now - _state["last_update"]) >= 0.25:
                _state["last_update"] = now
                elapsed = now - _state["start_time"]
                speed = (
                    _state["bytes_done"] / elapsed if elapsed > 1.0 else 0.0
                )
                on_progress(
                    _state["current_file"],
                    1,
                    total_files,
                    _state["bytes_done"],
                    total_bytes,
                    "downloading",
                    speed,
                )

        # ── Perform download ───────────────────────────────────────
        try:
            hf_tqdm.__init__ = _patched_init       # type: ignore[assignment]
            hf_tqdm.update = _patched_update        # type: ignore[assignment]

            allow_patterns = filenames or None

            path = snapshot_download(
                model_id,
                cache_dir=str(self.hub_cache),
                allow_patterns=allow_patterns,
            )
        except DownloadCancelled:
            raise
        except Exception:
            raise
        finally:
            # Always restore original tqdm methods
            hf_tqdm.__init__ = _orig_tqdm_init      # type: ignore[assignment]
            hf_tqdm.update = _orig_tqdm_update       # type: ignore[assignment]

        # ── Notify completion ──────────────────────────────────────
        if on_progress:
            for idx, f in enumerate(files_info, 1):
                on_progress(
                    f["name"], idx, total_files,
                    total_bytes, total_bytes, "done", 0.0,
                )

        return path

    def cancel_download(self) -> None:
        """Signal the in-progress download to stop."""
        self._cancel_event.set()

    @property
    def is_download_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    # ── Delete ─────────────────────────────────────────────────────────
    def delete_model(self, model_id: str) -> bool:
        try:
            cache_info = scan_cache_dir(self.hub_cache)
            for repo in cache_info.repos:
                if repo.repo_id == model_id and repo.repo_type == "model":
                    revisions = [r.commit_hash for r in repo.revisions]
                    strategy = cache_info.delete_revisions(*revisions)
                    strategy.execute()
                    # Remove .trash so files are permanently deleted
                    trash_dir = self.hub_cache / ".trash"
                    if trash_dir.exists():
                        shutil.rmtree(trash_dir, ignore_errors=True)
                    return True
            return False
        except Exception:
            return False

    # ── Total cache size ───────────────────────────────────────────────
    def total_cache_size(self) -> str:
        try:
            cache_info = scan_cache_dir(self.hub_cache)
            return self._human_size(cache_info.size_on_disk)
        except Exception:
            return "0 B"

    # ── Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _human_size(nbytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if nbytes < 1024:
                return f"{nbytes:.1f} {unit}"
            nbytes /= 1024  # type: ignore[assignment]
        return f"{nbytes:.1f} PB"
