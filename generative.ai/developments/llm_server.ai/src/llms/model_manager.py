"""Hugging Face model management — download, list, delete."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Enable hf_transfer for multi-connection parallel downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download, hf_hub_download


class DownloadCancelled(Exception):
    """Raised when a download is stopped by the user."""


class _ProgressTracker:
    """Collects real-time byte-level progress from tqdm bars.

    ``hf_hub_download`` creates one tqdm bar per file.  We monkey-patch
    ``tqdm.update()`` so that every chunk written triggers our callback
    with accurate bytes-done and speed.
    """

    def __init__(
        self,
        files_info: List[Dict[str, Any]],
        on_progress: Optional[Callable] = None,
    ) -> None:
        self.files_info = files_info
        self.on_progress = on_progress
        self._file_set = {f["name"] for f in files_info}
        self._total_bytes = sum(f["size"] for f in files_info)
        self._completed_bytes = 0        # bytes from fully-done files
        self._current_file_bytes = 0     # bytes of file being downloaded NOW
        self._current_fname: str = ""
        self._current_idx: int = 0
        self._done_files: set[str] = set()
        self._download_start = time.monotonic()
        self._lock = threading.Lock()

    @property
    def bytes_done(self) -> int:
        with self._lock:
            return self._completed_bytes + self._current_file_bytes

    @property
    def speed(self) -> float:
        elapsed = time.monotonic() - self._download_start
        return self.bytes_done / elapsed if elapsed > 0.5 else 0.0

    def file_start(self, fname: str, idx: int) -> None:
        with self._lock:
            self._current_fname = fname
            self._current_idx = idx
            self._current_file_bytes = 0

    def file_done(self, fname: str, fsize: int) -> None:
        with self._lock:
            self._completed_bytes += fsize
            self._current_file_bytes = 0
            self._done_files.add(fname)

    def tqdm_update(self, n_bytes: int) -> None:
        """Called from the patched tqdm.update with chunk size."""
        with self._lock:
            self._current_file_bytes += n_bytes
        # Fire callback
        if self.on_progress:
            bd = self.bytes_done
            pct = (bd / self._total_bytes * 100) if self._total_bytes else 0
            self.on_progress(
                self._current_fname,
                self._current_idx,
                len(self.files_info),
                bd,
                self._total_bytes,
                "downloading",
                self.speed,
            )


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
                    models.append(
                        {
                            "repo_id": repo.repo_id,
                            "size": repo.size_on_disk,
                            "size_str": self._human_size(repo.size_on_disk),
                            "nb_files": repo.nb_files,
                            "last_modified": repo.last_modified,
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

    def download_model_with_progress(
        self,
        model_id: str,
        on_progress: Optional[Callable] = None,
        on_file_list: Optional[Callable] = None,
    ) -> str:
        """Download with detailed per-file + byte-level progress.

        Uses tqdm monkey-patching to capture real-time byte progress
        from ``hf_hub_download`` regardless of backend (hf_transfer
        or plain requests).
        """
        self._cancel_event.clear()

        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)
        siblings = info.siblings or []

        # Build file manifest
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

        # Notify UI of the full file list
        if on_file_list:
            on_file_list(files_info)

        # ── Progress tracker ───────────────────────────────────────
        tracker = _ProgressTracker(files_info, on_progress)

        last_path = ""

        for idx, sibling in enumerate(siblings, 1):
            # ── Cancel gate ────────────────────────────────────────
            if self._cancel_event.is_set():
                raise DownloadCancelled(
                    f"Download of {model_id} stopped at file {idx}/{total_files}"
                )

            fname = sibling.rfilename
            fsize = getattr(sibling, "size", 0) or 0
            tracker.file_start(fname, idx)

            # Notify "downloading" status
            if on_progress:
                bd = tracker.bytes_done
                on_progress(
                    fname, idx, total_files, bd, total_bytes,
                    "downloading", tracker.speed,
                )

            # ── Monkey-patch tqdm to capture byte-level progress ──
            import tqdm as _tqdm_mod
            _orig_init = _tqdm_mod.tqdm.__init__
            _orig_update = _tqdm_mod.tqdm.update
            _this_tracker = tracker  # closure capture

            def _patched_init(self_bar, *a: Any, **kw: Any) -> None:
                _orig_init(self_bar, *a, **kw)
                self_bar._llm_patched = True

            def _patched_update(self_bar, n: int = 1) -> None:
                _orig_update(self_bar, n)
                if getattr(self_bar, "_llm_patched", False):
                    _this_tracker.tqdm_update(n)

            _tqdm_mod.tqdm.__init__ = _patched_init  # type: ignore[assignment]
            _tqdm_mod.tqdm.update = _patched_update  # type: ignore[assignment]

            try:
                last_path = hf_hub_download(
                    model_id,
                    filename=fname,
                    cache_dir=str(self.hub_cache),
                )
            finally:
                # Always restore tqdm
                _tqdm_mod.tqdm.__init__ = _orig_init  # type: ignore[assignment]
                _tqdm_mod.tqdm.update = _orig_update  # type: ignore[assignment]

            tracker.file_done(fname, fsize)

            # Notify "done" status
            if on_progress:
                bd = tracker.bytes_done
                on_progress(
                    fname, idx, total_files, bd, total_bytes,
                    "done", tracker.speed,
                )

        # Return the snapshot directory (parent of files)
        return str(Path(last_path).parent) if last_path else ""

    def cancel_download(self) -> None:
        """Signal the running download to stop after the current file."""
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
