"""Hugging Face model management — download, list, delete."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Enable hf_transfer for multi-connection parallel downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download, hf_hub_download


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
        """Download with detailed per-file progress.

        Parameters
        ----------
        on_file_list(files):
            Called once with the full list of
            ``[{"name": str, "size": int, "size_str": str}, ...]``
            before downloading begins.
        on_progress(file_name, file_idx, total_files, bytes_done, bytes_total, status):
            Called before each file starts (``status="downloading"``)
            and after it finishes (``status="done"``).
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

        bytes_done = 0
        last_path = ""
        download_start = time.monotonic()
        speed = 0.0  # bytes per second

        for idx, sibling in enumerate(siblings, 1):
            # ── Cancel gate ────────────────────────────────────────
            if self._cancel_event.is_set():
                raise DownloadCancelled(
                    f"Download of {model_id} stopped at file {idx}/{total_files}"
                )

            fname = sibling.rfilename
            fsize = getattr(sibling, "size", 0) or 0

            if on_progress:
                on_progress(fname, idx, total_files, bytes_done, total_bytes, "downloading", speed)

            file_start = time.monotonic()
            last_path = hf_hub_download(
                model_id,
                filename=fname,
                cache_dir=str(self.hub_cache),
            )
            file_elapsed = time.monotonic() - file_start
            bytes_done += fsize

            # Calculate speed: use per-file speed for responsiveness,
            # but fall back to overall average if file was cached (< 0.1s)
            if file_elapsed > 0.1 and fsize > 0:
                speed = fsize / file_elapsed
            elif bytes_done > 0:
                total_elapsed = time.monotonic() - download_start
                speed = bytes_done / total_elapsed if total_elapsed > 0 else 0.0

            if on_progress:
                on_progress(fname, idx, total_files, bytes_done, total_bytes, "done", speed)

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
