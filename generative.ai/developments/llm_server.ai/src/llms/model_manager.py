"""Hugging Face model management — download, list, delete."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download


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
        self._dl_process: Optional[subprocess.Popen] = None

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
        """Download using ``huggingface-cli download`` as a subprocess.

        Cancellation is 100 % reliable — ``process.terminate()`` closes
        all network connections instantly.

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

        # ── Build command ──────────────────────────────────────────
        hf_cli = str(Path(sys.executable).parent / "huggingface-cli")
        cmd = [hf_cli, "download", model_id]
        if filenames:
            cmd.extend(filenames)
        cmd.extend(["--cache-dir", str(self.hub_cache)])

        env = os.environ.copy()
        env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

        # ── Launch ─────────────────────────────────────────────────
        self._dl_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        start_time = time.monotonic()

        try:
            fd = self._dl_process.stderr.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            buf = ""

            while self._dl_process.poll() is None:
                # ── Cancel gate ────────────────────────────────────
                if self._cancel_event.is_set():
                    self._kill_dl_process()
                    raise DownloadCancelled(
                        f"Download of {model_id} cancelled"
                    )

                # ── Read stderr for tqdm progress ──────────────────
                try:
                    raw = os.read(fd, 8192)
                    if raw:
                        buf += raw.decode("utf-8", errors="replace")
                        parts = re.split(r"[\r\n]+", buf)
                        buf = parts[-1]
                        for part in parts[:-1]:
                            self._parse_cli_progress(
                                part, files_info, total_bytes,
                                total_files, start_time, on_progress,
                            )
                except (OSError, BlockingIOError):
                    pass

                time.sleep(0.15)

            # ── Process exited ─────────────────────────────────────
            retcode = self._dl_process.returncode

            if retcode != 0:
                if self._cancel_event.is_set():
                    raise DownloadCancelled(
                        f"Download of {model_id} cancelled"
                    )
                err = ""
                try:
                    rest = self._dl_process.stderr.read()
                    if rest:
                        err = rest.decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(
                    f"huggingface-cli download failed (exit {retcode}):\n{err}"
                )

            # stdout = local path printed by huggingface-cli
            path = ""
            try:
                path = (
                    self._dl_process.stdout
                    .read()
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except Exception:
                pass

            # Notify all files done
            if on_progress:
                for idx, f in enumerate(files_info, 1):
                    on_progress(
                        f["name"], idx, total_files,
                        total_bytes, total_bytes, "done", 0.0,
                    )

            return path or str(self.hub_cache)

        finally:
            self._dl_process = None

    # ── Subprocess helpers ─────────────────────────────────────────────

    def _kill_dl_process(self) -> None:
        """Send SIGTERM (then SIGKILL) to the download process group."""
        proc = self._dl_process
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _parse_cli_progress(
        line: str,
        files_info: List[Dict[str, Any]],
        total_bytes: int,
        total_files: int,
        start_time: float,
        on_progress: Optional[Callable],
    ) -> None:
        """Extract progress percentage from tqdm CLI output."""
        if not on_progress or "%" not in line:
            return
        m = re.search(r"(\d+)%\|", line)
        if not m:
            return
        pct = int(m.group(1))
        bytes_done = int(total_bytes * pct / 100)
        elapsed = time.monotonic() - start_time
        speed = bytes_done / elapsed if elapsed > 1.0 else 0.0

        # Try to extract filename from "filename:  XX%|…"
        fname = files_info[0]["name"] if files_info else ""
        fn_m = re.match(r"^(.*?):\s+\d+%", line)
        if fn_m:
            raw = fn_m.group(1).strip()
            for f in files_info:
                if f["name"] == raw or f["name"].endswith(raw):
                    fname = f["name"]
                    break

        on_progress(
            fname, 1, total_files, bytes_done, total_bytes,
            "downloading", speed,
        )

    def cancel_download(self) -> None:
        """Terminate the download subprocess immediately."""
        self._cancel_event.set()
        self._kill_dl_process()

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
