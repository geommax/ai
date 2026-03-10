"""
Downloader — download SafeTensors & GGUF models from HuggingFace Hub.

SafeTensors: uses snapshot_download (downloads all files, no memory loading).
GGUF:        uses hf_hub_download  (downloads a single selected .gguf file).
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    GatedRepoError,
)


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


# ── Repo inspection ─────────────────────────────────────────────────────

def list_gguf_files(repo_id: str) -> list[str]:
    """
    List all .gguf files in a HuggingFace repo.
    Returns a sorted list of filenames (e.g. ["model-Q4_K_M.gguf", ...]).
    Raises on invalid / gated repos.
    """
    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id)
    except RepositoryNotFoundError:
        raise ValueError(f"Repository '{repo_id}' not found on HuggingFace Hub.")
    except GatedRepoError:
        raise ValueError(
            f"Repository '{repo_id}' is gated. "
            "Please accept the license on HuggingFace and log in with `huggingface-cli login`."
        )

    gguf_files = sorted(f for f in all_files if f.lower().endswith(".gguf"))
    if not gguf_files:
        raise ValueError(f"No .gguf files found in '{repo_id}'.")
    return gguf_files


# ── SafeTensors download ────────────────────────────────────────────────

def download_safetensors(
    repo_id: str,
    cache_dir: str | Path | None = None,
) -> Generator[str, None, str]:
    """
    Download a full SafeTensors model repository (no memory loading).

    Yields real-time status messages.
    Returns final status string.
    """
    save_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    yield f"📦 Starting download: **{repo_id}** (SafeTensors)\n\nSave path: `{save_path}`"

    try:
        result_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(save_path),
            local_files_only=False,
        )
        msg = (
            f"✅ Download complete!\n\n"
            f"**Model:** {repo_id}\n"
            f"**Format:** SafeTensors\n"
            f"**Path:** `{result_path}`"
        )
        yield msg
        return msg

    except RepositoryNotFoundError:
        msg = f"❌ Repository '{repo_id}' not found on HuggingFace Hub."
        yield msg
        return msg
    except GatedRepoError:
        msg = (
            f"❌ Repository '{repo_id}' is gated.\n"
            "Please accept the license on HuggingFace and run `huggingface-cli login`."
        )
        yield msg
        return msg
    except Exception as exc:
        msg = f"❌ Download failed: {exc}"
        yield msg
        return msg


# ── GGUF download ────────────────────────────────────────────────────────

def download_gguf(
    repo_id: str,
    filename: str,
    cache_dir: str | Path | None = None,
) -> Generator[str, None, str]:
    """
    Download a single GGUF file from a HuggingFace repo.

    Yields real-time status messages.
    Returns final status string.
    """
    save_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    yield (
        f"📦 Starting download: **{repo_id}**\n\n"
        f"File: `{filename}`\n"
        f"Save path: `{save_path}`"
    )

    try:
        result_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(save_path),
            local_files_only=False,
        )
        msg = (
            f"✅ Download complete!\n\n"
            f"**Model:** {repo_id}\n"
            f"**File:** `{filename}`\n"
            f"**Format:** GGUF\n"
            f"**Path:** `{result_path}`"
        )
        yield msg
        return msg

    except EntryNotFoundError:
        msg = f"❌ File '{filename}' not found in '{repo_id}'."
        yield msg
        return msg
    except RepositoryNotFoundError:
        msg = f"❌ Repository '{repo_id}' not found on HuggingFace Hub."
        yield msg
        return msg
    except GatedRepoError:
        msg = (
            f"❌ Repository '{repo_id}' is gated.\n"
            "Please accept the license on HuggingFace and run `huggingface-cli login`."
        )
        yield msg
        return msg
    except Exception as exc:
        msg = f"❌ Download failed: {exc}"
        yield msg
        return msg
