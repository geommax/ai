"""
UI — Gradio interface for downloading HuggingFace models.

Two tabs:
  1. SafeTensors — enter repo ID, click Download.
  2. GGUF       — enter repo ID, auto-fetch .gguf file list, select & download.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .downloader import (
    DEFAULT_CACHE_DIR,
    download_gguf,
    download_safetensors,
    list_gguf_files,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_cache_dir(user_path: str) -> Path:
    """Return user-supplied path or fall back to the default HF cache."""
    stripped = user_path.strip()
    if stripped:
        return Path(stripped).expanduser().resolve()
    return DEFAULT_CACHE_DIR


# ── Builder ──────────────────────────────────────────────────────────────

def build_fetch_interface() -> gr.Blocks:
    """Build the Gradio Blocks app for model downloading."""

    with gr.Blocks(title="Model Downloader", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📥 HuggingFace Model Downloader")
        gr.Markdown(
            "Download models from HuggingFace Hub to your local cache.\n\n"
            "Choose the **SafeTensors** tab for standard Transformers models, "
            "or the **GGUF** tab for quantised llama.cpp models."
        )

        # ── Shared: save-path setting ────────────────────────────────
        with gr.Accordion("⚙️ Save Path Settings", open=False):
            cache_dir_input = gr.Textbox(
                label="Cache Directory",
                value=str(DEFAULT_CACHE_DIR),
                placeholder=str(DEFAULT_CACHE_DIR),
                info="Leave blank to use the default HuggingFace cache path.",
            )

        # ══════════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── Tab 1: SafeTensors ───────────────────────────────────
            with gr.Tab("🟢 SafeTensors"):
                gr.Markdown(
                    "Enter a HuggingFace **repo ID** (e.g. `Qwen/Qwen2.5-3B-Instruct`) "
                    "and click **Download**.\n\n"
                    "The full model snapshot will be saved to your cache directory — "
                    "no GPU memory is used."
                )

                st_repo_id = gr.Textbox(
                    label="Model Repo ID",
                    placeholder="e.g. Qwen/Qwen2.5-3B-Instruct",
                )
                st_download_btn = gr.Button("📥 Download SafeTensors", variant="primary")
                st_status = gr.Markdown("Waiting for download...")

            # ── Tab 2: GGUF ──────────────────────────────────────────
            with gr.Tab("🟠 GGUF"):
                gr.Markdown(
                    "**Step 1:** Enter a GGUF repo ID (e.g. `bartowski/Qwen2.5-3B-Instruct-GGUF`) "
                    "and click **Fetch File List**.\n\n"
                    "**Step 2:** Select the quantisation variant you want and click **Download GGUF**."
                )

                gguf_repo_id = gr.Textbox(
                    label="GGUF Repo ID",
                    placeholder="e.g. bartowski/Qwen2.5-3B-Instruct-GGUF",
                )
                fetch_btn = gr.Button("🔍 Fetch File List", variant="secondary")

                gguf_dropdown = gr.Dropdown(
                    choices=[],
                    label="Available GGUF Files",
                    info="Click 'Fetch File List' to populate this dropdown.",
                    interactive=True,
                )
                gguf_download_btn = gr.Button("📥 Download GGUF", variant="primary")
                gguf_status = gr.Markdown("Waiting for download...")

        # ══════════════════════════════════════════════════════════════
        # ── Event handlers ───────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════

        # -- SafeTensors download --
        def on_st_download(repo_id: str, cache_dir: str):
            repo_id = repo_id.strip()
            if not repo_id:
                yield "⚠️ Please enter a Model Repo ID."
                return
            save = _resolve_cache_dir(cache_dir)
            yield from download_safetensors(repo_id, save)

        st_download_btn.click(
            fn=on_st_download,
            inputs=[st_repo_id, cache_dir_input],
            outputs=st_status,
        )

        # -- GGUF: fetch file list --
        def on_fetch_files(repo_id: str):
            repo_id = repo_id.strip()
            if not repo_id:
                return (
                    gr.update(choices=[], value=None),
                    "⚠️ Please enter a GGUF Repo ID.",
                )
            try:
                files = list_gguf_files(repo_id)
                return (
                    gr.update(choices=files, value=files[0]),
                    f"✅ Found **{len(files)}** GGUF file(s) in `{repo_id}`.",
                )
            except ValueError as exc:
                return gr.update(choices=[], value=None), f"❌ {exc}"
            except Exception as exc:
                return gr.update(choices=[], value=None), f"❌ Error: {exc}"

        fetch_btn.click(
            fn=on_fetch_files,
            inputs=gguf_repo_id,
            outputs=[gguf_dropdown, gguf_status],
        )

        # -- GGUF: download selected file --
        def on_gguf_download(repo_id: str, filename: str | None, cache_dir: str):
            repo_id = repo_id.strip()
            if not repo_id:
                yield "⚠️ Please enter a GGUF Repo ID."
                return
            if not filename:
                yield "⚠️ Please fetch the file list first and select a GGUF file."
                return
            save = _resolve_cache_dir(cache_dir)
            yield from download_gguf(repo_id, filename, save)

        gguf_download_btn.click(
            fn=on_gguf_download,
            inputs=[gguf_repo_id, gguf_dropdown, cache_dir_input],
            outputs=gguf_status,
        )

    return demo
