"""
UI — Gradio Blocks interface for LLM Standalone.

Features:
  - Lazy model loading (load only after UI is up)
  - SafeTensors / GGUF format selection
  - Cached model list auto-scan
  - Dynamic load/unload status indicator
"""

from __future__ import annotations

import gradio as gr

from .config import APP_CONFIG, ModelFormat
from .model_manager import (
    CachedModel,
    is_loaded,
    get_state,
    load_model,
    scan_cached_models,
    unload_model,
)
from .generation import generate_response


# ── Helpers ──────────────────────────────────────────────────────────────

def _scan_and_group() -> tuple[list[CachedModel], dict[str, CachedModel]]:
    """Scan cache and build lookup dict keyed by display_name."""
    models = scan_cached_models()
    lookup = {m.display_name: m for m in models}
    return models, lookup


def _status_text() -> str:
    """Return a markdown status string."""
    if is_loaded():
        s = get_state()
        fmt_tag = "SafeTensors" if s.model_format == ModelFormat.SAFETENSORS else f"GGUF · `{s.gguf_filename}`"
        return (
            f"### ✅ Model Loaded\n"
            f"**{s.model_id}**\n\n"
            f"Format: {fmt_tag}"
        )
    return "### ⏳ No model loaded\nSelect a model and click **Load Model**."


# ── Main builder ─────────────────────────────────────────────────────────

def build_standalone_interface() -> gr.Blocks:
    """LLM standalone testing Gradio interface."""

    # Pre-scan cached models
    cached_models, model_lookup = _scan_and_group()
    display_names = [m.display_name for m in cached_models]

    with gr.Blocks(title="LLM Standalone Test", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 LLM Standalone Test")
        gr.Markdown(
            "A tool for testing the LLM's **raw knowledge** without a RAG pipeline.\n\n"
            "Select a model and click **Load Model**, then enter a prompt and click Generate."
        )

        # ── Top: Model Settings ──────────────────────────────────────
        with gr.Accordion("🔧 Model Settings", open=True):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=display_names,
                    value=display_names[0] if display_names else None,
                    label="Cached Model",
                    info="Previously downloaded models from ~/.cache/huggingface/hub/",
                    scale=3,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            # GGUF file selector (visible only when a GGUF model is chosen)
            gguf_dropdown = gr.Dropdown(
                choices=[],
                label="GGUF File",
                info="Select quantisation variant for GGUF models",
                visible=False,
                scale=3,
            )

            with gr.Row():
                load_btn = gr.Button("📥 Load Model", variant="primary", scale=2)
                unload_btn = gr.Button("🗑️ Unload Model", variant="secondary", scale=1)

            status_md = gr.Markdown(_status_text())

        # ── Middle: Prompt / Response ────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Ask the LLM anything to test its raw knowledge...",
                    lines=4,
                )
                generate_btn = gr.Button("🚀 Generate", variant="primary")
                response_output = gr.Textbox(
                    label="LLM Response",
                    lines=12,
                    interactive=False,
                )

            # ── Right: Generation Parameters ────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Generation Parameters")

                do_sample = gr.Checkbox(
                    label="do_sample",
                    value=APP_CONFIG.generation.do_sample,
                    info="True = sampling (creative), False = greedy (deterministic)",
                )
                temperature = gr.Slider(
                    minimum=0.01, maximum=2.0,
                    value=APP_CONFIG.generation.temperature,
                    step=0.01,
                    label="Temperature",
                    info="Higher = more random, Lower = more focused",
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0.0, maximum=1.0,
                    value=APP_CONFIG.generation.top_p,
                    step=0.01,
                    label="Top-p (nucleus sampling)",
                    info="Cumulative probability threshold",
                    interactive=True,
                )
                top_k = gr.Slider(
                    minimum=1, maximum=100,
                    value=APP_CONFIG.generation.top_k,
                    step=1,
                    label="Top-k",
                    info="Top-k tokens to sample from",
                    interactive=True,
                )
                max_new_tokens = gr.Slider(
                    minimum=16, maximum=2048,
                    value=APP_CONFIG.generation.max_new_tokens,
                    step=16,
                    label="Max New Tokens",
                    info="Maximum number of tokens to generate",
                    interactive=True,
                )

        gr.Markdown("---")
        gr.Markdown(
            "💡 **Tip:** When `do_sample=False` (greedy), temperature, top_p, and top_k "
            "have no effect. Enable `do_sample` ✅ for creative/diverse responses."
        )

        # ═══════════════════════════════════════════════════════════════
        # ── Event handlers ─────────────────────────────────────────────
        # ═══════════════════════════════════════════════════════════════

        # -- Refresh button: rescan cache --
        def on_refresh():
            nonlocal cached_models, model_lookup
            cached_models, model_lookup = _scan_and_group()
            names = [m.display_name for m in cached_models]
            return gr.update(choices=names, value=names[0] if names else None)

        refresh_btn.click(fn=on_refresh, outputs=model_dropdown)

        # -- Model dropdown change: show/hide GGUF selector --
        def on_model_change(display_name: str):
            if not display_name or display_name not in model_lookup:
                return gr.update(visible=False, choices=[], value=None)
            cm = model_lookup[display_name]
            if cm.model_format == ModelFormat.GGUF and cm.gguf_files:
                return gr.update(
                    visible=True,
                    choices=cm.gguf_files,
                    value=cm.gguf_files[0],
                )
            return gr.update(visible=False, choices=[], value=None)

        model_dropdown.change(
            fn=on_model_change,
            inputs=model_dropdown,
            outputs=gguf_dropdown,
        )

        # -- Load Model --
        def on_load(display_name: str, gguf_file: str | None):
            if not display_name or display_name not in model_lookup:
                return "### ❌ Please select a model."
            cm = model_lookup[display_name]
            msg = load_model(cm.model_id, cm.model_format, gguf_file or "")
            return f"{_status_text()}\n\n{msg}"

        load_btn.click(
            fn=on_load,
            inputs=[model_dropdown, gguf_dropdown],
            outputs=status_md,
        )

        # -- Unload Model --
        def on_unload():
            msg = unload_model()
            return f"{_status_text()}\n\n{msg}"

        unload_btn.click(fn=on_unload, outputs=status_md)

        # -- Toggle sampling param interactivity --
        def toggle_sampling_params(is_sampling: bool):
            interactive = gr.update(interactive=is_sampling)
            return interactive, interactive, interactive

        do_sample.change(
            fn=toggle_sampling_params,
            inputs=do_sample,
            outputs=[temperature, top_p, top_k],
        )

        # -- Generate --
        generate_btn.click(
            fn=generate_response,
            inputs=[prompt_input, do_sample, temperature, top_p, top_k, max_new_tokens],
            outputs=response_output,
        )
        prompt_input.submit(
            fn=generate_response,
            inputs=[prompt_input, do_sample, temperature, top_p, top_k, max_new_tokens],
            outputs=response_output,
        )

    return demo
