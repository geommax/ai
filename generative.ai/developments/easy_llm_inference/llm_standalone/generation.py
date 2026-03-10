"""
Generation — text generation logic for both SafeTensors and GGUF models.

Supports prompt templates: Auto (Chat Template), ChatML, Llama, Alpaca,
Vicuna, Phi, and Custom user-defined templates.
"""

from __future__ import annotations

from .config import ModelFormat, PROMPT_TEMPLATES, DEFAULT_SYSTEM_PROMPT
from .model_manager import get_state, is_loaded


# ── Prompt formatting ────────────────────────────────────────────────────

def _format_prompt(
    user_prompt: str,
    system_prompt: str,
    template_name: str,
    custom_template: str,
    tokenizer=None,
) -> str | list[dict[str, str]]:
    """
    Apply the selected prompt template.

    Returns:
        - A formatted string for raw-text generation, OR
        - A list[dict] messages for the "Auto" path (chat template / GGUF chat API).
    """
    system = system_prompt.strip() if system_prompt else DEFAULT_SYSTEM_PROMPT

    # ── Auto: use tokenizer chat template or return messages list ────
    if template_name == "Auto (Chat Template)":
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Fallback: return messages list (used by GGUF chat API)
        return messages

    # ── Named or Custom template ─────────────────────────────────────
    if template_name == "Custom":
        tpl = custom_template if custom_template and custom_template.strip() else PROMPT_TEMPLATES["Custom"]
    else:
        tpl = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["Custom"])

    return tpl.format(system=system, prompt=user_prompt)


# ── Public entry point ───────────────────────────────────────────────────

def generate_response(
    prompt: str,
    system_prompt: str,
    template_name: str,
    custom_template: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    """
    Generate a response using the currently loaded model.
    SafeTensors → transformers pipeline
    GGUF        → llama-cpp-python
    """
    if not prompt.strip():
        return "⚠️ Please enter a prompt."

    if not is_loaded():
        return "⚠️ No model loaded yet. Please select a model in the Model Settings panel and click **Load Model**."

    state = get_state()

    if state.model_format == ModelFormat.SAFETENSORS:
        return _generate_safetensors(
            state, prompt, system_prompt, template_name, custom_template,
            do_sample, temperature, top_p, top_k, max_new_tokens,
        )
    else:
        return _generate_gguf(
            state, prompt, system_prompt, template_name, custom_template,
            do_sample, temperature, top_p, top_k, max_new_tokens,
        )


# ── SafeTensors generation ───────────────────────────────────────────────

def _generate_safetensors(
    state,
    prompt: str,
    system_prompt: str,
    template_name: str,
    custom_template: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    from transformers import pipeline as hf_pipeline

    formatted = _format_prompt(
        prompt, system_prompt, template_name, custom_template,
        tokenizer=state.tokenizer,
    )

    # If _format_prompt returned a messages list (Auto without chat template),
    # fall back to a simple concatenation.
    if isinstance(formatted, list):
        formatted = "\n".join(m["content"] for m in formatted)

    gen_kwargs: dict = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = int(top_k)

    pipe = hf_pipeline(
        "text-generation",
        model=state.model,
        tokenizer=state.tokenizer,
        return_full_text=False,
        **gen_kwargs,
    )

    result = pipe(formatted)
    return result[0]["generated_text"]


# ── GGUF generation ─────────────────────────────────────────────────────

def _generate_gguf(
    state,
    prompt: str,
    system_prompt: str,
    template_name: str,
    custom_template: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    llm = state.model  # Llama instance

    gen_kwargs: dict = {
        "max_tokens": int(max_new_tokens),
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = int(top_k)
    else:
        gen_kwargs["temperature"] = 0.0

    formatted = _format_prompt(
        prompt, system_prompt, template_name, custom_template,
    )

    # Auto / messages-list path → use chat completion API
    if isinstance(formatted, list):
        response = llm.create_chat_completion(
            messages=formatted,
            **gen_kwargs,
        )
        return response["choices"][0]["message"]["content"]

    # Template-formatted string → use raw completion API
    response = llm(
        formatted,
        **gen_kwargs,
    )
    return response["choices"][0]["text"]
