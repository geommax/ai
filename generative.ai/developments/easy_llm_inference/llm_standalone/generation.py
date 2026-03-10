"""
Generation — text generation logic for both SafeTensors and GGUF models.
"""

from __future__ import annotations

from .config import ModelFormat
from .model_manager import get_state, is_loaded


def generate_response(
    prompt: str,
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
        return _generate_safetensors(state, prompt, do_sample, temperature, top_p, top_k, max_new_tokens)
    else:
        return _generate_gguf(state, prompt, do_sample, temperature, top_p, top_k, max_new_tokens)


# ── SafeTensors generation ───────────────────────────────────────────────

def _generate_safetensors(
    state,
    prompt: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    from transformers import pipeline as hf_pipeline

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

    result = pipe(prompt)
    return result[0]["generated_text"]


# ── GGUF generation ─────────────────────────────────────────────────────

def _generate_gguf(
    state,
    prompt: str,
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
        # Greedy: temperature=0 forces deterministic output
        gen_kwargs["temperature"] = 0.0

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        **gen_kwargs,
    )

    return response["choices"][0]["message"]["content"]
