"""
07 - LLM (Large Language Model)
Qwen 2.5 model ကို local cache ကနေ load လုပ်ပြီး LangChain pipeline အဖြစ် initialize လုပ်တယ်။

Supported Models:
  - Qwen/Qwen2.5-72B-Instruct — size ကြီးလေ Myanmar handling ကောင်းလေ
  - google/gemma-2-9b-it — Myanmar ကို training data ထဲ ပိုပါတယ်
  - meta-llama/Llama-3.1-8B-Instruct — multilingual ပိုကောင်းတယ်
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


def load_llm(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float | None = None,
    local_files_only: bool = True,
) -> HuggingFacePipeline:
    """
    HuggingFace model ကို load လုပ်ပြီး LangChain HuggingFacePipeline အဖြစ် return ပြန်ပေးတယ်။
    local_files_only=True ensures we don't re-download if already cached.

    Args:
        model_id: HuggingFace model identifier
        max_new_tokens: generate လုပ်မယ့် max token အရေအတွက်
        temperature: sampling temperature
        local_files_only: True ဆိုရင် local cache ကပဲ load လုပ်မယ်

    Returns:
        HuggingFacePipeline: LangChain LLM instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=local_files_only,
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    # temperature, top_p, top_k are only valid when do_sample=True
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **gen_kwargs,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm
