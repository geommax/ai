"""
09 - LLM Standalone (Without RAG Pipeline)
RAG pipeline မတွဲခင် LLM ရဲ့ raw knowledge ကို စစ်ဆေးဖို့ standalone script။

ရည်ရွယ်ချက်:
  - LLM ကို သီးသန့် test လုပ်ပြီး domain knowledge ရှိ/မရှိ စစ်ဆေးတယ်
  - RAG ထည့်ဖို့ လိုအပ်သလား evaluate လုပ်တယ်
  - Generation parameters (temperature, top_p, top_k, etc.) ကို tune လုပ်လို့ရတယ်

Usage:
  python 09_llm_standalone.py
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ── Model Loading ────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
)
print("Model loaded successfully!\n")


# ── Generation Function ─────────────────────────────────────────────────

def generate_response(
    prompt: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    """
    LLM ဆီကို prompt ပို့ပြီး response ပြန်ယူတယ်။
    RAG context မပါဘဲ model ရဲ့ built-in knowledge ကိုပဲ သုံးတယ်။
    """
    if not prompt.strip():
        return "⚠️ Please enter a prompt."

    gen_kwargs: dict = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }

    # do_sample=True မှသာ sampling parameters တွေ သက်ဆိုင်တယ်
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = int(top_k)

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **gen_kwargs,
    )

    result = text_gen(prompt)
    return result[0]["generated_text"]


# ── Gradio UI ────────────────────────────────────────────────────────────

def build_standalone_interface() -> gr.Blocks:
    """LLM standalone testing အတွက် Gradio interface တည်ဆောက်တယ်။"""

    with gr.Blocks(title="LLM Standalone Test") as demo:
        gr.Markdown("# 🧠 LLM Standalone Test")
        gr.Markdown(
            "RAG pipeline မပါဘဲ LLM ရဲ့ **raw knowledge** ကို စစ်ဆေးတဲ့ tool ဖြစ်ပါတယ်။\n\n"
            f"**Model:** `{MODEL_ID}`\n\n"
            "ဒီမှာ question မေးကြည့်ပြီး model က ဘာတွေသိလဲ၊ "
            "ကိုယ့် domain knowledge ပါသလား စစ်ဆေးနိုင်ပါတယ်။"
        )

        with gr.Row():
            # ── Left: Input / Output ──
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

            # ── Right: Generation Parameters ──
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Generation Parameters")

                do_sample = gr.Checkbox(
                    label="do_sample",
                    value=False,
                    info="True = sampling (creative), False = greedy (deterministic)",
                )
                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=2.0,
                    value=0.7,
                    step=0.01,
                    label="Temperature",
                    info="Higher = more random, Lower = more focused",
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    label="Top-p (nucleus sampling)",
                    info="Cumulative probability threshold",
                    interactive=True,
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Top-k tokens to sample from",
                    interactive=True,
                )
                max_new_tokens = gr.Slider(
                    minimum=16,
                    maximum=2048,
                    value=512,
                    step=16,
                    label="Max New Tokens",
                    info="Generate လုပ်မယ့် max token အရေအတွက်",
                    interactive=True,
                )

        # ── Toggle sampling params visibility ──
        def toggle_sampling_params(is_sampling: bool):
            interactive = gr.update(interactive=is_sampling)
            return interactive, interactive, interactive

        do_sample.change(
            fn=toggle_sampling_params,
            inputs=do_sample,
            outputs=[temperature, top_p, top_k],
        )

        # ── Generate button action ──
        generate_btn.click(
            fn=generate_response,
            inputs=[prompt_input, do_sample, temperature, top_p, top_k, max_new_tokens],
            outputs=response_output,
        )

        # ── Enter key shortcut ──
        prompt_input.submit(
            fn=generate_response,
            inputs=[prompt_input, do_sample, temperature, top_p, top_k, max_new_tokens],
            outputs=response_output,
        )

        gr.Markdown("---")
        gr.Markdown(
            "💡 **Tip:** `do_sample=False` (greedy) ဆိုရင် temperature, top_p, top_k တွေ "
            "effect မရှိပါဘူး။ Creative/diverse responses လိုချင်ရင် `do_sample` ကို ✅ ဖွင့်ပါ။"
        )

    return demo


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_standalone_interface()
    demo.launch(theme=gr.themes.Soft())
