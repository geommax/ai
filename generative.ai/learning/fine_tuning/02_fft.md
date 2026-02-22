# 02. Full Fine-Tuning (FFT) with Axolotl - RTX 3050 12GB

## RTX 3050 12GB VRAM နဲ့ Full Fine-Tuning

### ⚠️ VRAM Limitation ကို နားလည်ခြင်း

Full Fine-Tuning (FFT) မှာ model parameter **အားလုံး** ကို train လုပ်ရတဲ့အတွက် VRAM usage က LoRA/QLoRA ထက် **အများကြီး ပိုကုန်** ပါတယ်။

#### VRAM Usage Breakdown (Full Fine-Tuning)

```
Full Fine-Tuning VRAM = Model Weights + Gradients + Optimizer States + Activations

                        ┌─────────────────────────────────────────────────┐
                        │ Component         │ Memory (per 1B params)     │
                        ├───────────────────┼────────────────────────────┤
fp16/bf16 Training:     │ Model Weights     │ ~2 GB   (2 bytes/param)   │
                        │ Gradients         │ ~2 GB   (2 bytes/param)   │
                        │ Optimizer (AdamW) │ ~4 GB   (8 bytes/param)   │ ← 2 states
                        │ Activations       │ ~1-3 GB (varies)          │
                        ├───────────────────┼────────────────────────────┤
                        │ Total per 1B      │ ~9-11 GB                  │
                        └───────────────────┴────────────────────────────┘
```

#### RTX 3050 12GB နဲ့ Train နိုင်တဲ့ Model Size

| Model Size | FFT Memory (bf16) | RTX 3050 12GB | မှတ်ချက် |
|---|---|---|---|
| **0.1B - 0.2B** | ~1.5 - 2.5 GB | ✅ အဆင်ပြေ | SmolLM2-135M |
| **0.5B** | ~5 - 6 GB | ✅ အဆင်ပြေ | Qwen2.5-0.5B |
| **1B - 1.1B** | ~9 - 11 GB | ⚠️ Tight (gradient checkpointing လို) | TinyLlama-1.1B, Llama-3.2-1B |
| **1.5B** | ~14 - 16 GB | ❌ VRAM မလောက် | Qwen2.5-1.5B |
| **3B+** | ~27+ GB | ❌ မဖြစ်နိုင် | - |

> 💡 **12GB VRAM** နဲ့ full fine-tuning အတွက် **0.5B - 1B** model ကို recommend ပါတယ်။

---

## RTX 3050 GPU Specifications

```
┌──────────────────────────────────────────┐
│ NVIDIA GeForce RTX 3050                  │
├──────────────────────────────────────────┤
│ Architecture:    Ampere (SM 8.6)         │
│ VRAM:            12 GB GDDR6             │
│ CUDA Cores:      2560                    │
│ Memory Bus:      192-bit                 │
│ bf16 Support:    ✅ Yes                  │
│ Flash Attention: ✅ Yes (Ampere+)        │
│ Compute Cap:     8.6                     │
└──────────────────────────────────────────┘
```

---

## Step-by-Step Guide

### Step 1: Docker Container ဖွင့်ခြင်း

01_settingup.md မှာ Docker + NVIDIA Container Toolkit install ပြီးပြီဆိုရင်:

```bash
# Axolotl container ကို run ပါ
# -v flag နဲ့ local data folder ကို mount ပါ (dataset/config files အတွက်)
docker run --gpus '"all"' \
  --rm -it \
  --shm-size=4g \
  -v $(pwd)/workspace:/workspace/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8888:8888 \
  axolotlai/axolotl:main-latest
```

### Step 2: Container ထဲမှာ GPU စစ်ဆေးခြင်း

```bash
# GPU ရှိ/မရှိ စစ်ဆေး
nvidia-smi

# Expected output:
# NVIDIA GeForce RTX 3050 | 12GB
```

```bash
# CUDA + PyTorch compatibility စစ်ဆေး
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'Compute Capability: {torch.cuda.get_device_capability()}')
print(f'bf16 support: {torch.cuda.is_bf16_supported()}')
"
```

### Step 3: Hugging Face Login

```bash
# HF Token နဲ့ login (gated models ယူဖို့)
hf auth login --token $HF_TOKEN

# OR
# pip install huggingface_hub
# huggingface-cli login
```

---

## Full Fine-Tuning Test Runs

Config files နဲ့ dataset files တွေကို model folder တစ်ခုချင်းစီမှာ ခွဲထားပါတယ်။ **အစဉ်လိုက် run ပါ:**

### Docker Container စတင်ခြင်း (Test Run အားလုံးအတွက်)

```bash
docker run --gpus '"all"' --rm -it \
  --shm-size=4g \
  -v /home/mr_cobot/Desktop/dev_projects/ai/generative.ai/learning/fine_tuning:/workspace/data \
  -v /home/mr_cobot/.cache/huggingface:/root/.cache/huggingface \
  axolotlai/axolotl:main-latest
```

---

### Test Run ၁ — SmolLM2-135M (Smoke Test)

> 🧪 VRAM ~2GB | Pipeline စစ်ဖို့ အရင်ဆုံး run ပါ

📁 **Folder:** [SmolLM2-135M/](SmolLM2-135M/) — Config, dataset, README ပါပြီးသား

```bash
accelerate launch -m axolotl.cli.train /workspace/data/SmolLM2-135M/config.yml
```

---

### Test Run ၂ — Qwen2.5-0.5B (Recommended 🎯)

> 🎯 VRAM ~5-6GB | 12GB VRAM FFT sweet spot

📁 **Folder:** [Qwen2.5-0.5B/](Qwen2.5-0.5B/) — Config, dataset, README ပါပြီးသား

```bash
accelerate launch -m axolotl.cli.train /workspace/data/Qwen2.5-0.5B/config.yml
```

---

### Test Run ၃ — Llama-3.2-1B (Maximum ⚠️)

> ⚠️ VRAM ~9-11GB | 12GB limit နားကပ်၊ OOM ဖြစ်နိုင်
> 🔑 Gated model — HuggingFace login + access approval လိုအပ်

📁 **Folder:** [Llama-3.2-1B/](Llama-3.2-1B/) — Config, dataset, README ပါပြီးသား

```bash
# HuggingFace login (gated model အတွက် required)
huggingface-cli login

# Training
accelerate launch -m axolotl.cli.train /workspace/data/Llama-3.2-1B/config.yml
```

---

### Folder Structure

```
fine_tuning/
├── SmolLM2-135M/              ← Test Run ၁
│   ├── README.md
│   ├── config.yml
│   └── train.jsonl
├── Qwen2.5-0.5B/              ← Test Run ၂ (Recommended 🎯)
│   ├── README.md
│   ├── config.yml
│   └── train.jsonl
├── Llama-3.2-1B/              ← Test Run ၃ (Maximum ⚠️)
│   ├── README.md
│   ├── config.yml
│   └── train.jsonl
└── 02_fft.md                  ← ဒီဖိုင် (overview)
```

### VRAM Monitoring (Training run နေချိန်မှာ)

```bash
# နောက်ထပ် terminal ကနေ container ထဲဝင်ပါ
docker ps
docker exec -it <container_id> bash
watch -n 1 nvidia-smi
```

---

## Known Errors & Fixes

### ❌ Error: Tokenizer does not have a padding token

```
ValueError: Asking to pad but the tokenizer does not have a padding token.
Please select a token to use as `pad_token`
```

**အကြောင်းရင်း:** Model ရဲ့ tokenizer မှာ `pad_token` define မလုပ်ထားလို့ evaluation step မှာ batch padding လုပ်တဲ့အခါ ပျက်ပါတယ်။ SmolLM2, GPT-2 စတဲ့ models တွေမှာ ဒီ error ဖြစ်တတ်ပါတယ်။

**ဖြေရှင်းနည်း:** Config YAML မှာ `special_tokens` section ထည့်ပါ:

```yaml
# SmolLM2 / GPT-2 style models:
special_tokens:
  pad_token: "<|endoftext|>"

# Llama 3.x models:
special_tokens:
  pad_token: "<|finetune_right_pad_id|>"

# General fallback (eos_token ကို pad_token အဖြစ်သုံး):
special_tokens:
  pad_token: "</s>"
```

> 💡 **Model တစ်ခုချင်းစီရဲ့ special tokens ကို စစ်ဖို့:**
> ```python
> from transformers import AutoTokenizer
> tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
> print(f"eos: {tok.eos_token}, pad: {tok.pad_token}, bos: {tok.bos_token}")
> ```

---

## OOM (Out of Memory) ဖြစ်ရင် ဖြေရှင်းနည်း

### OOM Prevention Checklist

```
VRAM မလောက်ရင် ဒီ settings တွေကို အစဉ်လိုက် ပြင်ပါ:

Step 1: gradient_checkpointing: true        ← ~40% VRAM save
Step 2: micro_batch_size: 1                 ← Batch size minimum
Step 3: sequence_len ကို လျှော့ (1024→512→256)
Step 4: flash_attention: true               ← Attention memory save
Step 5: optimizer: adafactor                 ← AdamW ထက် memory နည်း
Step 6: sample_packing: false               ← Pack ဖြုတ်ကြည့်
Step 7: eval_batch_size: 1
Step 8: Model size ကို ပြောင်း (1B → 0.5B → 135M)
```

### Memory-Efficient Optimizer ပြောင်းခြင်း

AdamW optimizer ဟာ parameter 1 ခုအတွက် **8 bytes** (2 states × 4 bytes) သုံးပါတယ်။ Adafactor က **နည်းနည်း** ပိုသက်သာပါတယ်:

```yaml
# Option A: Standard AdamW (default)
optimizer: adamw_torch
# Memory: 8 bytes/param → 1B model = ~8 GB for optimizer alone

# Option B: Fused AdamW (slightly better)
optimizer: adamw_torch_fused

# Option C: Adafactor (less memory, no momentum states)
optimizer: adafactor
# Memory: ~4 bytes/param → 1B model = ~4 GB for optimizer

# Option D: 8-bit AdamW (significant savings)
optimizer: adamw_bnb_8bit
# Memory: ~4 bytes/param → Uses BitsAndBytes 8-bit states
```

---

## Training ပြီးရင် Inference Test

### 6.1: Trained Model Test

```bash
# Training ပြီးရင် inference test
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/workspace/data/test_fft/output_qwen25_05b'  # output path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Test inference
prompt = '### Instruction:\nWhat is the capital of Myanmar?\n\n### Response:\n'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
"
```

### 6.2: Axolotl Inference Command

```bash
# Axolotl ရဲ့ built-in inference
python -m axolotl.cli.inference /workspace/data/test_fft/config_qwen25_05b.yml \
  --lora_model_dir="/workspace/data/test_fft/output_qwen25_05b"
```

---

## VRAM Usage Summary — RTX 3050 12GB

```
┌──────────────────────────────────────────────────────────────────┐
│              RTX 3050 12GB — Full Fine-Tuning VRAM Map          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  0 GB  ├─────────────────────────────────────────────┤ 12 GB    │
│        │                                             │          │
│        │  SmolLM2-135M FFT                           │          │
│        │  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ~2 GB    │          │
│        │                                             │          │
│        │  Qwen2.5-0.5B FFT                           │          │
│        │  █████████████░░░░░░░░░░░░░░░░░░ ~5-6 GB   │ ← Sweet  │
│        │                                             │    Spot  │
│        │  Llama-3.2-1B FFT                           │          │
│        │  ██████████████████████████████░░ ~10-11 GB │ ← Tight  │
│        │                                             │          │
│        │  Qwen2.5-1.5B FFT                           │          │
│        │  ████████████████████████████████████ ~15 GB│ ← OOM ❌ │
│        │                                             │          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Full Fine-Tuning vs LoRA/QLoRA — 12GB GPU Comparison

| | Full Fine-Tuning | LoRA | QLoRA |
|---|---|---|---|
| **Max Model Size (12GB)** | ~1B | ~7B | ~7-8B |
| **Training Quality** | Best | Good | Good (slight loss) |
| **Trainable Params** | 100% | 0.1-10% | 0.1-10% |
| **Training Speed** | Slow | Fast | Medium |
| **VRAM Usage** | High | Medium | Low |
| **Use Case** | Small model mastery | Large model adaptation | Large model, low VRAM |

> 💡 **12GB VRAM Recommendation:**
> - **စမ်းသပ်ဖို့ / learning:** Full Fine-Tuning + small model (ဒီ guide)
> - **Production quality:** QLoRA + 7B-8B model (PEFT_Types.md ကို ကြည့်ပါ)

---

## Complete Workflow Summary

```bash
# ============================================================
# RTX 3050 12GB — Full Fine-Tuning Quick Start
# ============================================================

# 1. Container start
docker run --gpus '"all"' --rm -it \
  --shm-size=4g \
  -v $(pwd)/workspace:/workspace/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  axolotlai/axolotl:main-latest

# 2. GPU check
nvidia-smi

# 3. HuggingFace login (gated models အတွက်)
huggingface-cli login

# 4. Dataset + Config ပြင်ဆင် (Step 3.1, 3.2 ကို ကြည့်ပါ)
mkdir -p /workspace/data/test_fft
# ... create train.jsonl and config yml ...

# 5. Data preprocess (optional, validates config)
python -m axolotl.cli.preprocess /workspace/data/test_fft/config_qwen25_05b.yml

# 6. Train!
accelerate launch -m axolotl.cli.train /workspace/data/test_fft/config_qwen25_05b.yml

# 7. Monitor VRAM (another terminal)
watch -n 1 nvidia-smi

# 8. Inference test
python -m axolotl.cli.inference /workspace/data/test_fft/config_qwen25_05b.yml

# 9. Output ထုတ်ယူ (container ပြင်ပ)
# Trained model: /workspace/data/test_fft/output_qwen25_05b/
```

---

## Next Steps

- **LoRA/QLoRA** နဲ့ 7B-8B model train ချင်ရင် → [PEFT_Types.md](PEFT_Types.md) ကို ကြည့်ပါ
- **Dataset format** အသေးစိတ် → [Datasets.md](Datasets.md) ကို ကြည့်ပါ
- **Model ရွေးချယ်ခြင်း** → [Models.md](Models.md) ကို ကြည့်ပါ
