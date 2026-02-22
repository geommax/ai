# Llama-3.2-1B — Full Fine-Tuning (Maximum)

## Overview

| Item | Detail |
|---|---|
| **Model** | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| **Parameters** | 1.24B |
| **Training Type** | Full Fine-Tuning (FFT) |
| **Expected VRAM** | ~9-11 GB ⚠️ |
| **GPU** | RTX 3050 12GB |
| **ရည်ရွယ်ချက်** | 12GB VRAM limit test (tight fit) |

## ⚠️ Warning

ဒီ model က **12GB VRAM ရဲ့ limit နားကပ်** ပါတယ်။ OOM ဖြစ်နိုင်ခြေ ရှိပါတယ်။

- OOM ဖြစ်ရင် → `sequence_len` ကို 256 ထိ လျှော့ပါ
- ဒါလည်း မရရင် → Qwen2.5-0.5B ကို သုံးပါ
- **Llama 3.2 ဟာ gated model ဖြစ်ပါတယ်** — HuggingFace မှာ access request လုပ်ပြီး accept ရပါမယ်

## Folder Structure

```
Llama-3.2-1B/
├── README.md          ← ဒီဖိုင်
├── config.yml         ← Axolotl training config
├── train.jsonl        ← Test dataset (10 examples)
├── prepared/          ← (auto) Preprocessed data cache
└── output/            ← (auto) Trained model output
```

## Prerequisites

### Llama 3.2 Access Request

Llama 3.2 ဟာ gated model ဖြစ်တဲ့အတွက် HuggingFace မှာ access request လိုပါတယ်:

1. https://huggingface.co/meta-llama/Llama-3.2-1B သွားပါ
2. "Access Request" button နှိပ်ပါ
3. License agreement accept လုပ်ပါ
4. Approval ရဖို့ စောင့်ပါ (usually instant)

### HuggingFace Login

```bash
# Container ထဲမှာ login လုပ်ပါ
huggingface-cli login
# OR
hf auth login --token $HF_TOKEN
```

## Quick Start

### 1. Docker Container စတင်ခြင်း

```bash
docker run --gpus '"all"' --rm -it \
  --shm-size=4g \
  -v /home/mr_cobot/Desktop/dev_projects/ai/generative.ai/learning/fine_tuning:/workspace/data \
  -v /home/mr_cobot/.cache/huggingface:/root/.cache/huggingface \
  axolotlai/axolotl:main-latest
```

### 2. HuggingFace Login (REQUIRED — gated model)

```bash
huggingface-cli login
```

### 3. Preprocess (Optional)

```bash
accelerate launch -m axolotl.cli.preprocess /workspace/data/Llama-3.2-1B/config.yml
```

### 4. Training Run

```bash
accelerate launch -m axolotl.cli.train /workspace/data/Llama-3.2-1B/config.yml
```

### 5. VRAM Monitoring (⚠️ Recommended — OOM ဖြစ်နိုင်)

```bash
# နောက်ထပ် terminal မှာ — VRAM ကို closely monitor လုပ်ပါ
docker ps
docker exec -it <container_id> bash
watch -n 1 nvidia-smi
```

### 6. Inference Test

```bash
accelerate launch -m axolotl.cli.inference /workspace/data/Llama-3.2-1B/config.yml
```

## Config Key Settings

```yaml
base_model: meta-llama/Llama-3.2-1B
special_tokens:
  pad_token: "<|finetune_right_pad_id|>"   # ← Llama 3 native pad token
sequence_len: 512                           # ← 1024 ဆိုရင် OOM ဖြစ်နိုင်
micro_batch_size: 1                         # ← absolute minimum
gradient_accumulation_steps: 8
optimizer: adamw_torch_fused                # ← memory-efficient optimizer
gradient_checkpointing: true                # ← MUST for 1B FFT on 12GB
flash_attention: true
sample_packing: true
bf16: auto
```

## VRAM Estimate

```
┌─────────────────────────────────────────────────┐
│ Llama-3.2-1B FFT — VRAM Breakdown    ⚠️ TIGHT  │
├─────────────────────────────────────────────────┤
│ Model Weights (bf16):       ~2.5 GB             │
│ Gradients:                  ~2.5 GB             │
│ Optimizer States (AdamW):   ~5.0 GB             │
│ Activations + Overhead:     ~1.0-2.0 GB         │
├─────────────────────────────────────────────────┤
│ Total:                      ~9-11 GB / 12 GB    │
│ Headroom:                   ~1-3 GB ⚠️          │
└─────────────────────────────────────────────────┘
```

## OOM ဖြစ်ရင် ဖြေရှင်းနည်း

အောက်ပါ settings တွေကို **အစဉ်လိုက်** ပြင်ပါ:

### Fix 1: Sequence Length လျှော့

```yaml
sequence_len: 256       # 512 → 256
```

### Fix 2: Sample Packing ပိတ်

```yaml
sample_packing: false
```

### Fix 3: Optimizer ပြောင်း

```yaml
# Adafactor — optimizer states memory ~50% save
optimizer: adafactor

# OR 8-bit AdamW
optimizer: adamw_bnb_8bit
```

### Fix 4: ဒါတွေ အားလုံး မရရင်

→ **Qwen2.5-0.5B** folder ကို သုံးပါ (VRAM ~5-6GB ပဲ ကုန်ပါတယ်)

## Notes

- ⚠️ **VRAM limit** နားကပ်ပါတယ် — monitor closely
- 🔑 **Gated model** — HF login + access approval လိုပါတယ်
- SmolLM2-135M → Qwen2.5-0.5B → ဒီ model (**အစဉ်လိုက် run** ပါ)
- Production အတွက် 1B model FFT ထက် **QLoRA + 7B-8B model** ကို recommend ပါတယ်
