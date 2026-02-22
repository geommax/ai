# Qwen2.5-0.5B — Full Fine-Tuning (Recommended)

## Overview

| Item | Detail |
|---|---|
| **Model** | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| **Parameters** | 0.5B (494M) |
| **Training Type** | Full Fine-Tuning (FFT) |
| **Expected VRAM** | ~5-6 GB |
| **GPU** | RTX 3050 12GB |
| **ရည်ရွယ်ချက်** | 🎯 12GB VRAM FFT Sweet Spot |

## Folder Structure

```
Qwen2.5-0.5B/
├── README.md          ← ဒီဖိုင်
├── config.yml         ← Axolotl training config
├── train.jsonl        ← Test dataset (10 examples)
├── prepared/          ← (auto) Preprocessed data cache
└── output/            ← (auto) Trained model output
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

### 2. Preprocess (Optional)

```bash
accelerate launch -m axolotl.cli.preprocess /workspace/data/Qwen2.5-0.5B/config.yml
```

### 3. Training Run

```bash
accelerate launch -m axolotl.cli.train /workspace/data/Qwen2.5-0.5B/config.yml
```

### 4. VRAM Monitoring

```bash
# နောက်ထပ် terminal မှာ
docker ps
docker exec -it <container_id> bash
watch -n 1 nvidia-smi
```

### 5. Inference Test

```bash
# Interactive inference
accelerate launch -m axolotl.cli.inference /workspace/data/Qwen2.5-0.5B/config.yml

# OR Python script နဲ့
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = '/workspace/data/Qwen2.5-0.5B/output'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map='auto'
)

prompt = '### Instruction:\nWhat is the capital of Myanmar?\n\n### Response:\n'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

## Config Key Settings

```yaml
base_model: Qwen/Qwen2.5-0.5B
special_tokens:
  pad_token: "<|endoftext|>"
sequence_len: 1024
micro_batch_size: 1
gradient_accumulation_steps: 8       # effective batch = 8
gradient_checkpointing: true
flash_attention: true
sample_packing: true                 # GPU efficiency တိုးမြင့်
bf16: auto
```

## VRAM Estimate

```
┌─────────────────────────────────────────────────┐
│ Qwen2.5-0.5B FFT — VRAM Breakdown              │
├─────────────────────────────────────────────────┤
│ Model Weights (bf16):       ~1.0 GB             │
│ Gradients:                  ~1.0 GB             │
│ Optimizer States (AdamW):   ~2.0 GB             │
│ Activations + Overhead:     ~1.0-2.0 GB         │
├─────────────────────────────────────────────────┤
│ Total:                      ~5-6 GB / 12 GB     │
│ Headroom:                   ~6-7 GB ✅          │
└─────────────────────────────────────────────────┘
```

## ဘာကြောင့် Recommend လုပ်တာလဲ?

- **VRAM 50% ပဲ သုံး** → OOM risk မရှိ
- **Qwen2.5 architecture** → Multilingual ကောင်း၊ performance ကောင်း
- **sequence_len: 1024** ထိ သုံးနိုင် → longer context training
- **sample_packing** ဖွင့်လို့ ရ → training speed up

## OOM ဖြစ်ရင် (ဖြစ်နိုင်ခြေ နည်းပါတယ်)

```yaml
# sequence_len လျှော့
sequence_len: 512

# sample_packing ပိတ်
sample_packing: false
```

## Notes

- ✅ 12GB GPU အတွက် FFT **sweet spot** ဖြစ်ပါတယ်
- Production dataset (1K-10K examples) နဲ့ train ရင် meaningful results ရနိုင်ပါတယ်
- SmolLM2-135M smoke test ပြီးမှ ဒီ model ကို run ပါ
