# SmolLM2-135M — Full Fine-Tuning Test Run

## Overview

| Item | Detail |
|---|---|
| **Model** | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) |
| **Parameters** | 135M |
| **Training Type** | Full Fine-Tuning (FFT) |
| **Expected VRAM** | ~2-3 GB |
| **GPU** | RTX 3050 12GB |
| **ရည်ရွယ်ချက်** | Pipeline စစ်ဆေးခြင်း (smoke test) |

## Folder Structure

```
SmolLM2-135M/
├── README.md          ← ဒီဖိုင်
├── config.yml         ← Axolotl training config
├── train.jsonl        ← Test dataset (10 examples)
├── prepared/          ← (auto) Preprocessed data cache
└── output/            ← (auto) Trained model output
```

## Quick Start

### 1. Docker Container စတင်ခြင်း

```bash
# Host machine ကနေ run ပါ
# fine_tuning folder တစ်ခုလုံးကို mount လုပ်ပါတယ်
# HuggingFace cache ကို mount လုပ်ထားတဲ့အတွက် model ကို တစ်ခါပဲ download ဆွဲရပါမယ်
docker run --gpus '"all"' --rm -it \
  --shm-size=4g \
  -v /home/mr_cobot/Desktop/dev_projects/ai/generative.ai/learning/fine_tuning:/workspace/data \
  -v /home/mr_cobot/.cache/huggingface:/root/.cache/huggingface \
  axolotlai/axolotl:main-latest
```

### 2. Container ထဲမှာ GPU စစ်ဆေးခြင်း

```bash
nvidia-smi
```

### 3. Preprocess (Optional — Config validate ဖို့)

```bash
accelerate launch -m axolotl.cli.preprocess /workspace/data/SmolLM2-135M/config.yml
```

### 4. Training Run

```bash
accelerate launch -m axolotl.cli.train /workspace/data/SmolLM2-135M/config.yml
```

### 5. VRAM Monitoring (နောက်ထပ် terminal မှာ)

```bash
# Container ID ရှာ
docker ps

# Container ထဲဝင်
docker exec -it <container_id> bash

# VRAM ကြည့်
watch -n 1 nvidia-smi
```

### 6. Inference Test

```bash
accelerate launch -m axolotl.cli.inference /workspace/data/SmolLM2-135M/config.yml
```

## Config Key Settings

```yaml
base_model: HuggingFaceTB/SmolLM2-135M
special_tokens:
  pad_token: "<|endoftext|>"        # ← pad_token fix (required)
sequence_len: 512
micro_batch_size: 2
gradient_checkpointing: true
bf16: auto
```

## Known Issues

- **pad_token error** — SmolLM2 tokenizer မှာ pad_token မရှိပါ။ `special_tokens.pad_token: "<|endoftext|>"` ထည့်ပေးရပါမယ်။ Config မှာ ထည့်ပြီးသားပါ။

## Notes

- ဒီ model က **pipeline testing** အတွက်သာ ဖြစ်ပါတယ်
- Model အရမ်းသေးလို့ production-grade output ရဖို့ မမျှော်လင့်ပါနဲ့
- Config/data format အဆင်ပြေကြောင်း confirm ဖြစ်ပြီးရင် → Qwen2.5-0.5B ကို ဆက် run ပါ
