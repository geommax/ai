# Axolotl Framework — အလုပ်လုပ်ပုံ နှင့် Essential Commands

## Axolotl ဆိုတာ ဘာလဲ?

Axolotl ဟာ **LLM fine-tuning** အတွက် ရေးထားတဲ့ open-source framework ဖြစ်ပြီး၊ YAML config file တစ်ခုတည်းနဲ့ training pipeline တစ်ခုလုံးကို ထိန်းချုပ်နိုင်ပါတယ်။

```
GitHub: https://github.com/axolotl-ai-cloud/axolotl
```

---

## Axolotl Framework Architecture — အလုပ်လုပ်ပုံ

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        AXOLOTL FRAMEWORK OVERVIEW                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │  YAML Config │  ← တစ်ခုတည်းနဲ့ အားလုံး ထိန်းချုပ်                        │
│   │  (.yml file) │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │                   CONFIG PARSER                          │              │
│   │  Model config + Dataset config + Training config parse   │              │
│   └──────┬───────────────┬──────────────────┬────────────────┘              │
│          │               │                  │                                │
│          ▼               ▼                  ▼                                │
│   ┌────────────┐  ┌────────────┐    ┌──────────────┐                        │
│   │   Model    │  │  Dataset   │    │   Training   │                        │
│   │  Loading   │  │  Loading   │    │   Config     │                        │
│   │            │  │            │    │              │                        │
│   │ • HF Hub   │  │ • Local    │    │ • Optimizer  │                        │
│   │ • Local    │  │ • HF Hub   │    │ • Scheduler  │                        │
│   │ • 4/8-bit  │  │ • Multiple │    │ • Precision  │                        │
│   │ • Adapter  │  │   datasets │    │ • Batch size │                        │
│   └─────┬──────┘  └─────┬──────┘    └──────┬───────┘                        │
│         │               │                  │                                │
│         ▼               ▼                  │                                │
│   ┌────────────┐  ┌──────────────┐         │                                │
│   │ Tokenizer  │  │  Preprocess  │         │                                │
│   │  Loading   │  │  & Format    │         │                                │
│   │            │  │              │         │                                │
│   │ • Chat     │  │ • Tokenize   │         │                                │
│   │   template │  │ • Pack/Pad   │         │                                │
│   │ • Special  │  │ • Train/Val  │         │                                │
│   │   tokens   │  │   split      │         │                                │
│   └─────┬──────┘  └──────┬───────┘         │                                │
│         │               │                  │                                │
│         └───────┬───────┘                  │                                │
│                 │                          │                                │
│                 ▼                          ▼                                │
│   ┌──────────────────────────────────────────────────────┐                  │
│   │              🔥 TRAINING LOOP (HF Trainer)           │                  │
│   │                                                      │                  │
│   │   ┌──────────────────────────────────────────────┐  │                  │
│   │   │  for each epoch:                             │  │                  │
│   │   │    for each batch:                           │  │                  │
│   │   │      1. Forward Pass  → Loss 計算            │  │                  │
│   │   │      2. Backward Pass → Gradients 計算       │  │                  │
│   │   │      3. Optimizer Step → Weights Update      │  │                  │
│   │   │      4. Logging (loss, lr, VRAM)             │  │                  │
│   │   │      5. Eval (if eval_steps reached)         │  │                  │
│   │   │      6. Save checkpoint (if save_steps)      │  │                  │
│   │   └──────────────────────────────────────────────┘  │                  │
│   │                                                      │                  │
│   │   Powered by: 🤗 Transformers Trainer               │                  │
│   │              + 🚀 Accelerate (multi-GPU/DeepSpeed)  │                  │
│   │              + ⚡ FlashAttention                     │                  │
│   │              + 🔧 PEFT (LoRA/QLoRA)                 │                  │
│   └──────────────────────────┬───────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────┐                  │
│   │                    OUTPUT                            │                  │
│   │                                                      │                  │
│   │   📁 output_dir/                                    │                  │
│   │   ├── checkpoint-100/  (intermediate saves)         │                  │
│   │   ├── checkpoint-200/                               │                  │
│   │   ├── adapter_model.safetensors (if LoRA)           │                  │
│   │   ├── model.safetensors (if FFT)                    │                  │
│   │   ├── tokenizer.json                                │                  │
│   │   ├── config.json                                   │                  │
│   │   └── training_args.bin                             │                  │
│   └──────────────────────────────────────────────────────┘                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Axolotl CLI Command Pipeline — အလုပ်လုပ်ပုံ Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                    AXOLOTL COMMAND PIPELINE                        │
│                                                                    │
│    config.yml ── preprocess ──→ train ──→ inference / merge        │
│                                                                    │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│    │ 1.Config │───→│2.Preproc │───→│ 3.Train  │───→│4.Infer/  │   │
│    │  ရေးသား   │    │  Data    │    │  Model   │    │  Merge   │   │
│    └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│         │               │               │               │         │
│    YAML file       Tokenize &      Training loop    Test or       │
│    ပြင်ဆင်           validate        run            deploy       │
│                                                                    │
│    Optional:                                                       │
│    ┌──────────┐                                                    │
│    │ 5.Eval   │  ← Benchmark evaluation                          │
│    └──────────┘                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Essential Commands — အမြဲသုံးရမယ့် Commands

### 🔧 1. Preprocess (Data ပြင်ဆင်ခြင်း)

Training မစခင် dataset ကို tokenize + validate + cache လုပ်ပေးပါတယ်။

```bash
# Basic preprocess
python -m axolotl.cli.preprocess config.yml

# Accelerate နဲ့ (recommended)
accelerate launch -m axolotl.cli.preprocess config.yml
```

**ဘာလုပ်ပေးလဲ?**
```
preprocess command:
├── ✅ Config YAML ကို validate
├── ✅ Dataset load + format check
├── ✅ Tokenization (text → token IDs)
├── ✅ Sample packing (if enabled)
├── ✅ Train/Val split
├── ✅ Preprocessed data ကို disk မှာ cache
└── ✅ Token count / sequence length statistics ပြ
```

**ဘယ်အခါ သုံးသင့်လဲ?**
- Config ရေးပြီးတိုင်း (validate ဖို့)
- Dataset ကြီးရင် (preprocess တစ်ခါ run ပြီး cache ထားရင် training ပိုမြန်)
- Dataset format error ရှိ/မရှိ စစ်ဖို့

---

### 🚀 2. Train (Training Run)

Model training ကို စတင်ပါတယ်။ Axolotl ရဲ့ **အဓိက command** ဖြစ်ပါတယ်။

```bash
# Single GPU training
accelerate launch -m axolotl.cli.train config.yml

# OR (accelerate မသုံးဘဲ)
python -m axolotl.cli.train config.yml

# Multi-GPU training (2 GPUs)
accelerate launch --num_processes 2 -m axolotl.cli.train config.yml

# DeepSpeed နဲ့ training
accelerate launch --config_file deepspeed_config.yaml -m axolotl.cli.train config.yml

# Resume from checkpoint
accelerate launch -m axolotl.cli.train config.yml --resume_from_checkpoint /path/to/checkpoint
```

**ဘာလုပ်ပေးလဲ?**
```
train command:
├── 1. Config parse
├── 2. Model load (HF Hub / local)
├── 3. Tokenizer load
├── 4. Dataset load (preprocess if not cached)
├── 5. Adapter setup (if LoRA/QLoRA)
├── 6. Training loop start
│   ├── Forward pass
│   ├── Loss calculation
│   ├── Backward pass
│   ├── Optimizer step
│   ├── Logging (WandB / TensorBoard / console)
│   ├── Evaluation (periodic)
│   └── Checkpoint save (periodic)
├── 7. Final model save
└── 8. Training complete! 🎉
```

---

### 💬 3. Inference (စမ်းသပ်ခြင်း)

Train ပြီးသား model ကို interactive chat mode မှာ စမ်းကြည့်ပါတယ်။

```bash
# Basic inference (interactive prompt)
accelerate launch -m axolotl.cli.inference config.yml

# LoRA model inference (adapter path specify)
accelerate launch -m axolotl.cli.inference config.yml \
  --lora_model_dir="./output/checkpoint-final"

# Gradio UI နဲ့ inference
accelerate launch -m axolotl.cli.inference config.yml --gradio

# Specific prompt နဲ့ inference
accelerate launch -m axolotl.cli.inference config.yml \
  --prompter_type="alpaca" \
  --instruction="What is the capital of Myanmar?"
```

**ဘာလုပ်ပေးလဲ?**
```
inference command:
├── Model + Tokenizer load
├── Adapter merge (if LoRA)
├── Interactive mode start
│   ├── User prompt input ←──┐
│   ├── Tokenize              │
│   ├── Generate              │
│   ├── Decode + Display ─────┘
│   └── Loop until quit
└── OR Gradio web UI launch
```

---

### 🔗 4. Merge LoRA (Adapter ပေါင်းခြင်း)

LoRA/QLoRA adapter weights ကို base model ထဲ **merge** လုပ်ပြီး standalone model ဖန်တီးပါတယ်။

```bash
# LoRA adapter ကို base model ထဲ merge
python -m axolotl.cli.merge_lora config.yml \
  --lora_model_dir="./output"

# Output directory specify
python -m axolotl.cli.merge_lora config.yml \
  --lora_model_dir="./output" \
  --output_dir="./merged_model"
```

**ဘာလုပ်ပေးလဲ?**
```
merge_lora command:
├── Base model load
├── LoRA adapter load
├── Weights merge (W' = W + A×B)
├── Merged model save (safetensors)
└── Tokenizer + Config copy

Merged model ကို ဒီနေရာတွေမှာ သုံးနိုင်:
├── 🔄 GGUF convert → Ollama/llama.cpp
├── 📦 HF Hub upload
├── 🚀 vLLM / TGI serving
└── 🔧 ထပ် fine-tune
```

**ဘယ်အခါ merge လုပ်သင့်လဲ?**
- LoRA/QLoRA training ပြီးတိုင်း (deployment အတွက်)
- GGUF/AWQ/GPTQ convert မလုပ်ခင်
- HF Hub ကို upload မလုပ်ခင်

> ⚠️ Full Fine-Tuning (FFT) မှာ merge command **မလိုပါ** — model weights ကို directly save ထားပြီးသားပါ။

---

### 📊 5. Evaluate (အကဲဖြတ်ခြင်း)

Model ရဲ့ performance ကို benchmark datasets ပေါ်မှာ evaluate လုပ်ပါတယ်။

```bash
# Evaluation run
accelerate launch -m axolotl.cli.evaluate config.yml

# Specific eval dataset နဲ့
accelerate launch -m axolotl.cli.evaluate config.yml \
  --lora_model_dir="./output"
```

---

## Command Quick Reference Card

```
┌────────────────────────────────────────────────────────────────────────┐
│                     AXOLOTL COMMAND CHEAT SHEET                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  📋 PREPROCESS (validate + tokenize + cache)                          │
│  accelerate launch -m axolotl.cli.preprocess config.yml               │
│                                                                        │
│  🚀 TRAIN (model training)                                            │
│  accelerate launch -m axolotl.cli.train config.yml                    │
│                                                                        │
│  🔄 RESUME TRAINING (checkpoint ကနေ ဆက်)                              │
│  accelerate launch -m axolotl.cli.train config.yml \                  │
│    --resume_from_checkpoint output/checkpoint-500                     │
│                                                                        │
│  💬 INFERENCE (interactive test)                                       │
│  accelerate launch -m axolotl.cli.inference config.yml                │
│                                                                        │
│  🌐 INFERENCE + GRADIO UI                                             │
│  accelerate launch -m axolotl.cli.inference config.yml --gradio       │
│                                                                        │
│  🔗 MERGE LORA (adapter → full model)                                 │
│  python -m axolotl.cli.merge_lora config.yml \                        │
│    --lora_model_dir="./output"                                        │
│                                                                        │
│  📊 EVALUATE (benchmark test)                                         │
│  accelerate launch -m axolotl.cli.evaluate config.yml                 │
│                                                                        │
│  🐛 DEBUG (1 step train for testing)                                  │
│  accelerate launch -m axolotl.cli.train config.yml --debug            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Training Workflow Diagram — Config ကနေ Deployment အထိ

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE AXOLOTL WORKFLOW                                │
│                                                                             │
│  ╔══════════════╗                                                          │
│  ║  1. PREPARE  ║                                                          │
│  ╚══════╤═══════╝                                                          │
│         │                                                                   │
│         ├── 📝 config.yml ရေးသား                                            │
│         │     ├── base_model: meta-llama/Llama-3.2-1B                      │
│         │     ├── adapter: lora / qlora / (none=FFT)                       │
│         │     ├── datasets: [{path, type}]                                 │
│         │     └── training params (lr, epochs, batch...)                   │
│         │                                                                   │
│         ├── 📁 Dataset ပြင်ဆင် (JSONL/JSON/Parquet)                         │
│         │     ├── alpaca format: {instruction, input, output}              │
│         │     ├── sharegpt format: {conversations: [...]}                  │
│         │     └── completion format: {text: "..."}                         │
│         │                                                                   │
│         └── 🔑 HF Login: huggingface-cli login                            │
│                                                                             │
│  ╔══════════════╗                                                          │
│  ║ 2. VALIDATE  ║                                                          │
│  ╚══════╤═══════╝                                                          │
│         │                                                                   │
│         └── ⚙️ accelerate launch -m axolotl.cli.preprocess config.yml     │
│              ├── Config validation ✓                                       │
│              ├── Dataset format check ✓                                    │
│              ├── Tokenization ✓                                            │
│              └── Cache to disk ✓                                           │
│                                                                             │
│  ╔══════════════╗                                                          │
│  ║  3. TRAIN    ║                                                          │
│  ╚══════╤═══════╝                                                          │
│         │                                                                   │
│         └── 🚀 accelerate launch -m axolotl.cli.train config.yml          │
│              │                                                              │
│              │  Training Loop:                                              │
│              │  ┌─────────────────────────────────────────┐                │
│              │  │ Epoch 1/3 ████████████████████░░░ 80%   │                │
│              │  │ Loss: 2.34 → 1.12 → 0.67 → 0.45       │                │
│              │  │ LR:   2e-5 → 1.5e-5 → ... → 0         │                │
│              │  │ VRAM: 8.5 GB / 12 GB                    │                │
│              │  │ Speed: 2.3 samples/sec                  │                │
│              │  └─────────────────────────────────────────┘                │
│              │                                                              │
│              ├── 💾 Checkpoints saved: output/checkpoint-{N}/              │
│              └── ✅ Final model saved: output/                             │
│                                                                             │
│  ╔══════════════╗                                                          │
│  ║  4. TEST     ║                                                          │
│  ╚══════╤═══════╝                                                          │
│         │                                                                   │
│         ├── 💬 accelerate launch -m axolotl.cli.inference config.yml      │
│         │      > Prompt: What is AI?                                       │
│         │      > Response: AI is a branch of computer science...           │
│         │                                                                   │
│         └── 📊 accelerate launch -m axolotl.cli.evaluate config.yml       │
│                                                                             │
│  ╔══════════════╗                                                          │
│  ║  5. DEPLOY   ║                                                          │
│  ╚══════╤═══════╝                                                          │
│         │                                                                   │
│         ├── [If LoRA] 🔗 python -m axolotl.cli.merge_lora config.yml      │
│         │                                                                   │
│         ├── 📤 Upload to HF Hub                                            │
│         │     huggingface-cli upload ./output org/model-name               │
│         │                                                                   │
│         ├── 🔄 Convert to GGUF (for Ollama/llama.cpp)                     │
│         │     python convert_hf_to_gguf.py ./merged --outtype q4_k_m      │
│         │                                                                   │
│         └── 🌐 Serve with vLLM / TGI / Ollama                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Docker Container ထဲမှာ သုံးရမယ့် Commands (Step-by-Step)

### Container Start + Setup

```bash
# 1. Container ဖွင့်
docker run --gpus '"all"' --rm -it \
  --shm-size=4g \
  -v $(pwd)/workspace:/workspace/data \
  -p 7860:7860 \
  axolotlai/axolotl:main-latest

# 2. GPU check
nvidia-smi

# 3. HF Login
huggingface-cli login --token $HF_TOKEN

# 4. Working directory
cd /workspace
```

### Training Session

```bash
# 5. Config + Data ပြင်ဆင် (ကိုယ့် config.yml ကို /workspace/data/ မှာ ထားပါ)
ls /workspace/data/

# 6. Preprocess — Data validate + cache
accelerate launch -m axolotl.cli.preprocess /workspace/data/config.yml

# 7. Train — Training start
accelerate launch -m axolotl.cli.train /workspace/data/config.yml

# 8. VRAM Monitor (another terminal)
watch -n 1 nvidia-smi
```

### Post-Training

```bash
# 9. Inference — Model စမ်းသပ်
accelerate launch -m axolotl.cli.inference /workspace/data/config.yml

# 10. Gradio UI နဲ့ စမ်းသပ်
accelerate launch -m axolotl.cli.inference /workspace/data/config.yml --gradio

# 11. LoRA Merge (LoRA/QLoRA training ပြီးမှ)
python -m axolotl.cli.merge_lora /workspace/data/config.yml \
  --lora_model_dir="/workspace/data/output" \
  --output_dir="/workspace/data/merged_model"

# 12. Merged model ကို host machine ထဲ ကူးယူ
# (container ပြင်ပမှာ)
# docker cp <container_id>:/workspace/data/merged_model ./merged_model
```

---

## Config YAML — Essential Fields Guide

Config YAML ဟာ Axolotl ရဲ့ **အသက်** ဖြစ်ပါတယ်။ ဒီ fields တွေကို နားလည်ဖို့ လိုပါတယ်:

```yaml
# ═══════════════════════════════════════════
# 📌 MODEL SECTION — ဘယ် model ကို train မလဲ
# ═══════════════════════════════════════════
base_model: meta-llama/Llama-3.2-1B      # HF model name or local path
model_type: LlamaForCausalLM              # Model architecture class
tokenizer_type: AutoTokenizer
trust_remote_code: true                    # Custom model code ခွင့်ပြု

# ═══════════════════════════════════════════
# 📌 ADAPTER SECTION — Training method
# ═══════════════════════════════════════════
# adapter:                                # ← Comment out = Full Fine-Tuning
adapter: lora                             # lora / qlora
lora_r: 32                                # LoRA rank
lora_alpha: 16                            # Scaling factor
lora_dropout: 0.05
lora_target_linear: true                  # All linear layers ကို target
# load_in_4bit: true                      # QLoRA အတွက် uncomment

# ═══════════════════════════════════════════
# 📌 DATASET SECTION — Training data
# ═══════════════════════════════════════════
datasets:
  - path: ./data/train.jsonl              # Local file path
    type: alpaca                           # Format type
    # type: sharegpt                       # Chat format
    # conversation: chatml                 # Chat template

  # Multiple datasets ပေါင်းလို့ ရတယ်
  # - path: org/dataset_name              # HF Hub dataset
  #   type: sharegpt
  #   split: train

val_set_size: 0.05                        # 5% for validation
dataset_prepared_path: ./prepared          # Cache directory

# ═══════════════════════════════════════════
# 📌 TRAINING SECTION — Training hyperparameters
# ═══════════════════════════════════════════
output_dir: ./output                       # Output directory

sequence_len: 1024                         # Max token length
num_epochs: 3                              # Training epochs
micro_batch_size: 1                        # Batch per GPU (VRAM dependent)
gradient_accumulation_steps: 8             # Effective batch = micro × accum
eval_batch_size: 1

learning_rate: 2e-4                        # Learning rate (SFT default)
optimizer: adamw_torch                     # Optimizer
lr_scheduler: cosine                       # LR schedule
weight_decay: 0.01
warmup_ratio: 0.1                          # Warmup portion

# ═══════════════════════════════════════════
# 📌 PRECISION & MEMORY — GPU optimization
# ═══════════════════════════════════════════
bf16: auto                                 # bfloat16 (Ampere+ GPUs)
tf32: true                                 # TF32 (faster matrix math)
gradient_checkpointing: true               # Trade compute for memory
flash_attention: true                      # FlashAttention2
sample_packing: true                       # Pack short sequences
pad_to_sequence_len: true

# ═══════════════════════════════════════════
# 📌 LOGGING & SAVING
# ═══════════════════════════════════════════
logging_steps: 1                           # Log every N steps
eval_steps: 20                             # Evaluate every N steps
save_steps: 100                            # Save checkpoint every N steps
save_total_limit: 3                        # Keep N latest checkpoints

# Weights & Biases logging (optional)
# wandb_project: my-project
# wandb_run_id: run-001

# ═══════════════════════════════════════════
# 📌 SPECIAL FEATURES
# ═══════════════════════════════════════════
# chat_template: chatml                    # Chat template
# neftune_noise_alpha: 5                   # NEFTune noise
# rl: dpo                                  # DPO training
# dpo_beta: 0.1                            # DPO beta

seed: 42
strict: false
```

---

## Common Command Patterns

### Pattern 1: Quick Test (1-step debug)

Config မှန်/မမှန် **မြန်မြန် စစ်ဖို့**:

```bash
# Debug mode — 1 training step ပဲ run
accelerate launch -m axolotl.cli.train config.yml --debug

# ဒါက VRAM ဘယ်လောက်ကုန်လဲ၊ config error ရှိ/မရှိ ချက်ချင်း သိနိုင်
```

### Pattern 2: Preprocess → Train → Inference (Full Pipeline)

```bash
# Step 1: Validate & cache data
accelerate launch -m axolotl.cli.preprocess config.yml

# Step 2: Train
accelerate launch -m axolotl.cli.train config.yml

# Step 3: Test
accelerate launch -m axolotl.cli.inference config.yml
```

### Pattern 3: LoRA Training → Merge → Deploy

```bash
# Train with LoRA
accelerate launch -m axolotl.cli.train lora_config.yml

# Merge adapter into base model
python -m axolotl.cli.merge_lora lora_config.yml \
  --lora_model_dir="./output"

# Upload merged model
huggingface-cli upload ./merged_model your-username/model-name
```

### Pattern 4: Resume Interrupted Training

```bash
# Training ကျိုးသွားရင် checkpoint ကနေ ဆက်
accelerate launch -m axolotl.cli.train config.yml \
  --resume_from_checkpoint ./output/checkpoint-500
```

### Pattern 5: Multi-GPU Training

```bash
# 2 GPUs
accelerate launch --num_processes 2 -m axolotl.cli.train config.yml

# 4 GPUs
accelerate launch --num_processes 4 -m axolotl.cli.train config.yml

# DeepSpeed ZeRO-2 (multi-GPU memory optimization)
accelerate launch --use_deepspeed \
  --deepspeed_config_file ds_config.json \
  -m axolotl.cli.train config.yml
```

---

## Axolotl Internal Architecture — Technical Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AXOLOTL INTERNAL COMPONENTS                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  config.yml                                                          │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────────┐                                               │
│  │ axolotl.utils.   │                                               │
│  │ config.normalize │  ← Config parsing + validation                │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ├──────────────────────┬────────────────────┐              │
│           ▼                      ▼                    ▼              │
│  ┌────────────────┐    ┌─────────────────┐   ┌──────────────┐      │
│  │  Model Layer   │    │  Dataset Layer  │   │ Trainer Layer│      │
│  ├────────────────┤    ├─────────────────┤   ├──────────────┤      │
│  │                │    │                 │   │              │      │
│  │ transformers   │    │ datasets (HF)   │   │ HF Trainer   │      │
│  │ AutoModel      │    │ load_dataset()  │   │ + Accelerate │      │
│  │ ┌──────────┐   │    │ ┌───────────┐   │   │ ┌──────────┐│      │
│  │ │BitsAndByt│   │    │ │Prompters: │   │   │ │Callbacks:││      │
│  │ │es (4/8bit│   │    │ │ alpaca    │   │   │ │ logging  ││      │
│  │ │ quant)   │   │    │ │ sharegpt  │   │   │ │ saving   ││      │
│  │ └──────────┘   │    │ │ chat_tmpl │   │   │ │ eval     ││      │
│  │ ┌──────────┐   │    │ │ completn  │   │   │ │ early    ││      │
│  │ │PEFT      │   │    │ └───────────┘   │   │ │ stopping ││      │
│  │ │ LoRA     │   │    │ ┌───────────┐   │   │ └──────────┘│      │
│  │ │ QLoRA    │   │    │ │Tokenizer  │   │   │ ┌──────────┐│      │
│  │ │ DoRA     │   │    │ │ + Chat    │   │   │ │Optimizer:││      │
│  │ └──────────┘   │    │ │ template  │   │   │ │ AdamW    ││      │
│  │ ┌──────────┐   │    │ └───────────┘   │   │ │ Adafactor││      │
│  │ │Flash     │   │    │ ┌───────────┐   │   │ │ 8bit     ││      │
│  │ │Attention │   │    │ │Sample     │   │   │ └──────────┘│      │
│  │ │ 2        │   │    │ │ Packing   │   │   │              │      │
│  │ └──────────┘   │    │ └───────────┘   │   │              │      │
│  └────────────────┘    └─────────────────┘   └──────────────┘      │
│                                                                      │
│  Underlying Libraries:                                               │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ 🤗 Transformers │ 🤗 PEFT │ 🤗 Accelerate │ 🤗 Datasets │     │
│  │ PyTorch │ FlashAttention2 │ BitsAndBytes │ DeepSpeed     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting Commands

```bash
# ── GPU/CUDA Issues ──────────────────────────────
nvidia-smi                                    # GPU status
python -c "import torch; print(torch.cuda.is_available())"  # CUDA check

# ── VRAM Monitoring ──────────────────────────────
watch -n 1 nvidia-smi                         # Real-time VRAM
python -c "
import torch
print(f'{torch.cuda.memory_allocated()/1024**3:.1f}GB allocated')
print(f'{torch.cuda.memory_reserved()/1024**3:.1f}GB reserved')
"

# ── Config Validation ────────────────────────────
accelerate launch -m axolotl.cli.preprocess config.yml   # Validates config

# ── Dataset Debug ────────────────────────────────
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='train.jsonl')
print(ds)
print(ds['train'][0])
"

# ── Disk Space ───────────────────────────────────
df -h                                         # Disk usage
du -sh ./output/*                             # Output size

# ── Kill Zombie GPU Processes ────────────────────
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}

# ── Clear GPU Memory ─────────────────────────────
python -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"
```
