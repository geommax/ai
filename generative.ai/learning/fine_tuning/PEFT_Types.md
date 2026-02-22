# PEFT (Parameter-Efficient Fine-Tuning) အမျိုးအစားများ

## PEFT ဆိုတာဘာလဲ?

PEFT ဆိုတာ **Parameter-Efficient Fine-Tuning** ရဲ့ အတိုကောက်ဖြစ်ပြီး၊ LLM (Large Language Model) တစ်ခုလုံးကို fine-tune လုပ်မယ့်အစား **parameter အနည်းငယ်ကိုသာ** ပြင်ဆင်ပြီး fine-tune လုပ်တဲ့ နည်းလမ်းဖြစ်ပါတယ်။

### Full Fine-Tuning vs PEFT

| | Full Fine-Tuning | PEFT |
|---|---|---|
| **ပြင်ဆင်တဲ့ Parameters** | Model parameter အားလုံး | Parameter အနည်းငယ် (0.1% - 10%) |
| **GPU Memory** | အရမ်းများများလိုအပ် | နည်းနည်းပဲလိုအပ် |
| **Training Time** | ကြာတယ် | မြန်တယ် |
| **Catastrophic Forgetting** | ဖြစ်နိုင်ခြေများ | ဖြစ်နိုင်ခြေနည်း |
| **Storage** | Model တစ်ခုလုံး save ရတယ် | Adapter weights လေးပဲ save ရတယ် |

---

## PEFT အမျိုးအစားများ

PEFT methods တွေကို အဓိက **၃ မျိုး** ခွဲနိုင်ပါတယ်။

### 1. Additive Methods (ထပ်ထည့်တဲ့ နည်းလမ်းများ)

Original model ကို မပြင်ဘဲ **parameter အသစ်တွေ ထပ်ထည့်** ပြီး train လုပ်တဲ့ နည်းလမ်းဖြစ်ပါတယ်။

### 2. Selective Methods (ရွေးချယ်တဲ့ နည်းလမ်းများ)

Model ရဲ့ **parameter အချို့ကိုသာ ရွေးချယ်** ပြီး train လုပ်တဲ့ နည်းလမ်းဖြစ်ပါတယ်။

### 3. Reparameterization Methods (ပြန်လည်ဖွဲ့စည်းတဲ့ နည်းလမ်းများ)

Model ရဲ့ weight matrices တွေကို **low-rank representation** နဲ့ ပြန်လည်ဖွဲ့စည်းပြီး train လုပ်တဲ့ နည်းလမ်းဖြစ်ပါတယ်။

---

## PEFT နည်းလမ်းများ အသေးစိတ်

---

### 🔷 1. LoRA (Low-Rank Adaptation)

**အမျိုးအစား:** Reparameterization Method

#### အလုပ်လုပ်ပုံ

LoRA ဟာ model ရဲ့ weight matrix `W` ကို directly ပြင်မယ့်အစား၊ **low-rank decomposition** ကိုသုံးပြီး update matrix `ΔW` ကို `A × B` အဖြစ် ခွဲထုတ်ပါတယ်။

```
W' = W + ΔW = W + (A × B)
```

- `W` = Original weight matrix (frozen, train မလုပ်)
- `A` = Down-projection matrix (d × r)
- `B` = Up-projection matrix (r × d)
- `r` = Rank (LoRA rank, e.g., 8, 16, 32, 64)

#### ဥပမာ

Original weight matrix `W` ရဲ့ size က `4096 × 4096` ဆိုရင်:
- Full fine-tune: `4096 × 4096 = 16,777,216` parameters
- LoRA (r=8): `(4096 × 8) + (8 × 4096) = 65,536` parameters → **~0.4% သာ**

#### Key Hyperparameters

| Parameter | ရှင်းလင်းချက် | Common Values |
|---|---|---|
| `lora_r` | Rank - Low-rank matrix ရဲ့ dimension | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor (alpha/r = scaling) | 16, 32 |
| `lora_dropout` | Dropout rate for regularization | 0.05, 0.1 |
| `lora_target_modules` | LoRA apply လုပ်မယ့် layers | `q_proj`, `v_proj`, `k_proj`, `o_proj` |

#### Axolotl Config Example

```yaml
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj
```

---

### 🔷 2. QLoRA (Quantized LoRA)

**အမျိုးအစား:** Reparameterization Method + Quantization

#### အလုပ်လုပ်ပုံ

QLoRA ဟာ LoRA ရဲ့ extension ဖြစ်ပြီး၊ base model ကို **4-bit quantization** လုပ်ပြီးမှ LoRA adapters ထည့်ပါတယ်။

```
Frozen Base Model (4-bit quantized) + LoRA Adapters (trainable, fp16/bf16)
```

#### QLoRA ရဲ့ Key Innovations

1. **4-bit NormalFloat (NF4)** - Normal distribution အတွက် optimal quantization data type
2. **Double Quantization** - Quantization constants ကိုပါ ထပ် quantize လုပ်ပြီး memory ပိုသက်သာအောင်လုပ်ခြင်း
3. **Paged Optimizers** - GPU memory overflow ကို CPU RAM ကို page လုပ်ပြီး ဖြေရှင်းခြင်း

#### Memory Comparison

| Method | 7B Model Memory |
|---|---|
| Full Fine-Tuning (fp16) | ~28 GB |
| LoRA (fp16) | ~14 GB |
| QLoRA (4-bit) | ~6 GB |

#### Axolotl Config Example

```yaml
adapter: qlora
load_in_4bit: true
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
```

---

### 🔷 3. DoRA (Weight-Decomposed Low-Rank Adaptation)

**အမျိုးအစား:** Reparameterization Method

#### အလုပ်လုပ်ပုံ

DoRA ဟာ LoRA ကို improve လုပ်ထားတာဖြစ်ပြီး၊ weight matrix ကို **magnitude** နဲ့ **direction** ဆိုပြီး ၂ ပိုင်း ခွဲပါတယ်။

```
W' = m × (V + ΔV) / ||V + ΔV||
```

- `m` = Magnitude vector (trainable)
- `V` = Direction matrix (LoRA နဲ့ update)
- `ΔV` = LoRA update for direction

#### LoRA vs DoRA

- **LoRA**: Magnitude နဲ့ direction ကို တစ်ပြိုင်နက် update လုပ်တယ်
- **DoRA**: Magnitude နဲ့ direction ကို **သီးခြား** update လုပ်တယ် → full fine-tuning ရဲ့ learning pattern နဲ့ ပိုနီးစပ်

#### Axolotl Config Example

```yaml
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
peft_use_dora: true
```

---

### 🔷 4. Prompt Tuning

**အမျိုးအစား:** Additive Method (Soft Prompts)

#### အလုပ်လုပ်ပုံ

Model ရဲ့ input embedding layer ရဲ့ **ရှေ့ဆုံးမှာ** trainable virtual tokens (soft prompts) တွေ ထည့်ပြီး train လုပ်ပါတယ်။

```
Input = [Soft Prompt Tokens] + [Actual Input Tokens]
         (trainable)           (frozen embeddings)
```

- Soft prompt ရဲ့ embedding vectors တွေကိုသာ train လုပ်ပါတယ်
- Model weights အားလုံး frozen ဖြစ်ပါတယ်
- Task-specific soft prompts တွေကို swap လုပ်ပြီး multi-task serving လုပ်နိုင်ပါတယ်

#### Trainable Parameters

Soft prompt length = 20 tokens, embedding dim = 4096 ဆိုရင်:
- `20 × 4096 = 81,920` parameters သာ train ရပါတယ်

---

### 🔷 5. Prefix Tuning

**အမျိုးအစား:** Additive Method

#### အလုပ်လုပ်ပုံ

Prompt Tuning နဲ့ ဆင်တူပေမယ့်၊ **transformer ရဲ့ layer တိုင်းမှာ** trainable prefix vectors တွေ ထည့်ပါတယ်။

```
Layer_i_output = Attention(prefix_key_i, prefix_value_i, input)
```

- Input embedding layer မှာပဲ မဟုတ်ဘဲ **every layer** ရဲ့ key-value pairs မှာ prefix ထည့်ပါတယ်
- Prompt Tuning ထက် expressiveness ပိုကောင်းပါတယ်
- Parameters ပိုများပါတယ် (layer count × prefix_length × hidden_dim × 2)

---

### 🔷 6. P-Tuning v2

**အမျိုးအစား:** Additive Method

#### အလုပ်လုပ်ပုံ

Prefix Tuning ရဲ့ improved version ဖြစ်ပြီး:

- Deep prompt tuning: **Layer တိုင်းမှာ** trainable continuous prompts ထည့်ပါတယ်
- Reparameterization ကို optional ဖြစ်စေပါတယ် (MLP encoder မလိုအပ်)
- NLU tasks တွေမှာ full fine-tuning နဲ့ comparable performance ရပါတယ်

---

### 🔷 7. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**အမျိုးအစား:** Additive Method

#### အလုပ်လုပ်ပုံ

Model ရဲ့ activations (key, value, feedforward) တွေကို **learned vectors** နဲ့ element-wise multiply (rescale) လုပ်ပါတယ်။

```
k' = l_k ⊙ k    (key activations ကို rescale)
v' = l_v ⊙ v    (value activations ကို rescale)
ff' = l_ff ⊙ ff  (feedforward activations ကို rescale)
```

- `l_k`, `l_v`, `l_ff` = Learned rescaling vectors (trainable)
- LoRA ထက် trainable parameters **အများကြီး နည်းပါတယ်**
- Few-shot learning မှာ ကောင်းပါတယ်

---

### 🔷 8. Adapter Layers (Bottleneck Adapters)

**အမျိုးအစား:** Additive Method

#### အလုပ်လုပ်ပုံ

Transformer layer တိုင်းရဲ့ **attention နဲ့ feedforward sublayer ကြားမှာ** small bottleneck modules (adapters) ထည့်ပါတယ်။

```
Adapter(x) = x + f(x × W_down) × W_up

W_down: d → r  (down-project)
f: activation function (ReLU/GELU)
W_up: r → d    (up-project)
```

- Original model weights freeze ထားပြီး adapter layers ကိုသာ train လုပ်ပါတယ်
- Bottleneck dimension `r` ကို ချိန်ညှိနိုင်ပါတယ်
- Residual connection ပါဝင်ပါတယ်

---

### 🔷 9. LoftQ (LoRA-Fine-Tuning-aware Quantization)

**အမျိုးအစား:** Reparameterization + Quantization

#### အလုပ်လုပ်ပုံ

QLoRA ကို improve လုပ်ထားတဲ့ method ဖြစ်ပြီး:

- Quantization error ကို LoRA initialization မှာ compensate လုပ်ပါတယ်
- Quantized weight + LoRA ရဲ့ sum ဟာ original weight နဲ့ **ပိုနီးကပ်** အောင် initialize လုပ်ပါတယ်
- QLoRA ထက် convergence ပိုကောင်းပါတယ်

```
min ||W - (Q + AB)||  (Alternating optimization)
```

---

### 🔷 10. NEFTune (Noisy Embeddings for Fine-Tuning)

**အမျိုးအစား:** Training Technique (PEFT နဲ့ တွဲသုံးလို့ရ)

#### အလုပ်လုပ်ပုံ

Training input embeddings တွေမှာ **uniform random noise** ထည့်ပြီး train လုပ်ပါတယ်။

```
embedding_noisy = embedding + α × uniform_noise / √(L × d)
```

- `α` = Noise scale (neftune_noise_alpha, e.g., 5, 10, 15)
- `L` = Sequence length
- `d` = Embedding dimension
- Inference မှာ noise မထည့်ပါ

#### Axolotl Config Example

```yaml
neftune_noise_alpha: 5
```

---

### 🔷 11. ReLoRA (Stacked LoRA)

**အမျိုးအစား:** Reparameterization Method

#### အလုပ်လုပ်ပုံ

LoRA adapters ကို **periodically merge** လုပ်ပြီး **reset** ကာ ထပ်ခါထပ်ခါ train လုပ်ပါတယ်။

```
Loop:
  1. Train LoRA for N steps
  2. Merge: W = W + A × B
  3. Reset A, B to new initialization
  4. Repeat
```

- High-rank updates ကို low-rank LoRA အသုံးပြုပြီး approximate လုပ်နိုင်ပါတယ်
- Pre-training stage မှာ LoRA ကို ထိရောက်စွာ အသုံးပြုနိုင်ပါတယ်

#### Axolotl Config Example

```yaml
adapter: lora
relora_steps: 200
relora_warmup_steps: 50
```

---

## Axolotl မှာ အသုံးပြုလို့ရတဲ့ PEFT Methods အကျဉ်းချုပ်

| PEFT Method | Axolotl Support | Config Key | မှတ်ချက် |
|---|---|---|---|
| **LoRA** | ✅ Full Support | `adapter: lora` | အသုံးအများဆုံး PEFT method |
| **QLoRA** | ✅ Full Support | `adapter: qlora` | GPU memory နည်းတဲ့သူအတွက် အကောင်းဆုံး |
| **DoRA** | ✅ Support | `peft_use_dora: true` | LoRA ထက် performance ပိုကောင်း |
| **NEFTune** | ✅ Support | `neftune_noise_alpha: 5` | LoRA/QLoRA နဲ့ တွဲသုံးလို့ရ |
| **ReLoRA** | ✅ Support | `relora_steps: 200` | Stacked LoRA training |
| **LoftQ** | ✅ Support | `peft_use_loftq: true` | Better quantization-aware init |
| **Prompt Tuning** | ⚠️ Limited | PEFT library ကနေ | Axolotl direct config နည်း |
| **Prefix Tuning** | ⚠️ Limited | PEFT library ကနေ | Axolotl direct config နည်း |
| **IA³** | ⚠️ Limited | PEFT library ကနေ | Axolotl direct config နည်း |
| **Adapter Layers** | ❌ Not Direct | - | Axolotl မှာ native support မရှိ |

---

## Axolotl PEFT Config Template (Recommended)

### LoRA (Standard GPU - 24GB+)

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj

sequence_len: 4096
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_torch
lr_scheduler: cosine
bf16: auto
neftune_noise_alpha: 5
```

### QLoRA (Low VRAM GPU - 8GB+)

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
adapter: qlora
load_in_4bit: true
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj

sequence_len: 2048
gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 3
learning_rate: 2e-4
optimizer: paged_adamw_8bit
lr_scheduler: cosine
bf16: auto
neftune_noise_alpha: 5
```

---

## PEFT Method ရွေးချယ်ရာမှာ လမ်းညွှန်

```
GPU VRAM ဘယ်လောက်ရှိလဲ?
│
├── 8GB အောက် ──→ QLoRA (4-bit) + low batch size
│
├── 8-16GB ──→ QLoRA (4-bit) recommended
│
├── 16-24GB ──→ LoRA (fp16/bf16) or QLoRA
│
├── 24-48GB ──→ LoRA + DoRA ကို စမ်းကြည့်
│
└── 48GB+ ──→ LoRA / Full Fine-Tuning ရွေးချယ်နိုင်
```

### LoRA vs QLoRA ရွေးချယ်ခြင်း

- **QLoRA** ကို သုံးပါ → GPU memory **အကန့်အသတ်** ရှိရင်
- **LoRA** ကို သုံးပါ → GPU memory **လုံလောက်** ပြီး quality ကို ဦးစားပေးချင်ရင်
- **DoRA** ကို ထပ်ထည့်ပါ → LoRA ထက် **performance ပိုလိုချင်** ရင် (memory အနည်းငယ် ပိုကုန်)
- **NEFTune** ကို အမြဲတွဲသုံးပါ → **generalization ကောင်းစေ** တယ်
- **ReLoRA** ကို သုံးပါ → **pre-training** or **continued pre-training** လုပ်ချင်ရင်

---

## LoRA Target Modules - Model Architecture အလိုက်

| Model | Common Target Modules |
|---|---|
| **LLaMA / Llama 2/3** | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Mistral** | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **GPT-NeoX** | `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h` |
| **Falcon** | `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h` |
| **Phi-2/3** | `q_proj`, `v_proj`, `k_proj`, `dense`, `fc1`, `fc2` |
| **Gemma** | `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

> 💡 **Tip:** `lora_target_linear: true` ကို သုံးရင် **linear layer အားလုံးကို** auto-target လုပ်ပေးပြီး model-specific modules ကို manually specify မလိုတော့ပါ။
