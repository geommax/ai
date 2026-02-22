# AI Model အမျိုးအစားများ နှင့် Suffix ကြည့်ပြီး ခွဲခြားနည်း

## Model အမျိုးအစားများ (Categories)

AI Models တွေကို အဓိက **အမျိုးအစား ၅ မျိုး** ခွဲနိုင်ပါတယ်။

| Category | Description | Examples |
|---|---|---|
| **LLM** (Large Language Model) | Text generation, reasoning, chat | LLaMA, Mistral, GPT |
| **VLM** (Vision-Language Model) | Image + Text understanding | LLaVA, Qwen-VL, InternVL |
| **Speech/Audio Model** | Speech recognition, TTS, audio understanding | Whisper, Bark, SeamlessM4T |
| **Vision Model** | Image classification, detection, segmentation | ViT, DINO, SAM |
| **Multimodal Model** | Multiple modalities (text + image + audio + video) | GPT-4o, Gemini, Any-to-Any |

---

## Model Suffix ကြည့်ပြီး Type ခွဲခြားနည်း

Model name ရဲ့ **suffix** (နောက်ဆက်) ကို ကြည့်ရုံနဲ့ model ရဲ့ training stage, purpose, quantization level ကို ခွဲခြားနိုင်ပါတယ်။

---

### 🏷️ 1. Training Stage Suffixes

Model ကို ဘယ် training stage အထိ လုပ်ထားလဲ ပြတဲ့ suffixes:

| Suffix | အဓိပ္ပာယ် | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| **(suffix မရှိ)** | Base / Pretrained Model | Raw pretrained model, next-token prediction သာ train ထား | `meta-llama/Llama-3.1-8B` |
| `-base` | Base Model | Pretrained model ဖြစ်ကြောင်း explicitly ပြထား | `Qwen/Qwen2.5-7B-Base` |
| `-Instruct` | Instruction-tuned | Instruction following အတွက် fine-tune ထားပြီး | `meta-llama/Llama-3.1-8B-Instruct` |
| `-Chat` | Chat-optimized | Multi-turn chat conversation အတွက် optimize ထား | `Qwen/Qwen2-7B-Chat` |
| `-it` | Instruction-tuned | `-Instruct` ရဲ့ အတိုကောက် (Google models) | `google/gemma-2-9b-it` |
| `-hf` | Hugging Face format | HF Transformers library နဲ့ compatible format | `tiiuae/falcon-7b-hf` |

---

### 🏷️ 2. Alignment / Safety Suffixes

Model ကို alignment / safety training ဘယ်လောက်ထိ လုပ်ထားလဲ ပြတဲ့ suffixes:

| Suffix | အဓိပ္ပာယ် | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| `-RLHF` | Reinforcement Learning from Human Feedback | Human preference data နဲ့ align ထား | `Llama-2-7b-chat-RLHF` |
| `-DPO` | Direct Preference Optimization | RLHF ရဲ့ simpler alternative နဲ့ align ထား | `NousResearch/Hermes-2-Pro-Llama-3-8B-DPO` |
| `-ORPO` | Odds Ratio Preference Optimization | SFT + alignment ကို single stage မှာ လုပ် | `mlabonne/OrpoLlama-3-8B` |
| `-KTO` | Kahneman-Tversky Optimization | Unpaired preference data နဲ့ align ထား | `model-kto` |
| `-PPO` | Proximal Policy Optimization | Classic RL algorithm နဲ့ align ထား | `model-ppo` |
| `-SimPO` | Simple Preference Optimization | Reference-free preference optimization | `model-simpo` |

---

### 🏷️ 3. Fine-Tuning Method Suffixes

ဘယ် fine-tuning method သုံးထားလဲ ပြတဲ့ suffixes:

| Suffix | အဓိပ္ပာယ် | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| `-SFT` | Supervised Fine-Tuning | Labeled data နဲ့ supervised train ထား | `model-7B-SFT` |
| `-LoRA` | LoRA adapter | LoRA fine-tune ထားတဲ့ adapter weights | `model-7B-LoRA` |
| `-QLoRA` | Quantized LoRA | 4-bit quantized + LoRA | `model-7B-QLoRA` |
| `-merged` | Merged adapter | LoRA adapter ကို base model ထဲ merge ထားပြီး | `model-7B-LoRA-merged` |
| `-FT` | Fine-Tuned | Full fine-tuning လုပ်ထားတဲ့ model | `model-7B-FT` |
| `-adapter` | Adapter weights only | Adapter weights သီးသန့် (base model မပါ) | `model-7B-adapter` |

---

### 🏷️ 4. Quantization Suffixes

Model ရဲ့ precision / quantization level ပြတဲ့ suffixes:

| Suffix | အဓိပ္ပာယ် | Size Reduction | Quality | ဥပမာ |
|---|---|---|---|---|
| `-fp32` | 32-bit floating point | Baseline | Highest | `model-fp32` |
| `-fp16` | 16-bit floating point | 2× smaller | Near-original | `model-fp16` |
| `-bf16` | Brain floating point 16 | 2× smaller | Near-original (better range) | `model-bf16` |
| `-int8` | 8-bit integer | 4× smaller | Slight loss | `model-int8` |
| `-int4` | 4-bit integer | 8× smaller | Moderate loss | `model-int4` |
| `-GPTQ` | GPTQ quantization | 4-8× smaller | Good (post-training quant) | `TheBloke/Llama-2-7B-GPTQ` |
| `-AWQ` | Activation-aware Weight Quantization | 4-8× smaller | Better than GPTQ | `TheBloke/Llama-2-7B-AWQ` |
| `-GGUF` | GGML Universal Format | Variable | llama.cpp compatible | `model-Q4_K_M.gguf` |
| `-EXL2` | ExLlamaV2 format | Variable | ExLlamaV2 compatible | `model-EXL2` |
| `-bnb` / `-4bit` | BitsAndBytes quantization | 4-8× smaller | Runtime quantization | `model-bnb-4bit` |

#### GGUF Quantization Levels

GGUF files တွေမှာ quantization level ကို filename မှာ ပြပါတယ်:

| Quant Type | Bits | Quality | Size (7B Model) | အသုံးပြုသင့်တဲ့ အခြေအနေ |
|---|---|---|---|---|
| `Q2_K` | 2-bit | Low | ~2.8 GB | Memory အရမ်းနည်းတဲ့အခါ |
| `Q3_K_S/M/L` | 3-bit | Fair | ~3.2-3.8 GB | Mobile / Edge devices |
| `Q4_0` | 4-bit | Good | ~3.8 GB | Standard quantization |
| `Q4_K_S/M` | 4-bit | Better | ~3.8-4.1 GB | **အသုံးအများဆုံး (recommended)** |
| `Q5_0` | 5-bit | Very Good | ~4.6 GB | Quality ဦးစားပေးရင် |
| `Q5_K_S/M` | 5-bit | Very Good+ | ~4.6-4.8 GB | Quality + reasonable size |
| `Q6_K` | 6-bit | Excellent | ~5.5 GB | Near-original quality |
| `Q8_0` | 8-bit | Near-perfect | ~7.2 GB | Maximum quality quantization |
| `F16` | 16-bit | Original | ~14 GB | Full precision |

---

### 🏷️ 5. Model Size Suffixes

Model parameter count ပြတဲ့ suffixes:

| Suffix | အဓိပ္ပာယ် | ဥပမာ |
|---|---|---|
| `-1B`, `-3B`, `-7B`, `-8B` | Billion parameters | `Llama-3.1-8B` |
| `-0.5B`, `-1.5B` | Sub-billion / small models | `Qwen2.5-0.5B` |
| `-13B`, `-14B` | Medium models | `Llama-2-13B` |
| `-30B`, `-34B`, `-35B` | Large models | `Yi-34B` |
| `-70B`, `-72B` | Very large models | `Llama-3.1-70B` |
| `-405B` | Ultra-large models | `Llama-3.1-405B` |
| `-MoE` | Mixture of Experts | Active parameters < total | `Mixtral-8x7B` |
| `-A14B` | Active parameters (MoE) | `14B` active out of total | `Qwen2.5-A14B` |

---

### 🏷️ 6. Vision / Multimodal Suffixes

| Suffix | အဓိပ္ပာယ် | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| `-VL` | Vision-Language | Image + Text understanding | `Qwen/Qwen2-VL-7B-Instruct` |
| `-Vision` | Vision capable | Image understanding | `model-Vision` |
| `-LLaVA` | LLaVA architecture | Visual instruction tuning | `liuhaotian/llava-v1.6-mistral-7b` |
| `-MM` | Multimodal | Multiple modalities support | `model-MM` |
| `-Omni` | Omni-modal | Text + Image + Audio + Video | `model-Omni` |

---

### 🏷️ 7. Special Purpose Suffixes

| Suffix | အဓိပ္ပာယ် | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| `-Coder` / `-Code` | Code generation | Code generation အတွက် specialized | `Qwen2.5-Coder-7B` |
| `-Math` | Mathematics | Math reasoning အတွက် | `Qwen2.5-Math-7B` |
| `-Med` / `-Medical` | Medical domain | Medical knowledge | `model-Med` |
| `-Legal` | Legal domain | Legal text processing | `model-Legal` |
| `-Finance` | Finance domain | Financial analysis | `model-Finance` |
| `-RP` | Roleplay | Roleplay/Character chat | `model-RP` |
| `-Uncensored` | Uncensored | Safety filters ဖြုတ်ထား | `model-Uncensored` |
| `-Abliterated` | Abliterated | Refusal behavior ဖယ်ရှားထား | `model-abliterated` |
| `-Turbo` | Turbo/Fast | Speed optimized | `model-Turbo` |
| `-Mini` / `-Nano` | Small variant | Smaller, faster version | `Phi-3-mini` |
| `-Pro` / `-Plus` | Enhanced variant | Better performance version | `Gemma-2-Pro` |
| `-Preview` | Preview/Beta | Testing release | `model-Preview` |
| `-Long` | Long context | Extended context window | `model-Long` |

---

### 🏷️ 8. Version Suffixes

| Suffix | အဓိပ္ပာယ် | ဥပမာ |
|---|---|---|
| `-v1`, `-v1.5`, `-v2` | Version number | `llava-v1.6-mistral-7b` |
| `.1`, `.2`, `.3` | Sub-version (in model family) | `Llama-3.1`, `Qwen2.5` |
| `-2025xxxx` | Date-based version | `model-20250201` |

---

## Model Name Anatomy - ဥပမာနဲ့ ခွဲခြမ်းစိတ်ဖြာခြင်း

### ဥပမာ ၁: `meta-llama/Llama-3.1-8B-Instruct`

```
meta-llama  /  Llama-3.1  -  8B      -  Instruct
─────────     ─────────     ──         ────────
Organization   Model v3.1   8 Billion   Instruction-tuned
                             params
```

### ဥပမာ ၂: `TheBloke/Llama-2-13B-Chat-GPTQ`

```
TheBloke  /  Llama-2  -  13B       -  Chat    -  GPTQ
────────     ───────     ───          ────       ────
Quantizer    Model v2    13 Billion   Chat-opt   GPTQ quantized
```

### ဥပမာ ၃: `Qwen/Qwen2.5-72B-Instruct-AWQ`

```
Qwen  /  Qwen2.5  -  72B        -  Instruct        -  AWQ
────     ───────     ───            ────────           ───
Org      v2.5       72 Billion     Instruction-tuned   AWQ quantized
```

### ဥပမာ ၄: `NousResearch/Hermes-2-Pro-Llama-3-8B-DPO`

```
NousResearch / Hermes-2-Pro - Llama-3 - 8B  - DPO
────────────   ────────────   ───────   ──    ───
Organization   Fine-tune       Base     Size  Alignment
               name            model          method
```

### ဥပမာ ၅: `liuhaotian/llava-v1.6-mistral-7b-hf`

```
liuhaotian / llava-v1.6 - mistral - 7b - hf
──────────   ──────────   ───────   ──   ──
Org          VLM v1.6     Base LLM  Size HuggingFace
                          backbone       format
```

---

## LLM Models - အသေးစိတ် အမျိုးအစားများ

### Open-Source LLM Families

| Model Family | Organization | Sizes | License | ထူးခြားချက် |
|---|---|---|---|---|
| **LLaMA 3.1 / 3.2 / 3.3** | Meta | 1B, 3B, 8B, 70B, 405B | Llama License | အကျယ်ပြန့်ဆုံး open-source LLM |
| **Qwen 2.5** | Alibaba | 0.5B - 72B | Apache 2.0 / Qwen | Multilingual, Code, Math variants |
| **Mistral / Mixtral** | Mistral AI | 7B, 8x7B, 8x22B | Apache 2.0 | MoE architecture, efficient |
| **Gemma 2** | Google | 2B, 9B, 27B | Gemma License | Lightweight, efficient |
| **Phi-3 / Phi-4** | Microsoft | 3.8B, 7B, 14B | MIT | Small but powerful (SLM) |
| **Yi** | 01.AI | 6B, 9B, 34B | Apache 2.0 | Strong bilingual (EN/ZH) |
| **DeepSeek V3** | DeepSeek | 671B (MoE) | MIT | MoE, cost-efficient training |
| **Command R+** | Cohere | 35B, 104B | CC-BY-NC | RAG optimized |
| **OLMo** | AI2 | 1B, 7B, 13B | Apache 2.0 | Fully open (data + code + weights) |
| **Falcon** | TII | 7B, 40B, 180B | Apache 2.0 | Early open-source pioneer |
| **InternLM 2.5** | Shanghai AI Lab | 7B, 20B | Apache 2.0 | Strong reasoning |
| **StarCoder 2** | BigCode | 3B, 7B, 15B | BigCode OpenRAIL-M | Code generation specialized |

---

## Vision-Language Models (VLMs)

### VLM Architecture Types

```
┌─────────────────────────────────────────────────┐
│                   VLM Architecture               │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │  Vision   │───→│ Connector │───→│    LLM    │  │
│  │ Encoder   │    │ (Bridge)  │    │ Backbone  │  │
│  │ (ViT etc) │    │           │    │           │  │
│  └──────────┘    └──────────┘    └───────────┘  │
│                                                   │
│  Image input      Feature        Text output      │
│                   alignment                        │
└─────────────────────────────────────────────────┘
```

### Open-Source VLMs

| Model | Organization | LLM Backbone | Vision Encoder | Sizes | ထူးခြားချက် |
|---|---|---|---|---|---|
| **LLaVA 1.6 (NeXT)** | LLaVA Team | Mistral/Vicuna/LLaMA | CLIP ViT-L | 7B, 13B, 34B | Pioneer open VLM |
| **Qwen2-VL** | Alibaba | Qwen2 | ViT (native) | 2B, 7B, 72B | Video understanding ပါ |
| **InternVL 2.5** | Shanghai AI Lab | InternLM2 | InternViT-6B | 1B - 78B | Strong OCR + document |
| **Llama 3.2 Vision** | Meta | Llama 3.2 | ViT | 11B, 90B | Official Meta VLM |
| **Phi-3-Vision** | Microsoft | Phi-3 | CLIP ViT | 4.2B | Small but capable |
| **DeepSeek-VL 2** | DeepSeek | DeepSeek MoE | SigLIP | 4.5B, 28B | MoE VLM |
| **CogVLM2** | Zhipu AI | LLaMA2/ChatGLM | EVA2-CLIP | 19B | High-res understanding |
| **Idefics3** | Hugging Face | Llama 3.1 | SigLIP | 8B | Native HF integration |
| **MiniCPM-V** | OpenBMB | MiniCPM | SigLIP | 3B, 8B | Mobile-friendly VLM |
| **Pixtral** | Mistral AI | Mistral | Custom ViT | 12B | Mistral's VLM |

---

## Speech / Audio Models

### Speech Model Types

| Type | Description | Models |
|---|---|---|
| **ASR** (Automatic Speech Recognition) | Speech → Text | Whisper, wav2vec2, Conformer |
| **TTS** (Text-to-Speech) | Text → Speech | VITS, Bark, XTTS, F5-TTS |
| **Voice Cloning** | Voice replication | XTTS, OpenVoice, RVC |
| **Speech Translation** | Speech → Translated text | SeamlessM4T, Whisper |
| **Audio Understanding** | Audio analysis + QA | Qwen2-Audio, SALMONN |
| **Music Generation** | Text → Music | MusicGen, Stable Audio |
| **Sound Effect** | Text → Sound effects | AudioGen, Make-An-Audio |

### Key Speech/Audio Models

| Model | Organization | Task | Sizes | ထူးခြားချက် |
|---|---|---|---|---|
| **Whisper** | OpenAI | ASR + Translation | tiny - large-v3 | Multilingual ASR, 99 languages |
| **Seamless M4T v2** | Meta | Speech ↔ Text Translation | 2.3B | Multimodal translation |
| **Bark** | Suno | TTS | 1.3B | Multilingual, music, sound effects |
| **XTTS v2** | Coqui | TTS + Voice Clone | ~1B | 17 languages, voice cloning |
| **Wav2Vec 2.0** | Meta | ASR | 300M | Self-supervised speech |
| **Qwen2-Audio** | Alibaba | Audio Understanding | 7B | Audio QA, multi-type audio |
| **VALL-E X** | Microsoft | TTS + Clone | - | Zero-shot voice synthesis |
| **F5-TTS** | Community | TTS | ~300M | Fast, high quality |
| **Parler-TTS** | Hugging Face | TTS | 600M, 2.3B | Describable TTS |

---

## Vision Models (Image-only)

### Vision Model Types

| Type | Description | Models |
|---|---|---|
| **Classification** | Image → Label | ViT, ConvNeXt, EfficientNet |
| **Object Detection** | Image → Bounding boxes | YOLO, DETR, RT-DETR |
| **Segmentation** | Image → Pixel-level masks | SAM, Mask2Former |
| **Image Generation** | Text → Image | Stable Diffusion, FLUX, DALL-E |
| **Image Editing** | Image modification | InstructPix2Pix |
| **Super Resolution** | Low-res → High-res | Real-ESRGAN, SwinIR |
| **Depth Estimation** | Image → Depth map | Depth Anything, MiDaS |
| **OCR** | Image → Text extraction | TrOCR, PaddleOCR, EasyOCR |

### Key Vision Models

| Model | Organization | Task | ထူးခြားချက် |
|---|---|---|---|
| **Stable Diffusion XL/3** | Stability AI | Image Generation | Open-source image gen |
| **FLUX** | Black Forest Labs | Image Generation | SD successor, high quality |
| **SAM 2** | Meta | Segmentation | Segment Anything (image + video) |
| **YOLO v11** | Ultralytics | Object Detection | Real-time detection |
| **DINOv2** | Meta | Visual Features | Self-supervised vision backbone |
| **ViT** | Google | Classification | Vision Transformer |
| **Depth Anything v2** | HKU | Depth Estimation | Monocular depth |
| **RT-DETR** | Baidu | Object Detection | Real-time DETR |

---

## Embedding Models

| Model | Organization | Dimensions | Max Tokens | ထူးခြားချက် |
|---|---|---|---|---|
| **BGE-M3** | BAAI | 1024 | 8192 | Multilingual, multi-granularity |
| **E5-Mistral-7B** | Microsoft | 4096 | 32768 | LLM-based embedding |
| **GTE-Qwen2** | Alibaba | 768-1536 | 8192 | Strong multilingual |
| **Nomic-Embed-Text** | Nomic AI | 768 | 8192 | Open-source, efficient |
| **jina-embeddings-v3** | Jina AI | 1024 | 8192 | Task-specific embeddings |
| **Snowflake-Arctic-Embed** | Snowflake | 768-1024 | 512 | Retrieval optimized |

---

## Axolotl မှာ Train လို့ရမယ့် Model Types

### ✅ Fully Supported (Direct Training)

Axolotl ဟာ **Hugging Face Transformers** library ပေါ်မှာ အခြေခံထားတဲ့အတွက်, Transformers compatible ဖြစ်တဲ့ **causal LLM** (decoder-only) models အားလုံးကို train လို့ရပါတယ်။

| Model Architecture | Models | Axolotl Support Level |
|---|---|---|
| **LlamaForCausalLM** | LLaMA 2/3/3.1/3.2/3.3, CodeLlama, Vicuna, Yi | ✅ Full (First-class) |
| **MistralForCausalLM** | Mistral 7B, Zephyr | ✅ Full |
| **MixtralForCausalLM** | Mixtral 8x7B, 8x22B | ✅ Full (MoE support) |
| **Qwen2ForCausalLM** | Qwen2, Qwen2.5 series | ✅ Full |
| **GemmaForCausalLM** | Gemma, Gemma 2 | ✅ Full |
| **Phi3ForCausalLM** | Phi-3, Phi-3.5 | ✅ Full |
| **GPTNeoXForCausalLM** | Pythia, RedPajama | ✅ Full |
| **FalconForCausalLM** | Falcon 7B, 40B, 180B | ✅ Full |
| **GPT2LMHeadModel** | GPT-2 series | ✅ Full |
| **MPTForCausalLM** | MPT-7B, MPT-30B | ✅ Full |
| **StableLMForCausalLM** | StableLM 2 | ✅ Full |
| **InternLM2ForCausalLM** | InternLM 2, 2.5 | ✅ Full |
| **DeepseekV2ForCausalLM** | DeepSeek V2, V3 | ✅ Supported (MoE) |
| **CohereForCausalLM** | Command R/R+ | ✅ Supported |
| **OlmoForCausalLM** | OLMo | ✅ Supported |
| **StarCoder2ForCausalLM** | StarCoder 2 | ✅ Supported |
| **Starcoder2ForCausalLM** | StarCoder 2 | ✅ Supported |

### ⚠️ Partially Supported (VLMs - Multimodal)

Axolotl မှာ Vision-Language Models တချို့ကို train လို့ ရပါတယ် (experimental/growing support):

| Model | Architecture | Support Status | မှတ်ချက် |
|---|---|---|---|
| **LLaVA 1.5/1.6** | LlavaForConditionalGeneration | ⚠️ Supported | Visual instruction tuning |
| **Qwen2-VL** | Qwen2VLForConditionalGeneration | ⚠️ Experimental | Vision-Language training |
| **Pixtral** | PixtralForConditionalGeneration | ⚠️ Experimental | Mistral VLM |
| **Llama 3.2 Vision** | MllamaForConditionalGeneration | ⚠️ Experimental | Meta VLM |

#### Axolotl VLM Training Config Example

```yaml
base_model: llava-hf/llava-v1.6-mistral-7b-hf
model_type: LlavaForConditionalGeneration
adapter: lora
lora_r: 16
lora_alpha: 32

datasets:
  - path: dataset_path
    type: llava
```

### ❌ Not Supported (Direct Training)

| Model Type | Reason | Alternative |
|---|---|---|
| **Encoder-only** (BERT, RoBERTa) | Axolotl is for causal/autoregressive LMs | HF Trainer / custom script |
| **Encoder-Decoder** (T5, BART, mBART) | Architecture mismatch | HF Seq2SeqTrainer |
| **Speech Models** (Whisper, Wav2Vec) | Different modality | HF Trainer + custom data |
| **Diffusion Models** (SD, FLUX) | Completely different training paradigm | Kohya, diffusers library |
| **Embedding Models** (BGE, E5) | Different training objective | Sentence-transformers |
| **Vision-only** (ViT, YOLO, SAM) | Not language models | Timm, Ultralytics |
| **GGUF / GGML models** | Quantized inference format | Convert to HF format first |
| **GPTQ models** | Post-training quantized | အခက်အခဲရှိ (QLoRA ကို base model နဲ့ သုံးပါ) |
| **AWQ models** | Post-training quantized | QLoRA with base model instead |
| **EXL2 models** | ExLlamaV2 inference format | Not trainable |

---

## Axolotl Model Selection Guide

### Training လုပ်မယ့် Model ရွေးချယ်ခြင်း

```
ဘာ Task အတွက်လဲ?
│
├── 💬 General Chat / Instruction Following
│   ├── GPU 8-16GB  → Qwen2.5-3B / Phi-3-mini-4k / Gemma-2-2B
│   ├── GPU 24GB    → Llama-3.1-8B / Mistral-7B / Qwen2.5-7B
│   ├── GPU 48GB    → Qwen2.5-14B / InternLM2.5-20B
│   └── GPU 80GB+   → Llama-3.1-70B / Qwen2.5-72B
│
├── 💻 Code Generation
│   ├── Small      → Qwen2.5-Coder-3B / StarCoder2-3B
│   ├── Medium     → Qwen2.5-Coder-7B / CodeLlama-7B
│   └── Large      → Qwen2.5-Coder-14B+ / DeepSeek-Coder-V2
│
├── 🔢 Math / Reasoning
│   ├── Small      → Qwen2.5-Math-1.5B
│   ├── Medium     → Qwen2.5-Math-7B
│   └── Large      → Qwen2.5-Math-72B
│
├── 🖼️ Vision + Language (VLM)
│   ├── Small      → MiniCPM-V-2.5 / Phi-3-Vision
│   ├── Medium     → LLaVA-v1.6-7B / Qwen2-VL-7B
│   └── Large      → InternVL2.5-78B / Qwen2-VL-72B
│
└── 🌍 Multilingual
    ├── Small      → Qwen2.5-3B
    ├── Medium     → Qwen2.5-7B / Llama-3.1-8B
    └── Large      → Qwen2.5-72B
```

### Base Model vs Instruct Model ရွေးချယ်ခြင်း

| Scenario | ရွေးချယ်ရမယ့် Model | အကြောင်းပြချက် |
|---|---|---|
| Custom chat style ဖန်တီးချင်ရင် | **Base model** | Instruct training ရဲ့ bias မရှိ |
| Domain-specific knowledge ထည့်ချင်ရင် | **Base model** | Clean continued pretraining |
| Existing chat format ကို ပြင်ချင်ရင် | **Instruct model** | Chat capability ရှိပြီးသား |
| Task-specific fine-tune ချင်ရင် | **Instruct model** | Instruction following ability ရှိပြီးသား |
| DPO/RLHF alignment လုပ်ချင်ရင် | **SFT model** | SFT ပြီးသား model ကို align |

---

## Model Format Compatibility

### Axolotl Compatible Formats

| Format | Trainable? | Load Method | မှတ်ချက် |
|---|---|---|---|
| **HuggingFace safetensors** | ✅ Yes | `base_model: org/model` | **Recommended format** |
| **HuggingFace bin (pytorch)** | ✅ Yes | `base_model: org/model` | Legacy format |
| **BitsAndBytes 4-bit** | ✅ Yes (QLoRA) | `load_in_4bit: true` | Runtime quantization |
| **BitsAndBytes 8-bit** | ✅ Yes | `load_in_8bit: true` | Runtime quantization |
| **GPTQ** | ⚠️ Limited | `gptq: true` | Training quality concerns |
| **AWQ** | ❌ No | - | Inference only format |
| **GGUF** | ❌ No | - | llama.cpp format |
| **EXL2** | ❌ No | - | ExLlamaV2 format |
| **ONNX** | ❌ No | - | Inference only format |
| **TensorRT** | ❌ No | - | NVIDIA inference only |

### Axolotl Inference/Serving Compatible Tools

Fine-tune ပြီးတဲ့ model ကို serve/deploy လုပ်ဖို့:

| Tool | Format Required | Speed | ထူးခြားချက် |
|---|---|---|---|
| **vLLM** | HF / AWQ / GPTQ | Fast | Production serving, batching |
| **llama.cpp** | GGUF | Medium | CPU/Metal inference |
| **TGI** | HF / AWQ / GPTQ | Fast | HuggingFace serving |
| **Ollama** | GGUF | Easy | Local deployment |
| **ExLlamaV2** | EXL2 / GPTQ | Very Fast | Consumer GPU optimized |
| **SGLang** | HF | Very Fast | Structured generation |

> 💡 **Tip:** Axolotl နဲ့ train ပြီးတဲ့ LoRA adapter ကို base model ထဲ merge လုပ်ပြီး →  ကြိုက်တဲ့ format (GGUF, AWQ, GPTQ, EXL2) ကို convert လုပ်ပြီး deploy လုပ်နိုင်ပါတယ်။

```bash
# Merge LoRA adapter into base model
python -m axolotl.cli.merge_lora your_config.yml --lora_model_dir="./outputs"

# Convert to GGUF for Ollama/llama.cpp
python convert_hf_to_gguf.py ./merged_model --outtype q4_k_m
```
