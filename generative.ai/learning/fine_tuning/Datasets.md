# Dataset Format Preparation - Training Process Type အလိုက်

## Overview

Fine-tuning လုပ်တဲ့အခါ **Training Process Type** ပေါ်မူတည်ပြီး dataset format ပြောင်းပေးရပါတယ်။ Model type တူတူပဲ ဖြစ်ပေမယ့် training objective ကွာရင် dataset format ကွာပါတယ်။

```
Dataset Format ကို ဘာက ဆုံးဖြတ်လဲ?
│
├── 1. Training Process Type (CPT, SFT, DPO, ORPO...)
├── 2. Model Type (LLM, VLM, Speech...)
└── 3. Axolotl Dataset Type Configuration
```

---

## Training Process Types နှင့် Dataset Formats

### 📊 Training Process Type Summary

| Training Type | ရည်ရွယ်ချက် | Dataset Format | Axolotl Support |
|---|---|---|---|
| **CPT** (Continued Pre-Training) | Domain knowledge ထည့်ခြင်း | Raw text / Completion | ✅ |
| **SFT** (Supervised Fine-Tuning) | Instruction following သင်ခြင်း | Instruction-Response pairs | ✅ |
| **DPO** (Direct Preference Optimization) | Human preference alignment | Chosen/Rejected pairs | ✅ |
| **ORPO** (Odds Ratio Preference Optimization) | SFT + Alignment တစ်ခါတည်း | Chosen/Rejected pairs | ✅ |
| **KTO** (Kahneman-Tversky Optimization) | Unpaired preference alignment | Completion + Label (true/false) | ✅ |
| **RLHF** (Reward Model Training) | Reward model train ခြင်း | Chosen/Rejected pairs | ⚠️ Limited |
| **VLM SFT** | Vision-Language fine-tuning | Image + Conversation | ✅ Experimental |

---

## 1. Continued Pre-Training (CPT) - Dataset Format

### ရည်ရွယ်ချက်

Model ကို **domain-specific knowledge** (ဥပမာ: Medical, Legal, Finance, Myanmar Language) ထပ်ထည့်သင်ပေးဖို့ ဖြစ်ပါတယ်။ Instruction format မလိုဘဲ **raw text** ပဲ လိုအပ်ပါတယ်။

### Dataset Format

#### Format A: Plain Text (Completion)

```json
{"text": "ဤသည်မှာ ပထမ document ၏ အပြည့်အစုံ text ဖြစ်ပါသည်။ Document တစ်ခုလုံးကို single text field မှာ ထည့်ပါ။"}
{"text": "ဒုတိယ document ၏ text ဖြစ်ပါသည်။ ဒီ format မှာ instruction/response ခွဲစရာ မလိုပါ။"}
{"text": "The quick brown fox jumps over the lazy dog. This is a sample document for continued pretraining."}
```

#### Format B: Pretraining Corpus (Large Text Blocks)

```json
{"text": "Chapter 1: Introduction to Machine Learning\n\nMachine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms identify patterns in data and make decisions with minimal human intervention.\n\n## Types of Machine Learning\n\n### Supervised Learning\nSupervised learning involves training a model on labeled data..."}
```

### Axolotl Config (CPT)

```yaml
base_model: meta-llama/Llama-3.1-8B

# CPT specific settings
datasets:
  - path: ./data/pretrain_corpus.jsonl
    type: completion           # ← CPT အတွက် completion type သုံး
    field: text                # ← text field name

# CPT Training Settings
learning_rate: 2e-5            # SFT ထက် learning rate နိမ့်သင့်
num_epochs: 1                  # CPT မှာ 1-2 epochs လောက်ပဲ
sequence_len: 4096
sample_packing: true           # Short texts တွေကို pack ပြီး efficiency တိုးမြင့်
```

### CPT Dataset Preparation Tips

```
⚠️ CPT အတွက် သတိထားရမယ့် အချက်များ:

1. Data Quality: Noisy/duplicate data ဖယ်ရှားပါ
2. Data Size: Model size ရဲ့ 1-10% tokens လောက် လိုအပ်
   - 7B model → 1B-10B tokens
   - 70B model → 10B-100B tokens
3. Learning Rate: SFT (2e-4) ထက် နိမ့်ပါ (1e-5 ~ 5e-5)
4. Epochs: 1-2 epochs ပဲ (overfit မဖြစ်အောင်)
5. No special tokens: Chat template, instruction tags မလို
```

---

## 2. Supervised Fine-Tuning (SFT) - Dataset Format

### ရည်ရွယ်ချက်

Model ကို **instruction following**, **chat**, **task-specific** abilities သင်ပေးဖို့ ဖြစ်ပါတယ်။

### Axolotl SFT Dataset Types

Axolotl မှာ SFT dataset format **အမျိုးမျိုး** support လုပ်ပါတယ်:

---

### 📝 Type 1: `alpaca` Format

**Single-turn instruction-response** format ဖြစ်ပြီး အရိုးရှင်းဆုံး SFT format ဖြစ်ပါတယ်။

#### Dataset Structure

```json
{
  "instruction": "Translate the following English text to Myanmar.",
  "input": "Hello, how are you?",
  "output": "မင်္ဂလာပါ၊ နေကောင်းလား?"
}
```

```json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "Artificial intelligence (AI) has transformed numerous industries over the past decade. From healthcare to finance, AI systems are being deployed to automate tasks, analyze data, and make predictions. The technology continues to evolve rapidly, with new breakthroughs in natural language processing, computer vision, and robotics emerging regularly.",
  "output": "AI has significantly impacted multiple industries in recent years. It is being used for automation, data analysis, and predictions across sectors like healthcare and finance. The field continues to advance rapidly with breakthroughs in NLP, computer vision, and robotics."
}
```

#### Input Field မလိုတဲ့ ဥပမာ

```json
{
  "instruction": "What is the capital of Myanmar?",
  "input": "",
  "output": "Myanmar (Burma) ရဲ့ မြို့တော်က နေပြည်တော် (Naypyidaw) ဖြစ်ပါတယ်။"
}
```

#### Axolotl Config

```yaml
datasets:
  - path: ./data/my_alpaca_data.jsonl
    type: alpaca

# Custom field names သုံးချင်ရင်
datasets:
  - path: ./data/custom_data.jsonl
    type:
      system_prompt: "You are a helpful assistant."
      field_instruction: question       # instruction field name ပြောင်း
      field_input: context              # input field name ပြောင်း
      field_output: answer              # output field name ပြောင်း
      format: "[INST] {instruction} {input} [/INST]"
      no_input_format: "[INST] {instruction} [/INST]"
```

---

### 💬 Type 2: `sharegpt` Format

**Multi-turn conversation** format ဖြစ်ပြီး chat models train ဖို့ အသုံးအများဆုံး format ဖြစ်ပါတယ်။

#### Dataset Structure

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful Myanmar language assistant."},
    {"from": "human", "value": "မင်္ဂလာပါ။ Python programming ကို ဘယ်လို စလေ့လာရမလဲ?"},
    {"from": "gpt", "value": "မင်္ဂလာပါ! Python ကို စလေ့လာဖို့ အဆင့်တွေကို ပြောပြပါမယ်:\n\n1. **Python Install လုပ်ပါ** - python.org ကနေ download လုပ်ပါ\n2. **Basic Syntax** - variables, data types, operators\n3. **Control Flow** - if/else, for/while loops\n4. **Functions** - def keyword နဲ့ function ရေးပါ\n5. **Practice** - LeetCode, HackerRank မှာ လေ့ကျင့်ပါ"},
    {"from": "human", "value": "IDE ဘာသုံးရင် ကောင်းမလဲ?"},
    {"from": "gpt", "value": "Python အတွက် IDE recommendation:\n\n1. **VS Code** - Lightweight, extensions များ, အကြံပြုချင်ဆုံး\n2. **PyCharm** - Full-featured Python IDE\n3. **Jupyter Notebook** - Data Science/ML အတွက်\n\nအစပိုင်း learner ဆိုရင် **VS Code** ကို recommend ပါတယ်။"}
  ]
}
```

#### Multi-turn Conversation (ရှည်တဲ့ ဥပမာ)

```json
{
  "conversations": [
    {"from": "system", "value": "You are an expert Python tutor."},
    {"from": "human", "value": "What is a list comprehension?"},
    {"from": "gpt", "value": "A list comprehension is a concise way to create lists in Python.\n\nBasic syntax: `[expression for item in iterable if condition]`\n\nExample:\n```python\nsquares = [x**2 for x in range(10)]\n# Result: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n```"},
    {"from": "human", "value": "Can you show a more complex example?"},
    {"from": "gpt", "value": "Sure! Here are more complex examples:\n\n```python\n# Nested list comprehension\nmatrix = [[1,2,3],[4,5,6],[7,8,9]]\nflat = [x for row in matrix for x in row]\n# [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n# With condition\neven_squares = [x**2 for x in range(20) if x % 2 == 0]\n# [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]\n\n# Dictionary comprehension\nword_lengths = {word: len(word) for word in ['hello', 'world']}\n# {'hello': 5, 'world': 5}\n```"}
  ]
}
```

#### Axolotl Config

```yaml
datasets:
  - path: ./data/my_chat_data.jsonl
    type: sharegpt
    conversation: chatml       # ← Chat template format

# conversation options:
# chatml     → <|im_start|>role\ncontent<|im_end|>
# llama3     → Llama 3 format
# mistral    → Mistral format
# gemma      → Gemma format
# vicuna     → Vicuna format
```

#### ShareGPT Field Name ပြောင်းလဲခြင်း

Default field names မဟုတ်ရင် mapping လုပ်ပေးနိုင်ပါတယ်:

```yaml
datasets:
  - path: ./data/custom_chat.jsonl
    type: sharegpt
    conversation: chatml
    field_messages: messages          # default: conversations
    message_field_role: role          # default: from
    message_field_content: content    # default: value
    roles:
      user:                           # ← "from" field values mapping
        - human
        - user
      assistant:
        - gpt
        - assistant
      system:
        - system
```

ဥပမာ - OpenAI format dataset ကို sharegpt type နဲ့ သုံးခြင်း:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
  ]
}
```

```yaml
datasets:
  - path: ./data/openai_format.jsonl
    type: sharegpt
    conversation: chatml
    field_messages: messages
    message_field_role: role
    message_field_content: content
```

---

### 🏷️ Type 3: `chat_template` Format

Model ရဲ့ **native chat template** (tokenizer_config.json ထဲက) ကို auto-detect လုပ်ပြီး format ချတဲ့ နည်းလမ်းဖြစ်ပါတယ်။ **Axolotl မှာ recommended approach** ဖြစ်ပါတယ်။

#### Dataset Structure (OpenAI Messages Format)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a branch of AI that enables systems to learn from data and improve without explicit programming."}
  ]
}
```

#### Axolotl Config

```yaml
chat_template: chatml              # or: llama3, mistral, gemma, tokenizer_default
datasets:
  - path: ./data/messages_data.jsonl
    type: chat_template
    field_messages: messages        # messages field name
    message_field_role: role
    message_field_content: content
    roles:
      user:
        - user
      assistant:
        - assistant
      system:
        - system
```

#### `chat_template` vs `sharegpt` ဘယ်ဟာ သုံးသင့်လဲ?

| Feature | `sharegpt` | `chat_template` |
|---|---|---|
| Chat template | Manually specify (`conversation:`) | Auto from tokenizer |
| Flexibility | More manual control | More automatic |
| Model compatibility | Need correct conversation type | Auto-detect |
| Recommended for | Custom formats | Standard training |

---

### 📄 Type 4: `completion` Format

**Raw text completion** format - CPT (Continued Pre-Training) နဲ့ raw text generation အတွက်:

```json
{"text": "Once upon a time, there was a small village nestled in the mountains of Myanmar. The villagers lived peacefully, growing rice in the terraced fields that cascaded down the hillsides."}
```

```yaml
datasets:
  - path: ./data/corpus.jsonl
    type: completion
    field: text
```

---

### 🔧 Type 5: `input_output` Format

**Flexible instruction format** - input/output ကို explicitly ခွဲပြီး ပေးတဲ့ format:

```json
{
  "input": "Below is an instruction. Write a response.\n\n### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
  "output": "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can be both 0 and 1 simultaneously, allowing them to solve certain problems much faster."
}
```

```yaml
datasets:
  - path: ./data/input_output.jsonl
    type: input_output
```

---

### 🛠️ Type 6: Custom Prompt Template

ကိုယ်ပိုင် prompt template ဖန်တီးလို့ရပါတယ်:

```yaml
datasets:
  - path: ./data/custom.jsonl
    type:
      system_prompt: "You are an expert translator."
      field_instruction: source_text
      field_input: context
      field_output: translation
      format: "<|start_header_id|>user<|end_header_id|>\n\nTranslate this: {instruction}\nContext: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      no_input_format: "<|start_header_id|>user<|end_header_id|>\n\nTranslate this: {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

---

## 3. DPO (Direct Preference Optimization) - Dataset Format

### ရည်ရွယ်ချက်

SFT ပြီးသား model ကို **human preference** နဲ့ align ဖို့ ဖြစ်ပါတယ်။ "ဘယ် response က ပိုကောင်းလဲ" ဆိုတာ သင်ပေးခြင်းဖြစ်ပါတယ်။

### DPO Dataset Format

DPO အတွက် **chosen** (preferred) နှင့် **rejected** (non-preferred) response pair လိုအပ်ပါတယ်:

#### Format A: ShareGPT Style DPO

```json
{
  "chosen": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Explain gravity to a 5-year-old."},
    {"from": "gpt", "value": "Imagine the Earth is like a big magnet, but instead of pulling metal, it pulls everything! That's why when you throw a ball up, it comes back down. The Earth is giving it a big hug and pulling it back! 🌍"}
  ],
  "rejected": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Explain gravity to a 5-year-old."},
    {"from": "gpt", "value": "Gravity is a fundamental force described by Einstein's General Theory of Relativity, where mass curves spacetime, creating what we perceive as gravitational attraction proportional to the inverse square of distance."}
  ]
}
```

> **chosen** = ရှင်းလင်းပြီး 5 နှစ်သား နားလည်နိုင် → ✅ ပိုကောင်း
> **rejected** = Technical ဘာသာစကား၊ ကလေး နားမလည် → ❌ မကောင်း

#### Format B: Instruction Style DPO

```json
{
  "prompt": "What is the best way to learn programming?",
  "chosen": "Start with a beginner-friendly language like Python. Follow structured courses, build small projects, practice daily, and join coding communities for support.",
  "rejected": "Just read documentation."
}
```

### Axolotl DPO Config

```yaml
# DPO training rl config
rl: dpo

chat_template: chatml
datasets:
  - path: ./data/dpo_data.jsonl
    type: chat_template.default
    field_messages: chosen
    message_field_role: from
    message_field_content: value
    roles:
      user:
        - human
      assistant:
        - gpt
      system:
        - system

# OR simpler sharegpt DPO format
datasets:
  - path: ./data/dpo_data.jsonl
    type: sharegpt.default
    split: train

# DPO specific parameters
rl: dpo
dpo_beta: 0.1                      # KL divergence penalty (0.1-0.5)
```

### DPO vs SFT Dataset Comparison

```
SFT Dataset:                         DPO Dataset:
┌─────────────────────┐              ┌──────────────────────────────┐
│ instruction: "..."  │              │ prompt: "..."                │
│ output: "good ans"  │              │ chosen: "good answer"    ✅  │
│                     │              │ rejected: "bad answer"   ❌  │
└─────────────────────┘              └──────────────────────────────┘
     Single answer                        Answer pair (good vs bad)
```

---

## 4. ORPO (Odds Ratio Preference Optimization) - Dataset Format

### ရည်ရွယ်ချက်

**SFT + DPO ကို single training step** မှာ တစ်ခါတည်း လုပ်ခြင်းဖြစ်ပါတယ်။ SFT model သီးသန့် train စရာ မလိုတော့ပါ။

### ORPO Dataset Format

DPO နဲ့ format **တူတူပဲ** ဖြစ်ပါတယ် (chosen/rejected pairs):

```json
{
  "chosen": [
    {"from": "system", "value": "You are a helpful coding assistant."},
    {"from": "human", "value": "Write a Python function to check if a number is prime."},
    {"from": "gpt", "value": "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\nThis function checks divisibility up to √n for efficiency."}
  ],
  "rejected": [
    {"from": "system", "value": "You are a helpful coding assistant."},
    {"from": "human", "value": "Write a Python function to check if a number is prime."},
    {"from": "gpt", "value": "```python\ndef is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n```"}
  ]
}
```

### Axolotl ORPO Config

```yaml
rl: orpo

chat_template: chatml
datasets:
  - path: ./data/orpo_data.jsonl
    type: chat_template.default
    field_messages: chosen
    message_field_role: from
    message_field_content: value
    roles:
      user:
        - human
      assistant:
        - gpt
      system:
        - system

orpo_alpha: 0.1                     # ORPO loss weight
```

---

## 5. KTO (Kahneman-Tversky Optimization) - Dataset Format

### ရည်ရွယ်ချက်

DPO မှာ chosen/rejected **pair** လိုအပ်ပေမယ့်၊ KTO မှာ **unpaired** data (individual completion + good/bad label) နဲ့ alignment train လုပ်နိုင်ပါတယ်။

### KTO Dataset Format

```json
{
  "prompt": "What is the meaning of life?",
  "completion": "The meaning of life is a philosophical question that has been pondered throughout human history. Different perspectives include finding purpose through relationships, personal growth, contribution to society, or spiritual fulfillment.",
  "label": true
}
```

```json
{
  "prompt": "What is the meaning of life?",
  "completion": "42",
  "label": false
}
```

> `label: true` = ကောင်းတဲ့ response ✅
> `label: false` = မကောင်းတဲ့ response ❌
> pair ချိတ်စရာ **မလိုပါ**

### KTO vs DPO Dataset Comparison

```
DPO (Paired):                        KTO (Unpaired):
┌──────────────────────┐              ┌───────────────────────┐
│ prompt: "question"   │              │ prompt: "question"    │
│ chosen: "good" ──┐   │              │ completion: "answer"  │
│ rejected: "bad"──┘   │              │ label: true / false   │
│    ↑ Must be paired  │              │    ↑ Independent      │
└──────────────────────┘              └───────────────────────┘
```

### Axolotl KTO Config

```yaml
rl: kto

datasets:
  - path: ./data/kto_data.jsonl
    type: ...  # standard format
    split: train

kto_desirable_weight: 1.0
kto_undesirable_weight: 1.0
```

---

## 6. VLM (Vision-Language Model) SFT - Dataset Format

### ရည်ရွယ်ချက်

**Image + Text** understanding/generation train ဖို့ ဖြစ်ပါတယ်။

### VLM Dataset Format

#### LLaVA Style Format

```json
{
  "id": "image_001",
  "image": "images/photo_001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat do you see in this image?"},
    {"from": "gpt", "value": "The image shows a beautiful sunset over the Irrawaddy River in Myanmar. The sky is painted in shades of orange and pink, with traditional boats silhouetted against the horizon."},
    {"from": "human", "value": "What time of day was this photo likely taken?"},
    {"from": "gpt", "value": "Based on the low angle of the sun and the warm colors, this photo was likely taken during golden hour, approximately 30-45 minutes before sunset."}
  ]
}
```

#### Multi-Image Format

```json
{
  "id": "multi_img_001",
  "images": ["images/before.jpg", "images/after.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\nCompare these two images and describe the differences."},
    {"from": "gpt", "value": "The first image shows the building before renovation, while the second shows it after. Key differences include..."}
  ]
}
```

### Axolotl VLM Config

```yaml
base_model: llava-hf/llava-v1.6-mistral-7b-hf
model_type: LlavaForConditionalGeneration

adapter: lora
lora_r: 16
lora_alpha: 32

datasets:
  - path: ./data/vlm_data.jsonl
    type: llava

# Image processing settings
image_folder: ./data/images/
```

---

## 7. Function Calling / Tool Use - Dataset Format

### ရည်ရွယ်ချက်

Model ကို **function/tool calling** ability သင်ပေးဖို့ ဖြစ်ပါတယ်။

### Function Calling Dataset Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following functions:\n\n{\"name\": \"get_weather\", \"description\": \"Get weather for a location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"location\"]}}"
    },
    {
      "role": "user",
      "content": "What's the weather in Yangon?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Yangon\", \"unit\": \"celsius\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "get_weather",
      "content": "{\"temperature\": 32, \"condition\": \"Partly Cloudy\", \"humidity\": 78}"
    },
    {
      "role": "assistant",
      "content": "The weather in Yangon is currently 32°C and partly cloudy with 78% humidity."
    }
  ]
}
```

### Axolotl Config

```yaml
chat_template: chatml
datasets:
  - path: ./data/function_calling.jsonl
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content
    roles:
      user:
        - user
      assistant:
        - assistant
      system:
        - system
      tool:
        - tool
```

---

## Training Process Type အလိုက် Dataset Format ပြောင်းလဲမှု Summary

### 🔄 Same Data, Different Formats

တူညီတဲ့ task အတွက် training process type ပြောင်းရင် dataset format ဘယ်လို ပြောင်းရလဲ:

#### ဥပမာ Task: "Myanmar ဘာသာ ဘာသာပြန်"

**CPT (Domain Knowledge):**
```json
{"text": "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံတစ်ခုဖြစ်သည်။ မြို့တော်မှာ နေပြည်တော်ဖြစ်ပြီး..."}
```

**SFT (Instruction Following):**
```json
{
  "instruction": "Translate to Myanmar",
  "input": "The weather is nice today.",
  "output": "ဒီနေ့ ရာသီဥတု ကောင်းပါတယ်။"
}
```

**SFT Chat (Multi-turn):**
```json
{
  "conversations": [
    {"from": "human", "value": "Translate 'Good morning' to Myanmar"},
    {"from": "gpt", "value": "မင်္ဂလာ မနက်ခင်းပါ (Mingalar Manekhinbar)"},
    {"from": "human", "value": "How about 'Thank you'?"},
    {"from": "gpt", "value": "ကျေးဇူးတင်ပါတယ် (Kyay Zu Tin Par Tal)"}
  ]
}
```

**DPO (Preference Alignment):**
```json
{
  "prompt": "Translate 'I love Myanmar' to Burmese",
  "chosen": "ကျွန်တော် မြန်မာကို ချစ်ပါတယ်။ (Kyun Daw Myanmar Ko Chit Par Tal)",
  "rejected": "I love Myanmar = မြန်မာ ချစ်"
}
```

**KTO (Unpaired Preference):**
```json
{"prompt": "Translate 'Hello' to Myanmar", "completion": "မင်္ဂလာပါ", "label": true}
{"prompt": "Translate 'Hello' to Myanmar", "completion": "ဟယ်လို", "label": false}
```

---

## Dataset Preparation Pipeline

### Step 1: Raw Data Collection

```
Data Sources:
├── 📁 Local files (CSV, JSON, TXT, PDF)
├── 🤗 Hugging Face Hub datasets
├── 🌐 Web scraping / crawling
├── 📊 API responses (OpenAI, Claude, etc.)
└── 👥 Human annotation
```

### Step 2: Data Cleaning & Processing

```python
# data_preparation.py - Dataset preparation script example

import json

def prepare_alpaca_format(raw_data):
    """Raw data ကို Alpaca format ပြောင်းခြင်း"""
    formatted = []
    for item in raw_data:
        formatted.append({
            "instruction": item["question"],
            "input": item.get("context", ""),
            "output": item["answer"]
        })
    return formatted

def prepare_sharegpt_format(raw_data):
    """Raw data ကို ShareGPT format ပြောင်းခြင်း"""
    formatted = []
    for item in raw_data:
        conversations = []
        if "system" in item:
            conversations.append({"from": "system", "value": item["system"]})
        conversations.append({"from": "human", "value": item["question"]})
        conversations.append({"from": "gpt", "value": item["answer"]})
        formatted.append({"conversations": conversations})
    return formatted

def prepare_dpo_format(raw_data):
    """Raw data ကို DPO format ပြောင်းခြင်း"""
    formatted = []
    for item in raw_data:
        formatted.append({
            "chosen": [
                {"from": "human", "value": item["question"]},
                {"from": "gpt", "value": item["good_answer"]}
            ],
            "rejected": [
                {"from": "human", "value": item["question"]},
                {"from": "gpt", "value": item["bad_answer"]}
            ]
        })
    return formatted

def save_jsonl(data, output_path):
    """JSONL format ဖြင့် save ခြင်း"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Usage
raw = json.load(open("raw_data.json"))
alpaca = prepare_alpaca_format(raw)
save_jsonl(alpaca, "train_alpaca.jsonl")
```

### Step 3: Data Validation

```python
def validate_alpaca(filepath):
    """Alpaca format dataset ကို validate ခြင်း"""
    errors = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                if "instruction" not in item:
                    errors.append(f"Line {i+1}: Missing 'instruction'")
                if "output" not in item:
                    errors.append(f"Line {i+1}: Missing 'output'")
                if not item.get("output", "").strip():
                    errors.append(f"Line {i+1}: Empty 'output'")
            except json.JSONDecodeError:
                errors.append(f"Line {i+1}: Invalid JSON")
    return errors

def validate_sharegpt(filepath):
    """ShareGPT format dataset ကို validate ခြင်း"""
    errors = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                convos = item.get("conversations", [])
                if not convos:
                    errors.append(f"Line {i+1}: Empty conversations")
                for j, msg in enumerate(convos):
                    if "from" not in msg or "value" not in msg:
                        errors.append(f"Line {i+1}, msg {j}: Missing from/value")
                    if msg.get("from") not in ["system", "human", "gpt"]:
                        errors.append(f"Line {i+1}, msg {j}: Invalid role '{msg.get('from')}'")
            except json.JSONDecodeError:
                errors.append(f"Line {i+1}: Invalid JSON")
    return errors

def validate_dpo(filepath):
    """DPO format dataset ကို validate ခြင်း"""
    errors = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                if "chosen" not in item:
                    errors.append(f"Line {i+1}: Missing 'chosen'")
                if "rejected" not in item:
                    errors.append(f"Line {i+1}: Missing 'rejected'")
            except json.JSONDecodeError:
                errors.append(f"Line {i+1}: Invalid JSON")
    return errors
```

### Step 4: Data Quality Checks

```
✅ Quality Checklist:
├── ☐ Duplicate entries ရှိ/မရှိ စစ်ဆေး
├── ☐ Empty/null fields ရှိ/မရှိ စစ်ဆေး
├── ☐ Token length distribution ကြည့် (too short/too long)
├── ☐ Language consistency စစ်ဆေး
├── ☐ JSON format validity စစ်ဆေး
├── ☐ Special characters / encoding issues စစ်ဆေး
├── ☐ Train/eval split ခွဲထား (90/10 or 95/5)
└── ☐ Sensitive/harmful content စစ်ဆေး
```

---

## Axolotl Dataset Configuration - Advanced Features

### Multiple Datasets ပေါင်းစည်းခြင်း

Axolotl မှာ dataset **အများကြီးကို** တစ်ပြိုင်နက် ပေါင်းသုံးလို့ ရပါတယ်:

```yaml
datasets:
  # Dataset 1: Alpaca format
  - path: ./data/general_instructions.jsonl
    type: alpaca
    split: train

  # Dataset 2: ShareGPT format (local file)
  - path: ./data/chat_conversations.jsonl
    type: sharegpt
    conversation: chatml

  # Dataset 3: Hugging Face Hub dataset
  - path: teknium/OpenHermes-2.5
    type: sharegpt
    split: train

  # Dataset 4: Completion (pretraining data)
  - path: ./data/domain_corpus.jsonl
    type: completion
    field: text

# Evaluation dataset
val_set_size: 0.05              # 5% for validation
```

### Dataset Sharding & Sampling

```yaml
datasets:
  - path: ./data/large_dataset.jsonl
    type: sharegpt
    shards: 10                    # Split into 10 shards for large datasets

  - path: ./data/small_high_quality.jsonl
    type: alpaca
    # Data ကို ထပ်ခါထပ်ခါ sample လုပ်
```

### Sample Packing

짧은 sequences တွေကို **pack** ပြီး GPU efficiency မြင့်အောင် လုပ်ခြင်း:

```yaml
sample_packing: true              # Short samples ကို pack
pad_to_sequence_len: true         # Pad to max sequence length
```

```
Without packing:                   With packing:
┌────────┬──────────┐              ┌────────┬────────┬──────┐
│ Sample1│ PADDING  │              │ Sample1│Sample2 │Samp3 │
├────────┼──────────┤              ├────────┼────────┼──────┤
│ Sample2│ PADDING  │     →        │Sample4 │ Sample5│Smp6  │
├────────┼──────────┤              └────────┴────────┴──────┘
│ Sample3│ PADDING  │              GPU utilization: ~95%
└────────┴──────────┘
GPU utilization: ~40%
```

---

## File Formats Supported

### Axolotl Data File Types

| Format | Extension | ရှင်းလင်းချက် | ဥပမာ |
|---|---|---|---|
| **JSONL** | `.jsonl` | Line-delimited JSON (recommended) | `data.jsonl` |
| **JSON** | `.json` | JSON array | `data.json` |
| **Parquet** | `.parquet` | Columnar binary format | `data.parquet` |
| **CSV** | `.csv` | Comma-separated values | `data.csv` |
| **HuggingFace Dataset** | - | Hub dataset path | `org/dataset_name` |
| **Arrow** | `.arrow` | Apache Arrow format | `data.arrow` |

### JSONL vs JSON

```
JSONL (Recommended ✅):              JSON:
{"instruction":"...", "output":".."}  [
{"instruction":"...", "output":".."}    {"instruction":"...", "output":".."},
{"instruction":"...", "output":".."}    {"instruction":"...", "output":".."},
                                        {"instruction":"...", "output":".."}
                                      ]
↑ Line by line, streamable            ↑ Full file load needed
↑ Memory efficient                    ↑ Memory heavy for large files
```

---

## Training Process Pipeline - Dataset Format Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Complete Training Pipeline                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: CPT (Continued Pre-Training)                                  │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ Dataset: completion format (raw text)            │                   │
│  │ {"text": "domain knowledge corpus..."}           │                   │
│  │ Goal: Domain knowledge ထည့်သွင်း                  │                   │
│  └──────────────────┬───────────────────────────────┘                   │
│                     ↓                                                    │
│  Stage 2: SFT (Supervised Fine-Tuning)                                  │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ Dataset: alpaca / sharegpt / chat_template       │                   │
│  │ {"instruction":"...", "output":"..."}             │                   │
│  │ Goal: Instruction following ability               │                   │
│  └──────────────────┬───────────────────────────────┘                   │
│                     ↓                                                    │
│  Stage 3: Preference Alignment (DPO/ORPO/KTO)                          │
│  ┌──────────────────────────────────────────────────┐                   │
│  │ Dataset: chosen/rejected pairs (DPO/ORPO)        │                   │
│  │          completion + label (KTO)                 │                   │
│  │ Goal: Human preference alignment                  │                   │
│  └──────────────────────────────────────────────────┘                   │
│                                                                          │
│  💡 Stage 3 ကို DPO/ORPO/KTO ထဲက တစ်ခုပဲ ရွေးပါ                        │
│  💡 ORPO ဆိုရင် Stage 2+3 ကို combine လုပ်ပြီး                          │
│     SFT model သီးသန့် train စရာ မလို                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Training Type → Dataset Format → Axolotl Config

| Training Type | Dataset Format | Axolotl `type` | Axolotl `rl` | Key Fields |
|---|---|---|---|---|
| **CPT** | Raw text | `completion` | - | `text` |
| **SFT (single-turn)** | Instruction pairs | `alpaca` | - | `instruction`, `input`, `output` |
| **SFT (multi-turn)** | Conversations | `sharegpt` | - | `conversations[{from, value}]` |
| **SFT (auto template)** | Messages | `chat_template` | - | `messages[{role, content}]` |
| **SFT (flexible)** | Input/Output | `input_output` | - | `input`, `output` |
| **DPO** | Preference pairs | `sharegpt.default` / `chat_template.default` | `dpo` | `chosen[]`, `rejected[]` |
| **ORPO** | Preference pairs | `sharegpt.default` / `chat_template.default` | `orpo` | `chosen[]`, `rejected[]` |
| **KTO** | Labeled completions | custom | `kto` | `prompt`, `completion`, `label` |
| **VLM SFT** | Image + Conversation | `llava` | - | `image`, `conversations[]` |
| **Function Calling** | Tool use conversations | `chat_template` | - | `messages[]` with `tool_calls` |

---

## Common Mistakes & Troubleshooting

### ❌ Dataset Format Errors

| Error | ဖြစ်တတ်တဲ့ အကြောင်းရင်း | ဖြေရှင်းနည်း |
|---|---|---|
| `KeyError: 'instruction'` | Field name မှား | `field_instruction` parameter စစ်ပါ |
| `KeyError: 'conversations'` | ShareGPT field name မှား | `field_messages` parameter စစ်ပါ |
| `Invalid role` | Role name မှား (ဥပမာ `user` vs `human`) | `roles` mapping စစ်ပါ |
| `Empty response` | Output/value field ဗလာ | Data cleaning လုပ်ပါ |
| `Token length exceeded` | Sequence ရှည်လွန်း | `sequence_len` တိုးပါ သို့ data ဖြတ်ပါ |
| `JSON decode error` | JSONL format မှား | JSON validity စစ်ပါ |
| `DPO missing chosen/rejected` | DPO field မရှိ | chosen/rejected fields ထည့်ပါ |
| `Tokenizer chat template not found` | Chat template config မရှိ | `chat_template` specify လုပ်ပါ |

### ✅ Best Practices

```
Dataset Quality Tips:
├── 1. Data Size: 1K-100K examples (SFT), 10K+ (DPO)
├── 2. Quality > Quantity: ကောင်းတဲ့ 1K ဟာ ညံ့တဲ့ 100K ထက် ပိုကောင်း
├── 3. Diversity: Task/topic variety ရှိပါစေ
├── 4. Consistency: Format/style consistent ဖြစ်ပါစေ
├── 5. Deduplication: Duplicate data ဖယ်ရှားပါ
├── 6. Validation split: 5-10% eval data ခွဲထားပါ
├── 7. Token length: Model ရဲ့ max_seq_len ထက် မကျော်ပါစေ
└── 8. System prompt: Consistent system prompt သုံးပါ
```
