# TODO — RAG QA Bot Pipeline

## ✅ Completed

- [x] PDF / Multi-format document loading (`01_load_sources.py`)
- [x] Text chunking / ingestion (`02_ingestion.py`)
- [x] Vector embedding model (`03_vector_embedding.py`)
- [x] ChromaDB vector store (`04_chromadb_managements.py`)
- [x] QA Chain — Retriever + LLM (`05_chain.py`)
- [x] Retrieval with similarity scores (`06_retrieval.py`)
- [x] LLM loading — Qwen 2.5 (`07_llm.py`)
- [x] Gradio UI with debug info (`08_gradio.py`)

---

## 📌 TODO

### 1. Chat History — FileChatMessageHistory

**ရည်ရွယ်ချက်:**
အခု pipeline မှာ question တစ်ခုမေးတိုင်း conversation context ပျောက်သွားတယ်။
Chat history ထည့်ခြင်းဖြင့် model က ယခင်မေးခွန်း/အဖြေတွေကို
သိနေပြီး follow-up questions တွေကို context-aware ဖြေနိုင်မယ်။

**လုပ်ရမယ့်အရာ:**
- [ ] `langchain_community.chat_message_histories.FileChatMessageHistory` integrate လုပ်
- [ ] JSON file ထဲ persistent chat history သိမ်း
- [ ] Session management — session ID အလိုက် history ခွဲသိမ်း
- [ ] Gradio UI မှာ chat history ပြ / clear button ထည့်
- [ ] `05_chain.py` မှာ `create_retrieval_chain` ကို history-aware retriever နဲ့ ပြောင်း

**သက်ဆိုင်ရာ files:**
- `05_chain.py` — chain logic ပြင်ရမယ်
- `main.py` — history state manage လုပ်ရမယ်
- `08_gradio.py` — chat history UI ထည့်ရမယ်

---

### 2. Persistent Vector Database (ChromaDB)

**ရည်ရွယ်ချက်:**
အခု ChromaDB က in-memory ဖြစ်နေတာကြောင့် app restart လုပ်တိုင်း
document ကို ပြန် process လုပ်ရတယ်။ Persistent storage ထည့်ခြင်းဖြင့်
embed လုပ်ထားပြီးသား chunks တွေကို disk ထဲ သိမ်းထားပြီး ပြန်သုံးလို့ရမယ်။

**လုပ်ရမယ့်အရာ:**
- [ ] ChromaDB `persist_directory` parameter ထည့်ပြီး disk ထဲ store လုပ်
- [ ] App startup မှာ existing collection ရှိရင် ပြန် load လုပ်
- [ ] Collection management — list, delete, rename collections
- [ ] Duplicate document detection — တူညီတဲ့ file ကို ထပ်ပြီး embed မလုပ်
- [ ] Gradio UI မှာ collection manager panel ထည့်

**သက်ဆိုင်ရာ files:**
- `04_chromadb_managements.py` — persistent storage logic ထည့်ရမယ်
- `main.py` — startup load logic ထည့်ရမယ်
- `08_gradio.py` — collection manager UI ထည့်ရမယ်

---

### 3. Knowledge Graph Visualization

**ရည်ရွယ်ချက်:**
RAG pipeline ရဲ့ data flow ကို interactive graph ပုံစံ visualize
လုပ်ခြင်းဖြင့် document → chunks → embeddings → retrieval → prompt → answer
flow တစ်ခုလုံးကို မြင်နိုင်အောင် ဖန်တီးမယ်။ Debugging နဲ့ pipeline
behavior နားလည်ဖို့ အထောက်အကူဖြစ်စေမယ်။

**လုပ်ရမယ့်အရာ:**
- [ ] `networkx` + `pyvis` သုံးပြီး interactive HTML graph generate လုပ်
- [ ] Document ingestion phase — Source → Chunks → Embeddings nodes ပြ
- [ ] Query phase — Question → Retrieved Chunks (with scores) → Augmented Prompt → Answer nodes ပြ
- [ ] Node hover info — chunk preview, score, page number, metadata
- [ ] Gradio UI မှာ graph tab ထည့်ပြီး `gr.HTML()` နဲ့ ပြ

**ဆွဲထုတ်ရမယ့် data sources:**

| Data                              | Source File                  |
| --------------------------------- | ---------------------------- |
| Document → Chunks                 | `01_load_sources.py` + `02_ingestion.py` |
| Chunks → Embeddings               | `04_chromadb_managements.py` |
| Query → Retrieved Chunks + Scores | `06_retrieval.py`            |
| Augmented Prompt                   | `05_chain.py`                |
| Final Answer                       | `main.py`                    |

**Dependencies:**
```bash
pip install pyvis networkx
```

---

### 4. Multi-Modal RAG Pipeline

**ရည်ရွယ်ချက်:**
Text-only RAG pipeline ကို multi-modal အဆင့်ထိ တိုးချဲ့ခြင်းဖြင့်
images, tables, diagrams စတဲ့ non-text content တွေကိုပါ
နားလည်ပြီး answer ထုတ်ပေးနိုင်တဲ့ system ဖန်တီးမယ်။

**လုပ်ရမယ့်အရာ:**
- [ ] Vision-Language Model (VLM) integrate — image understanding
- [ ] PDF ထဲက images/tables ကို extract လုပ်ပြီး သီးသန့် process လုပ်
- [ ] Image embeddings — CLIP / multi-modal embedding model သုံး
- [ ] Table extraction — `camelot` / `tabula-py` သုံးပြီး structured data extract
- [ ] Multi-modal retrieval — text + image ပေါင်းပြီး retrieve လုပ်
- [ ] Audio input support (optional) — Whisper STT → text → RAG pipeline

**ဖြစ်နိုင်ချေ tech stack:**

| Component         | Options                                |
| ----------------- | -------------------------------------- |
| VLM               | Qwen2.5-VL, LLaVA, InternVL           |
| Image Embedding   | CLIP, SigLIP                           |
| Table Extraction  | camelot, tabula-py, unstructured       |
| Audio (optional)  | Whisper, faster-whisper                |

**သက်ဆိုင်ရာ files:**
- `01_load_sources.py` — image/table extraction logic ချဲ့ရမယ်
- `03_vector_embedding.py` — multi-modal embedding support ထည့်ရမယ်
- `04_chromadb_managements.py` — image vectors store ထည့်ရမယ်
- `07_llm.py` — VLM support ထည့်ရမယ်

---

## 🗂️ Current Project Structure

```
langchain/rag/
├── 01_load_sources.py       # Multi-format document loader
├── 02_ingestion.py          # Text chunking
├── 03_vector_embedding.py   # Embedding model
├── 04_chromadb_managements.py # ChromaDB vector store
├── 05_chain.py              # QA chain (retriever + LLM)
├── 06_retrieval.py          # Retrieval with scores
├── 07_llm.py                # LLM loading (Qwen 2.5)
├── 08_gradio.py             # Gradio UI
├── main.py                  # Entry point
├── model_download.md        # HuggingFace model download guide
└── TODO.md                  # ← This file
```
