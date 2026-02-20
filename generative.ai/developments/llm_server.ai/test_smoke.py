#!/usr/bin/env python3
"""Quick smoke-test: list cached models, load Qwen, generate text."""

import sys

print("=" * 60)
print("  LLM Server.AI — Smoke Test")
print("=" * 60)

# ── 1. Model Manager ──────────────────────────────────────────────
from src.llms import ModelManager

mm = ModelManager()
models = mm.list_downloaded_models()
print("\n📦 Downloaded Models:")
for m in models:
    print(f"   {m['repo_id']:40s}  {m['size_str']:>10s}  files={m['nb_files']}")
print(f"\n   Total cache: {mm.total_cache_size()}")

# ── 2. Database / API Keys ────────────────────────────────────────
from src.config import DB_FILE
from src.database import Database
from src.apis.keys import generate_api_key

db = Database(str(DB_FILE))
key = generate_api_key()
db.add_key("smoke-test", key)
print(f"\n🔑 Generated test key: {key[:20]}...")
print(f"   Active keys: {db.key_count()}")
assert db.validate_key(key), "Key validation failed!"
print("   ✓ Key validated OK")

# ── 3. Inference Engine — load Qwen ───────────────────────────────
from src.llms import InferenceEngine

engine = InferenceEngine()
print(f"\n💻 Device: {engine.device_info()}")

model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"\n🧠 Loading {model_id}...")
engine.load_model(model_id)
print(f"   ✓ Model loaded on {engine.device}")

# ── 4. Generate ───────────────────────────────────────────────────
prompt = "What is the capital of Myanmar?"
print(f"\n📝 Prompt: {prompt}")
response = engine.generate(prompt, max_tokens=64, temperature=0.7)
print(f"🤖 Response: {response}")

# ── 5. Chat Generate ─────────────────────────────────────────────
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in Burmese."},
]
print(f"\n💬 Chat test...")
chat_resp = engine.chat_generate(messages, max_tokens=64, temperature=0.7)
print(f"🤖 Chat Response: {chat_resp}")

# ── Cleanup ───────────────────────────────────────────────────────
engine.unload_model()
db.delete_key(db.get_keys()[0]["id"])  # clean test key
print("\n✅ All smoke tests passed!")
