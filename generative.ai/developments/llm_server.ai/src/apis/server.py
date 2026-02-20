"""FastAPI server — OpenAI-compatible local LLM endpoint."""

from __future__ import annotations

import time
import uuid
import threading
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Pydantic schemas ───────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str = "stop"


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage = Field(default_factory=lambda: ChatMessage())
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Build the FastAPI app ─────────────────────────────────────────────

def create_api(
    inference_engine: Any,
    db: Any,
    config: Any,
) -> FastAPI:
    """Create and return a configured FastAPI application."""

    app = FastAPI(
        title="LLM Server.AI",
        version="1.0.0",
        description="Local LLM inference server (OpenAI-compatible)",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth dependency ────────────────────────────────────────────
    async def verify_api_key(
        authorization: Optional[str] = Header(None),
    ) -> str:
        if authorization is None:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="Invalid Authorization header")
        if not db.validate_key(token):
            raise HTTPException(status_code=403, detail="Invalid API key")
        return token

    # ── Routes ─────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models(_key: str = Depends(verify_api_key)):
        model_id = inference_engine.model_id or "none"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(
        req: CompletionRequest,
        _key: str = Depends(verify_api_key),
    ):
        if not inference_engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        text = inference_engine.generate(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=req.do_sample,
        )
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=inference_engine.model_id or "",
            choices=[CompletionChoice(text=text)],
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(
        req: ChatCompletionRequest,
        _key: str = Depends(verify_api_key),
    ):
        if not inference_engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        text = inference_engine.chat_generate(
            messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=req.do_sample,
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=inference_engine.model_id or "",
            choices=[
                ChatChoice(message=ChatMessage(role="assistant", content=text))
            ],
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": inference_engine.is_loaded,
            "model_id": inference_engine.model_id,
        }

    return app


# ── Threaded server runner ─────────────────────────────────────────────

class ServerThread:
    """Run uvicorn in a daemon thread so the TUI stays responsive."""

    def __init__(
        self,
        app: FastAPI,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        self.config = uvicorn.Config(
            app, host=host, port=port, log_level="warning"
        )
        self.server = uvicorn.Server(self.config)
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._thread = threading.Thread(target=self.server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
