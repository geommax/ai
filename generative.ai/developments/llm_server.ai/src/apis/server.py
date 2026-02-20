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
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None


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
    ) -> int:
        """Validate the Bearer token and return the key's database id."""
        if authorization is None:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="Invalid Authorization header")
        key_id = db.validate_key_get_id(token)
        if key_id is None:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return key_id

    # ── Param resolver (merge client overrides with server defaults) ─
    def _resolve_params(req: Any) -> Dict[str, Any]:
        """Merge request params with server-configured tuning defaults.

        If the client provides a value it overrides; if omitted (None)
        the server's ``config.tuning`` value is used.
        """
        t = config.tuning
        return {
            "max_tokens": req.max_tokens if req.max_tokens is not None else t.max_tokens,
            "temperature": req.temperature if req.temperature is not None else t.temperature,
            "top_p": req.top_p if req.top_p is not None else t.top_p,
            "top_k": req.top_k if req.top_k is not None else t.top_k,
            "repetition_penalty": req.repetition_penalty if req.repetition_penalty is not None else t.repetition_penalty,
            "do_sample": req.do_sample if req.do_sample is not None else t.do_sample,
        }

    # ── Routes ─────────────────────────────────────────────────────

    @app.get("/")
    async def root():
        return {
            "name": "LLM Server.AI",
            "version": "1.0.0",
            "status": "running",
            "model_loaded": inference_engine.is_loaded,
            "model_id": inference_engine.model_id,
            "endpoints": [
                "/health",
                "/v1/models",
                "/v1/completions",
                "/v1/chat/completions",
            ],
        }

    @app.get("/v1/models")
    async def list_models(key_id: int = Depends(verify_api_key)):
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
        key_id: int = Depends(verify_api_key),
    ):
        if not inference_engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        params = _resolve_params(req)
        text = inference_engine.generate(req.prompt, **params)

        # Rough token estimates (words ÷ 0.75)
        prompt_tok = max(1, len(req.prompt.split()))
        completion_tok = max(1, len(text.split()))
        total_tok = prompt_tok + completion_tok

        # Record usage
        try:
            db.record_usage(key_id, "/v1/completions", prompt_tok, completion_tok, total_tok)
        except Exception:
            pass  # never fail the request because of metering

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=inference_engine.model_id or "",
            choices=[CompletionChoice(text=text)],
            usage=UsageInfo(
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                total_tokens=total_tok,
            ),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(
        req: ChatCompletionRequest,
        key_id: int = Depends(verify_api_key),
    ):
        if not inference_engine.is_loaded:
            raise HTTPException(status_code=503, detail="No model loaded")

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        params = _resolve_params(req)
        text = inference_engine.chat_generate(messages, **params)

        # Rough token estimates
        prompt_text = " ".join(m.content for m in req.messages)
        prompt_tok = max(1, len(prompt_text.split()))
        completion_tok = max(1, len(text.split()))
        total_tok = prompt_tok + completion_tok

        # Record usage
        try:
            db.record_usage(key_id, "/v1/chat/completions", prompt_tok, completion_tok, total_tok)
        except Exception:
            pass

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=inference_engine.model_id or "",
            choices=[
                ChatChoice(message=ChatMessage(role="assistant", content=text))
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                total_tokens=total_tok,
            ),
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
