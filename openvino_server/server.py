"""
FastAPI server exposing an OpenAI-compatible /v1/chat/completions endpoint.
Wraps a locally-loaded OpenVINO Qwen2.5-VL-3B model.

Usage:
    python server.py --model-dir ./Qwen2.5-VL-3B-Instruct-ov --device CPU --port 8000
"""

import argparse
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Optional, Union

from model import QwenVLOpenVINO

# ---------------------------------------------------------------------------
# Global model instance (loaded once on startup)
# ---------------------------------------------------------------------------
_model: Optional[QwenVLOpenVINO] = None
_model_name: str = "qwen2.5-vl-3b"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global _model
    args = app.state.args
    print(f"[server] Loading model from '{args.model_dir}' on device '{args.device}'...")
    _model = QwenVLOpenVINO(model_dir=args.model_dir, device=args.device)
    yield
    # shutdown - nothing to clean up for OV


# ---------------------------------------------------------------------------
# Request / Response schemas  (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class TextContent(BaseModel):
    type: str = "text"
    text: str

class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = "auto"

class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: ImageURL

class Message(BaseModel):
    role: str
    content: Union[str, list[Union[TextContent, ImageContent, dict]]]

class ChatCompletionRequest(BaseModel):
    model: str = _model_name
    messages: list[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    # extra fields UI-TARS may send - just accept and ignore
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Any] = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen2.5-VL OpenVINO Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_raw_messages(messages: list[Message]) -> list[dict]:
    """Convert Pydantic Message objects back to plain dicts for the model."""
    result = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part)
                else:
                    parts.append(part.model_dump())
            result.append({"role": msg.role, "content": parts})
        else:
            result.append({"role": msg.role, "content": content})
    return result


@app.get("/")
async def root():
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if request.stream:
        # UI-TARS doesn't require streaming for basic use; return a clear error
        raise HTTPException(status_code=400, detail="Streaming not supported. Set stream=false.")

    try:
        raw_messages = _to_raw_messages(request.messages)
        text = _model.generate(
            messages=raw_messages,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            # token counts not tracked for OV GenAI; return rough estimates
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL OpenVINO FastAPI Server")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to exported OpenVINO model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "NPU", "AUTO"],
        help="OpenVINO device to run inference on",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="qwen2.5-vl-3b")
    args = parser.parse_args()

    _model_name = args.model_name

    app.state.args = args
    app.router.lifespan_context = lifespan

    print(f"[server] Starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")