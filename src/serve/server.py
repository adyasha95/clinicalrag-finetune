"""OpenAI-compatible inference server powered by vLLM.

Serves the merged fine-tuned model (base + LoRA adapter) via an
/v1/chat/completions endpoint that is a drop-in replacement for the
Anthropic API's OpenAI-compatible route.

Usage:
    python3 -m src.serve.server \
        --model checkpoints/dpo-adapter \
        --host 0.0.0.0 \
        --port 8000

    # Or using the merged model path:
    python3 -m src.serve.server --model /path/to/merged-model

Environment variables:
    FINETUNED_MODEL_PATH   Override --model from env
    SERVER_HOST            Override --host from env (default: 0.0.0.0)
    SERVER_PORT            Override --port from env (default: 8000)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Request / response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "finetuned-clinical"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False
    stop: list[str] | None = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ClinicalRAG Fine-tuned Model Server", version="1.0.0")
_engine: AsyncLLMEngine | None = None
_model_name: str = "finetuned-clinical"


def _apply_chat_template(messages: list[ChatMessage]) -> str:
    """Convert messages to Mistral instruct format."""
    parts = []
    system_content = ""

    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
        elif msg.role == "user":
            if system_content:
                parts.append(
                    f"<s>[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{msg.content} [/INST] "
                )
                system_content = ""
            else:
                if parts:
                    # Continuation turn — close prior response and open new
                    parts.append(f"<s>[INST] {msg.content} [/INST] ")
                else:
                    parts.append(f"<s>[INST] {msg.content} [/INST] ")
        elif msg.role == "assistant":
            parts.append(f"{msg.content}</s>")

    return "".join(parts)


def _make_choice(text: str, finish_reason: str = "stop") -> dict:
    return {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": finish_reason,
    }


async def _stream_response(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
    model_name: str,
) -> AsyncIterator[str]:
    created = int(time.time())
    async for output in _engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            delta = output.outputs[0].text
            chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": _model_name})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": _model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "clinicalrag-finetune",
        }],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    prompt = _apply_chat_template(request.messages)
    request_id = str(uuid.uuid4())

    stop_tokens = request.stop or []
    stop_tokens = list(set(stop_tokens + ["</s>", "[INST]"]))

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=stop_tokens,
    )

    if request.stream:
        return StreamingResponse(
            _stream_response(request_id, prompt, sampling_params, _model_name),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full output
    outputs = None
    async for out in _engine.generate(prompt, sampling_params, request_id):
        outputs = out

    if outputs is None or not outputs.outputs:
        raise HTTPException(status_code=500, detail="No output generated")

    generated_text = outputs.outputs[0].text
    finish_reason = outputs.outputs[0].finish_reason or "stop"

    prompt_tokens = len(outputs.prompt_token_ids)
    completion_tokens = len(outputs.outputs[0].token_ids)

    return JSONResponse({
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [_make_choice(generated_text, finish_reason)],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


# ── Startup ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "FINETUNED_MODEL_PATH",
            str(PROJECT_ROOT / "checkpoints" / "dpo-adapter"),
        ),
    )
    p.add_argument("--host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "8000")))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    global _engine, _model_name

    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            "Run the training pipeline first or set FINETUNED_MODEL_PATH."
        )

    _model_name = model_path.name

    print(f"Loading model from {model_path}…")
    engine_args = AsyncEngineArgs(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=False,
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)

    print(f"Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
