"""
server.py -- OpenAI-compatible /v1/chat/completions server for the MALoRA model.

Built directly against what openharness's openai_client.py actually sends
(confirmed by reading that file, not guessed):
  - endpoint: POST /v1/chat/completions (base_url + /v1 + /chat/completions)
  - always streaming ("stream": True, hardcoded on their side)
  - stream_options.include_usage present UNLESS tools are in the request
  - messages: already OpenAI-shaped (system/user/assistant/tool roles)
  - tools: OpenAI function-calling format
  - max_tokens (our model name won't match gpt-5*/o1*/o3*/o4*, so it's always
    max_tokens, never max_completion_tokens, on their side -- we accept both
    anyway for robustness / manual curl testing)

Tool-call format: confirmed by actually rendering Qwen2.5-Coder-3B-Instruct's
own chat template with tools= -- it's Hermes-style <tool_call>{...}</tool_call>
tags, same template on both the Coder and plain Instruct variants.

Run:
    MALORA_HF_FOLDER=malora_50k_1ep_aton uvicorn server:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role":"user","content":"def hello():"}], "stream": false}'
"""

import os
import json
import time
import uuid
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inference import (
    load_model,
    generate,
    generate_stream,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
)

log = logging.getLogger("malora.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── config, from env vars so the checkpoint can change per-deployment without
# touching this file. no default on the folder -- fail loudly at startup
# rather than silently loading the wrong checkpoint.
HF_FOLDER = os.environ.get("MALORA_HF_FOLDER")
if not HF_FOLDER:
    raise RuntimeError(
        "MALORA_HF_FOLDER env var is required (e.g. 'malora_50k_1ep_aton'). "
        "There's no default on purpose -- the checkpoint changes across runs."
    )
ATTN_ON = os.environ.get("MALORA_ATTN_ON", "1") == "1"
MODEL_NAME_FOR_RESPONSES = os.environ.get("MALORA_MODEL_NAME", "malora")

# one model instance for the whole process, loaded once at startup 
_model = None
_tokenizer = None

# real generation must be single-flight: one model, one GPU, one shared set of
# compiled graphs / KV cache. this lock serializes every request (streaming or
# not) so two concurrent calls never touch model.generate() at the same time.
# it's a plain threading.Lock, not asyncio.Lock, because the actual generation
# runs on background threads (Starlette's threadpool for the sync generator
# below, plus generate_stream's own internal thread) -- not on the event loop.
_generation_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    log.info("Loading model: folder=%s attn_on=%s", HF_FOLDER, ATTN_ON)
    _model, _tokenizer = load_model(HF_FOLDER, attn_on=ATTN_ON)
    log.info("Model ready, accepting requests.")
    yield
    log.info("Shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    # reachable at all only once lifespan startup (i.e. load_model) has
    # finished -- ASGI servers don't accept connections until then, so a
    # 200 here already means "warm". this is what modal_deploy.py will poll.
    return {"status": "ok"}


#  tool-call tag parsing 
# confirmed tag format by actually rendering the tokenizer's chat template
# with tools= a few turns back: <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _parse_tool_calls(token_iter):
    """
    Wraps a stream of text pieces (or a single full string, wrapped in an
    iterator), watching for <tool_call>...</tool_call> blocks -- same idea as
    the harness's own <think>-block stripping, applied to a different tag.
    Yields ("text", str) for plain content and ("tool_call", dict) for a
    fully parsed tool call, in the order they appeared.
    """
    buffer = ""
    in_tool_call = False
    for token in token_iter:
        buffer += token
        while True:
            if not in_tool_call:
                idx = buffer.find(TOOL_CALL_OPEN)
                if idx == -1:
                    # hold back a tail as long as the open tag, in case it's
                    # split across two token pieces from the streamer
                    safe_len = max(0, len(buffer) - len(TOOL_CALL_OPEN))
                    if safe_len:
                        yield ("text", buffer[:safe_len])
                        buffer = buffer[safe_len:]
                    break
                if idx > 0:
                    yield ("text", buffer[:idx])
                buffer = buffer[idx + len(TOOL_CALL_OPEN):]
                in_tool_call = True
            else:
                idx = buffer.find(TOOL_CALL_CLOSE)
                if idx == -1:
                    break  # wait for more tokens before deciding anything
                raw = buffer[:idx].strip()
                buffer = buffer[idx + len(TOOL_CALL_CLOSE):]
                in_tool_call = False
                try:
                    yield ("tool_call", json.loads(raw))
                except json.JSONDecodeError:
                    # malformed -- surface as text rather than silently
                    # dropping whatever the model actually said
                    yield ("text", f"{TOOL_CALL_OPEN}{raw}{TOOL_CALL_CLOSE}")
    if buffer:
        yield ("text", buffer)


def _extract_tool_calls(full_text: str):
    """Same parser as above, applied to one already-complete string (non-streaming path)."""
    text_parts, tool_calls = [], []
    for kind, value in _parse_tool_calls(iter([full_text])):
        (text_parts if kind == "text" else tool_calls).append(value)
    return "".join(text_parts), tool_calls


# messages/tools prep for the chat template
def _prepare_messages(messages):
    """
    OpenAI's wire format encodes tool_call arguments as a JSON STRING (both in
    requests and in what we'd echo back for prior turns), but the tokenizer's
    chat template does `tool_call.arguments | tojson`, which expects an
    actual object -- feeding it a string double-encodes it. Parse it back
    into a dict here so a multi-turn conversation that already includes a
    prior tool call renders correctly instead of producing broken JSON.
    """
    prepared = []
    for msg in messages:
        msg = dict(msg)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            new_calls = []
            for tc in tool_calls:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        pass  # leave as-is rather than crash the request
                tc["function"] = fn
                new_calls.append(tc)
            msg["tool_calls"] = new_calls
        prepared.append(msg)
    return prepared


def _build_prompt_and_tokens(body):
    messages = body.get("messages")
    if not messages:
        raise ValueError("request body must include a non-empty 'messages' list")
    tools = body.get("tools") or None
    prepared = _prepare_messages(messages)
    formatted = _tokenizer.apply_chat_template(
        prepared, tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(_tokenizer(formatted, add_special_tokens=False)["input_ids"])
    return formatted, prompt_tokens


def _gen_kwargs_from_body(body):
    return dict(
        max_new_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_NEW_TOKENS,
        temperature=body.get("temperature", DEFAULT_TEMPERATURE),
        top_p=body.get("top_p", DEFAULT_TOP_P),
        top_k=body.get("top_k", DEFAULT_TOP_K),
        repetition_penalty=body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
    )


# OpenAI chunk/response builders 
def _chunk(request_id, created, model_name, delta=None, finish_reason=None, usage=None):
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
    }
    if usage is not None:
        # matches real OpenAI behavior for stream_options.include_usage: one
        # extra terminal chunk with empty choices and populated usage
        payload["choices"] = []
        payload["usage"] = usage
    else:
        payload["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    return payload


def _sse(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _new_tool_call_id():
    return f"call_{uuid.uuid4().hex[:24]}"


# streaming path 
def _stream_events(body, request_id, model_name):
    """
    Plain (sync) generator -- Starlette's StreamingResponse detects it's not
    an async iterator and runs it via its own threadpool automatically, so
    this doesn't block the event loop. The generation_lock is acquired here,
    inside the generator, so it's held for the actual full duration of
    generation (acquiring it in the endpoint function itself wouldn't work --
    that coroutine returns as soon as the StreamingResponse object is built,
    long before this generator actually runs).
    """
    created = int(time.time())
    include_usage = bool(body.get("stream_options", {}).get("include_usage")) and not body.get("tools")

    try:
        formatted, prompt_tokens = _build_prompt_and_tokens(body)
    except ValueError as e:
        yield _sse({"error": {"message": str(e), "type": "invalid_request_error"}})
        yield "data: [DONE]\n\n"
        return

    gen_kwargs = _gen_kwargs_from_body(body)

    # real OpenAI sends a role-only chunk first, before any content
    yield _sse(_chunk(request_id, created, model_name, delta={"role": "assistant"}))

    text_parts = []
    saw_tool_call = False
    tool_index = 0

    with _generation_lock:
        try:
            token_iter = generate_stream(_model, _tokenizer, formatted, skip_formatting=True, **gen_kwargs)
            for kind, value in _parse_tool_calls(token_iter):
                if kind == "text":
                    if value:
                        text_parts.append(value)
                        yield _sse(_chunk(request_id, created, model_name, delta={"content": value}))
                else:
                    saw_tool_call = True
                    tc = {
                        "index": tool_index,
                        "id": _new_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": value.get("name", ""),
                            "arguments": json.dumps(value.get("arguments", {})),
                        },
                    }
                    tool_index += 1
                    yield _sse(_chunk(request_id, created, model_name, delta={"tool_calls": [tc]}))
        except Exception as e:
            log.exception("generation failed mid-stream (request_id=%s)", request_id)
            yield _sse({"error": {"message": str(e), "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return

    generated_text = "".join(text_parts)
    completion_tokens = (
        len(_tokenizer(generated_text, add_special_tokens=False)["input_ids"]) if generated_text else 0
    )
    if saw_tool_call:
        finish_reason = "tool_calls"
    elif completion_tokens >= gen_kwargs["max_new_tokens"]:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    yield _sse(_chunk(request_id, created, model_name, delta={}, finish_reason=finish_reason))

    if include_usage:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield _sse(_chunk(request_id, created, model_name, usage=usage))

    yield "data: [DONE]\n\n"


# non-streaming path
def _complete(body, request_id, model_name):
    """Runs on a worker thread via asyncio.to_thread -- keeps the event loop free."""
    formatted, prompt_tokens = _build_prompt_and_tokens(body)
    gen_kwargs = _gen_kwargs_from_body(body)

    with _generation_lock:
        raw_text = generate(_model, _tokenizer, formatted, skip_formatting=True, **gen_kwargs)

    content, tool_calls = _extract_tool_calls(raw_text)
    completion_tokens = len(_tokenizer(raw_text, add_special_tokens=False)["input_ids"]) if raw_text else 0

    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": _new_tool_call_id(),
                "type": "function",
                "function": {"name": tc.get("name", ""), "arguments": json.dumps(tc.get("arguments", {}))},
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"
    elif completion_tokens >= gen_kwargs["max_new_tokens"]:
        finish_reason = "length"
    else:
        finish_reason = "stop"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_name = body.get("model") or MODEL_NAME_FOR_RESPONSES
    stream = bool(body.get("stream", False))

    log.info(
        "request id=%s stream=%s n_messages=%d has_tools=%s",
        request_id, stream, len(body.get("messages") or []), bool(body.get("tools")),
    )

    if not body.get("messages"):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages is required and must be non-empty", "type": "invalid_request_error"}},
        )

    if stream:
        return StreamingResponse(_stream_events(body, request_id, model_name), media_type="text/event-stream")

    try:
        result = await asyncio.to_thread(_complete, body, request_id, model_name)
        return JSONResponse(result)
    except Exception as e:
        log.exception("request failed (request_id=%s)", request_id)
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "server_error"}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)