import os 
import json 
import time 
import uuid
import asyncio
import uvicorn 
import logging 
import threading 
import queue
from contextlib import asynccontextmanager
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse
from starlette.requests import ClientDisconnect

from inference import load_model,generate,generate_stream
from inference import(
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
)

log=logging.getLogger("malora.server")
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

# some configs and args form terminal related to the model/checkpoint details
HF_FOLDER = os.environ.get("MALORA_HF_FOLDER")
if not HF_FOLDER:
    raise RuntimeError(
        "MALORA_HF_FOLDER env var is required (e.g. 'malora_50k_1ep_aton'). "
        "There's no default on purpose -- the checkpoint changes across runs."
    )
ATTN_ON = os.environ.get("MALORA_ATTN_ON", "1") == "1"
MODEL_NAME_FOR_RESPONSES = os.environ.get("MALORA_MODEL_NAME", "malora")

_model=None
_tokenizer=None
_telemetry=None

# a lock here so that no two concurrent locks call model.generate()  at the same time 
_generation_lock=threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model,_tokenizer,_telemetry
    log.info("Loading model: folder=%s attn_on=%s", HF_FOLDER, ATTN_ON)
    _model,_tokenizer,_telemetry=load_model(HF_FOLDER, attn_on=ATTN_ON)
    log.info("Model ready, accepting requests.")
    yield
    log.info("shutting down")


app=FastAPI(lifespan=lifespan)

# just an endpoint to check if everything is okay or not 
@app.get("/health")
async def health():
    # reachable at all only once lifespan startup (i.e. load_model) has
    # finished -- ASGI servers don't accept connections until then, so a
    # 200 here already means "warm". this is what modal_deploy.py will poll.
    return {"status": "ok"}

# some tool call parsing 
# confirmed tag format by actually rendering the tokenizer's chat template
# with tools= a few turns back: <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
TOOL_CALL_OPEN  = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

def parse_tool_calls(token_iter):
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

def extract_tool_calls(full_text: str):
    """Same parser as above, applied to one already-complete string (non-streaming path)."""
    text_parts, tool_calls = [], []
    for kind, value in parse_tool_calls(iter([full_text])):
        (text_parts if kind == "text" else tool_calls).append(value)
    return "".join(text_parts), tool_calls

# messages/tools prep for the chat template 
def prepare_messages(messages):
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

def build_prompt_and_tokens(body):
    messages = body.get("messages")
    if not messages:
        raise ValueError("request body must include a non-empty 'messages' list")
    tools    = body.get("tools") or None
    prepared = prepare_messages(messages)
    formatted = _tokenizer.apply_chat_template(
        prepared, tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(_tokenizer(formatted, add_special_tokens=False)["input_ids"])
    return formatted, prompt_tokens

def gen_kwargs_from_body(body):
    # fix: `a or b or c` treats an explicit 0 the same as "not provided" --
    # use is-not-None checks so max_tokens=0 (unusual, but valid per the
    # OpenAI spec) isn't silently overridden with our default.
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_NEW_TOKENS
    max_tokens=min(int(max_tokens),4096)    
    return dict(
        max_new_tokens=max_tokens,
        temperature=body.get("temperature", DEFAULT_TEMPERATURE),
        top_p=body.get("top_p", DEFAULT_TOP_P),
        top_k=body.get("top_k", DEFAULT_TOP_K),
        repetition_penalty=body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY),
    )

# Openai chunk/response builders 
def chunk(request_id, created, model_name, delta=None, finish_reason=None, usage=None):
    payload = {
        "id":      request_id,
        "object":  "chat.completion.chunk",
        "created": created,
        "model":   model_name,
    }
    if usage is not None:
        # matches real OpenAI behavior for stream_options.include_usage: one
        # extra terminal chunk with empty choices and populated usage
        payload["choices"] = []
        payload["usage"]   = usage
    else:
        payload["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    return payload

def sse(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def new_tool_call_id():
    return f"call_{uuid.uuid4().hex[:24]}"

# streaming path 
def stream_events(body,request_id,model_name,telemetry=None):
    """
    Plain (sync) generator -- Starlette's StreamingResponse detects it's not
    an async iterator and runs it via its own threadpool automatically, so
    this doesn't block the event loop. The generation_lock is acquired here,
    inside the generator, so it's held for the actual full duration of
    generation (acquiring it in the endpoint function itself wouldn't work --
    that coroutine returns as soon as the StreamingResponse object is built,
    long before this generator actually runs).
    """
    created      = int(time.time())
    include_usage = bool(body.get("stream_options", {}).get("include_usage"))

    try:
        formatted, prompt_tokens = build_prompt_and_tokens(body)
    except Exception as e:
        yield sse({"error": {"message": str(e), "type": "invalid_request_error"}})
        yield "data: [DONE]\n\n"
        return

    gen_kwargs = gen_kwargs_from_body(body)

    # real OpenAI sends a role-only chunk first, before any content
    yield sse(chunk(request_id, created, model_name, delta={"role": "assistant"}))

    text_parts     = []
    saw_tool_call  = False
    tool_index     = 0
    result_info: dict = {}

    with _generation_lock:
        try:
            token_iter = generate_stream(
                _model, _tokenizer, formatted, skip_formatting=True,
                result_info=result_info, 
                telemetry=telemetry,
                **gen_kwargs,
            )
            for kind, value in parse_tool_calls(token_iter):
                if kind == "text":
                    if value:
                        text_parts.append(value)
                        yield sse(chunk(request_id, created, model_name, delta={"content": value}))
                else:
                    saw_tool_call = True
                    tc = {
                        "index": tool_index,
                        "id":    new_tool_call_id(),
                        "type":  "function",
                        "function": {
                            "name":      value.get("name", ""),
                            "arguments": json.dumps(value.get("arguments", {})),
                        },
                    }
                    tool_index += 1
                    yield sse(chunk(request_id, created, model_name, delta={"tool_calls": [tc]}))
        except Exception as e:
            log.exception("generation failed mid-stream (request_id=%s)", request_id)
            yield sse({"error": {"message": str(e), "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return

    completion_tokens = result_info.get("completion_tokens", 0)
    prompt_tokens = result_info.get("prompt_tokens", prompt_tokens)   # prefer the exact count

    if saw_tool_call:
        finish_reason = "tool_calls"
    else:
        finish_reason = result_info.get("finish_reason", "stop")

    yield sse(chunk(request_id, created, model_name, delta={}, finish_reason=finish_reason))

    if include_usage:
        usage = {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        }
        yield sse(chunk(request_id, created, model_name, usage=usage))

    yield "data: [DONE]\n\n"

# non streaming path 
def complete(body, request_id, model_name,telemetry=None):
    """Runs on a worker thread via asyncio.to_thread -- keeps the event loop free."""
    formatted, prompt_tokens = build_prompt_and_tokens(body)
    gen_kwargs = gen_kwargs_from_body(body)

    result_info: dict = {}   # fix: same reasoning as the streaming path above
    with _generation_lock:
        raw_text = generate(
            _model, _tokenizer, formatted, skip_formatting=True,
            result_info=result_info, 
            telemetry=telemetry,
            **gen_kwargs,
        )

    content, tool_calls = extract_tool_calls(raw_text)
    completion_tokens = result_info.get("completion_tokens", 0)
    prompt_tokens = result_info.get("prompt_tokens", prompt_tokens)

    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id":   new_tool_call_id(),
                "type": "function",
                "function": {
                    "name":      tc.get("name", ""),
                    "arguments": json.dumps(tc.get("arguments", {})),
                },
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"
    else:
        finish_reason = result_info.get("finish_reason", "stop")

    return {
        "id":      request_id,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }

# endpoint 
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body       = await request.json()
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_name = body.get("model") or MODEL_NAME_FOR_RESPONSES
    stream     = bool(body.get("stream", False))

    log.info(
        "request id=%s stream=%s n_messages=%d has_tools=%s",
        request_id, stream, len(body.get("messages") or []), bool(body.get("tools")),
    )

    if not body.get("messages"):
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": "messages is required and must be non-empty",
                "type":    "invalid_request_error",
            }},
        )

    if stream:
        return StreamingResponse(
            stream_events(body, request_id, model_name,telemetry=_telemetry),
            media_type="text/event-stream",
        )

    try:
        result = await asyncio.to_thread(complete, body, request_id, model_name,telemetry=_telemetry)
        return JSONResponse(result)
    except Exception as e:
        log.exception("request failed (request_id=%s)", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )

# telemetry endpoint 
@app.get("/telemetry/stream")
async def telemetry_stream():
    """Server-Sent Events endpoint for live routing telemetry."""
    def event_generator():
        while True:
            try:
                payload=_telemetry.queue.get(timeout=1.0)
                yield f"data: {json.dumps(payload)}\n\n"
            except queue.Empty:
                yield ":keepalive\n\n"
            except ClientDisconnect:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )            


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
