"""
Exercises the deployed MALoRA server the same way openharness's
openai_client.py actually calls it -- same SDK, same stream=True +
stream_options pattern, same tools shape -- not just a generic curl.

    pip install openai
    python test_deployment.py

Edit BASE_URL below if your Modal URL changes.
"""

import asyncio
import json
import time

import httpx
from openai import AsyncOpenAI

BASE_URL = "https://ppp3work--malora-server-serve.modal.run"
API_KEY = "not-needed"   l


def check_health():
    print("=" * 70) 
    print("1. /health  (cold start can take several minutes -- checkpoint")
    print("   download + base model load + apply_compile + warmup passes)")
    print("=" * 70)
    t0 = time.time()
    max_wait = 600  # 10 minutes -- generous, given the warmup cost we measured
    poll_every = 10
    while True:
        elapsed = time.time() - t0
        if elapsed > max_wait:
            print(f"Still not up after {max_wait}s -- check `modal app logs malora-server`")
            return
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=15)
            print(f"status={r.status_code}  body={r.text}  ({elapsed:.0f}s elapsed)")
            return
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
            print(f"  ({elapsed:.0f}s elapsed) not ready yet ({type(e).__name__}), retrying...")
            time.sleep(poll_every)
    print()


async def check_nonstream(client: AsyncOpenAI):
    print("=" * 70)
    print("2. non-streaming chat completion")
    print("=" * 70)
    t0 = time.time()
    resp = await client.chat.completions.create(
        model="malora",
        messages=[{"role": "user", "content": "def fibonacci(n):"}],
        max_tokens=100,
        stream=False,
    )
    dt = time.time() - t0
    choice = resp.choices[0]
    print(f"({dt:.1f}s) finish_reason={choice.finish_reason}")
    print(f"usage={resp.usage}")
    print("content:")
    print(choice.message.content)
    print()


async def check_stream(client: AsyncOpenAI):
    print("=" * 70)
    print("3. streaming chat completion (mirrors the harness's actual call)")
    print("=" * 70)
    t0 = time.time()
    first_token_time = None
    text_parts = []
    finish_reason = None
    usage = None

    stream = await client.chat.completions.create(
        model="malora",
        messages=[{"role": "user", "content": "Write a Python function to reverse a linked list."}],
        max_tokens=200,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if not chunk.choices:
            if chunk.usage:
                usage = chunk.usage
            continue
        delta = chunk.choices[0].delta
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            text_parts.append(delta.content)

    total_time = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time else None
    print(f"time-to-first-token: {ttft:.2f}s" if ttft else "no tokens received!")
    print(f"total time: {total_time:.2f}s")
    print(f"finish_reason={finish_reason}  usage={usage}")
    print("content:")
    print("".join(text_parts))
    print()


async def check_tool_calling(client: AsyncOpenAI):
    print("=" * 70)
    print("4. tool-calling (streaming, same shape as check_tool_template.py)")
    print("=" * 70)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    t0 = time.time()
    text_parts = []
    tool_calls = {}
    finish_reason = None

    # note: no stream_options here -- matches the harness's own behavior of
    # dropping it whenever tools are present
    stream = await client.chat.completions.create(
        model="malora",
        messages=[{"role": "user", "content": "What's the weather like in Paris right now?"}],
        max_tokens=200,
        stream=True,
        tools=tools,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta.content:
            text_parts.append(delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments

    print(f"({time.time() - t0:.1f}s) finish_reason={finish_reason}")
    print(f"plain text content: {''.join(text_parts)!r}")
    if tool_calls:
        print("tool calls detected:")
        for tc in tool_calls.values():
            print(f"  name={tc['name']!r}")
            try:
                print(f"  arguments={json.loads(tc['arguments'])}")
            except json.JSONDecodeError:
                print(f"  arguments (unparsed)={tc['arguments']!r}")
    else:
        print("NO tool call detected -- model answered in plain text instead.")
        print("expected given this checkpoint (see earlier note): it's only ever")
        print("seen OpenCodeInstruct in training, never tool-call trajectories.")
    print()


async def main():
    check_health()
    client = AsyncOpenAI(api_key=API_KEY, base_url=f"{BASE_URL}/v1")
    await check_nonstream(client)
    await check_stream(client)
    await check_tool_calling(client)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
