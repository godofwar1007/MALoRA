
"""
Visualize per‑token expert routing across layers.
Usage: python visualize_routing.py --prompt "def fibonacci(n):" --base-url https://mc250041030--malora-server-serve.modal.run
"""

import asyncio
import json
import time
import argparse
import sys

import httpx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from openai import AsyncOpenAI

# Defaults 
DEFAULT_BASE_URL = "https://mc250041030--malora-server-serve.modal.run"
DEFAULT_PROMPT = "def fibonacci(n):"
DEFAULT_MODEL = "malora"
DEFAULT_MAX_TOKENS = 30

# Telemetry collector 
class TelemetryCollector:
    def __init__(self):
        self.events = []  # list of (token_index, layer_experts)

    async def listen(self, base_url):
        """Listen to /telemetry/stream and parse SSE events."""
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{base_url}/v1/telemetry/stream", timeout=None) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        payload = json.loads(data)
                        if payload.get("type") == "routing":
                            token_idx = payload["token_index"]
                            layer_experts = payload.get("layer_experts")
                            if layer_experts is not None:
                                self.events.append((token_idx, layer_experts))
                    except json.JSONDecodeError:
                        continue

    def get_matrix(self, num_layers=36):
        self.events.sort(key=lambda x: x[0])
        max_tokens = max([idx for idx, _ in self.events]) if self.events else 0
        matrix = np.full((num_layers, max_tokens + 1), -1, dtype=int)
        for token_idx, layer_experts in self.events:
            for layer_idx, expert_id in enumerate(layer_experts):
                if layer_idx < num_layers:
                    matrix[layer_idx, token_idx] = expert_id
        return matrix

    def get_routing_summary(self, num_layers=36):
        self.events.sort(key=lambda x: x[0])
        lines = ["Per‑token expert routing (layer → expert ID):"]
        for token_idx, layer_experts in self.events:
            layer_str = " ".join(f"L{l:02d}:E{exp}" for l, exp in enumerate(layer_experts[:num_layers]))
            lines.append(f"Token {token_idx:3d}: {layer_str}")
        return "\n".join(lines)

# Main 
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Modal endpoint (without /v1)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max generation tokens")
    parser.add_argument("--output-png", default="routing_heatmap.png", help="Output PNG")
    parser.add_argument("--output-txt", default="routing_summary.txt", help="Output text summary")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    api_url = f"{base_url}/v1"

    collector = TelemetryCollector()
    telemetry_task = asyncio.create_task(collector.listen(base_url))

    # Wait for telemetry to connect
    await asyncio.sleep(1)

    client = AsyncOpenAI(
        api_key="not-needed",
        base_url=api_url,
        timeout=300  # 5 minutes for cold start
    )

    print(f"Sending prompt: {args.prompt}")
    stream = await client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    generated_text = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            generated_text += chunk.choices[0].delta.content

    # Give telemetry a moment to finish
    await asyncio.sleep(2)
    telemetry_task.cancel()

    matrix = collector.get_matrix(num_layers=36)
    if matrix.size == 0:
        print("No telemetry data received.")
        return

    # Heatmap
    valid_mask = matrix[0, :] != -1
    matrix = matrix[:, valid_mask]
    num_tokens = matrix.shape[1]

    fig, ax = plt.subplots(figsize=(max(8, num_tokens * 0.3), 8))
    cmap = plt.colormaps.get_cmap("tab10")
    norm = plt.Normalize(vmin=0, vmax=7)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Layer index")
    ax.set_title(f"Expert Routing Across Layers\nPrompt: {args.prompt[:50]}...")
    cbar = fig.colorbar(im, ax=ax, ticks=range(8), label="Expert ID")
    patches = [mpatches.Patch(color=cmap(i), label=f"Expert {i}") for i in range(8)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {args.output_png}")

    # Combined summary
    combined = [
        "="*70,
        f"PROMPT: {args.prompt}",
        "="*70,
        "GENERATED CODE:",
        generated_text.strip() if generated_text else "(None)",
        "="*70,
        collector.get_routing_summary(),
        "="*70
    ]
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"Summary saved to {args.output_txt}")

    # Show first 200 chars
    print("\nGenerated text preview:\n", generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)

if __name__ == "__main__":
    asyncio.run(main())
