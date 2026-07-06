"""
eval_malora_v3.py — eval script for MALoRA models (shared S_A decomposition)

Forked from eval_malorav2.py. Changes from that file:
1. Config fields updated for MALoRA:
       moe_config.experts_rank   → REMOVED
       moe_config.shared_rank    → NEW (d, S_A rank, default 16)
       moe_config.expert_rank    → NEW (r_bar, P_t/B_bar_t rank, default 16)
       moe_config.experts_dropout → NEW (shared dropout rate, default 0.05)
2. LoRA key detection in step 6 updated to look for MALoRA-specific keys:
       gate_SA / up_SA / down_SA  (shared down-projection matrices)
       B_bar / .P.                (per-expert private matrices)
   in addition to the original 'lora' / 'router' checks.
3. enable_input_require_grads() removed — not needed at inference, only
   required for gradient flow during training.
4. Loading signature counts updated — MALoRA has different key counts than
   MoE-LoRA due to the S_A modules and renamed P/B_bar matrices.

Usage:
    # eval a MALoRA aton run
    python eval_malora_v3.py --folder malora_50k_1ep_aton --attn-on --dataset humaneval

    # eval a specific checkpoint
    python eval_malora_v3.py --folder malora_50k_1ep_aton/checkpoint-1000 --attn-on --dataset both

    # sanity check only (3 inference prompts, no benchmark)
    python eval_malora_v3.py --folder malora_50k_1ep_aton --attn-on --sanity-only

    # eval an atoff run (no attention LoRA)
    python eval_malora_v3.py --folder malora_50k_1ep_atoff --dataset humaneval
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# ── HF config ─────────────────────────────────────────────────────────────────
HF_TOKEN    = ""   # update if expired
HF_REPO_ID  = "godofwar1007/moelora"
BASE_MODEL  = "Qwen/Qwen2.5-Coder-3B-Instruct"

MAX_NEW_TOKENS = 512
OUTPUT_DIR     = "eval_outputs"

# ── MALoRA hyperparameters (must match what was used during training) ──────────
# If you trained with different values, change these to match training_config.py
SHARED_RANK      = 16     # d — S_A's rank
EXPERT_RANK      = 16     # r_bar — P_t / B_bar_t rank
EXPERTS_DROPOUT  = 0.05   # shared dropout rate
EXPERTS_SCALE    = 1.0
ATTENTION_RANK   = 32
NUM_EXPERTS      = 8
NUM_EXPERTS_PER_TOK = 2

# ── make sure local architecture files are importable ─────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(hf_folder: str, attn_on: bool):
    """
    Downloads MALoRA checkpoint from HF Hub and loads it correctly.

    hf_folder: subfolder inside godofwar1007/moelora, e.g. 'malora_50k_1ep_aton'
               or 'malora_50k_1ep_aton/checkpoint-1000'
    attn_on:   whether this checkpoint was trained with attention LoRA enabled
    """
    print(f"\n{'='*60}")
    print(f"Loading MALoRA checkpoint: {HF_REPO_ID}/{hf_folder}")
    print(f"Attention LoRA: {'ON' if attn_on else 'OFF'}")
    print(f"shared_rank={SHARED_RANK}  expert_rank={EXPERT_RANK}")
    print(f"{'='*60}\n")

    # ── step 1: download subfolder from HF ───────────────────────────────────
    print("Downloading checkpoint from HF Hub...")
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        allow_patterns=[f"{hf_folder}/*", f"{hf_folder}/model.safetensors"],
        local_dir=f"./hf_cache/{hf_folder.replace('/', '_')}",
    )
    ckpt_path = os.path.join(local_dir, hf_folder, "model.safetensors")

    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(local_dir, "model.safetensors")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"model.safetensors not found.\n"
            f"Looked at:\n"
            f"  {os.path.join(local_dir, hf_folder, 'model.safetensors')}\n"
            f"  {os.path.join(local_dir, 'model.safetensors')}\n"
            f"Files in local_dir: {os.listdir(local_dir)}"
        )

    print(f"Checkpoint found at: {ckpt_path}")
    print(f"Size: {os.path.getsize(ckpt_path)/1e9:.2f} GB\n")

    # ── step 2: tokenizer ─────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── step 3: base model ────────────────────────────────────────────────────
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = True

    # ── step 4: wrap with MALoRA LoraMoeModel ────────────────────────────────
    print("Wrapping with MALoRA LoraMoeModel...")
    moe_config = LoraMoeConfig.from_pretrained(BASE_MODEL)

    # MALoRA-specific fields (replaces single experts_rank from MoE-LoRA)
    moe_config.shared_rank          = SHARED_RANK
    moe_config.expert_rank          = EXPERT_RANK
    moe_config.experts_dropout      = EXPERTS_DROPOUT
    moe_config.experts_scale        = EXPERTS_SCALE

    # unchanged fields
    moe_config.attention_rank       = ATTENTION_RANK
    moe_config.num_experts_per_tok  = NUM_EXPERTS_PER_TOK
    moe_config.num_local_experts    = NUM_EXPERTS
    moe_config.output_router_logits = False    # off for inference
    moe_config.router_aux_loss_coef = 0.001
    moe_config.use_attention_lora   = attn_on

    moe_model = LoraMoeModel(base_model, moe_config)

    # ── step 5: load saved weights ────────────────────────────────────────────
    print("Loading saved weights...")
    saved_sd = load_file(ckpt_path, device="cpu")
    model_sd = moe_model.state_dict()

    print(f"  Checkpoint keys: {len(saved_sd)}")
    print(f"  Model keys:      {len(model_sd)}")

    # try direct match first
    matched   = {k: v for k, v in saved_sd.items() if k in model_sd}
    unmatched = [k for k in saved_sd if k not in model_sd]

    if len(matched) == 0:
        print("  Direct keys didn't match, trying prefix remapping...")
        remapped = {}
        for k, v in saved_sd.items():
            new_k = "base_model." + k
            if new_k in model_sd:
                remapped[new_k] = v
                continue
            if k.startswith("base_model."):
                new_k = k[len("base_model."):]
                if new_k in model_sd:
                    remapped[new_k] = v
                    continue
            remapped[k] = v
        matched   = {k: v for k, v in remapped.items() if k in model_sd}
        unmatched = [k for k in remapped if k not in model_sd]

    print(f"  Matched:   {len(matched)} keys")
    print(f"  Unmatched: {len(unmatched)} keys")
    if unmatched[:5]:
        print(f"  Sample unmatched: {unmatched[:5]}")

    missing = moe_model.load_state_dict(matched, strict=False)
    print(f"  Missing from checkpoint (frozen base weights expected): {len(missing.missing_keys)}")

    # ── step 6: verify MALoRA weights loaded ──────────────────────────────────
    # MALoRA key patterns differ from MoE-LoRA:
    #   MoE-LoRA:  gate_lora.A / gate_lora.B  (per expert, independent)
    #   MALoRA:    gate_SA.proj               (shared S_A matrix)
    #              gate_lora.P / gate_lora.B_bar  (private per expert)
    # We check for all of these plus router and attention LoRA keys.
    malora_key_patterns = ['lora', 'router', 'gate_SA', 'up_SA', 'down_SA', 'B_bar', '.P.']
    lora_keys_loaded = [
        k for k in matched
        if any(pat in k for pat in malora_key_patterns)
    ]
    print(f"  MALoRA/router keys loaded: {len(lora_keys_loaded)}")

    # break down by type so you can spot missing categories immediately
    sa_keys   = [k for k in matched if any(x in k for x in ['gate_SA', 'up_SA', 'down_SA'])]
    P_keys    = [k for k in matched if '.P.' in k]
    Bbar_keys = [k for k in matched if 'B_bar' in k]
    router_keys = [k for k in matched if 'router' in k]
    attn_keys = [k for k in matched if 'lora' in k and 'self_attn' in k]

    print(f"    S_A (shared subspace):   {len(sa_keys)}")
    print(f"    P_t (private coeff):     {len(P_keys)}")
    print(f"    B_bar (private up-proj): {len(Bbar_keys)}")
    print(f"    Router:                  {len(router_keys)}")
    print(f"    Attention LoRA:          {len(attn_keys)}")

    # warn loudly if any critical category is missing
    if len(sa_keys) == 0:
        print("\n  WARNING: No S_A keys found — shared down-projection matrices missing!")
        print("  This means either the checkpoint isn't a MALoRA checkpoint, or")
        print("  the key names changed. Check checkpoint key samples below.")
        print("  Checkpoint key samples:", list(saved_sd.keys())[:10])

    if len(P_keys) == 0 or len(Bbar_keys) == 0:
        print("\n  WARNING: P_t or B_bar keys missing — expert adapters may not have loaded.")

    if attn_on and len(attn_keys) == 0:
        print("\n  WARNING: --attn-on was set but no attention LoRA keys were found.")
        print("  Are you sure this checkpoint was trained with attention LoRA enabled?")

    moe_model.eval()
    print("\nModel ready.\n")
    return moe_model, tokenizer


# ── sanity check inference ────────────────────────────────────────────────────

def sanity_check(model, tokenizer):
    print("Running sanity check inference...")
    test_prompts = [
        "def fibonacci(n):",
        "def binary_search(arr, target):",
        "# Write a function to reverse a string\ndef reverse_string(s):",
    ]
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write a complete, correct Python function. Return only the code, no explanation."},
            {"role": "user",   "content": f"Complete this Python function:\n\n{prompt}"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {completion[:200]}")
        print("-" * 40)


# ── generation ────────────────────────────────────────────────────────────────

def strip_markdown(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


@torch.inference_mode()
def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


# ── HumanEval+ ───────────────────────────────────────────────────────────────

def run_humaneval(model, tokenizer, tag: str):
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    print(f"\nHumanEval+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Complete the function below. Return ONLY the function body — no explanation, no markdown, no extra text."},
            {"role": "user",   "content": f"Complete this Python function:\n\n{problem['prompt']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        start = time.time()
        completion = strip_markdown(generate(model, tokenizer, prompt))
        elapsed = time.time() - start

        results.append({"task_id": task_id, "completion": completion})

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(problems)}] last: {elapsed:.1f}s")

    out_path = os.path.join(OUTPUT_DIR, f"humaneval_{tag}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}")
    return out_path


# ── MBPP+ ─────────────────────────────────────────────────────────────────────

def run_mbpp(model, tokenizer, tag: str):
    from evalplus.data import get_mbpp_plus
    problems = get_mbpp_plus()
    print(f"\nMBPP+: {len(problems)} problems")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i, (task_id, problem) in enumerate(problems.items()):
        tests = "\n".join(problem.get("test_list", [])[:2])
        messages = [
            {"role": "system", "content": "You are an expert Python programmer. Write a Python function that solves the given task. Return ONLY the function — no explanation, no markdown, no extra text."},
            {"role": "user",   "content": f"Task: {problem['prompt']}\n\nExample tests:\n{tests}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = strip_markdown(generate(model, tokenizer, prompt))
        results.append({"task_id": task_id, "completion": completion})

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(problems)}]")

    out_path = os.path.join(OUTPUT_DIR, f"mbpp_{tag}.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved → {out_path}")
    return out_path


# ── scoring ───────────────────────────────────────────────────────────────────

def score(dataset: str, completions_path: str):
    print(f"\nScoring {dataset}...")
    os.system(f"python -m evalplus.evaluate --dataset {dataset} --samples {completions_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",      required=True,
                        help="HF subfolder, e.g. 'malora_50k_1ep_aton' or 'malora_50k_1ep_aton/checkpoint-1000'")
    parser.add_argument("--dataset",     choices=["humaneval", "mbpp", "both"], default="both")
    parser.add_argument("--attn-on",     action="store_true",
                        help="set use_attention_lora=True (use for aton runs)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="run 3 inference prompts only, skip benchmarks")
    # optional overrides for non-default rank configs
    parser.add_argument("--shared-rank", type=int, default=SHARED_RANK,
                        help=f"S_A rank used during training (default: {SHARED_RANK})")
    parser.add_argument("--expert-rank", type=int, default=EXPERT_RANK,
                        help=f"P_t/B_bar rank used during training (default: {EXPERT_RANK})")
    args = parser.parse_args()

    # allow rank overrides via CLI without changing the file
    SHARED_RANK = args.shared_rank
    EXPERT_RANK = args.expert_rank

    tag = args.folder.replace("/", "_")

    model, tokenizer = load_model(args.folder, attn_on=args.attn_on)
    sanity_check(model, tokenizer)

    if args.sanity_only:
        print("\nSanity check done. Exiting (--sanity-only was set).")
        exit(0)

    if args.dataset in ("humaneval", "both"):
        score("humaneval", run_humaneval(model, tokenizer, tag))

    if args.dataset in ("mbpp", "both"):
        score("mbpp", run_mbpp(model, tokenizer, tag))