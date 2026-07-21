import os
import sys 
import torch
import argparse
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

from configuration_lora_moe import LoraMoeConfig
from modelling import LoraMoeModel

from transformers import TextIteratorStreamer
import threading 
import queue   # fix: needed to catch the streamer timeout below
import torch._dynamo

load_dotenv()   # fix: this was imported but never called, so HF_TOKEN from .env was never actually loaded into os.environ

torch.backends.cuda.matmul.allow_tf32 = True   # fix: free throughput on Ampere+ (A10G/L40S/H100), no downside for bf16 weights
torch.backends.cudnn.allow_tf32 = True

# config 
BASE_MODEL  = "Qwen/Qwen2.5-Coder-3B-Instruct"
HF_REPO_ID  = "godofwar1007/moelora"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

# some defaults 
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE    = 0.2      # low temp for coding = more deterministic
DEFAULT_TOP_P          = 0.95
DEFAULT_TOP_K          = 50
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_STREAM_TIMEOUT = 120   # fix: seconds of silence on the streamer queue before we give up

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write clean, correct, and well-commented code. "
    "Think step by step. Return only the code unless explanation is asked for."
)

# defining some functions to do compile and warmup 
def apply_compile(moe_model):
    """
    here i have applied seletive compile to the malora model
    compilation is applied to the attention and the mlp layers since they dont have any graph breaks 
    left the dipatch loop as it has unavoidable data dependent graph breaks 
    """
    print("Applying torch.compile to linear layers...")
    compiled = 0

    for layer in moe_model.base_model.model.layers:

        if hasattr(layer,'_has_attn_lora') and layer._has_attn_lora:
            base_attn=layer.self_attn.base_attn
        else:
            base_attn=layer.self_attn

        base_attn.q_proj=torch.compile(base_attn.q_proj,fullgraph=False)
        base_attn.k_proj=torch.compile(base_attn.k_proj,fullgraph=False)
        base_attn.v_proj=torch.compile(base_attn.v_proj,fullgraph=False)
        base_attn.o_proj=torch.compile(base_attn.o_proj,fullgraph=False)
        compiled+=4


        # frozen mlp projections -- here dynamic = True
        layer.mlp.gate_proj=torch.compile(layer.mlp.gate_proj,dynamic=True,fullgraph=False)
        layer.mlp.up_proj=torch.compile(layer.mlp.up_proj,dynamic=True,fullgraph=False)
        layer.mlp.down_proj=torch.compile(layer.mlp.down_proj,dynamic=True,fullgraph=False)
        compiled+=3

        # shared SA matrices 
        block=layer.lora_moe_block
        block.gate_SA.proj=torch.compile(block.gate_SA.proj,dynamic=True,fullgraph=False)
        block.up_SA.proj=torch.compile(block.up_SA.proj,dynamic=True,fullgraph=False)
        block.down_SA.proj=torch.compile(block.down_SA.proj,dynamic=True,fullgraph=False)
        compiled+=3

        block.router.gate=torch.compile(block.router.gate,dynamic=True,fullgraph=False)
        compiled+=1

        # P and B matrices for all the individual experts 
        for expert in block.lora_experts:
            for adapter in [expert.gate_lora, expert.up_lora, expert.down_lora]:
                adapter.P=torch.compile(adapter.P,dynamic=True,fullgraph=False)
                adapter.B_bar=torch.compile(adapter.B_bar,dynamic=True,fullgraph=False)
                compiled+=2

        # dispatch loop in block.router.forward() is not compiled
        # mask.any() is data-dependent control flow that causes unavoidable
        # graph breaks. The linear layers it calls are compiled above.

    print(f"  Compiled {compiled} linear layers across {len(moe_model.base_model.model.layers)} decoder layers")
    return moe_model

def warmup(moe_model,tokenizer):
    """
    Run dummy forward passes to trigger actual compilation.
    torch.compile is JIT — compilation happens on first call.
    Without warmup, the first real user request pays 30-60s compilation cost.
    Run this immediately after apply_compile(), before serving any requests.
    """

    print("Running warmup passes (triggers torch.compile JIT compilation)...")
    device=next(moe_model.parameters()).device

    dummy_input=tokenizer(
        "def warmup():",
        return_tensors="pt"
    ).to(device)

    for i in range(3):
        with torch.inference_mode():   # fix: was no_grad, missed in the earlier inference_mode sweep
            moe_model.generate(
                **dummy_input,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        print(f"  Warmup pass {i+1}/3 done")

    long_dummy_input=tokenizer(
        "def process_data(items):\n    results = []\n    for item in items:\n"
        "        if item is None:\n            continue\n        results.append(item * 2)\n"
        "    return results\n\n# Write a function that takes a list of dictionaries and\n"
        "# returns a new list containing only the dictionaries where a given key\n"
        "# matches a given value, handling missing keys gracefully.\ndef filter_by_key(",
        return_tensors="pt"
    ).to(device)
    with torch.inference_mode():   # fix: same miss as the short warmup pass above
        moe_model.generate(
            **long_dummy_input,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("  Warmup pass for longer prefill length done")

    print("Warmup complete — model ready for inference\n")


# stream generation function 
def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float  = DEFAULT_TEMPERATURE,
    top_p: float        = DEFAULT_TOP_P,
    top_k: int          = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    system_prompt: str  = SYSTEM_PROMPT,   # fix: caller (server.py/harness) can now override this
    skip_formatting: bool = False,  # fix(server.py): when True, `prompt` is already a fully
                                    # rendered chat-template string (built by server.py via
                                    # apply_chat_template, with the full multi-turn conversation +
                                    # tools) -- skip build_prompt entirely instead of re-wrapping
                                    # it as a single user turn under our own SYSTEM_PROMPT.
    result_info: dict = None,      # fix(server.py): server.py needs prompt/completion token
                                    # counts and finish_reason (stop vs length). model.generate()
                                    # still returns the full sequence even with a streamer
                                    # attached -- pass a dict in and it gets filled before return.
):
    """
    Streaming generation - yields tokens as they are produced.
    this will be used in the server file i guess 
    """

    device=next(model.parameters()).device
    formatted = prompt if skip_formatting else build_prompt(prompt,tokenizer,system_prompt)
    inputs=tokenizer(formatted,return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    streamer=TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,    # so it doenst yeild the input prompt back
        skip_special_tokens=True,
        timeout=DEFAULT_STREAM_TIMEOUT,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    exception_box = {}
    seq_box = {}
    def _run_generate():
        try:
            with torch.inference_mode():
                out = model.generate(**generation_kwargs)
                seq_box["sequences"] = out   # fix(server.py): generate() still returns this even with a streamer attached
        except Exception as e:
            exception_box["error"] = e

    thread=threading.Thread(target=_run_generate,daemon=True)
    thread.start()

    try:
        for token in streamer:
            yield token 
    except queue.Empty:
        thread.join(timeout=1)
        if "error" in exception_box:
            raise exception_box["error"]
        raise TimeoutError(f"generate_stream: no token for {DEFAULT_STREAM_TIMEOUT}s, generation likely hung")

    thread.join()    
    if "error" in exception_box:
        raise exception_box["error"]

    # fix(server.py): fill in result_info for the caller now that generation is done
    if result_info is not None and "sequences" in seq_box:
        gen_ids = seq_box["sequences"][0][input_len:]
        last_tok = gen_ids[-1].item() if len(gen_ids) > 0 else None
        result_info["prompt_tokens"] = input_len
        result_info["completion_tokens"] = len(gen_ids)
        result_info["finish_reason"] = "stop" if last_tok == tokenizer.eos_token_id else "length"

# loading the model 
def load_model(hf_folder: str, attn_on: bool = True):
    """
    this just downloads the checkpoint form the hf repo and loads it i tried
    to keep it to be similar to the eval pattern 
    """
    
    print(f"\n{'='*60}")
    print(f"Loading: {HF_REPO_ID}/{hf_folder}")
    print(f"{'='*60}\n")
    
    print("Downloading checkpoint from HF Hub...")
    local_dir=snapshot_download(
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
            f"Tried:\n"
            f"  {os.path.join(local_dir, hf_folder, 'model.safetensors')}\n"
            f"  {os.path.join(local_dir, 'model.safetensors')}\n"
            f"Files in cache: {os.listdir(local_dir)}"
        )
 
    print(f"Checkpoint: {ckpt_path}")
    print(f"Size: {os.path.getsize(ckpt_path)/1e9:.2f} GB\n")

    print("Loading tokenizer .....")
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    tokenizer.padding_side="left" 

    print("Loading base model...")
    base_model=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print(f"Wrapping with MALoRA (attention LoRA {'ON' if attn_on else 'OFF'})...")
    moe_config = LoraMoeConfig.from_pretrained(BASE_MODEL)
    moe_config.shared_rank          = 16
    moe_config.expert_rank          = 16
    moe_config.experts_dropout      = 0.05
    moe_config.attention_rank       = 32
    moe_config.experts_scale        = 1.0
    moe_config.num_experts_per_tok  = 2
    moe_config.num_local_experts    = 8
    moe_config.output_router_logits = False
    moe_config.router_aux_loss_coef = 0.001
    moe_config.use_attention_lora   = attn_on   # fix(server.py): was hardcoded True -- now the
                                                 # caller decides (server.py reads MALORA_ATTN_ON)

    moe_model=LoraMoeModel(base_model,moe_config)

    print("Loading saved LoRA weights...")
    saved_sd = load_file(ckpt_path, device="cpu")
    model_sd = moe_model.state_dict()
 
    print(f"  Checkpoint keys: {len(saved_sd)}")
    print(f"  Model keys:      {len(model_sd)}")
    
    matched   = {k: v for k, v in saved_sd.items() if k in model_sd}
    unmatched = [k for k in saved_sd if k not in model_sd]

    if len(matched) == 0:
        print("  Direct keys didn't match — trying prefix remapping...")
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
    if unmatched[:3]:
        print(f"  Sample unmatched: {unmatched[:3]}")                                           
    
    load_result = moe_model.load_state_dict(matched,strict=False)
    print(f"  Missing from checkpoint (frozen base weights expected): {len(load_result.missing_keys)}")

    lora_keys = [k for k in matched if "lora" in k.lower() or "router" in k.lower() or "gate_SA" in k]
    print(f"  LoRA/router keys loaded: {len(lora_keys)}")
    if len(lora_keys) == 0:
        print("\n  ⚠️  WARNING: No LoRA keys found — checkpoint may not match this config.")
        print(f"  Checkpoint samples: {list(saved_sd.keys())[:5]}")
        print(f"  Model samples:      {list(model_sd.keys())[:5]}")

    moe_model.eval()
    moe_model.base_model.config.use_cache=True

    device=next(moe_model.parameters()).device
    print(f"\nModel loaded on: {device}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
    
    apply_compile(moe_model)
    warmup(moe_model,tokenizer)
    return moe_model, tokenizer

# generation
def build_prompt(user_message: str, tokenizer, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Apply Qwen2.5's chat template correctly.
    """
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float  = DEFAULT_TEMPERATURE,
    top_p: float        = DEFAULT_TOP_P,
    top_k: int          = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    system_prompt: str  = SYSTEM_PROMPT,
    skip_formatting: bool = False,  # fix(server.py): same as generate_stream() above
    result_info: dict = None,      # fix(server.py): same as generate_stream() above
) -> str:
    """
    Run generation on a single prompt string.
    Returns only the generated text (not the input prompt).
    """
    device = next(model.parameters()).device
 
    formatted = prompt if skip_formatting else build_prompt(prompt, tokenizer, system_prompt)
    inputs    = tokenizer(formatted, return_tensors="pt").to(device)
 
    input_len = inputs["input_ids"].shape[1]
 
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    outputs = model.generate(**inputs, **gen_kwargs)

    if result_info is not None:
        gen_ids_for_info = outputs[0][input_len:]
        last_tok = gen_ids_for_info[-1].item() if len(gen_ids_for_info) > 0 else None
        result_info["prompt_tokens"] = input_len
        result_info["completion_tokens"] = len(gen_ids_for_info)
        result_info["finish_reason"] = "stop" if last_tok == tokenizer.eos_token_id else "length"
 
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float  = DEFAULT_TEMPERATURE,
    top_p: float        = DEFAULT_TOP_P,
    top_k: int          = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    system_prompt: str  = SYSTEM_PROMPT,
) -> list[str]:
    """
    Batch generation — more efficient when you have multiple prompts.
    Returns a list of generated strings, one per prompt.
    """
    device = next(model.parameters()).device
 
    formatted = [build_prompt(p, tokenizer, system_prompt) for p in prompts]
    inputs    = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
 
    input_len = inputs["input_ids"].shape[1]
 
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    outputs = model.generate(**inputs, **gen_kwargs)
 
    results = []
    for i, out in enumerate(outputs):
        gen_tokens = out[input_len:]
        results.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
 
    return results

def run_sanity_check(model, tokenizer):
    """
    Three quick test prompts to verify the model is generating correctly
    before you run anything real.
    """
    print("\n" + "="*60)
    print("SANITY CHECK")
    print("="*60)
 
    test_cases = [
        "Write a Python function that returns the nth Fibonacci number.",
        "Write a Python function to check if a string is a palindrome.",
        "What does the Python `zip()` function do? Give an example.",
    ]
 
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {prompt}")
        print("-" * 40)
        result = generate(model, tokenizer, prompt, max_new_tokens=256)
        print(result)
 
    print("\n" + "="*60)
    print("Sanity check done.")
    print("="*60 + "\n")


def interactive_loop(model, tokenizer, args):
    """
    REPL-style interactive generation.
    Type your prompt, get output. Type 'exit' or Ctrl+C to quit.
    """
    print("\nInteractive mode. Type 'exit' to quit, 'clear' to reset.")
    print("Generation params: max_new_tokens={}, temperature={}, top_p={}\n".format(
        args.max_new_tokens, args.temperature, args.top_p
    ))
 
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
 
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break
 
        result = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{result}\n")

def main():

    parser=argparse.ArgumentParser(description="Malora inference script")          

    parser.add_argument(
        "--folder", required=True,
        help="HF subfolder inside godofwar1007/moelora, e.g. 'malora_opencode_50k_1ep/checkpoint-1200'"
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Single prompt to run (non-interactive mode)"
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Path to a file with one prompt per line (batch mode)"
    )
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="Run sanity check and exit"
    )
    parser.add_argument("--max-new-tokens",      type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",         type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p",               type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k",               type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--repetition-penalty",  type=float, default=DEFAULT_REPETITION_PENALTY)

    args=parser.parse_args()

    model, tokenizer = load_model(args.folder)  
    
    run_sanity_check(model,tokenizer)

    if args.sanity_only:
        print("--sanity-only flag set. Exiting.")
        return 

    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"ERROR: prompt file not found: {args.prompt_file}")
            sys.exit(1)
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"\nRunning batch generation on {len(prompts)} prompts...")
        results = generate_batch(
            model, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"\n[{i}] {prompt}")
            print("-" * 60)
            print(result)
        return 
    
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 60)
        result = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(result)
        return
    
    interactive_loop(model,tokenizer,args)

if __name__ == "__main__":
    main()