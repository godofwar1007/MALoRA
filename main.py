# ── main.py ───────────────────────────────────────────────────────────────────

import os
import random

from datasets import Dataset, interleave_datasets
from transformers import AutoTokenizer

from training_config import TrainingConfig          # TrainingConfig lives here, NOT in config.py
from config import DATASET_CONFIGS, TOTAL_SAMPLES, EVAL_FRACTION

# ── active loaders ─────────────────────────────────────────────────────────────
# Only import loaders that are actually used in DATASET_CONFIGS.
# Add imports here when enabling new datasets in config.py.
from loaders.opencode_instruct import OpenCodeInstructLoader

# ── commented-out loaders (available if you switch datasets in config.py) ─────
# from loaders.codecontestsloader import CodeContestsLoader
# from loaders.codealpacaloader   import CodeAlpacaLoader
# from loaders.python_instruct    import PythonInstructionsLoader
# from loaders.code_search_net    import CodeSearchNetLoader
# from loaders.codefeedback       import CodeFeedbackLoader
# from loaders.leetcode_loader    import LeetCodeLoader
# from loaders.magic_coder_oss    import MagicoderOSSLoader
# from loaders.sql                import SqlCreateContextLoaderBi

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("WARNING: HF_TOKEN not set — gated datasets will fail")

# ── tokenizer ──────────────────────────────────────────────────────────────────
conf      = TrainingConfig()
tokenizer = AutoTokenizer.from_pretrained(conf.MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── loaders ────────────────────────────────────────────────────────────────────
# Add an entry here for every dataset name used in DATASET_CONFIGS.
# If a name appears in DATASET_CONFIGS but not here, make_dataset() raises
# a clear KeyError immediately rather than silently skipping it.
LOADER_ARGS = dict(tokenizer=tokenizer, context_length=conf.CONTEXT_LENGTH)

LOADERS = {
    "opencode_instruct": OpenCodeInstructLoader(**LOADER_ARGS),

    # ── uncomment to re-enable ─────────────────────────────────────────────
    # "code_contests":       CodeContestsLoader(**LOADER_ARGS),
    # "codealpaca":          CodeAlpacaLoader(**LOADER_ARGS),
    # "python_instructions": PythonInstructionsLoader(**LOADER_ARGS),
    # "magicoder":           MagicoderOSSLoader(**LOADER_ARGS),
    # "code_feedback":       CodeFeedbackLoader(**LOADER_ARGS),
    # "leetcode":            LeetCodeLoader(**LOADER_ARGS),
    # "code_search_net":     CodeSearchNetLoader(**LOADER_ARGS),
    # "sql_context":         SqlCreateContextLoaderBi(**LOADER_ARGS),
}


# ── main ───────────────────────────────────────────────────────────────────────
def make_dataset():
    all_train = []
    all_eval  = []

    for cfg in DATASET_CONFIGS:
        total_alloc = int(TOTAL_SAMPLES * cfg.weight)
        eval_alloc  = max(1, int(total_alloc * EVAL_FRACTION))
        train_alloc = total_alloc - eval_alloc

        print(
            f"\n[{cfg.name}] "
            f"total={total_alloc} | train={train_alloc} | eval={eval_alloc}"
        )

        if cfg.name not in LOADERS:
            raise KeyError(
                f"Missing loader for dataset '{cfg.name}'. "
                f"Add it to LOADERS in main.py and uncomment its import."
            )

        raw = LOADERS[cfg.name].collect(total_alloc)
        n   = len(raw["input_ids"])

        if n == 0:
            raise RuntimeError(
                f"{cfg.name} produced zero samples.\n"
                f"If using OpenCodeInstruct and the filter is too strict, "
                f"lower MIN_LLM_SCORE in loaders/opencode_instruct.py from 4.5 to 4.0."
            )

        print(f"[{cfg.name}] collected {n} samples")

        # shuffle BEFORE train/eval split so eval isn't just the tail end
        shuffled = Dataset.from_dict(raw).shuffle(seed=42)
        raw = {
            "input_ids":      shuffled["input_ids"],
            "attention_mask": shuffled["attention_mask"],
            "labels":         shuffled["labels"],
        }

        actual_eval  = min(eval_alloc, n)
        actual_train = n - actual_eval

        print(
            f"[{cfg.name}] "
            f"collected={n} | train={actual_train} | eval={actual_eval}"
        )

        all_train.append(
            Dataset.from_dict({k: v[:actual_train] for k, v in raw.items()})
        )
        all_eval.append(
            Dataset.from_dict({k: v[actual_train:] for k, v in raw.items()})
        )

    # ── combine across datasets ────────────────────────────────────────────────
    # For single dataset (weight=1.0): interleave is a no-op, just wraps it.
    # For multiple datasets: interleave proportionally by weight, stop when
    # the smallest dataset is exhausted, then shuffle the combined result.
    target_features = all_train[0].features
    all_train = [ds.cast(target_features) for ds in all_train]
    all_eval  = [ds.cast(target_features) for ds in all_eval]

    weights = [cfg.weight for cfg in DATASET_CONFIGS]

    train_dataset = interleave_datasets(
        all_train,
        probabilities=weights,
        seed=42,
        stopping_strategy="first_exhausted",
    ).shuffle(seed=42)

    eval_dataset = interleave_datasets(
        all_eval,
        probabilities=weights,
        seed=42,
        stopping_strategy="first_exhausted",
    ).shuffle(seed=42)

    print(f"\n=== Dataset Summary ===")
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── validation ─────────────────────────────────────────────────────────────
    sample = train_dataset[0]
    assert "input_ids"      in sample
    assert "attention_mask" in sample
    assert "labels"         in sample
    assert len(sample["input_ids"]) == len(sample["labels"]), \
        f"Length mismatch: input_ids={len(sample['input_ids'])} labels={len(sample['labels'])}"
    assert any(x != -100 for x in sample["labels"]), \
        "All labels are -100 — label masking is broken"
    print("Validation passed.")

    # ── save ───────────────────────────────────────────────────────────────────
    # Saves to data/train and data/eval — this is what gemini.py loads from.
    # The old OpenCodeInstruct version saved to data/composite_agentic_sft
    # which caused a path mismatch with gemini.py. Fixed here — one save path,
    # consistent with what gemini.py expects.
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/eval",  exist_ok=True)
    train_dataset.save_to_disk("data/train")
    eval_dataset.save_to_disk("data/eval")
    print("Saved to data/train and data/eval")

    return train_dataset, eval_dataset


if __name__ == "__main__":
    make_dataset()