# ── config.py ─────────────────────────────────────────────────────────────────
# Dataset configuration for MALoRA training.
# TrainingConfig (hyperparameters) lives in training_config.py — not here.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CommitPackFTConfig:
    """
    CommitPackFT-specific config — ignored by all other loaders.
    Kept here so the DatasetConfig dataclass can reference it without
    breaking if someone re-enables commitpackft in DATASET_CONFIGS later.
    """
    min_samples    : int            = 4000
    priority_langs : set            = field(default_factory=lambda: {"python", "java", "javascript", "cpp"})
    fmt_split      : tuple          = (0.6, 0.4)
    lang_counts    : Optional[dict] = None
    langs_to_keep  : set            = field(default_factory=lambda: {
        "python", "javascript", "typescript", "go",
        "java", "c#", "shell", "css", "scss", "cpp",
    })


@dataclass
class DatasetConfig:
    name    : str
    weight  : float
    commitpackft: CommitPackFTConfig = field(default_factory=CommitPackFTConfig)


# ── DATASET CONFIG ─────────────────────────────────────────────────────────────
# nvidia/OpenCodeInstruct — execution-verified coding samples.
#
# WHY THIS DATASET:
# Every sample was actually executed and verified to pass unit tests.
# Quality filter applied inside the loader:
#     average_test_score >= 0.8  (passes 80%+ unit tests)
#     llm_judgement avg  >= 4.5  (LLM scores all 3 criteria >= 4.5/5)
#
# HOW TO SWITCH DATASETS:
# Comment out the opencode_instruct line and uncomment/add others.
# Weights must sum to 1.0 — the assert below enforces this.
# Each name must have a corresponding loader in LOADERS in main.py.

TOTAL_SAMPLES  = 50000   # total samples across all datasets combined
EVAL_FRACTION  = 0.05    # 5% held out for eval → 47500 train / 2500 eval

DATASET_CONFIGS = [
    DatasetConfig(name="opencode_instruct", weight=1.0),

    # ── other datasets available (commented out) ──────────────────────────
    # DatasetConfig(name="magicoder",           weight=0.55),  # ise-uiuc/Magicoder-Evol-Instruct-110K
    # DatasetConfig(name="magicoder_oss",       weight=0.28),  # ise-uiuc/Magicoder-OSS-Instruct-75K
    # DatasetConfig(name="python_instructions", weight=0.17),
    # DatasetConfig(name="code_contests",       weight=0.15),
    # DatasetConfig(name="codealpaca",          weight=0.15),
    # DatasetConfig(name="leetcode",            weight=0.15),
    # DatasetConfig(name="code_feedback",       weight=0.10),
    # DatasetConfig(name="code_search_net",     weight=0.10),
    # DatasetConfig(name="sql_context",         weight=0.10),
]

assert abs(sum(d.weight for d in DATASET_CONFIGS) - 1.0) < 1e-6, \
    "DATASET_CONFIGS weights must sum to 1.0"