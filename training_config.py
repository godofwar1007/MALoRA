from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:

    # MALoRA parameters
    # NOTE: the old MoE-LoRA codebase had a single EXPERTS_RANK. MALoRA
    # splits this into two independent values — see configuration_lora_moe.py
    # and peft_experts.py for the full mechanism:
    #   SHARED_RANK -> d, the rank of S_A (one shared subspace per
    #                  gate/up/down per layer, reused by all experts)
    #   EXPERT_RANK -> r_bar, the EXPANDED per-expert rank used by both
    #                  P_t (private coefficient matrix) and B_bar_t
    #                  (private, expanded up-projection)
    # These are independent knobs, not required to be equal — an asymmetric
    # config (e.g. SHARED_RANK < EXPERT_RANK) is the whole point of MALoRA's
    # "capacity reallocated from shared A to expanded B" idea. Defaulted
    # equal here as a reasonable starting point; tune based on actual
    # trainable-param-count measurements rather than the paper's lambda
    # formula, which doesn't translate cleanly to this model's dimensions
    # (see prior discussion — at Qwen2.5-3B's hidden/intermediate sizes,
    # the formula doesn't reliably predict savings or costs).
    SHARED_RANK: int         = 16         # d — shared subspace rank (S_A)
    EXPERT_RANK: int         = 16         # r_bar — expanded per-expert rank (P_t, B_bar_t)
    ATTENTION_RANK: int      = 32         # Attention LoRA rank — unrelated to MALoRA decomposition
    EXPERTS_SCALE: float     = 1.0
    EXPERTS_DROPOUT: float   = 0.05       # single source of truth — shared by MALoRALinear AND
                                           # DispatchMoERouter's hoisted gate/up projection, so they
                                           # can't silently drift apart (see peft_experts.py)
    NUM_EXPERTS_PER_TOK: int = 2          # top-2 routing
    NUM_EXPERTS: int         = 8          # total experts
    ROUTER_AUX_COEF: float   = 0.001      # was 0.01, too high causes over-regularization

    # Training parameters
    SEED: int            = 42
    NUM_EPOCHS: int      = 1
    TRAIN_BATCH: int     = 24
    EVAL_BATCH: int      = 24
    CONTEXT_LENGTH: int  = 1024
    LR: float            = 2e-4
    EVAL_STEPS: int      = 100
    GRAD_ACCUM: int      = 2
    MAX_STEPS: int       = -1             # -1 for real training(full scale)
    LOGGING_STEPS: int   = 25

    # Model parameters
    USE_8BIT_ADAM: bool  = True
    MIXED_PRECISION: str = "bf16"
    QUANTIZE: bool       = False          # full bf16, no 4-bit quantization
    MODEL_ID: str        = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # Logging parameters
    RESUME_FROM: str | None   = None      # set to checkpoint path to resume e.g "./outputs/malora/checkpoint-400"
    OUTPUT_DIR: str           = "./outputs/malora"
    NUM_CHECKPOINT_LIMIT: int = 2
    LOGDIR: str               = "./logs"
    RUN_NAME: str             = "malora-run1"
    PROJECT_NAME: str         = "malora"