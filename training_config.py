from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:

    # ── MALoRA architecture parameters ────────────────────────────────────────
    # SHARED_RANK -> d, rank of S_A (shared subspace, one per gate/up/down per
    #                layer, reused by all experts)
    # EXPERT_RANK -> r_bar, expanded per-expert rank used by both P_t (private
    #                coefficient matrix) and B_bar_t (private up-projection)
    # These are independent — asymmetric configs are the point of MALoRA.
    # Defaulted equal here as a starting point; tune based on trainable param
    # count measurements rather than the paper's lambda formula (doesn't
    # translate cleanly to Qwen2.5-3B's dimensions — see prior discussion).
    SHARED_RANK: int         = 16         # d — shared subspace rank (S_A)
    EXPERT_RANK: int         = 16         # r_bar — expanded per-expert rank (P_t, B_bar_t)
    ATTENTION_RANK: int      = 32         # attention LoRA rank — unrelated to MALoRA decomposition
    EXPERTS_SCALE: float     = 1.0
    EXPERTS_DROPOUT: float   = 0.05       # single source of truth shared by MALoRALinear
                                           # AND DispatchMoERouter's gate/up projection
    NUM_EXPERTS_PER_TOK: int = 2          # top-2 routing
    NUM_EXPERTS: int         = 8          # total experts
    ROUTER_AUX_COEF: float   = 0.001     # was 0.01 — too high causes over-regularization

    # ── Training parameters ───────────────────────────────────────────────────
    # These match the MoE-LoRA run that first beat baseline on all metrics.
    # The two critical fixes vs all prior runs:
    #   LR: 2e-4 → 1e-5  (was 40x too high — thrashed alignment faster than
    #                      the model could absorb new signal)
    #   CONTEXT_LENGTH: 1024 → 2048  (matches paper setup, captures full
    #                                  solutions without mid-truncation)
    SEED: int            = 42
    NUM_EPOCHS: int      = 1
    TRAIN_BATCH: int     = 16
    EVAL_BATCH: int      = 16
    CONTEXT_LENGTH: int  = 2048
    LR: float            = 1e-5
    EVAL_STEPS: int      = 200
    GRAD_ACCUM: int      = 2
    MAX_STEPS: int       = -1             # -1 for full training
    LOGGING_STEPS: int   = 25

    # ── Model parameters ──────────────────────────────────────────────────────
    USE_8BIT_ADAM: bool  = True
    MIXED_PRECISION: str = "bf16"
    QUANTIZE: bool       = False          # full bf16 — H100 has headroom
    MODEL_ID: str        = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # ── Checkpoint / logging ──────────────────────────────────────────────────
    RESUME_FROM: str | None   = None
    OUTPUT_DIR: str           = "./outputs/malora"
    NUM_CHECKPOINT_LIMIT: int = 2
    LOGDIR: str               = "./logs"
    RUN_NAME: str             = "malora-opencode-run1"
    PROJECT_NAME: str         = "malora"