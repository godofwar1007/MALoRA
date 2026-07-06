"""
MALoRA Model Configuration

Extends Qwen2Config with MoE + Asymmetric LoRA (MALoRA) parameters.

Architecture recap (per Wang et al., 2024 — MALoRA):
    MoE-LoRA expert:   delta_W_t = B_t @ A_t
    MALoRA expert:     delta_W_t = B_bar_t @ P_t @ S_A

    - S_A     : layer-shared down-projection subspace (one per gate/up/down per layer),
                shape [shared_rank, in_features]. Replaces the per-expert A_t entirely.
    - P_t     : tiny private coefficient matrix per expert, shape [expert_rank, shared_rank].
                This is what used to be a full A_t — now just projects within S_A's subspace.
    - B_bar_t : per-expert up-projection, shape [out_features, expert_rank]. Rank EXPANDED
                relative to vanilla MoE-LoRA's B_t — the capacity saved by sharing A is
                reallocated here, since experts showed much more divergence in B than in A.

Attention LoRA (Q/K/V/O) is unrelated to the MALoRA decomposition — the paper's
contribution is specific to the MoE/MLP expert side. Attention stays standard LoRA.
"""

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config


class LoraMoeConfig(Qwen2Config):
    r"""
    Configuration class for MALoRA (Mixture of Asymmetric LoRA) models.

    Extends Qwen2Config with parameters for:
    - MoE routing (num_local_experts, num_experts_per_tok)
    - MALoRA MLP expert decomposition (shared_rank / S_A, expert_rank / P_t + B_bar_t, experts_scale)
    - Attention LoRA adapters (attention_rank) — standard LoRA, untouched by MALoRA
    - Load balancing (router_aux_loss_coef)

    Args:
        shared_rank: Rank d of the layer-shared down-projection subspace S_A.
            One S_A is created per projection type (gate/up/down) per layer, and
            reused by every expert in that layer. This replaces the independent
            per-expert A_t of vanilla MoE-LoRA. Default 16.
        expert_rank: Rank r_bar of each expert's private up-projection B_bar_t
            (and the row-dim of its coefficient matrix P_t). This is the EXPANDED
            rank — capacity reallocated from the now-shared down-projection.
            Should be >= shared_rank for the asymmetry to make sense. Default 16.
        attention_rank: Rank of standard LoRA matrices for attention Q/K/V/O.
            Independent of the MALoRA MLP decomposition above. Default 32.
        experts_scale: Scaling factor applied to LoRA/MALoRA outputs. Default 1.0.
        num_experts_per_tok: Top-k experts activated per token. Default 2.
        num_local_experts: Total number of experts per MoE layer. Default 8.
        output_router_logits: Return router logits for aux loss. Default False.
        router_aux_loss_coef: Load balancing loss weight. Default 0.001.
        use_attention_lora: Whether to apply LoRA to attention layers. Default True.
    """

    def __init__(
        self,
        shared_rank: int = 16,
        expert_rank: int = 16,
        attention_rank: int = 32,
        experts_scale: float = 1.0,
        experts_dropout: float = 0.05,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        use_attention_lora: bool = True,
        **kwargs,
    ):
        # MALoRA MLP expert params — replaces old single `experts_rank`
        # shared_rank  -> dimension d of S_A (shared across all experts in a layer)
        # expert_rank  -> dimension r_bar of P_t (rows) and B_bar_t (cols), private per expert
        self.shared_rank = shared_rank
        self.expert_rank = expert_rank
        self.experts_scale = experts_scale
        # single source of truth for MALoRA expert dropout — used both by
        # MALoRALinear.forward() (down_lora's path, which computes S_A
        # itself) and by DispatchMoERouter's hoisted gate_SA/up_SA
        # projection (which can't call a per-expert dropout, since it runs
        # ONCE for all tokens before dispatch — see DispatchMoERouter for
        # why these must use the same rate rather than two independently
        # hardcoded values that could silently drift apart).
        self.experts_dropout = experts_dropout

        # Attention LoRA params — standard LoRA, not part of the MALoRA decomposition
        self.attention_rank = attention_rank
        self.use_attention_lora = use_attention_lora

        # MoE routing params
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(**kwargs)