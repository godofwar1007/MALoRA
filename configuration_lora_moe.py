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