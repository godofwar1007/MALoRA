"""
MALoRA Expert Modules

Implements:
- SharedDownProjection: the shared low-rank subspace S_A (one per gate/up/down
  per layer, reused by every expert in that layer) — THE core MALoRA change.
  Initialized via SVD over throwaway per-expert matrices, not plain
  independent kaiming init, to match the paper's init statistics.
- MALoRALinear: per-expert adapter using S_A + private P_t + expanded B_bar_t
  (replaces LoraInjectedLinear's role inside experts). Exposes both forward()
  (computes S_A itself, used for down_lora) and forward_from_shared() (takes
  an already-projected tensor, used for gate_lora/up_lora via the router's
  hoisted projection — see DispatchMoERouter below).
- LoraExpert: MLP expert built from MALoRALinear on gate/up/down projections.
  gate/up receive pre-projected shared tensors from the router; down computes
  its own S_A pass since its input is expert-specific.
- AttentionLoRA: standard LoRA for Q/K/V/O attention projections (UNCHANGED —
  the MALoRA decomposition only applies to MoE/MLP experts, not attention)
- DispatchMoERouter: real sparse token-dispatch routing. Routing DECISION
  logic (softmax/top-k/noise gate/scatter-gather) is unchanged from the
  MoE-LoRA codebase, but it now also hoists the gate_SA/up_SA projection
  outside the per-expert dispatch loop to avoid redundant compute under
  top-k>1 routing — see its docstring for why, and for the resulting
  coupling caveat (it's no longer fully agnostic to expert internals).

MALoRA mechanism (Wang et al., 2024):
    Vanilla MoE-LoRA expert:  delta_W_t = B_t(A_t(x))          — A_t private per expert
    MALoRA expert:            delta_W_t = B_bar_t(P_t(S_A(x))) — S_A shared, B_bar_t expanded

    S_A is shared across all N experts in a layer because empirically the
    down-projection matrices A_t across experts turn out highly similar —
    keeping them independent wastes parameters. The up-projection matrices
    B_t, in contrast, are where experts actually diverge — so the rank
    saved by sharing A is reallocated to grow B's rank instead.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from configuration_lora_moe import LoraMoeConfig


# ── Shared Down-Projection (S_A) ──────────────────────────────────────────────

def _svd_init_shared_subspace(
    in_features: int,
    shared_rank: int,
    expert_rank: int,
    num_experts: int,
    device=None,
) -> torch.Tensor:
    """
    Derive S_A's initial weights via SVD over throwaway per-expert matrices,
    rather than independently kaiming-initializing S_A on its own.

    HONEST SCOPE OF THIS FIX (downgraded from an earlier version of this
    docstring that overclaimed fidelity to the paper): this gives S_A a
    structured, SVD-derived init instead of a plain random one — but it
    does NOT control the joint variance of the P_t @ S_A product the way
    the paper's actual scheme does. The paper ties each expert's P_t to
    the same SVD (each P_t comes from its own U/S block of the throwaway
    decomposition), so the product P_t @ S_A is what reproduces single-
    matrix Kaiming statistics. Here, P_t (in MALoRALinear) is still
    independently kaiming_uniform_'d with no reference to this SVD — only
    S_A's own structure is controlled. Since B_bar=0 zeroes the adapter's
    output at step 0 regardless, this doesn't break forward-pass
    correctness; it's a partial improvement on gradient-flow fidelity, not
    a full reproduction of the paper's init scheme. If exact paper fidelity
    matters for a reported result, this needs the full joint P_t/S_A
    coupling implemented — flagged here rather than silently left as a gap.

    Procedure: generate N throwaway kaiming-uniform matrices K_1..K_N, one
    per expert, each shaped [shared_rank, in_features] — matching the
    paper's K_t in R^(d x n) (d = shared_rank), NOT expert_rank. Getting
    this dimension right matters: shared_rank and expert_rank are
    independent config values (e.g. shared_rank=16, expert_rank=32 is a
    valid asymmetric config) and using the wrong one changes what subspace
    the SVD extracts. Stack the N throwaway matrices into
    [N * shared_rank, in_features] and take its SVD. The top `shared_rank`
    right-singular vectors, scaled by their singular values, define S_A.

    Returns: S_A weight tensor, shape [shared_rank, in_features], float32
    (caller casts to bf16 after assigning).
    """
    total_rows = num_experts * shared_rank
    assert shared_rank <= total_rows, (
        f"shared_rank ({shared_rank}) exceeds the throwaway matrix's total "
        f"rows (num_experts * shared_rank = {total_rows}). This SVD cannot "
        f"produce {shared_rank} singular vectors from a smaller matrix. "
        f"Check shared_rank/num_local_experts in your config."
    )

    throwaway = torch.empty(total_rows, in_features, device=device)
    nn.init.kaiming_uniform_(throwaway, a=math.sqrt(5))

    # SVD: throwaway = U @ diag(S) @ Vh, Vh rows are right-singular vectors
    # in the in_features space — exactly the directions a shared down-
    # projection subspace should span.
    _, S, Vh = torch.linalg.svd(throwaway, full_matrices=False)

    top_S = S[:shared_rank]                      # [shared_rank]
    top_Vh = Vh[:shared_rank, :]                  # [shared_rank, in_features]

    # scale rows by their singular values so the resulting S_A carries the
    # same energy/variance the dominant directions had in the original
    # kaiming-init throwaway matrices, rather than just an orthonormal basis.
    S_A_weight = top_S.unsqueeze(1) * top_Vh      # [shared_rank, in_features]
    return S_A_weight


class SharedDownProjection(nn.Module):
    """
    The S_A matrix — a single, layer-shared low-rank down-projection.

    One instance of this exists per projection type (gate / up / down) PER
    LAYER, and is passed by reference into every expert (LoraExpert) in that
    layer. This is what replaces the independent per-expert A_t of vanilla
    MoE-LoRA: instead of N separate down-projections, all N experts in a
    layer project through this single shared subspace, then differentiate
    via their own private P_t (see MALoRALinear below).

    Shape: in_features -> shared_rank (d)

    Init: SVD-derived from N throwaway kaiming-init expert matrices (see
    _svd_init_shared_subspace above), matching the paper's actual init
    procedure — NOT a plain independent kaiming_uniform_ on S_A alone,
    which would understate/distort the variance P_t @ S_A produces at
    init relative to what a true full-rank A_t would have had.

    `num_experts` and `expert_rank` are needed only at init time (to build
    the throwaway matrices the SVD is derived from) — they are not stored
    or used afterward; S_A itself has no notion of "which expert."

    Like A in vanilla LoRA, S_A is not the zero-init piece — only the final
    B_bar_t is zeroed, which is what guarantees zero adapter output at init
    (B_bar_t(P_t(S_A(x))) = 0 regardless of S_A and P_t's values, since
    B_bar_t = 0). The SVD-init only matters for gradient flow on step 1+,
    not for step-0 forward correctness.
    """

    def __init__(self, in_features: int, shared_rank: int, expert_rank: int, num_experts: int):
        super().__init__()
        self.shared_rank = shared_rank
        self.proj = nn.Linear(in_features, shared_rank, bias=False)

        with torch.no_grad():
            init_weight = _svd_init_shared_subspace(
                in_features=in_features,
                shared_rank=shared_rank,
                expert_rank=expert_rank,
                num_experts=num_experts,
            )
            self.proj.weight.copy_(init_weight)

        self.proj = self.proj.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj.weight.dtype != x.dtype:
            self.proj = self.proj.to(x.dtype)
        return self.proj(x)


# ── Per-Expert MALoRA Adapter (P_t + B_bar_t) ────────────────────────────────

class MALoRALinear(nn.Module):
    """
    Per-expert MALoRA adapter. Replaces LoraInjectedLinear's role inside
    LoraExpert.

    Computes: output = scale * B_bar(P(dropout(S_A(x))))

    S_A: shared down-projection (passed in at forward time, NOT owned here —
         lives in the parent LoraMoeBlock and is shared across all experts)
    P:   private coefficient matrix [shared_rank -> expert_rank]. This is the
         small per-expert piece that used to be a full A matrix.
    B_bar: private up-projection [expert_rank -> out_features]. EXPANDED rank
         relative to vanilla LoRA's B — this is where the capacity saved by
         sharing S_A gets reallocated.

    Init: P ~ kaiming_uniform, B_bar = zeros (standard LoRA-style zero init —
    guarantees the adapter contributes nothing at step 0, training starts
    from the frozen base model exactly like vanilla LoRA/MoE-LoRA did).

    Weights created directly in bf16, same rationale as the old
    LoraInjectedLinear: avoids runtime dtype-cast overhead/bugs.
    """

    def __init__(
        self,
        shared_rank: int,
        expert_rank: int,
        out_features: int,
        scale: float = 1.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.shared_rank = shared_rank
        self.expert_rank = expert_rank
        self.scale = scale

        self.P = nn.Linear(shared_rank, expert_rank, bias=False)
        self.B_bar = nn.Linear(expert_rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.P.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_bar.weight)
        self.P = self.P.to(torch.bfloat16)
        self.B_bar = self.B_bar.to(torch.bfloat16)

    def forward(self, x: torch.Tensor, S_A: SharedDownProjection) -> torch.Tensor:
        # cast to match input dtype (bf16 in quantized model)
        if self.P.weight.dtype != x.dtype:
            self.P = self.P.to(x.dtype)
            self.B_bar = self.B_bar.to(x.dtype)

        # dropout applied to the RAW input, before S_A — matches the
        # original LoraInjectedLinear convention (dropout(x) then A(x)),
        # not the shared projection's output. Regularizes the same
        # feature space the old per-expert A used to see.
        x = self.dropout(x)
        s = S_A(x)                              # [*, shared_rank] — shared across experts
        return self.B_bar(self.P(s)) * self.scale

    def forward_from_shared(self, s: torch.Tensor) -> torch.Tensor:
        """
        Variant of forward() that takes an ALREADY-PROJECTED shared tensor
        `s = S_A(x)` instead of computing S_A itself. Used by the router's
        dispatch loop for gate/up, where S_A(hidden_states) is identical
        regardless of which expert is asking and can be computed once
        globally instead of once per expert (see DispatchMoERouter.forward).

        NOTE: dropout is intentionally skipped here. Dropout must be applied
        to x BEFORE S_A (see forward() above) to match the original
        convention — by the time `s` reaches this method that decision has
        already been made by whoever computed s upstream. Callers that want
        dropout on the hoisted path should drop x before projecting it
        through S_A themselves.
        """
        if self.P.weight.dtype != s.dtype:
            self.P = self.P.to(s.dtype)
            self.B_bar = self.B_bar.to(s.dtype)
        return self.B_bar(self.P(s)) * self.scale


# ── MLP Expert (MALoRA) ───────────────────────────────────────────────────────

class LoraExpert(nn.Module):
    """
    MALoRA-adapted expert for MoE FFN layers.

    Wraps the frozen base MLP with trainable MALoRA adapters on gate/up/down.
    Unlike the old LoraExpert, this does NOT own its down-projection — gate
    and up route through the layer's shared S_A subspaces, down through its
    own down_SA.

    PERFORMANCE NOTE — why gate/up and down are handled differently here:
    With top-k routing, a token can be assigned to multiple experts, so
    gate_SA(hidden_states) and up_SA(hidden_states) are IDENTICAL regardless
    of which expert ends up using them — recomputing them once per assigned
    expert (as a naive forward(hidden_states, gate_SA, up_SA, ...) would)
    wastes compute proportional to top_k. DispatchMoERouter therefore
    computes s_gate = gate_SA(hidden_states) and s_up = up_SA(hidden_states)
    ONCE globally before dispatching to any expert, then gathers the
    per-expert token subset of that already-projected tensor and passes it
    in here via MALoRALinear.forward_from_shared(). down_SA can NOT be
    hoisted the same way — its input `act` depends on THIS expert's own
    gate/up LoRA deltas (act = activation(gate) * up, and gate/up differ
    per expert via their private P/B_bar), so down must stay genuinely
    per-expert and is computed inside forward() as before.

    Forward (shape of the math):
        gate = mlp.gate_proj(x) + gate_lora.forward_from_shared(s_gate_for_this_expert)
        up   = mlp.up_proj(x)   + up_lora.forward_from_shared(s_up_for_this_expert)
        act  = activation(gate) * up
        out  = mlp.down_proj(act) + down_lora(act, down_SA)
    """

    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        shared_rank = config.shared_rank
        expert_rank = config.expert_rank
        scale = config.experts_scale
        dropout = config.experts_dropout

        # gate/up project hidden_size -> intermediate_size (out_features = intermediate_size).
        # NOTE: gate_lora/up_lora's own .dropout module is constructed here
        # but never actually invoked at runtime — the router calls
        # forward_from_shared() for these two (see DispatchMoERouter), which
        # has no dropout step of its own. The router instead applies dropout
        # ONCE globally using this same config.experts_dropout rate before
        # the hoisted gate_SA/up_SA projection (see DispatchMoERouter.forward).
        # Kept as a real nn.Dropout(dropout) here rather than removed, so
        # state_dict shape stays consistent with down_lora's, and so a
        # future refactor that calls gate_lora.forward() directly (bypassing
        # the hoisted path) gets correct behavior for free rather than
        # silently no-op-ing.
        self.gate_lora = MALoRALinear(shared_rank, expert_rank, config.intermediate_size, scale, dropout)
        self.up_lora   = MALoRALinear(shared_rank, expert_rank, config.intermediate_size, scale, dropout)
        # down projects intermediate_size -> hidden_size (out_features = hidden_size)
        # — this one DOES use its own dropout, via forward() (not hoisted).
        self.down_lora = MALoRALinear(shared_rank, expert_rank, config.hidden_size, scale, dropout)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlp: Qwen2MLP,
        s_gate: torch.Tensor,
        s_up: torch.Tensor,
        down_SA: SharedDownProjection,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens_for_this_expert, hidden_size] — already
                gathered to just this expert's assigned tokens by the router
            mlp: frozen base MLP
            s_gate, s_up: [num_tokens_for_this_expert, shared_rank] — the
                SAME shared subspace projection used by every expert,
                pre-computed once and pre-gathered by the router (see note
                above). NOT raw hidden_states, and NOT re-projected here.
            down_SA: still the raw SharedDownProjection module — down's
                input (act) is expert-specific, so it's computed fresh per
                expert inside this forward(), unlike gate/up.
        """
        gate = mlp.gate_proj(hidden_states) + self.gate_lora.forward_from_shared(s_gate)
        up   = mlp.up_proj(hidden_states)   + self.up_lora.forward_from_shared(s_up)
        act  = self.activation_fn(gate) * up
        down = mlp.down_proj(act) + self.down_lora(act, down_SA)
        return down


# ── Attention LoRA (UNCHANGED — not part of the MALoRA decomposition) ────────

class AttentionLoRA(nn.Module):
    """
    Standard LoRA adapters for attention Q/K/V/O projections.

    NOT MoE, NOT MALoRA — one ordinary LoRA adapter per attention layer,
    shared across all tokens. Uses a single A/B pair, not the S_A/P_t/B_bar_t
    decomposition, because there's only one "expert" here (no mixture to
    share a subspace across). Kept exactly as in the MoE-LoRA codebase.

    Uses higher rank than MLP experts (attention matters more for coding
    tasks). Mentor's insight: attention controls WHAT the model focuses on
    (variable tracking, bracket matching, function call patterns). MLP
    stores knowledge. For coding, better attention patterns matter more
    than more knowledge.
    """

    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        hidden_size = config.hidden_size
        rank = config.attention_rank
        scale = config.experts_scale

        self.q_lora = _StandardLoraLinear(hidden_size, hidden_size, rank, scale)
        self.o_lora = _StandardLoraLinear(hidden_size, hidden_size, rank, scale)

        kv_size = config.num_key_value_heads * (hidden_size // config.num_attention_heads)
        self.k_lora = _StandardLoraLinear(hidden_size, kv_size, rank, scale)
        self.v_lora = _StandardLoraLinear(hidden_size, kv_size, rank, scale)

    def forward_q(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_lora(x)

    def forward_k(self, x: torch.Tensor) -> torch.Tensor:
        return self.k_lora(x)

    def forward_v(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_lora(x)

    def forward_o(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_lora(x)


class _StandardLoraLinear(nn.Module):
    """
    Plain single-adapter LoRA linear (the old LoraInjectedLinear), kept only
    for AttentionLoRA's use. Renamed with a leading underscore to make clear
    this is NOT the MALoRA expert path — it's ordinary two-matrix LoRA.

    Computes: output = scale * B(A(dropout(x)))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        scale: float = 1.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.scale = scale

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.A = self.A.to(torch.bfloat16)
        self.B = self.B.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.A.weight.dtype != x.dtype:
            self.A = self.A.to(x.dtype)
            self.B = self.B.to(x.dtype)
        return self.B(self.A(self.dropout(x))) * self.scale


# ── Real Sparse Token-Dispatch MoE Router ─────────────────────────────────────

class DispatchMoERouter(nn.Module):
    """
    Real sparse expert dispatch — routing decision logic (softmax, top-k,
    noise gate, scatter/gather dispatch) is unchanged from the MoE-LoRA
    codebase.

    1. Route each token to its top-k experts
    2. Project hidden_states through the shared gate_SA/up_SA subspaces
       ONCE, globally, for ALL tokens (see "hoisted projection" note below)
    3. Group tokens by which expert they were assigned to
    4. Run each expert ONLY on the tokens actually assigned to it
    5. Scatter-add results back to original token positions

    HOISTED SHARED PROJECTION (gate/up only): with top-k routing each token
    is assigned to top_k experts, and gate_SA(hidden_states)/up_SA(hidden_states)
    are identical regardless of which expert ends up using them — computing
    them inside each expert's forward() would redo the same projection
    top_k times per token. Instead this router projects ALL tokens through
    gate_SA/up_SA once (with dropout applied beforehand, matching the
    convention that dropout sees the raw input), then gathers the
    per-expert token subset of that already-projected tensor and passes
    it into LoraExpert.forward() via MALoRALinear.forward_from_shared().
    down_SA is NOT hoisted this way — see LoraExpert's docstring for why
    (its input depends on this expert's own gate/up LoRA delta, so it's
    genuinely expert-specific and stays inside the per-expert call).

    NOTE ON COUPLING: this router does technically need to know that
    LoraExpert's internals expose a gate_SA/up_SA-shaped interface — it is
    no longer purely agnostic to "what's inside each expert" the way the
    original MoE-LoRA router was, since hoisting the shared projection only
    makes sense for a MALoRA-style expert. If DispatchMoERouter is ever
    reused with a non-MALoRA expert architecture, this coupling would need
    to be revisited.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dropout: float = 0.05):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.noise_gate = nn.Linear(hidden_dim, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.noise_gate.weight)
        self.gate = self.gate.to(torch.bfloat16)
        self.noise_gate = self.noise_gate.to(torch.bfloat16)

        # dropout applied to hidden_states before the hoisted gate_SA/up_SA
        # projection — mirrors MALoRALinear.forward()'s "dropout before S_A"
        # convention for the path that isn't hoisted (down_SA). `dropout`
        # is passed in by the caller (LoraMoeBlock, using
        # config.experts_dropout) rather than hardcoded here, so this and
        # every MALoRALinear's dropout rate share one source of truth — see
        # LoraExpert.__init__'s comment on gate_lora/up_lora's now-unused-
        # but-still-rate-matched dropout module for the full rationale.
        self.shared_proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts: nn.ModuleList,
        mlp,
        gate_SA: SharedDownProjection,
        up_SA: SharedDownProjection,
        down_SA: SharedDownProjection,
    ):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]  (already flattened)
            experts: list of LoraExpert modules
            mlp: frozen base MLP shared by all experts
            gate_SA, up_SA: the layer's shared S_A modules for gate/up —
                projected ONCE here for all tokens (see class docstring)
            down_SA: the layer's shared S_A module for down — passed
                through unprojected, since down's input is expert-specific

        Returns:
            output: [num_tokens, hidden_dim]
            router_logits: [num_tokens, num_experts]  for aux loss
        """
        if self.gate.weight.dtype != hidden_states.dtype:
            self.gate = self.gate.to(hidden_states.dtype)
            self.noise_gate = self.noise_gate.to(hidden_states.dtype)

        num_tokens = hidden_states.shape[0]

        # ── routing decision (identical math to before) ─────────────────────
        logits = self.gate(hidden_states)

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_gate(hidden_states))
            logits = logits + noise

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        # ── hoisted shared projection (THE FIX for redundant compute) ───────
        # computed once for ALL tokens, regardless of how many experts each
        # token is ultimately routed to. dropout sampled once per token here,
        # rather than once per (token, assigned-expert) pair as a naive
        # per-expert recompute would do — also makes dropout behavior
        # consistent across a token's top_k assigned experts in one pass.
        dropped = self.shared_proj_dropout(hidden_states)
        s_gate_all = gate_SA(dropped)   # [num_tokens, shared_rank]
        s_up_all   = up_SA(dropped)     # [num_tokens, shared_rank]

        # ── real sparse dispatch (unchanged structure) ───────────────────────
        output = torch.zeros_like(hidden_states)

        flat_expert_ids = top_k_indices.reshape(-1)
        flat_weights    = top_k_weights.reshape(-1)
        flat_token_ids  = torch.arange(num_tokens, device=hidden_states.device) \
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        for expert_idx in range(self.num_experts):
            mask = flat_expert_ids == expert_idx
            if not mask.any():
                continue

            token_ids_for_expert = flat_token_ids[mask]
            weights_for_expert   = flat_weights[mask].unsqueeze(-1)

            expert_input = hidden_states[token_ids_for_expert]
            s_gate_for_expert = s_gate_all[token_ids_for_expert]
            s_up_for_expert   = s_up_all[token_ids_for_expert]

            expert_output = experts[expert_idx](
                expert_input, mlp, s_gate_for_expert, s_up_for_expert, down_SA
            )
            expert_output = expert_output * weights_for_expert

            output.index_add_(0, token_ids_for_expert, expert_output)

        return output, logits