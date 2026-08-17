"""PyTorch port of MagicLens's custom transformer building blocks.

Ported from the official JAX/Flax release (see
magiclens_reference/layers.py for the source of truth). Three details
here are non-standard and must match exactly for the parity check:

1. LayerNorm applies `(1 + scale)`, not `scale` (see MLLayerNorm).
2. Attention logits are soft-capped before softmax: `50 * tanh(logits/50)`.
3. The pooling attention (only) uses a learned per-dimension query scale
   instead of the usual `1/sqrt(head_dim)`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOGIT_CAP = 50.0
_R_SOFTPLUS_0 = 1.442695041


class MLLayerNorm(nn.Module):
    """LayerNorm with MagicLens's `(1 + scale)` convention."""

    def __init__(self, dim: int, eps: float = 1e-6, use_scale: bool = True, use_bias: bool = True):
        super().__init__()
        self.eps = eps
        self.use_scale = use_scale
        self.use_bias = use_bias
        if use_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(var + self.eps)
        if self.use_scale:
            normed = normed * (1 + self.scale)
        if self.use_bias:
            normed = normed + self.bias
        return normed


class MLMultiHeadAttention(nn.Module):
    """Multi-head dot-product attention with logit capping and optional
    per-dimension query scaling (both non-standard vs. `nn.MultiheadAttention`).

    `dim_per_head` is independent of `num_heads` in MagicLens (the pooling
    attention projects to a *larger* total dim than `input_dim`), so it must
    be passed explicitly rather than derived as `input_dim // num_heads`.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dim_per_head: int,
        use_bias: bool = True,
        use_per_dim_scale: bool = False,
        logit_cap: float = _LOGIT_CAP,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.use_per_dim_scale = use_per_dim_scale
        self.logit_cap = logit_cap
        proj_dim = num_heads * dim_per_head

        self.q_proj = nn.Linear(input_dim, proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(input_dim, proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(input_dim, proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(proj_dim, input_dim, bias=use_bias)

        if use_per_dim_scale:
            self.per_dim_scale = nn.Parameter(torch.ones(dim_per_head))

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor):
        B, Tq, _ = q_in.shape
        Tk = k_in.shape[1]
        H, Dh = self.num_heads, self.dim_per_head

        q = self.q_proj(q_in).view(B, Tq, H, Dh)
        k = self.k_proj(k_in).view(B, Tk, H, Dh)
        v = self.v_proj(v_in).view(B, Tk, H, Dh)

        if self.use_per_dim_scale:
            scale = (_R_SOFTPLUS_0 / (Dh ** 0.5)) * F.softplus(self.per_dim_scale)
            q = q * scale
        else:
            q = q * (Dh ** -0.5)

        logits = torch.einsum("bthd,bshd->bhts", q, k)
        logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        probs = torch.softmax(logits, dim=-1)

        out = torch.einsum("bhts,bshd->bthd", probs, v)
        out = out.reshape(B, Tq, H * Dh)
        return self.out_proj(out), probs


class MLTransformerFFN(nn.Module):
    """Pre-LN feed-forward block with its own residual connection."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = None, use_bias: bool = True, add_skip_connection: bool = True):
        super().__init__()
        output_dim = output_dim or input_dim
        self.ln = MLLayerNorm(input_dim)
        self.ffn1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.ffn2 = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        self.add_skip_connection = add_skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        x = F.relu(self.ffn1(x))
        x = self.ffn2(x)
        if self.add_skip_connection:
            x = x + residual
        return x


class MLTransformerLayer(nn.Module):
    """One block: Pre-LN self-attention (+residual) -> Pre-LN FFN (+residual)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, use_bias: bool = True, add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super().__init__()
        self.add_skip_connection = add_skip_connection
        self.layer_norm = MLLayerNorm(input_dim)
        self.self_attention = MLMultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dim_per_head=input_dim // num_heads,
            use_bias=use_bias,
            use_per_dim_scale=use_per_dim_scale,
        )
        self.ff_layer = MLTransformerFFN(input_dim, hidden_dim, input_dim, use_bias, add_skip_connection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm)
        if self.add_skip_connection:
            attn_out = attn_out + x
        return self.ff_layer(attn_out)


class MLStackedTransformer(nn.Module):
    """MagicLens's `multimodal_encoder`: N stacked MLTransformerLayer blocks."""

    def __init__(self, num_layers: int, num_heads: int, input_dim: int, hidden_dim: int, use_bias: bool = True, add_skip_connection: bool = True, use_per_dim_scale: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            MLTransformerLayer(input_dim, hidden_dim, num_heads, use_bias, add_skip_connection, use_per_dim_scale)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLAttenTokenPoolingLayer(nn.Module):
    """MagicLens's `contrastive_multimodal_pooler`: a learned query attends
    over the fused sequence and pools it down to `num_query_tokens` (=1)."""

    def __init__(
        self,
        input_dim: int,
        query_dim: int = None,
        hidden_dim: int = 0,
        num_heads: int = 1,
        num_query_tokens: int = 1,
        use_bias: bool = True,
        use_per_dim_scale: bool = True,
    ):
        super().__init__()
        query_dim = query_dim or input_dim
        ff_hidden_dim = hidden_dim if hidden_dim > 0 else 4 * input_dim
        dim_per_head = ff_hidden_dim // num_heads

        self.pool_attn = MLMultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dim_per_head=dim_per_head,
            use_bias=use_bias,
            use_per_dim_scale=use_per_dim_scale,
        )
        self.pool_attn_ln = MLLayerNorm(query_dim, eps=1e-6)
        self.pooling_attn_query = nn.Parameter(torch.empty(num_query_tokens, query_dim))
        nn.init.normal_(self.pooling_attn_query, std=0.02)  # overwritten by loaded weights

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        B = embeds.shape[0]
        query = self.pooling_attn_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, embeds, embeds)
        return self.pool_attn_ln(pooled)
