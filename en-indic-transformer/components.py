from typing import Literal
import torch
from torch import nn


class MLP(nn.Module):
    """Experimental"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    """Experiemental, Need to test and modify."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        attn_type: Literal["self-attention", "cross-attention"] = "self-attention",
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads
        self.attn_type = attn_type

        self.wq = nn.Linear(in_dim, out_dim, bias=bias)
        self.wk = nn.Linear(in_dim, out_dim, bias=bias)
        self.wv = nn.Linear(in_dim, out_dim, bias=bias)
        self.proj = nn.Linear(out_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # this moves the tensor between devices automatically. Usage: self.mask
        # This creates a matrix of (context_length, context_length) with upper triangle
        # filled with True. This is for masking in causal attention during training and
        # during first iteration during inference(I think).
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
            ),
        )

        # Experimental - can be used later for key-value caching
        self.inference = False

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Forward method"""
        if not self.inference:
            return self.train_attn(x)
        # If not training, then check attention type whether to use cross-attention or self-attention
        # for inference.
        elif self.attn_type == "cross-attn":
            return self.cross_attn(x, y)
        return self.inference_attn(x)  # Self-attention by default

    def cross_attn(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Need to implement"""
        return x

    def inference_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Need to implement"""
        return x

    def train_attn(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Experimental"""
        b, context, _emb_dim = x.shape  # (b, context, in_dim)

        queries = self.wq(x)  # (b, context, out_dim)

        if y is not None:
            keys = self.wk(y)  # (b, context, out_dim)
            values = self.wv(y)  # (b, context, out_dim)

        else:
            keys = self.wk(x)  # (b, context, out_dim)
            values = self.wv(x)  # (b, context, out_dim)

        # Reshape queries, keys, values for multi-head attention.
        # (b, context, out_dim) -> (b, num_heads, context, head_dim)
        queries = queries.view(b, self.num_heads, context, self.head_dim)
        keys = keys.view(b, self.num_heads, context, self.head_dim)
        values = values.view(b, self.num_heads, context, self.head_dim)

        # (b, num_heads, context, head_dim) @ (b, num_heads, head_dim, context)
        # => (b, num_heads, context, context)
        attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim**0.5)

        # causal attention. Fill all the upper traingle matrix with torch.inf,
        # so that softmax can output 0.
        attn_scores = attn_scores.masked_fill(
            getattr(self, "mask")[:context, :context], -torch.inf
        )  # self.mask[:context, :context] -> Get the values/ sub-matrix till the current context.

        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, context, context) @ (b, num_heads, context, head_dim)
        # => (b, num_heads, context, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, context, head_dim) => (b, context, out_dim)
        context_vec = context_vec.view(b, context, self.num_heads * self.head_dim)

        context_vec = self.proj(context_vec)

        return context_vec


class DecoderLayer:
    pass


class Decoder:
    pass


class EnocoderLayer:
    pass


class Encoder:
    pass


class Transformer:
    pass


class DecoderOnly:
    pass
