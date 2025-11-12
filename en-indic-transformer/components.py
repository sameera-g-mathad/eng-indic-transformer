from typing import Literal
import torch
from torch import nn


# My implementation and explainations based on what I have learnt
# from Build a Large Language Model and other youtube videos.


class LayerNorm(nn.Module):
    """
    Implementation follows from the book Build a Large Language Model.
    This is done to ensure there is no exploding or vanishing gradients
    during traing. Each token is normalized to have a 0 mean and 1 variance.
    Scale and Shift are been added following the lessons from the book.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(in_dim))
        self.shift = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / std
        return self.scale * x + self.shift


class MLP(nn.Module):
    """
    Feed forward layer that produces the transformed
    output embeddings. Part of the transformer design.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    """
    Explanation follows from the book Build a Large Language Model.

    Self-attention: The purpose of self-attention is to create
    a context vector for each input vector, where context vector
    can be thouhgt of as an enriched vector that contains information
    about its corresponding input vector and all other tokens in
    the sequence. This is done in three steps.
    1. Attention scores: Calculate the dot product between input vector
    and all other vectors in the sequence.
    2. Attention weights: Take the weighted average(softmax) to assign a
    weight between 0 and 1 (probability).
    3. Context vector: Take the sum of all the input vectors multiplied
    with their attention weights to output a new vector which is the context
    vector.
    (In a nutshell - input @ intput.T @ input).
    Weights are used to project input into query, key and value to learn
    better representation.

    Causal attention: To prevent the token from attending future tokens, they
    are masked.

    Cross-attention: The same but the keys and values come from encoder. No
    masking here.

    Multi-head attention: The embedding dim (emb_dim) is split into (num_heads, head_dim)
    for the embedding to learn different representation. Each head would attend
    differently like grammer, semantics etc.

    """

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
        # If not training, then check attention type whether to use
        # cross-attention or self-attention for inference.
        elif self.attn_type == "cross-attn":
            return self.cross_attn(x, y)
        return self.inference_attn(x)  # Self-attention by default

    def cross_attn(
        self, query: torch.Tensor, kv: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Need to implement"""
        return query

    def inference_attn(self, query: torch.Tensor) -> torch.Tensor:
        """Need to implement"""
        return query

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
        if self.attn_type == "self-attention":
            # self.mask[:context, :context] -> Get the values/ sub-matrix
            #  till the current context.
            attn_scores = attn_scores.masked_fill(
                getattr(self, "mask")[:context, :context], -torch.inf
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, context, context) @ (b, num_heads, context, head_dim)
        # => (b, num_heads, context, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, context, head_dim) => (b, context, out_dim)
        context_vec = context_vec.view(b, context, self.num_heads * self.head_dim)

        context_vec = self.proj(context_vec)

        return context_vec


class DecoderLayer(nn.Module):
    """
    Class consists of components of each
    encoder layer. Multihead attention, Cross Attention,
    Layer Normalization, Feed-forward components.
    """

    def __init__(
        self,
        context_length: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.mlp = MLP(emb_dim, emb_dim * 4)
        self.attn = MultiHeadAttention(
            in_dim=emb_dim,
            out_dim=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            attn_type="self-attention",
        )
        self.c_attn = MultiHeadAttention(
            in_dim=emb_dim,
            out_dim=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            attn_type="cross-attention",
        )
        self.norm1 = LayerNorm(in_dim=emb_dim)
        self.norm2 = LayerNorm(in_dim=emb_dim)
        self.norm3 = LayerNorm(in_dim=emb_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        x = self.attn(x) + x
        x = self.norm1(x)
        x = self.c_attn(x, y) + x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.norm3(x)

        return x


class Decoder:
    """
    Decoder is supposed to process both the output
    of the final layer of encoder and the target vectors
    to predict the next word in the sequence.
    Consists of a token embedding layer for source inputs,
    positional embeddings to add position for decoder to learn
    about the structure of the input, and num_layers * Dncoder
    layer, where the input passes through multihead attention,
    cross attention, layer normalization, and feedforward network.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.pos_embeddings = nn.Embedding(context_length, emb_dim)
        self.decoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    context_length=context_length,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        _, context = x.shape
        x = self.token_embeddings(x) + self.pos_embeddings(torch.arange(context))
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, y)
        return x


class EncoderLayer(nn.Module):
    """
    Class consists of components of each
    encoder layer. Multihead attention, Layer
    Normalization, Feed-forward components.
    """

    def __init__(
        self,
        context_length: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.mlp = MLP(emb_dim, emb_dim * 4)
        self.attn = MultiHeadAttention(
            in_dim=emb_dim,
            out_dim=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            attn_type="self-attention",
        )
        self.norm1 = LayerNorm(in_dim=emb_dim)
        self.norm2 = LayerNorm(in_dim=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        x = self.attn(x) + x
        x = self.norm1(x)
        x = self.mlp(x) + x
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    The encoder is supposed to process the source
    vectors and provide an intermediate representations
    for the decoder to predict next word in the sequence.
    Consists of a token embedding layer for source inputs,
    positional embeddings to add position for encoder to learn
    about the structure of the input, and num_layers * Encoder
    layer, where the input passes through multihead attention,
    layer normalization, and feedforward network.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.pos_embeddings = nn.Embedding(context_length, emb_dim)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    context_length=context_length,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Experimental"""
        _, context = x.shape
        x = self.token_embeddings(x) + self.pos_embeddings(torch.arange(0, context))
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        return x


class Transformer(nn.Module):
    """
    Transformer consists of both encoder and decoder.
    The inputs are first passed into encoder to get the
    representations of the encoder input. In this case,
    it would be english sentence. The output of the final
    layer of the encoder is passed to the decoder.
    Encoder here consists of multi-head attention and feedforward
    networks and two normalization layers.
    Decoder here consists of multi-head attention, cross-attention,
    and feedforward networks and three normalization layers.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        emb_dim: int,
        enc_layers: int,
        dec_layers: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
    ):
        super().__init__()
        # Create an Encoder.
        self.encoder = Encoder(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=emb_dim,
            num_layers=enc_layers,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )
        # Create a decoder.
        self.decoder = Encoder(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=emb_dim,
            num_layers=dec_layers,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

        # Final layer to project each embedding into vocab size
        self.final_layer = nn.Linear(emb_dim, vocab_size, bias=bias)

        self.inference = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Experimental
        """
        # pass the input into encoder.
        y = self.encoder(y)

        # pass the targets into decoder
        x = self.decoder(x, y)

        # pass the targets into final layer
        # to transform the dimension to vocab
        # size
        x = self.final_layer(x)

        # return raw logits.
        return x
