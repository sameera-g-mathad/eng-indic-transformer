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
        """
        :param in_dim: The dimension to initialize weights
                       for scaling and shifting the inputs.
                       Usually same size as the embedding
                       dimension of input.
        :type in_dim: int.
        """
        super().__init__()
        # a small epsilon value to prevent divide by zero error
        self.eps = 1e-5
        # two parameters that the model will learn to scale
        # the inputs
        self.scale = nn.Parameter(torch.ones(in_dim))
        self.shift = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each channel/emb_dim of the inputs
        to have zero mean and a unit standard deviation.
        Finally scale and shift the inputs by learable parameters.

        :param x: input to LayerNorm
        :type x: torch.Tensor

        :returns: Returns a layer normalized tensor.
        :rtype: torch.Tensor.
        """
        # find mean for the each token's emb_dim.
        mean = torch.mean(x, dim=-1, keepdim=True)

        # find std for the each token's emb_dim.
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)

        # Normalize each token's embeddings.
        x = (x - mean) / (std + self.eps)  # add eps to prevent errors.

        # return the scaled result.
        return self.scale * x + self.shift


class MLP(nn.Module):
    """
    Feed forward layer that produces the transformed
    output embeddings. Part of the transformer design.
    """

    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: The in_dim to initialize the linear
                       weights to match the embedding dimenision.
        :type in_dim: int.
        :param out_dim: The out_dim to initialize the linear
                        weights.
        :type out_dim: int.

        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the result of the feedforward network.

        :param x: The inputs to the feed-forward module.
        :type x: torch.Tensor.

        :returns: The output tensors.
        :rtype: torch.Tensor.
        """
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    """
    Explanation follows from the book Build a Large Language Model.

    Self-attention: The purpose of self-attention is to create
    a context vector for each input vector, where context vector
    can be thought of as an enriched vector that contains information
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
        layer_type: Literal["decoder", "encoder"] = "decoder",
    ):
        """
         :param in_dim: The in_dim to initialize the linear
                       weights of query, key and values to match the
                       embedding dimenision.
        :type in_dim: int.
        :param out_dim: The out_dim to initialize the linear
                       weights of query, key and values to match the
                       embedding dimenision.
        :type out_dim: int.
        :param context_length: Maximum allowed context length to allow masking.
                               and used for output projection layer.
        :type context_length: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        :param attn_type: Value to tell the model what kind of layer this is, i.e either
                          a self-attention or cross attention layer.
        :type attn_type: Literal["self-attention", "cross-attention"]
        :param layer_type: Used during inference as encoder doesn't need masking and caching.
        :type layer_type: Literal["decoder", "encoder"]
        """
        super().__init__()
        # the out_dim.
        self.out_dim = out_dim

        # number of heads that user define.
        self.num_heads = num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # the head_dim after multi-head projection.
        self.head_dim = out_dim // num_heads

        # attentin type, either encoder/decoder.
        self.attn_type = attn_type

        # model type, either encoder/decoder.
        self.layer_type = layer_type

        # learnable weights for query, key, value, output projection.
        self.wq = nn.Linear(in_dim, out_dim, bias=bias)
        self.wk = nn.Linear(in_dim, out_dim, bias=bias)
        self.wv = nn.Linear(in_dim, out_dim, bias=bias)
        self.proj = nn.Linear(out_dim, out_dim, bias=bias)

        # dropout.
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

        # caches for key-value caching.
        # persistent prevents from saving into state_dict
        self.register_buffer("cache_key", None, persistent=False)
        self.register_buffer("cache_val", None, persistent=False)

    def reset_cache(self):
        """
        To reset the caches before every inference
        run.
        """
        setattr(self, "cache_key", None)
        setattr(self, "cache_val", None)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, inference: bool = False
    ) -> torch.Tensor:
        """
        Forward method of Mutli-head attention layer.

        :param x: Inputs from the decoder or encoder depending on the layer
                  using it.
        :type x: torch.Tensor.
        :param y: Could be the same inputs in case of encoder and self-attention,
                  but changes for cross attention.
        :type y: torch.Tensor.
        :param inference: Boolean value that can be used for caching.
        :type inference: bool.
        """
        # if inference:
        #     return self.inference_attn(x, y)

        return self.train_attn(x, y)

    def calc_attn_scores(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Method to calculate attn_scores. Takes both queries and keys.
        Formula: (QK_T) / (√out_dim)

        :param queries: Projected queries part of the inputs.
        :type queries: torch.Tensor.
        :param keys: Projected keys part of the inputs.
        :type keys: torch.Tensor.

        :returns: Calculated attention scores.
        :rtype: torch.Tensor.
        """
        # (b, num_heads, context, head_dim) @ (b, num_heads, head_dim, context)
        # => (b, num_heads, context, context)
        attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim**0.5)
        return attn_scores

    def calc_context_vec(
        self, attn_scores: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Method to calculate attention weights and finally context vector.
        Takes the pre-calculated attention scores and values and
        returns the context vector.

        :param attn_scores: Attention scores calculated.
        :type attn_scores: torch.Tensor.

        :param values: Projected values part of the inputs.
        :type values: torch.Tensor.

        :returns: Calculated attention weights.
        :rtype: torch.Tensor.
        """

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        b, _, attn_context, _ = (
            attn_weights.shape
        )  # b, heads, query_context, keys_context

        # (b, num_heads, context, context) @ (b, num_heads, context, head_dim)
        # => (b, num_heads, context, head_dim)
        context_vec = attn_weights @ values

        # (b, num_heads, attn_context, head_dim) => (b, attn_context, out_dim)
        context_vec = context_vec.view(b, attn_context, self.num_heads * self.head_dim)

        context_vec = self.proj(context_vec)

        return context_vec

    def inference_attn(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Method used during inference to use key value caching. Used only
        by decoder.
        Receives new query everytime for processing. Apart from the initial
        query that is set to keys and values, subsequent queries are projected for
        key_new and val_new and concatenated with respective caches to prevent
        recomputation. The concatenation happens only for self attention and is skipped
        for cross attention.
        Note: Masking is skipped for self attention as well. This is because the attention
        score generated will be for the latest query which represent the last element of the
        attention score.
        ex: attn_score_dim = 1 x 1 x 5 => b x query_context x key_context.

        :param query: latest query to be processed by the decoder.
        :type query: torch.Tensor.
        :param kv: kv part of the decoder to be projected into keys and values to be cached.
        :type kv: torch.Tensor.

        :returns: Attention aware input.
        :rtype: torch.Tensor.
        """

        b, context_q, _ = query.shape

        # project query.
        query = self.wq(query)

        # if cache is empty, fill them
        if getattr(self, "cache_key") is None or getattr(self, "cache_val") is None:
            self.cache_key = self.wk(kv)
            self.cache_val = self.wv(kv)

        # calculate key_new and val_new and concatenate
        # with key and value caches along context dim.
        elif self.attn_type == "self-attention":
            key_new = self.wk(query)
            val_new = self.wv(query)
            self.cache_key = torch.cat([self.cache_key, key_new], dim=-2)
            self.cache_val = torch.cat([self.cache_val, val_new], dim=-2)

        # take a reference.
        keys = self.cache_key
        values = self.cache_val

        # take the length of key's context.
        # this is because encoder context may or
        # may not equal decoder context.
        context_kv = keys.shape[-2]

        # project q,k,v to multiple heads
        queries = query.view(b, self.num_heads, context_q, self.head_dim)
        keys = keys.view(b, self.num_heads, context_kv, self.head_dim)
        values = values.view(b, self.num_heads, context_kv, self.head_dim)

        # calculate attention_scores.
        # (b, num_heads, context, head_dim) @ (b, num_heads, head_dim, context)
        # => (b, num_heads, context, context)
        attn_scores = self.calc_attn_scores(queries=queries, keys=keys)

        # return context vector
        return self.calc_context_vec(attn_scores=attn_scores, values=values)

    def train_attn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Multi Masked/Cross Attention used to compute the context vectors
        of the inputs. This method is used by both encoder and decoder.
        Receives the entire query(x) and kv(y) during training and also
        during inference by encoder. Masking is applied only for decoder
        self-attention as decoder should attend to previous tokens.
        Takes two inputs x and y, because decoder takes both input from
        encoder and decoder.
        For encoder pass the same input twice.

        :param x: Inputs from decoder or encoder using attention.
        :type x: torch.Tensor.
        :param y: Same input as x for encoder and decoder, but differs
                  for cross-attention.
        :type y: torch.Tensor.
        """
        b, context, _emb_dim = x.shape  # (b, context, in_dim)

        queries = self.wq(x)  # (b, context, out_dim)

        keys = self.wk(y)  # (b, context, out_dim)
        values = self.wv(y)  # (b, context, out_dim)
        context_kv = y.shape[1]  # get the context of encoder kv

        # else:
        #     keys = self.wk(x)  # (b, context, out_dim)
        #     values = self.wv(x)  # (b, context, out_dim)
        # context_kv = context

        # Reshape queries, keys, values for multi-head attention.
        # (b, context, out_dim) -> (b, num_heads, context, head_dim)
        queries = queries.view(b, self.num_heads, context, self.head_dim)
        keys = keys.view(b, self.num_heads, context_kv, self.head_dim)
        values = values.view(b, self.num_heads, context_kv, self.head_dim)

        # (b, num_heads, context, head_dim) @ (b, num_heads, head_dim, context)
        # => (b, num_heads, context, context)
        attn_scores = self.calc_attn_scores(queries=queries, keys=keys)

        # causal attention. Fill all the upper traingle matrix with torch.inf,
        # so that softmax can output 0.
        if self.attn_type == "self-attention" and self.layer_type == "decoder":
            # self.mask[:context, :context] -> Get the values/ sub-matrix
            #  till the current context.
            attn_scores = attn_scores.masked_fill(
                getattr(self, "mask")[:context, :context_kv], -torch.inf
            )

        return self.calc_context_vec(attn_scores=attn_scores, values=values)


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
        """
        :param context_length: Maximum allowed context length to allow masking.
                               and used for output projection layer.
        :type context_length: int.
        :param emb_dim: Embedding dimension of inputs.
        :type emb_dim: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        """
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
            layer_type="decoder",
        )
        self.c_attn = MultiHeadAttention(
            in_dim=emb_dim,
            out_dim=emb_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            attn_type="cross-attention",
            layer_type="decoder",
        )
        self.norm1 = LayerNorm(in_dim=emb_dim)
        self.norm2 = LayerNorm(in_dim=emb_dim)
        self.norm3 = LayerNorm(in_dim=emb_dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, inference: bool
    ) -> torch.Tensor:
        """
        Computes masked-multi attention + cross-attention + feed-forward.
        Normalization is applied after each operation.

        :param x: The target sequence to be passed to decoder.
        :type x: torch.Tensor.
        :param y: The transformed input sequence for cross-attention.
        :type y: torch.Tensor.
        :param inference: Boolean value that can be used for caching.
        :type inference: bool.

        :returns: The output of the single decoder layer.
        :rtype: torch.Tensor.
        """
        x = self.attn(x, x, inference) + x
        x = self.norm1(x)
        x = self.c_attn(x, y, inference) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = self.norm3(x)

        return x

    def reset_cache(self):
        """
        Reset the caches of attention and
        cross-attention blocks.
        """
        self.attn.reset_cache()
        self.c_attn.reset_cache()


class Decoder(nn.Module):
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
        """
        :param vocab_size: To create a embedding layer.
        :type vocab_size: int
        :param context_length: Maximum allowed context length to allow masking.
                               and used for output projection layer. Also, to
                               create positional embeddings.
        :type context_length: int.
        :param emb_dim: Embedding dimension of inputs.
        :type emb_dim: int.
        :param num_layers: Number of decoder layers to create.
        :type num_layers: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.pos_embeddings = nn.Embedding(context_length, emb_dim)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    context_length=context_length,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        # Final layer to project each embedding into vocab size
        self.final_layer = nn.Linear(emb_dim, vocab_size, bias=bias)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, inference: bool = False
    ) -> torch.Tensor:
        """
        Computes token_embeddings, positional_embeddings, and transformer(decoder layer)
        for each layer in decoder.

        :param x: The target sequence to be passed to decoder consisting of num_layer
                  decoder layers.
        :type x: torch.Tensor.
        :param y: The transformed input sequence for cross-attention.
        :type y: torch.Tensor.
        :param inference: Boolean value that can be used for caching.
        :type inference: bool.

        :returns: The output of the decoder.
        :rtype: torch.Tensor.
        """
        _, context = x.shape
        x = self.token_embeddings(x) + self.pos_embeddings(
            torch.arange(0, context, device=x.device)
        )
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, y, inference)

        # pass the targets into final layer
        # to transform the dimension to vocab
        # size
        x = self.final_layer(x)

        return x

    def reset_cache(self):
        """
        Reset caches for every decoder layer.
        """
        for dec_layer in self.decoder_layers:
            # to avoid pylance warning.
            dec_layer.reset_cache()  # type: ignore


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
        """
        :param context_length: To create positional embeddings and pass to
                               attention layer to make class generic.
        :type context_length: int.
        :param emb_dim: Embedding dimension of inputs sequence.
        :type emb_dim: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        """
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
            layer_type="encoder",
        )
        self.norm1 = LayerNorm(in_dim=emb_dim)
        self.norm2 = LayerNorm(in_dim=emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes masked-multi attention + feed-forward.
        Normalization is applied after each operation.

        :param x: Input to the encoder layer.
        :type x: torch.Tensor.
        :returns: The transformed x(inputs) of the single encoder layer.
        :rypte: torch.Tensor.
        """
        x = self.attn(x, x) + x
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
        """
        :param vocab_size: To create a embedding layer.
        :type vocab_size: int
        :param context_length: To create positional embeddings and pass to
                               attention layer to make class generic.
        :type context_length: int.
        :param emb_dim: Embedding dimension of inputs sequence.
        :type emb_dim: int.
        :param num_layers: Number of encoder layers to create.
        :type num_layers: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        """
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
        """
        Computes token_embeddings, positional_embeddings, and transformer(decoder layer)
        for each layer in decoder.

        :param x: Input to the encoder consisting of num_layers encoder layers.
        :type x: torch.Tensor.
        :returns: The transformed x(inputs) of the single encoder layer.
        :rypte: torch.Tensor.
        """
        _, context = x.shape
        x = self.token_embeddings(x) + self.pos_embeddings(
            torch.arange(0, context, device=x.device)
        )
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
        """
        :param vocab_size: To create a embedding layer.
        :type vocab_size: int
        :param context_length: Maximum allowed context length to allow masking.
                               and used for output projection layer. Also, to
                               create positional embeddings.
        :type context_length: int.
        :param emb_dim: Embedding dimension of inputs.
        :type emb_dim: int.
        :param enc_layer: Number of encoder layers to create.
        :type enc_layer: int.
        :param dec_layer: Number of decoder layers to create.
        :type dec_layer: int.
        :param num_heads: The number of heads the embedding dimension needs to
                          be projected. Usually even number and embedding dimension
                          must be divisible by num_heads.
        :type num_heads: int.
        :param dropout: Value used by the dropout layer.
        :type dropout: float.
        :param bias: Boolean value whether each weights in the layer needs bias or not.
        :type bias: bool.
        """
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
        self.decoder = Decoder(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=emb_dim,
            num_layers=dec_layers,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
        )

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Method to only encode the src vector druing inference.

        :param src: Source sequence to pass to encoder during inference.
        :type src: torch.Tensor.
        :returns: Output of encoder.
        :rtype: torch.Tensor.
        """
        return self.encoder(src)

    def decode(
        self, target: torch.Tensor, memory: torch.Tensor, inference: bool
    ) -> torch.Tensor:
        """
        Method to only decode the target vector during inference.

        :param target: Target sequence to pass to decoder during inference.
        :type target: torch.Tensor.
        :param memory: The output of encoder to be passed as memory for
                       cross attention.
        :type memory: torch.Tensor.
        :param inference: Boolean value that can be used for caching.
        :type inference: bool.

        :returns: Output of decoder.
        :rtype: torch.Tensor.
        """
        return self.decoder(target, memory, inference=inference)

    def reset_cache(self):
        """
        Reset the available caches.
        """
        self.decoder.reset_cache()

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply encoder and decoder during training.

        :param src: Source sequence to pass to the encoder.
        :type src: torch.Tensor.

        :param target: Target sequence to pass to the decoder.
        :type target: torch.Tensor.

        :returns: The output of final layer after encoding and decoding
                  of source and target sequences.
        :rtype: torch.Tensor.
        """
        # pass the input into encoder.
        y = self.encoder(src)

        # pass the targets into decoder
        x = self.decoder(target, y)

        # return raw logits.
        return x
