# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from llama.generation import Generation
from llama.lora import Linear as LoRALinear


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 8
    max_seq_len: int = 1024


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability (avoid devided by 0).
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions for ROPE.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # Rotary frequency
    t = torch.arange(end, device=freqs.device)  # Position of each token (until "end"). type: ignore
    freqs = torch.outer(t, freqs).float()  # Shape [max_seq_len, dim//2]. Where max_seq_len = end. type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 (complex numbers: Norm=torch.ones_like(freqs)=1, Angles=freqs)

    return freqs_cis  # [max_seq_len * 2, dim // 2]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim

    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given Query 'xq' and Key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified Query tensor and Key tensor with rotary embeddings.
    """
    # After reshape, shapes of xq = xk = [B, seq_len, H, head_dim/2, 2]
    # With torch.view_as_complex() -> become: [B, seq_len, H, head_dim/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # [1, seq_len, 1, head_dim/2]

    # After torch.view_as_real() -> become: [B, seq_len, H, head_dim/2, 2]
    # And after flatten(3) -> become: [B, seq_len, H, head_dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module without KV cache."""
    def __init__(self, args: ModelArgs, layer_idx: int=None):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            Removed: cache_k (torch.Tensor): Cached keys for attention.
            Removed: cache_v (torch.Tensor): Cached values for attention.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_idx = layer_idx

        # self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wq = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, r=16, lora_alpha=32, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = LoRALinear(args.dim, self.n_kv_heads * self.head_dim, r=16, lora_alpha=32, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_cache: bool = True,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (`torch.Tensor`): Input tensor.
            start_pos (`int`): Starting position for caching.
            freqs_cis (`torch.Tensor`): Precomputed frequency tensor.
            mask (`torch.Tensor`, `optional`): Attention mask tensor.
            use_cache (`bool`): Use KV Cache or not.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        bsz, q_len, _ = x.shape  # [B, q_len, hidden_dim]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, q_len, self.n_local_heads, self.head_dim)  # [B, q_len, H, head_dim]
        xk = xk.view(bsz, q_len, self.n_local_kv_heads, self.head_dim)  # [B, q_len, H, head_dim]: n_local_kv_heads = n_local_heads = 32
        xv = xv.view(bsz, q_len, self.n_local_kv_heads, self.head_dim)  # [B, q_len, H, head_dim]: n_local_kv_heads = n_local_heads = 32

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training and use_cache:
            self.cache_k = self.cache_k.to(xq.dtype)
            self.cache_v = self.cache_v.to(xq.dtype)
            self.cache_k[:bsz, start_pos : start_pos + q_len] = xk
            self.cache_v[:bsz, start_pos : start_pos + q_len] = xv

            keys = self.cache_k[:bsz, : start_pos + q_len]
            values = self.cache_v[:bsz, : start_pos + q_len]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # [B, seq_len, H, head_dim]
        values = repeat_kv(values, self.n_rep)  # [B, seq_len, H, head_dim]

        query_states = xq.transpose(1, 2)  # [B, H, q_len, head_dim]
        key_states = keys.transpose(1, 2)  # [B, H, seq_len, head_dim]
        value_states = values.transpose(1, 2)  # [B, H, seq_len, head_dim]

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)  # [B, H, q_len, seq_len]

        if mask is not None:
            scores = scores + mask  # [B, H, q_len, seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(query_states)
        output = torch.matmul(scores, value_states)  # [B, H, q_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module (LlamaMLP -> Gated Linear Unit (GLU)).

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)  # Reduce to 2/3 (less operations)

        # custom dim factor multiplier (if you want to increase it)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_idx (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_idx (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_idx = layer_idx

        self.attention = Attention(args, layer_idx=layer_idx)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_cache: bool = True,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
            use_cache (`bool`): Use KV Cache or not.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, use_cache
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama(Generation):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_idx in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_idx, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, use_cache: bool = True):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.
            use_cache (`bool`): Use KV Cache or not.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # !!! 99% of the time of "RuntimeError: CUDA error: device-side assert triggered" comes from indices
        # that are out of range for an nn.Embedding (**negative pad ID**, BOS/EOS = 32000, or an OOV token) !!!
        # Refer: 1. https://github.com/meta-llama/llama/issues/380 2. https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
        bsz, q_len = tokens.shape
        h = self.tok_embeddings(tokens)  # [B, q_len, hidden_dim]
        self.freqs_cis = self.freqs_cis.to(h.device)  # [max_seq_len * 2, dim // 2] = [512, 64]
        freqs_cis = self.freqs_cis[start_pos : start_pos + q_len]  # [q_len, 64]

        mask = None
        if q_len > 1:
            mask = torch.full(
                (q_len, q_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)  # only retain upper triangle as -inf

            # Useless code -> only changed the dtype to float16:
            mask = torch.hstack([
                torch.zeros((q_len, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)

        h = self.norm(h)
        output = self.output(h).float()

        return output
