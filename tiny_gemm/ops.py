import math

import torch

from triton_fused_transformer import fused_attention, fused_ffn

_lib = torch.library.Library("tiny_gemm", "DEF")

_lib.define(
    "fused_attention(Tensor q, Tensor k, Tensor v, bool causal, float dropout_p) -> Tensor"
)
_lib.define(
    "fused_ffn(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2, int activation) -> Tensor"
)


def _attention_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    if q.ndim != 4:
        raise RuntimeError("q must be [B, H, N, D]")
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("k and v must match q shape")
    return torch.empty_like(q)


def _ffn_meta(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    activation: int,
) -> torch.Tensor:
    if x.ndim != 3:
        raise RuntimeError("x must be [B, N, D]")
    d_model = x.shape[2]
    if w1.shape[0] != d_model or w2.shape[1] != d_model:
        raise RuntimeError("w1/w2 shapes must align with x")
    return torch.empty_like(x)


def _attention_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.dropout(attn_weights, p=dropout_p, train=False)
    return torch.matmul(attn_weights, v)


def _ffn_cpu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    activation: int,
) -> torch.Tensor:
    hidden = torch.matmul(x, w1) + b1
    if activation == 0:
        hidden = torch.nn.functional.gelu(hidden)
    elif activation == 1:
        hidden = torch.nn.functional.relu(hidden)
    elif activation == 2:
        hidden = torch.nn.functional.silu(hidden)
    return torch.matmul(hidden, w2) + b2


@torch.library.impl(_lib, "fused_attention", "Meta")
def _attention_meta_impl(q, k, v, causal, dropout_p):
    return _attention_meta(q, k, v, causal, dropout_p)


@torch.library.impl(_lib, "fused_attention", "CPU")
def _attention_cpu_impl(q, k, v, causal, dropout_p):
    return _attention_cpu(q, k, v, causal, dropout_p)


@torch.library.impl(_lib, "fused_attention", "CUDA")
def _attention_cuda_impl(q, k, v, causal, dropout_p):
    return fused_attention(q, k, v, causal=causal, dropout_p=dropout_p)


@torch.library.impl(_lib, "fused_ffn", "Meta")
def _ffn_meta_impl(x, w1, b1, w2, b2, activation):
    return _ffn_meta(x, w1, b1, w2, b2, activation)


@torch.library.impl(_lib, "fused_ffn", "CPU")
def _ffn_cpu_impl(x, w1, b1, w2, b2, activation):
    return _ffn_cpu(x, w1, b1, w2, b2, activation)


@torch.library.impl(_lib, "fused_ffn", "CUDA")
def _ffn_cuda_impl(x, w1, b1, w2, b2, activation):
    activation_map = {"gelu": 0, "relu": 1, "silu": 2}
    activation_str = {v: k for k, v in activation_map.items()}.get(int(activation), "gelu")
    return fused_ffn(x, w1, b1, w2, b2, activation=activation_str)
