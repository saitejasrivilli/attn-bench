"""
kernels/cuda_baseline.py

FlashAttention-2 baseline wrapper.
Provides the same interface as TritonAttention so both can be swapped
transparently in benchmarks.
"""

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input


class FlashAttentionBaseline(nn.Module):
    """
    FlashAttention-2 wrapper (CUDA kernel by Tri Dao).
    Input/output format: (batch, num_heads, seq_len, head_dim)

    Acts as the reference implementation for correctness checks and
    the performance baseline for all speedup numbers.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = True,
    ):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.causal       = causal
        self.sm_scale     = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,   # (batch, num_heads, seq_len, head_dim) BHSD
        k: torch.Tensor,   # (batch, num_kv_heads, seq_len, head_dim)
        v: torch.Tensor,   # (batch, num_kv_heads, seq_len, head_dim)
    ) -> torch.Tensor:
        # flash_attn_func expects BSH format: (batch, seq, heads, head_dim)
        q_bsh = q.transpose(1, 2)
        k_bsh = k.transpose(1, 2)
        v_bsh = v.transpose(1, 2)

        out_bsh = flash_attn_func(
            q_bsh, k_bsh, v_bsh,
            softmax_scale=self.sm_scale,
            causal=self.causal,
        )
        return out_bsh.transpose(1, 2)   # back to BHSD


class FlashAttentionVarLen(nn.Module):
    """
    Variable-length FlashAttention-2 for batches with different sequence lengths.
    Used internally by vLLM and SGLang — exposed here for benchmarking padded vs
    unpadded throughput.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, causal: bool = True):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.causal       = causal
        self.sm_scale     = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,          # (total_tokens, num_heads, head_dim)
        k: torch.Tensor,          # (total_tokens, num_kv_heads, head_dim)
        v: torch.Tensor,          # (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.sm_scale,
            causal=self.causal,
        )
