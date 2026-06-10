"""
kernels/triton_attention.py

Custom Triton multi-head attention kernel.
- Tiled computation (BLOCK_M × BLOCK_N tiles)
- Causal masking (lower-triangular)
- Fused online softmax (numerically stable, single-pass)
- GQA support (num_kv_heads < num_heads)
- Compatible with vLLM and SGLang custom attention APIs

References:
  - Dao et al., FlashAttention-2 (2023)
  - Triton tutorial: fused attention
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """
    Grid: (batch * num_heads, ceil(N_CTX / BLOCK_M))
    Each program handles one BLOCK_M × N_CTX tile for one (batch, head).
    """
    # ── program ids ───────────────────────────────────────────────────────────
    # grid = (batch * num_heads, cdiv(seq_len, BLOCK_M))
    off_bh   = tl.program_id(0)
    start_m  = tl.program_id(1)
    off_b    = off_bh // H
    off_h    = off_bh  % H

    # GQA: map query head → kv head
    off_kv_h = off_h * NUM_KV_HEADS // NUM_HEADS

    # ── base pointers ─────────────────────────────────────────────────────────
    Q_ptr  = Q  + off_b * stride_qb + off_h    * stride_qh
    K_ptr  = K  + off_b * stride_kb + off_kv_h * stride_kh
    V_ptr  = V  + off_b * stride_vb + off_kv_h * stride_vh
    Out_ptr = Out + off_b * stride_ob + off_h   * stride_oh

    # ── offsets ───────────────────────────────────────────────────────────────
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # ── load Q block ──────────────────────────────────────────────────────────
    q = tl.load(
        Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    ).to(tl.float32)

    # ── online softmax accumulators ───────────────────────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ── causal: only attend to tokens <= current position ─────────────────────
    lo = 0
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # load K block
        k = tl.load(
            K_ptr + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=offs_n_curr[None, :] < N_CTX,
            other=0.0,
        ).to(tl.float32)

        # QK^T scaled
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # causal mask
        if CAUSAL:
            mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # online softmax update — FA2 Algorithm 1 (Dao 2023)
        m_ij   = tl.max(qk, axis=1)
        m_new  = tl.maximum(m_i, m_ij)

        # p must be normalised by m_new (global max), NOT m_ij (tile max).
        # Using m_ij here misses the exp(m_ij - m_new) beta factor and produces
        # wrong cross-tile accumulation when the old max > the current tile max.
        # Guard: qk=-inf gives qk-m_new=-inf (or nan when m_new=-inf too);
        # clamp to -100 so exp ≈ 0 safely.
        p      = tl.exp(tl.maximum(qk - m_new[:, None], -100.0))
        l_ij   = tl.sum(p, axis=1)

        # alpha rescales the old accumulator to the new global max.
        # Guard: m_i=-inf on first iteration → alpha should be 0 (acc is 0 anyway).
        alpha  = tl.exp(tl.maximum(m_i - m_new, -100.0))
        m_i    = m_new
        l_i    = l_i * alpha + l_ij
        acc    = acc * alpha[:, None]

        # load V block
        v = tl.load(
            V_ptr + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n_curr[:, None] < N_CTX,
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)

    # ── normalize ─────────────────────────────────────────────────────────────
    acc = acc / l_i[:, None]

    # ── store output ──────────────────────────────────────────────────────────
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16),
        mask=offs_m[:, None] < N_CTX,
    )


# ── Python wrapper ────────────────────────────────────────────────────────────

class TritonAttention(torch.nn.Module):
    """
    Drop-in replacement for FlashAttention-2 using a custom Triton kernel.
    Supports GQA (num_kv_heads < num_heads).

    Args:
        num_heads:    Total query heads
        num_kv_heads: KV heads (< num_heads for GQA, == num_heads for MHA)
        head_dim:     Dimension per head
        causal:       Apply causal mask
        block_m:      Tile size along sequence (query) dimension
        block_n:      Tile size along sequence (key) dimension
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = True,
        block_m: int = 128,
        block_n: int = 64,
    ):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.causal       = causal
        self.block_m      = block_m
        self.block_n      = block_n
        self.sm_scale     = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,   # (batch, num_heads, seq_len, head_dim)
        k: torch.Tensor,   # (batch, num_kv_heads, seq_len, head_dim)
        v: torch.Tensor,   # (batch, num_kv_heads, seq_len, head_dim)
    ) -> torch.Tensor:

        assert q.dtype == torch.float16, "Triton kernel requires float16"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

        batch, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)

        grid = (batch * num_heads, triton.cdiv(seq_len, self.block_m))

        _attn_fwd_kernel[grid](
            q, k, v, out,
            self.sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, num_heads, seq_len,
            HEAD_DIM=head_dim,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            CAUSAL=self.causal,
            NUM_KV_HEADS=self.num_kv_heads,
            NUM_HEADS=num_heads,
        )
        return out


# ── Autotuned variant ─────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=2, num_stages=5),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=2, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["N_CTX", "HEAD_DIM", "NUM_HEADS"],
)
@triton.jit
def _attn_fwd_autotuned(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    """Same kernel body, Triton handles tile-size search via autotuning."""
    _attn_fwd_kernel(
        Q, K, V, Out, sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=CAUSAL,
        NUM_KV_HEADS=NUM_KV_HEADS,
        NUM_HEADS=NUM_HEADS,
    )


# ── Numerical correctness check ───────────────────────────────────────────────

def verify_correctness(
    batch: int = 2,
    num_heads: int = 8,
    num_kv_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    device: str = "cuda",
    atol: float = 1e-2,
):
    """
    Compares Triton output against FlashAttention-2 reference.
    Prints max absolute error and pass/fail.
    """
    from flash_attn import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, num_heads,    head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    # FlashAttention-2 reference (BSH format)
    ref_out = flash_attn_func(q, k, v, causal=True)

    # Triton kernel (BHSD format)
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()

    triton_attn = TritonAttention(num_heads, num_kv_heads, head_dim, causal=True)
    triton_out  = triton_attn(q_t, k_t, v_t).transpose(1, 2)

    max_err = (triton_out - ref_out).abs().max().item()
    passed  = max_err < atol

    print(f"Correctness check | max_abs_err={max_err:.6f} | atol={atol} | {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    verify_correctness()
