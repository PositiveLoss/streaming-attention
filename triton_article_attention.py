"""
Triton CUDA kernels for the article-style Taylor + residual-coreset causal
attention approximation.

This file is intentionally separate from the reference implementation
(`streaming_taylor_coreset_attention.py`) and the vectorized PyTorch fast path
(`fast_article_attention.py`).  It is a GPU-first implementation meant for
latency experiments.

Implemented kernels
-------------------
1. `triton_article_causal_attention`:
   A fused causal kernel for the article-style decomposition

       exp(x) = Taylor_<=degree(x) + residual_>degree(x).

   The low-degree Taylor term is evaluated exactly against the full causal
   prefix.  The residual term is evaluated exactly inside the current local
   block and approximately against deterministic segment-mean block coresets for
   completed previous blocks.  Set `compress_stride=1` to make the residual
   exact as well; in that mode the output should match exact causal softmax
   attention up to floating-point error.

2. `triton_flash_causal_attention`:
   A compact FlashAttention-style exact causal baseline implemented in Triton.
   It does not materialize the N x N score matrix.

Important caveat
----------------
The low-degree Taylor term in this Triton file is fused directly over the causal
prefix rather than using the streaming moment state from the paper.  This keeps
GPU code compact and fast for moderate N while avoiding score-matrix materializa-
tion, but it is an offline fused kernel, not a literal online streaming-state
kernel.  The reference implementation remains the closer data-structure match;
this file is optimized for GPU benchmarking.

Requirements
------------
    pip install triton

A CUDA-enabled PyTorch build is also required.  The current execution container
used to generate this file does not have Triton/CUDA installed, so the module is
written with optional imports and runtime guards.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

try:  # Optional so the file can still be imported on CPU-only machines.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on machines without Triton.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _require_triton() -> None:
    if triton is None or tl is None:
        raise ImportError(
            "Triton is not installed. Install it in a CUDA-enabled environment, "
            "for example: pip install triton"
        )


def _require_cuda(*tensors: Tensor) -> None:
    for x in tensors:
        if not x.is_cuda:
            raise ValueError("Triton kernels require CUDA tensors")


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _attention_scale(scale: Optional[float], dim: int) -> float:
    return float(1.0 / math.sqrt(dim) if scale is None else scale)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(num_bytes)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def safe_relative_l2(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    return (torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(b), min=eps)).item()


def exp_taylor(x: Tensor, degree: int) -> Tensor:
    out = torch.ones_like(x)
    term = torch.ones_like(x)
    for r in range(1, degree + 1):
        term = term * x / float(r)
        out = out + term
    return out


def torch_classical_causal_attention(q: Tensor, k: Tensor, v: Tensor, *, scale: Optional[float] = None) -> Tensor:
    """Exact PyTorch reference for [N,D] or [B,H,N,D] tensors."""
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shape")
    if q.ndim == 2:
        n, dim = q.shape
        s = _attention_scale(scale, dim)
        logits = (q @ k.T) * s
        mask = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(logits, dim=-1) @ v
    if q.ndim == 4:
        b, h, n, dim = q.shape
        if v.shape[:3] != q.shape[:3]:
            raise ValueError("v must have shape [B,H,N,Dv]")
        s = _attention_scale(scale, dim)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) * s
        mask = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask[None, None, :, :], float("-inf"))
        return torch.softmax(logits, dim=-1) @ v
    raise ValueError("q/k/v must be [N,D] or [B,H,N,D]")


def _flatten_bh_qkv(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, ...], bool]:
    """Return contiguous [BH,N,D], [BH,N,D], [BH,N,Dv] tensors."""
    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.ndim == 2:
        if v.ndim != 2 or v.shape[0] != q.shape[0]:
            raise ValueError("for [N,D] q/k, v must be [N,Dv]")
        return q[None, :, :].contiguous(), k[None, :, :].contiguous(), v[None, :, :].contiguous(), tuple(q.shape), False
    if q.ndim == 4:
        if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
            raise ValueError("for [B,H,N,D] q/k, v must be [B,H,N,Dv]")
        b, h, n, d = q.shape
        q3 = q.contiguous().view(b * h, n, d)
        k3 = k.contiguous().view(b * h, n, d)
        v3 = v.contiguous().view(b * h, n, v.shape[-1])
        return q3, k3, v3, tuple(q.shape), True
    raise ValueError("q/k/v must be [N,D] or [B,H,N,D]")


def _unflatten_output(out3: Tensor, original_q_shape: Tuple[int, ...], had_bh: bool) -> Tensor:
    if not had_bh:
        return out3[0]
    b, h, n, _ = original_q_shape
    return out3.view(b, h, n, out3.shape[-1])


@dataclass(frozen=True)
class TritonArticleStats:
    degree: int
    block_size: int
    compress_stride: int
    coreset_items_per_sequence: int
    coreset_bytes_per_sequence: int
    kv_bytes_per_sequence: int
    score_matrix_bytes_per_sequence: int
    output_bytes_per_sequence: int


# -----------------------------------------------------------------------------
# Segment-mean residual coreset construction
# -----------------------------------------------------------------------------


@torch.no_grad()
def build_segment_mean_coreset(
    k_bhn: Tensor,
    v_bhn: Tensor,
    *,
    block_size: int,
    compress_stride: int,
) -> Tuple[Tensor, Tensor, int]:
    """Build deterministic residual reps for completed blocks.

    Args:
        k_bhn: contiguous [BH,N,D] keys.
        v_bhn: contiguous [BH,N,Dv] values.
        block_size: local block whose residual is kept exact.
        compress_stride: number of adjacent tokens averaged into one residual
            representative for completed blocks.  `compress_stride=1` disables
            compression and makes the residual exact.

    Returns:
        rep_k: [BH,R,D]
        rep_v: [BH,R,Dv]
        reps_per_block: block_size // compress_stride

    Only blocks that can be used by a later query are represented.  For example,
    if N is exactly divisible by block_size, the final full block is not placed
    in the completed-block coreset because no query occurs after its end.
    """
    if k_bhn.ndim != 3 or v_bhn.ndim != 3:
        raise ValueError("k_bhn and v_bhn must be [BH,N,D] and [BH,N,Dv]")
    if k_bhn.shape[:2] != v_bhn.shape[:2]:
        raise ValueError("k_bhn and v_bhn must have the same [BH,N]")
    if block_size <= 0 or compress_stride <= 0:
        raise ValueError("block_size and compress_stride must be positive")
    if block_size % compress_stride != 0:
        raise ValueError("for the Triton kernel, block_size must be divisible by compress_stride")

    bh, n, dim = k_bhn.shape
    dv = v_bhn.shape[-1]
    reps_per_block = block_size // compress_stride
    completed_blocks_with_later_query = max(0, (n - 1) // block_size)
    if completed_blocks_with_later_query == 0:
        return (
            torch.empty(bh, 0, dim, device=k_bhn.device, dtype=k_bhn.dtype),
            torch.empty(bh, 0, dv, device=v_bhn.device, dtype=v_bhn.dtype),
            reps_per_block,
        )

    usable = completed_blocks_with_later_query * block_size
    kk = k_bhn[:, :usable, :].reshape(
        bh, completed_blocks_with_later_query, reps_per_block, compress_stride, dim
    )
    vv = v_bhn[:, :usable, :].reshape(
        bh, completed_blocks_with_later_query, reps_per_block, compress_stride, dv
    )
    rep_k = kk.mean(dim=3).reshape(bh, completed_blocks_with_later_query * reps_per_block, dim).contiguous()
    rep_v = vv.mean(dim=3).reshape(bh, completed_blocks_with_later_query * reps_per_block, dv).contiguous()
    return rep_k, rep_v, reps_per_block


# -----------------------------------------------------------------------------
# Triton kernels
# -----------------------------------------------------------------------------


if triton is not None and tl is not None:

    @triton.jit
    def _taylor_poly(x, DEGREE: tl.constexpr):
        y = tl.full(x.shape, 1.0, tl.float32)
        if DEGREE >= 1:
            y += x
        if DEGREE >= 2:
            y += 0.5 * x * x
        return y


    @triton.jit
    def _article_direct_kernel(
        Q,
        K,
        V,
        RK,
        RV,
        O,
        N: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        R: tl.constexpr,
        SCALE: tl.constexpr,
        DENOM_EPS: tl.constexpr,
        ARTICLE_BLOCK_SIZE: tl.constexpr,
        REPS_PER_BLOCK: tl.constexpr,
        COMPRESS_STRIDE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_R: tl.constexpr,
        DEGREE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_dv = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        q = tl.load(
            Q + pid_bh * N * D + offs_m[:, None] * D + offs_d[None, :],
            mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        low_den = tl.zeros((BLOCK_M,), tl.float32)
        low_acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)
        tail_den = tl.zeros((BLOCK_M,), tl.float32)
        tail_acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

        # Exact low-degree Taylor term against the full causal prefix, plus exact
        # residual within the current article block.
        for start_n in tl.range(0, N, BLOCK_N, num_stages=3):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + pid_bh * N * D + offs_n[:, None] * D + offs_d[None, :],
                mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + pid_bh * N * DV + offs_n[:, None] * DV + offs_dv[None, :],
                mask=(offs_n[:, None] < N) & (offs_dv[None, :] < DV),
                other=0.0,
            ).to(tl.float32)

            x = tl.dot(q, tl.trans(k), input_precision="tf32") * SCALE
            causal = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < N) & (offs_m[:, None] < N)
            taylor = tl.where(causal, _taylor_poly(x, DEGREE), 0.0)
            low_den += tl.sum(taylor, axis=1)
            low_acc += tl.dot(taylor, v, input_precision="tf32")

            same_article_block = (offs_n[None, :] // ARTICLE_BLOCK_SIZE) == (offs_m[:, None] // ARTICLE_BLOCK_SIZE)
            local_residual_mask = causal & same_article_block
            # Clamp only the exponential input to keep pathological non-high-
            # temperature tests from overflowing the tail term.
            exp_x = tl.exp(tl.minimum(x, 80.0))
            residual = tl.where(local_residual_mask, exp_x - _taylor_poly(x, DEGREE), 0.0)
            tail_den += tl.sum(residual, axis=1)
            tail_acc += tl.dot(residual, v, input_precision="tf32")

        # Residual for completed previous blocks, using compressed reps.
        for start_r in tl.range(0, R, BLOCK_R, num_stages=3):
            offs_r = start_r + tl.arange(0, BLOCK_R)
            rk = tl.load(
                RK + pid_bh * R * D + offs_r[:, None] * D + offs_d[None, :],
                mask=(offs_r[:, None] < R) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            rv = tl.load(
                RV + pid_bh * R * DV + offs_r[:, None] * DV + offs_dv[None, :],
                mask=(offs_r[:, None] < R) & (offs_dv[None, :] < DV),
                other=0.0,
            ).to(tl.float32)

            x = tl.dot(q, tl.trans(rk), input_precision="tf32") * SCALE
            rep_block = offs_r // REPS_PER_BLOCK
            active_from = (rep_block + 1) * ARTICLE_BLOCK_SIZE
            active = (offs_m[:, None] >= active_from[None, :]) & (offs_m[:, None] < N) & (offs_r[None, :] < R)
            exp_x = tl.exp(tl.minimum(x, 80.0))
            residual = tl.where(active, exp_x - _taylor_poly(x, DEGREE), 0.0) * COMPRESS_STRIDE
            tail_den += tl.sum(residual, axis=1)
            tail_acc += tl.dot(residual, rv, input_precision="tf32")

        den = low_den + tail_den
        acc = low_acc + tail_acc
        out = acc / tl.maximum(den[:, None], DENOM_EPS)

        tl.store(
            O + pid_bh * N * DV + offs_m[:, None] * DV + offs_dv[None, :],
            out,
            mask=(offs_m[:, None] < N) & (offs_dv[None, :] < DV),
        )


    @triton.jit
    def _flash_causal_kernel(
        Q,
        K,
        V,
        O,
        N: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_dv = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

        q = tl.load(
            Q + pid_bh * N * D + offs_m[:, None] * D + offs_d[None, :],
            mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

        for start_n in tl.range(0, N, BLOCK_N, num_stages=3):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + pid_bh * N * D + offs_n[:, None] * D + offs_d[None, :],
                mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                V + pid_bh * N * DV + offs_n[:, None] * DV + offs_dv[None, :],
                mask=(offs_n[:, None] < N) & (offs_dv[None, :] < DV),
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k), input_precision="tf32") * SCALE
            causal = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < N) & (offs_m[:, None] < N)
            scores = tl.where(causal, scores, -float("inf"))

            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_m)
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32")
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        out = acc / tl.maximum(l_i[:, None], 1.0e-20)
        tl.store(
            O + pid_bh * N * DV + offs_m[:, None] * DV + offs_dv[None, :],
            out,
            mask=(offs_m[:, None] < N) & (offs_dv[None, :] < DV),
        )

else:  # pragma: no cover - only used when Triton is unavailable.
    _article_direct_kernel = None
    _flash_causal_kernel = None


# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------


@torch.no_grad()
def triton_article_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    block_size: int = 128,
    compress_stride: int = 2,
    scale: Optional[float] = None,
    block_m: int = 16,
    block_n: int = 64,
    block_r: int = 64,
    value_block: Optional[int] = None,
    denominator_eps: float = 1.0e-12,
    output_dtype: Optional[torch.dtype] = torch.float32,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, TritonArticleStats]]:
    """Approximate causal attention using fused Triton CUDA kernels.

    Args:
        q, k, v: `[N,D]` or `[B,H,N,D]` tensors.  q/k must have the same
            shape.  v must share the same leading dimensions and sequence length.
        degree: Taylor degree.  The kernel supports 0, 1, and 2.
        block_size: residual is exact within each local block of this size.
        compress_stride: number of adjacent tokens averaged into one completed-
            block residual representative.  `1` disables residual compression and
            should match exact softmax attention up to roundoff.
        scale: attention scale; default is `1/sqrt(D)`.
        block_m, block_n, block_r: Triton tile sizes for query rows, exact-key
            tiles, and residual-representative tiles.
        value_block: value-dimension tile size.  Defaults to the next power of
            two of Dv, capped at 64.
        output_dtype: dtype of the output tensor.  `torch.float32` is the default
            because the kernel accumulates in fp32.  Pass `q.dtype` to store half
            precision outputs.
        return_stats: return rough memory/state statistics with the output.
    """
    _require_triton()
    _require_cuda(q, k, v)
    if degree not in (0, 1, 2):
        raise NotImplementedError("Triton article kernel currently supports degree 0, 1, or 2")
    if block_size <= 0 or compress_stride <= 0:
        raise ValueError("block_size and compress_stride must be positive")
    if block_size % compress_stride != 0:
        raise ValueError("block_size must be divisible by compress_stride")

    q3, k3, v3, original_shape, had_bh = _flatten_bh_qkv(q, k, v)
    _require_cuda(q3, k3, v3)
    bh, n, dim = q3.shape
    dv = v3.shape[-1]
    if dim > 128:
        raise ValueError("this compact kernel supports D <= 128; increase BLOCK_D logic for larger heads")
    if dv > 256:
        raise ValueError("this compact kernel supports Dv <= 256")

    # Contiguous fp16/bf16/fp32 inputs are supported.  Accumulation is fp32.
    if q3.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        q3 = q3.float()
        k3 = k3.float()
    if v3.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        v3 = v3.float()

    q3 = q3.contiguous()
    k3 = k3.contiguous()
    v3 = v3.contiguous()

    rep_k, rep_v, reps_per_block = build_segment_mean_coreset(
        k3,
        v3,
        block_size=block_size,
        compress_stride=compress_stride,
    )
    r = rep_k.shape[1]

    block_d = _next_power_of_2(dim)
    if value_block is None:
        block_dv = min(64, _next_power_of_2(dv))
    else:
        block_dv = _next_power_of_2(value_block)
    block_dv = max(1, min(block_dv, 256))

    s = _attention_scale(scale, dim)
    out_dtype = output_dtype or torch.float32
    out3 = torch.empty((bh, n, dv), device=q3.device, dtype=out_dtype)

    grid = (_ceil_div(n, block_m), bh, _ceil_div(dv, block_dv))
    assert _article_direct_kernel is not None
    _article_direct_kernel[grid](
        q3,
        k3,
        v3,
        rep_k,
        rep_v,
        out3,
        N=n,
        D=dim,
        DV=dv,
        R=r,
        SCALE=s,
        DENOM_EPS=float(denominator_eps),
        ARTICLE_BLOCK_SIZE=block_size,
        REPS_PER_BLOCK=reps_per_block,
        COMPRESS_STRIDE=compress_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_R=block_r,
        DEGREE=degree,
        num_warps=4 if block_m <= 16 else 8,
        num_stages=3,
    )

    out = _unflatten_output(out3, original_shape, had_bh)
    if not return_stats:
        return out

    elem_kv = q3.element_size()
    elem_out = out3.element_size()
    stats = TritonArticleStats(
        degree=degree,
        block_size=block_size,
        compress_stride=compress_stride,
        coreset_items_per_sequence=int(r),
        coreset_bytes_per_sequence=int(r * (dim + dv) * elem_kv),
        kv_bytes_per_sequence=int(n * (dim + dv) * elem_kv),
        score_matrix_bytes_per_sequence=int(n * n * elem_kv),
        output_bytes_per_sequence=int(n * dv * elem_out),
    )
    return out, stats


@torch.no_grad()
def triton_flash_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: Optional[float] = None,
    block_m: int = 16,
    block_n: int = 64,
    value_block: Optional[int] = None,
    output_dtype: Optional[torch.dtype] = torch.float32,
) -> Tensor:
    """Exact causal attention baseline using a FlashAttention-style Triton kernel.

    This avoids materializing the N x N score matrix.  It supports `[N,D]` and
    `[B,H,N,D]` tensors and accumulates in fp32.
    """
    _require_triton()
    _require_cuda(q, k, v)
    q3, k3, v3, original_shape, had_bh = _flatten_bh_qkv(q, k, v)
    bh, n, dim = q3.shape
    dv = v3.shape[-1]
    if dim > 128:
        raise ValueError("this compact kernel supports D <= 128")
    if dv > 256:
        raise ValueError("this compact kernel supports Dv <= 256")

    q3 = q3.contiguous()
    k3 = k3.contiguous()
    v3 = v3.contiguous()
    block_d = _next_power_of_2(dim)
    if value_block is None:
        block_dv = min(64, _next_power_of_2(dv))
    else:
        block_dv = _next_power_of_2(value_block)
    block_dv = max(1, min(block_dv, 256))
    s = _attention_scale(scale, dim)
    out3 = torch.empty((bh, n, dv), device=q3.device, dtype=output_dtype or torch.float32)

    grid = (_ceil_div(n, block_m), bh, _ceil_div(dv, block_dv))
    assert _flash_causal_kernel is not None
    _flash_causal_kernel[grid](
        q3,
        k3,
        v3,
        out3,
        N=n,
        D=dim,
        DV=dv,
        SCALE=s,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        num_warps=4 if block_m <= 16 else 8,
        num_stages=3,
    )
    return _unflatten_output(out3, original_shape, had_bh)


# -----------------------------------------------------------------------------
# Demo / benchmark CLI
# -----------------------------------------------------------------------------


def _make_data(
    n: int,
    d: int,
    dv: int,
    *,
    batch: int,
    heads: int,
    radius: float,
    data: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shape_q = (batch, heads, n, d) if batch * heads > 1 else (n, d)
    shape_v = (batch, heads, n, dv) if batch * heads > 1 else (n, dv)
    q = torch.randn(shape_q, device=device, dtype=dtype, generator=g)
    k = torch.randn(shape_q, device=device, dtype=dtype, generator=g)
    v = torch.randn(shape_v, device=device, dtype=dtype, generator=g)

    if data == "unit_ball":
        q_norm = torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12)
        k_norm = torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12)
        rq = torch.rand((*q.shape[:-1], 1), device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
        rk = torch.rand((*k.shape[:-1], 1), device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
        q = q / q_norm * rq
        k = k / k_norm * rk
    elif data == "unit_sphere":
        q = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12) * radius
        k = k / torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12) * radius
    elif data == "normal":
        pass
    else:
        raise ValueError("data must be one of {'unit_ball', 'unit_sphere', 'normal'}")
    return q, k, v


def _resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(table)}")
    return table[key]


def _bench_cuda_ms(fn, repeats: int) -> float:
    # CUDA events avoid measuring host-side scheduling latency.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def _demo(args: argparse.Namespace) -> None:
    _require_triton()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)
    scale = args.scale if args.scale is not None else (1.0 / math.sqrt(args.d) if args.standard_scale else 1.0)

    headers = [
        "n",
        "rel_l2_article",
        "max_abs_article",
        "rel_l2_flash",
        "torch_classical_ms",
        "triton_flash_ms",
        "triton_article_ms",
        "coreset_items",
        "coreset_bytes",
        "kv_bytes",
        "score_matrix",
    ]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    for n in args.sweep_n:
        q, k, v = _make_data(
            n,
            args.d,
            args.dv,
            batch=args.batch,
            heads=args.heads,
            radius=args.radius,
            data=args.data,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        exact = torch_classical_causal_attention(q.float(), k.float(), v.float(), scale=scale)
        flash = triton_flash_causal_attention(
            q,
            k,
            v,
            scale=scale,
            block_m=args.block_m,
            block_n=args.block_n,
            value_block=args.value_block,
            output_dtype=torch.float32,
        )
        article, stats = triton_article_causal_attention(
            q,
            k,
            v,
            degree=args.degree,
            block_size=args.block_size,
            compress_stride=args.compress_stride,
            scale=scale,
            block_m=args.block_m,
            block_n=args.block_n,
            block_r=args.block_r,
            value_block=args.value_block,
            output_dtype=torch.float32,
            return_stats=True,
        )

        torch_ms = _bench_cuda_ms(lambda: torch_classical_causal_attention(q.float(), k.float(), v.float(), scale=scale), args.repeats)
        flash_ms = _bench_cuda_ms(
            lambda: triton_flash_causal_attention(
                q,
                k,
                v,
                scale=scale,
                block_m=args.block_m,
                block_n=args.block_n,
                value_block=args.value_block,
                output_dtype=torch.float32,
            ),
            args.repeats,
        )
        article_ms = _bench_cuda_ms(
            lambda: triton_article_causal_attention(
                q,
                k,
                v,
                degree=args.degree,
                block_size=args.block_size,
                compress_stride=args.compress_stride,
                scale=scale,
                block_m=args.block_m,
                block_n=args.block_n,
                block_r=args.block_r,
                value_block=args.value_block,
                output_dtype=torch.float32,
            ),
            args.repeats,
        )

        print(
            " | ".join(
                [
                    str(n),
                    f"{safe_relative_l2(article, exact):.4g}",
                    f"{(article - exact).abs().max().item():.4g}",
                    f"{safe_relative_l2(flash, exact):.4g}",
                    f"{torch_ms:.3f}",
                    f"{flash_ms:.3f}",
                    f"{article_ms:.3f}",
                    str(stats.coreset_items_per_sequence),
                    _human_bytes(stats.coreset_bytes_per_sequence),
                    _human_bytes(stats.kv_bytes_per_sequence),
                    _human_bytes(stats.score_matrix_bytes_per_sequence),
                ]
            )
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton kernels for article-style Taylor + residual-coreset attention")
    parser.add_argument("--sweep-n", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--dv", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--degree", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--compress-stride", type=int, default=2, help="1 disables residual compression and should match exact attention")
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-r", type=int, default=64)
    parser.add_argument("--value-block", type=int, default=None)
    parser.add_argument("--data", choices=["unit_ball", "unit_sphere", "normal"], default="unit_ball")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--standard-scale", action="store_true", help="Use 1/sqrt(d) when --scale is omitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "fp32", "float16", "fp16", "half", "bfloat16", "bf16"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    _demo(_parse_args())
