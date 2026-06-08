"""
Fast PyTorch implementation of the article-style Taylor + residual-coreset
attention approximation.

This file is intentionally separate from `streaming_taylor_coreset_attention.py`.
The original module is a literal streaming reference implementation; this module
uses vectorized cumulative sums and chunked matrix multiplies to be much faster
for benchmarking and experimentation.

Implemented approximation:
  softmax numerator/denominator are decomposed as

      exp(<q,k>) = Taylor_<=t(<q,k>) + Residual_>t(<q,k>).

  * The Taylor part is exact for degrees 0, 1, and 2, using a symmetric
    monomial feature map and prefix cumulative sums.
  * The residual part is approximated by a fast block coreset:
      - previous completed blocks are represented by half-sized sampled pairs;
      - the current block prefix is evaluated exactly;
      - `compressor="none"` evaluates the residual exactly against all previous
        keys and should match classical causal attention up to roundoff.

The block coreset is not byte-for-byte the same as the merge-and-reduce object in
`streaming_taylor_coreset_attention.py`; it is a faster vectorized variant meant
for speed/memory sweeps.  The reference implementation remains the closer match
to the article's streaming data structure.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    key = str(dtype).lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(table)}")
    return table[key]


def _attention_scale(scale: Optional[float], dim: int) -> float:
    return float(1.0 / math.sqrt(dim) if scale is None else scale)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(num_bytes)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def exp_taylor(logits: Tensor, degree: int) -> Tensor:
    """Elementwise sum_{r=0}^degree logits**r / r!."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    out = torch.ones_like(logits)
    term = torch.ones_like(logits)
    for r in range(1, degree + 1):
        term = term * logits / float(r)
        out = out + term
    return out


def safe_relative_l2(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    return (torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(b), min=eps)).item()


# -----------------------------------------------------------------------------
# Exact classical baseline
# -----------------------------------------------------------------------------


@torch.no_grad()
def classical_causal_attention(q: Tensor, k: Tensor, v: Tensor, *, scale: Optional[float] = None) -> Tensor:
    """Exact causal softmax attention for [N,D] tensors."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("classical_causal_attention expects q,k,v with shapes [N,D], [N,D], [N,Dv]")
    if q.shape != k.shape or v.shape[0] != k.shape[0]:
        raise ValueError("q and k must match, and v must have the same sequence length")

    n, dim = q.shape
    s = _attention_scale(scale, dim)
    logits = (q @ k.T) * s
    mask = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


# -----------------------------------------------------------------------------
# Fast low-degree Taylor moments
# -----------------------------------------------------------------------------


def num_taylor_features(dim: int, degree: int) -> int:
    """Number of symmetric Taylor features for degree <= 2."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree > 2:
        raise NotImplementedError("fast implementation currently supports degree 0, 1, or 2")
    total = 1
    if degree >= 1:
        total += dim
    if degree >= 2:
        total += dim * (dim + 1) // 2
    return total


def _taylor_features_degree012(x: Tensor, degree: int, *, key_side: bool) -> Tensor:
    """Symmetric monomial features through degree 2.

    For keys, degree-2 diagonal terms include the coefficient 1/2 so that

        <phi_key(k), phi_query(q)> = 1 + k·q + (k·q)^2/2.

    The caller should pass scaled queries, i.e. q_eff = scale * q.
    """
    if degree > 2:
        raise NotImplementedError("fast implementation currently supports degree 0, 1, or 2")
    if x.ndim < 2:
        raise ValueError("x must have shape [..., N, D] or [N, D]")

    leading = x.shape[:-1]
    dim = x.shape[-1]
    feats: List[Tensor] = [torch.ones(*leading, 1, device=x.device, dtype=x.dtype)]

    if degree >= 1:
        feats.append(x)

    if degree >= 2:
        row, col = torch.triu_indices(dim, dim, device=x.device)
        quad = x[..., row] * x[..., col]
        if key_side:
            coeff = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
            coeff = torch.where(row == col, coeff * 0.5, coeff)
            quad = quad * coeff
        feats.append(quad)

    return torch.cat(feats, dim=-1)


@torch.no_grad()
def low_degree_taylor_causal_sums_prefix(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, int]:
    """Exact causal Taylor sums via full prefix tensors.

    This path is highly vectorized and may be best on GPU, but on CPU the large
    [N, feature_count, Dv] cumulative sum can be slower than the streaming path.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("low_degree_taylor_causal_sums_prefix expects [N,D], [N,D], [N,Dv]")
    if q.shape != k.shape or v.shape[0] != k.shape[0]:
        raise ValueError("q and k must match, and v must have the same sequence length")

    _, dim = q.shape
    s = _attention_scale(scale, dim)
    q_eff = q * s

    f_key = _taylor_features_degree012(k, degree, key_side=True)
    f_query = _taylor_features_degree012(q_eff, degree, key_side=False)

    prefix_key = f_key.cumsum(dim=0)
    low_den = (f_query * prefix_key).sum(dim=-1)
    prefix_key_value = (f_key[:, :, None] * v[:, None, :]).cumsum(dim=0)
    low_num = (f_query[:, :, None] * prefix_key_value).sum(dim=1)
    return low_num, low_den, int(f_key.shape[-1])


@torch.no_grad()
def low_degree_taylor_causal_sums_stream(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, int]:
    """Exact causal Taylor sums using a tight moment loop.

    This keeps only the current Taylor moments, so it is both closer to the
    article's streaming state and usually much faster on CPU than materializing
    all prefix value-moment tensors.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("low_degree_taylor_causal_sums_stream expects [N,D], [N,D], [N,Dv]")
    if q.shape != k.shape or v.shape[0] != k.shape[0]:
        raise ValueError("q and k must match, and v must have the same sequence length")

    n, dim = q.shape
    value_dim = v.shape[-1]
    s = _attention_scale(scale, dim)
    q_eff = q * s

    f_key = _taylor_features_degree012(k, degree, key_side=True)
    f_query = _taylor_features_degree012(q_eff, degree, key_side=False)
    feature_count = int(f_key.shape[-1])

    den_state = torch.zeros(feature_count, device=q.device, dtype=q.dtype)
    num_state = torch.zeros(feature_count, value_dim, device=q.device, dtype=q.dtype)
    low_den = torch.empty(n, device=q.device, dtype=q.dtype)
    low_num = torch.empty(n, value_dim, device=q.device, dtype=q.dtype)

    for i in range(n):
        fi = f_key[i]
        den_state.add_(fi)
        num_state.add_(fi[:, None] * v[i][None, :])
        qi = f_query[i]
        low_den[i] = torch.dot(qi, den_state)
        low_num[i] = qi @ num_state

    return low_num, low_den, feature_count


@torch.no_grad()
def low_degree_taylor_causal_sums(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    scale: Optional[float] = None,
    mode: str = "auto",
) -> Tuple[Tensor, Tensor, int]:
    """Exact causal Taylor numerator and denominator through degree <= 2.

    mode:
        "stream" keeps a moment state and is usually fastest on CPU.
        "prefix" materializes prefix tensors and can be better on GPU.
        "auto" uses "stream" on CPU and "prefix" on CUDA.
    """
    if mode == "auto":
        mode = "prefix" if q.device.type == "cuda" else "stream"
    if mode == "stream":
        return low_degree_taylor_causal_sums_stream(q, k, v, degree=degree, scale=scale)
    if mode == "prefix":
        return low_degree_taylor_causal_sums_prefix(q, k, v, degree=degree, scale=scale)
    raise ValueError("mode must be one of {'auto', 'stream', 'prefix'}")


# -----------------------------------------------------------------------------
# Fast block residual coreset
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FastArticleStats:
    degree: int
    block_size: int
    feature_count: int
    coreset_items: int
    approx_streaming_state_bytes: int
    scratch_low_prefix_bytes: int
    scratch_residual_matrix_bytes: int


@dataclass(frozen=True)
class _BlockReps:
    keys: Tensor
    values: Tensor
    weights: Tensor
    active_from: Tensor


@torch.no_grad()
def _compress_one_completed_block(
    keys: Tensor,
    values: Tensor,
    *,
    compressor: str,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compress one already-completed block to roughly half its size."""
    n, dim = keys.shape
    if n == 0:
        empty_w = torch.empty(0, device=keys.device, dtype=keys.dtype)
        return keys, values, empty_w

    if compressor == "random":
        m = (n + 1) // 2
        idx = torch.randperm(n, generator=generator, device=keys.device)[:m]
        weight = torch.full((m,), float(n) / float(m), device=keys.device, dtype=keys.dtype)
        return keys[idx], values[idx], weight

    if compressor == "sorted_pair":
        if n <= 2:
            m = (n + 1) // 2
            idx = torch.randperm(n, generator=generator, device=keys.device)[:m]
            weight = torch.full((m,), float(n) / float(m), device=keys.device, dtype=keys.dtype)
            return keys[idx], values[idx], weight

        direction = torch.randn(dim, generator=generator, device=keys.device, dtype=keys.dtype)
        direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-12)
        order = torch.argsort(keys @ direction)
        kk = keys[order]
        vv = values[order]

        pair_count = n // 2
        a = torch.arange(0, 2 * pair_count, 2, device=keys.device)
        b = a + 1
        choose_a = torch.rand(pair_count, generator=generator, device=keys.device, dtype=keys.dtype) < 0.5
        idx = torch.where(choose_a, a, b)

        rep_k = kk[idx]
        rep_v = vv[idx]
        rep_w = torch.full((pair_count,), 2.0, device=keys.device, dtype=keys.dtype)

        if n % 2 == 1:
            rep_k = torch.cat([rep_k, kk[-1:]], dim=0)
            rep_v = torch.cat([rep_v, vv[-1:]], dim=0)
            rep_w = torch.cat([rep_w, torch.ones(1, device=keys.device, dtype=keys.dtype)], dim=0)
        return rep_k, rep_v, rep_w

    raise ValueError("compressor must be one of {'sorted_pair', 'random', 'none'}")


@torch.no_grad()
def _build_completed_block_reps(
    k: Tensor,
    v: Tensor,
    *,
    block_size: int,
    compressor: str,
    seed: int,
) -> _BlockReps:
    """Build coreset reps for blocks that have at least one later query.

    A block [s,e) is represented for queries i >= e.  Queries inside [s,e) use
    the exact current-block residual instead, so there is no double counting.
    """
    n, dim = k.shape
    value_dim = v.shape[-1]
    device = k.device
    dtype = k.dtype

    if compressor == "none":
        pos = torch.arange(n, device=device)
        return _BlockReps(
            keys=k,
            values=v,
            weights=torch.ones(n, device=device, dtype=dtype),
            active_from=pos,
        )

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    rep_keys: List[Tensor] = []
    rep_values: List[Tensor] = []
    rep_weights: List[Tensor] = []
    rep_active: List[Tensor] = []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # No later query can use the final block's completed representation.
        if end >= n:
            break
        kk, vv, ww = _compress_one_completed_block(
            k[start:end],
            v[start:end],
            compressor=compressor,
            generator=generator,
        )
        if ww.numel() == 0:
            continue
        rep_keys.append(kk)
        rep_values.append(vv)
        rep_weights.append(ww)
        rep_active.append(torch.full((ww.numel(),), end, device=device, dtype=torch.long))

    if not rep_keys:
        return _BlockReps(
            keys=torch.empty(0, dim, device=device, dtype=dtype),
            values=torch.empty(0, value_dim, device=device, dtype=dtype),
            weights=torch.empty(0, device=device, dtype=dtype),
            active_from=torch.empty(0, device=device, dtype=torch.long),
        )

    return _BlockReps(
        keys=torch.cat(rep_keys, dim=0),
        values=torch.cat(rep_values, dim=0),
        weights=torch.cat(rep_weights, dim=0),
        active_from=torch.cat(rep_active, dim=0),
    )


@torch.no_grad()
def _full_residual_tail(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int,
    scale: float,
    query_chunk_size: int,
) -> Tuple[Tensor, Tensor, int]:
    """Exact residual term for all causal prefixes, evaluated in chunks."""
    n = q.shape[0]
    value_dim = v.shape[-1]
    tail_den = torch.zeros(n, device=q.device, dtype=q.dtype)
    tail_num = torch.zeros(n, value_dim, device=q.device, dtype=q.dtype)
    key_pos = torch.arange(n, device=q.device)

    for start in range(0, n, query_chunk_size):
        end = min(start + query_chunk_size, n)
        logits = (q[start:end] @ k.T) * scale
        residual = torch.exp(logits) - exp_taylor(logits, degree)
        query_pos = torch.arange(start, end, device=q.device)
        causal = key_pos[None, :] <= query_pos[:, None]
        residual = residual.masked_fill(~causal, 0.0)
        tail_den[start:end] = residual.sum(dim=-1)
        tail_num[start:end] = residual @ v

    return tail_num, tail_den, n


@torch.no_grad()
def _block_coreset_residual_tail(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int,
    block_size: int,
    scale: float,
    compressor: str,
    seed: int,
    query_chunk_size: int,
) -> Tuple[Tensor, Tensor, int, int]:
    """Approximate residual via completed block reps + exact current block."""
    n = q.shape[0]
    value_dim = v.shape[-1]
    tail_den = torch.zeros(n, device=q.device, dtype=q.dtype)
    tail_num = torch.zeros(n, value_dim, device=q.device, dtype=q.dtype)

    reps = _build_completed_block_reps(k, v, block_size=block_size, compressor=compressor, seed=seed)

    # Previous completed blocks: one big q x reps matmul, chunked by queries.
    if reps.weights.numel() > 0:
        for start in range(0, n, query_chunk_size):
            end = min(start + query_chunk_size, n)
            logits = (q[start:end] @ reps.keys.T) * scale
            residual = torch.exp(logits) - exp_taylor(logits, degree)
            query_pos = torch.arange(start, end, device=q.device)
            active = query_pos[:, None] >= reps.active_from[None, :]
            residual = residual.masked_fill(~active, 0.0)
            weighted = residual * reps.weights[None, :]
            tail_den[start:end] += weighted.sum(dim=-1)
            tail_num[start:end] += weighted @ reps.values

    # Current block prefix: exact residual inside each block.
    max_block_scratch = 0
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        bs = end - start
        logits = (q[start:end] @ k[start:end].T) * scale
        residual = torch.exp(logits) - exp_taylor(logits, degree)
        causal = torch.ones(bs, bs, device=q.device, dtype=torch.bool).tril()
        residual = residual.masked_fill(~causal, 0.0)
        tail_den[start:end] += residual.sum(dim=-1)
        tail_num[start:end] += residual @ v[start:end]
        max_block_scratch = max(max_block_scratch, bs * bs * q.element_size())

    return tail_num, tail_den, int(reps.weights.numel()), max_block_scratch


# -----------------------------------------------------------------------------
# Public fast attention function
# -----------------------------------------------------------------------------


@torch.no_grad()
def fast_article_causal_attention_single(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    block_size: int = 128,
    scale: Optional[float] = None,
    compressor: str = "sorted_pair",
    seed: int = 0,
    query_chunk_size: int = 2048,
    low_mode: str = "auto",
    denominator_eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, FastArticleStats]]:
    """Fast approximate causal attention for one [N,D] sequence.

    Args:
        q, k, v: tensors with shapes [N,D], [N,D], [N,Dv].
        degree: Taylor degree, currently 0, 1, or 2.
        block_size: size of the exact current-block window and compression unit.
        scale: attention scale.  If omitted, uses 1/sqrt(D).
        compressor: "sorted_pair", "random", or "none".  "none" computes the
            residual exactly and should reproduce classical attention.
        seed: RNG seed for randomized block compression.
        query_chunk_size: chunk size for q x coreset matrix multiplies.
        low_mode: "auto", "stream", or "prefix" for the Taylor moment path.
        return_stats: when True, also return rough state/scratch statistics.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q,k,v must have shapes [N,D], [N,D], [N,Dv]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if v.shape[0] != k.shape[0]:
        raise ValueError("v must have the same sequence length as q/k")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n, dim = q.shape
    value_dim = v.shape[-1]
    s = _attention_scale(scale, dim)

    low_num, low_den, feature_count = low_degree_taylor_causal_sums(
        q, k, v, degree=degree, scale=s, mode=low_mode
    )

    if compressor == "none":
        tail_num, tail_den, coreset_items = _full_residual_tail(
            q,
            k,
            v,
            degree=degree,
            scale=s,
            query_chunk_size=query_chunk_size,
        )
        residual_scratch_bytes = min(query_chunk_size, n) * n * q.element_size()
    else:
        tail_num, tail_den, coreset_items, block_scratch_bytes = _block_coreset_residual_tail(
            q,
            k,
            v,
            degree=degree,
            block_size=block_size,
            scale=s,
            compressor=compressor,
            seed=seed,
            query_chunk_size=query_chunk_size,
        )
        reps_scratch_bytes = min(query_chunk_size, n) * max(coreset_items, 1) * q.element_size()
        residual_scratch_bytes = max(reps_scratch_bytes, block_scratch_bytes)

    den = low_den + tail_den
    num = low_num + tail_num
    out = num / torch.clamp(den[:, None], min=denominator_eps)

    if not return_stats:
        return out

    elem = q.element_size()
    # Streaming state estimate: current Taylor moments + compressed residual reps
    # + one uncompressed current block.  This excludes temporary vectorized prefix
    # arrays used internally for speed.
    low_state_bytes = feature_count * (1 + value_dim) * elem
    residual_state_bytes = (
        coreset_items * (dim + value_dim + 1) * elem
        + min(block_size, n) * (dim + value_dim) * elem
    )
    approx_state = low_state_bytes + residual_state_bytes

    # Scratch estimate for the low Taylor path.  Streaming mode keeps only the
    # current moments; prefix mode materializes all prefix value-moment tensors.
    resolved_low_mode = "prefix" if (low_mode == "auto" and q.device.type == "cuda") else ("stream" if low_mode == "auto" else low_mode)
    if resolved_low_mode == "stream":
        low_prefix_bytes = feature_count * (1 + value_dim) * elem
    else:
        low_prefix_bytes = n * feature_count * (1 + value_dim) * elem

    stats = FastArticleStats(
        degree=degree,
        block_size=block_size,
        feature_count=feature_count,
        coreset_items=coreset_items,
        approx_streaming_state_bytes=int(approx_state),
        scratch_low_prefix_bytes=int(low_prefix_bytes),
        scratch_residual_matrix_bytes=int(residual_scratch_bytes),
    )
    return out, stats


@torch.no_grad()
def fast_article_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    block_size: int = 128,
    scale: Optional[float] = None,
    compressor: str = "sorted_pair",
    seed: int = 0,
    query_chunk_size: int = 2048,
    low_mode: str = "auto",
    denominator_eps: float = 1e-12,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, FastArticleStats]]:
    """Fast approximate causal attention for [N,D] or [B,H,N,D] tensors.

    Batched/multi-head inputs loop over the flattened leading dimensions, but
    each sequence is processed with vectorized cumulative sums and matmuls.
    """
    if q.ndim == 2:
        return fast_article_causal_attention_single(
            q,
            k,
            v,
            degree=degree,
            block_size=block_size,
            scale=scale,
            compressor=compressor,
            seed=seed,
            query_chunk_size=query_chunk_size,
            low_mode=low_mode,
            denominator_eps=denominator_eps,
            return_stats=return_stats,
        )

    if q.ndim != 4:
        raise ValueError("q/k/v must be [N,D] or [B,H,N,D]")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shape [B,H,N,D]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B,H,N,Dv]")

    bsz, heads, n, dim = q.shape
    value_dim = v.shape[-1]
    out = torch.empty(bsz, heads, n, value_dim, device=q.device, dtype=q.dtype)
    stats_list: List[FastArticleStats] = []

    for b in range(bsz):
        for h in range(heads):
            result = fast_article_causal_attention_single(
                q[b, h],
                k[b, h],
                v[b, h],
                degree=degree,
                block_size=block_size,
                scale=scale,
                compressor=compressor,
                seed=seed + b * heads + h,
                query_chunk_size=query_chunk_size,
                low_mode=low_mode,
                denominator_eps=denominator_eps,
                return_stats=return_stats,
            )
            if return_stats:
                seq_out, seq_stats = result  # type: ignore[misc]
                out[b, h] = seq_out
                stats_list.append(seq_stats)
            else:
                out[b, h] = result  # type: ignore[assignment]

    if not return_stats:
        return out

    # Aggregate conservative maxima over sequences.
    stats = FastArticleStats(
        degree=degree,
        block_size=block_size,
        feature_count=stats_list[0].feature_count if stats_list else num_taylor_features(dim, degree),
        coreset_items=max((s.coreset_items for s in stats_list), default=0),
        approx_streaming_state_bytes=max((s.approx_streaming_state_bytes for s in stats_list), default=0),
        scratch_low_prefix_bytes=max((s.scratch_low_prefix_bytes for s in stats_list), default=0),
        scratch_residual_matrix_bytes=max((s.scratch_residual_matrix_bytes for s in stats_list), default=0),
    )
    return out, stats


# -----------------------------------------------------------------------------
# Tiny built-in benchmark CLI
# -----------------------------------------------------------------------------


def _make_data(
    n: int,
    d: int,
    dv: int,
    *,
    radius: float,
    data: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    k = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    v = torch.randn(n, dv, device=device, dtype=dtype, generator=g)

    if data == "unit_ball":
        q_norm = torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12)
        k_norm = torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12)
        rq = torch.rand(n, 1, device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
        rk = torch.rand(n, 1, device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
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


def _time_call(fn, repeats: int, device: torch.device) -> Tuple[float, float]:
    fn()
    _sync(device)
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), min(times)


def _demo(args: argparse.Namespace) -> None:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    scale = args.scale if args.scale is not None else (1.0 / math.sqrt(args.d) if args.standard_scale else 1.0)

    rows = []
    for n in args.sweep_n:
        q, k, v = _make_data(
            n,
            args.d,
            args.dv,
            radius=args.radius,
            data=args.data,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        exact = classical_causal_attention(q, k, v, scale=scale)
        fast_result = fast_article_causal_attention(
            q,
            k,
            v,
            degree=args.degree,
            block_size=args.block_size,
            scale=scale,
            compressor=args.compressor,
            seed=args.seed,
            query_chunk_size=args.query_chunk_size,
            low_mode=args.low_mode,
            return_stats=True,
        )
        approx, stats = fast_result  # type: ignore[misc]

        classical_ms, _ = _time_call(lambda: classical_causal_attention(q, k, v, scale=scale), args.repeats, device)
        fast_ms, _ = _time_call(
            lambda: fast_article_causal_attention(
                q,
                k,
                v,
                degree=args.degree,
                block_size=args.block_size,
                scale=scale,
                compressor=args.compressor,
                seed=args.seed,
                query_chunk_size=args.query_chunk_size,
                low_mode=args.low_mode,
            ),
            args.repeats,
            device,
        )

        rows.append(
            {
                "n": n,
                "rel_l2": safe_relative_l2(approx, exact),
                "max_abs": (approx - exact).abs().max().item(),
                "classical_ms": classical_ms,
                "fast_ms": fast_ms,
                "approx_state": stats.approx_streaming_state_bytes,
                "low_scratch": stats.scratch_low_prefix_bytes,
                "residual_scratch": stats.scratch_residual_matrix_bytes,
                "score_matrix": n * n * torch.tensor([], dtype=dtype).element_size(),
                "coreset_items": stats.coreset_items,
            }
        )

    headers = [
        "n",
        "rel_l2",
        "max_abs",
        "classical_ms",
        "fast_ms",
        "approx_state",
        "low_scratch",
        "resid_scratch",
        "score_matrix",
        "coreset_items",
    ]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for r in rows:
        print(
            " | ".join(
                [
                    str(r["n"]),
                    f"{r['rel_l2']:.4g}",
                    f"{r['max_abs']:.4g}",
                    f"{r['classical_ms']:.3f}",
                    f"{r['fast_ms']:.3f}",
                    _human_bytes(int(r["approx_state"])),
                    _human_bytes(int(r["low_scratch"])),
                    _human_bytes(int(r["residual_scratch"])),
                    _human_bytes(int(r["score_matrix"])),
                    str(r["coreset_items"]),
                ]
            )
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast vectorized Taylor + block-coreset attention benchmark")
    parser.add_argument("--sweep-n", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--dv", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--compressor", choices=["sorted_pair", "random", "none"], default="sorted_pair")
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--low-mode", choices=["auto", "stream", "prefix"], default="auto")
    parser.add_argument("--threads", type=int, default=None, help="Optional torch.set_num_threads value; try 1 on CPU")
    parser.add_argument("--data", choices=["unit_ball", "unit_sphere", "normal"], default="unit_ball")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--standard-scale", action="store_true", help="Use 1/sqrt(d) when --scale is omitted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    _demo(parser.parse_args())
