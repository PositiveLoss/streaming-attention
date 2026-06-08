"""
PyTorch implementation of the constructive high-temperature streaming-attention
scheme from "Towards Tight Bounds for Streaming Attention" (arXiv:2606.07205v1).

The paper's upper-bound construction decomposes the softmax kernel

    exp(<k, q>) = exp_{<=t}(<k, q>) + exp_{>t}(<k, q>)

and stores:
  1. exact low-degree Taylor moments for exp_{<=t};
  2. a weighted merge-and-reduce coreset for the high-degree residual exp_{>t}.

This file implements that construction for causal inference experiments.  The
low-degree part uses the symmetric monomial basis, so degree-l storage is
C(d + l - 1, l), matching the paper's compressed monomial count rather than the
larger d**l tensor-power basis.

The paper's theoretical BaseCompress primitive is an abstract discrepancy
minimization routine.  For an executable PyTorch implementation we provide
practical unbiased compressors, with `sorted_pair` as the default.  This keeps
the same merge-and-reduce structure but does not claim the exact theorem's
worst-case probability bound.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
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
        raise ValueError(f"Unsupported dtype {dtype!r}. Use one of {sorted(table)}")
    return table[key]


def _num_combinations_with_replacement(dim: int, degree: int) -> int:
    if degree == 0:
        return 1
    return math.comb(dim + degree - 1, degree)


def _product_factorial_inverse(combo: Tuple[int, ...]) -> float:
    """Return 1 / prod_j alpha_j! for a sorted multi-index combo."""
    if not combo:
        return 1.0
    inv = 1.0
    run_value = combo[0]
    run_count = 0
    for idx in combo:
        if idx == run_value:
            run_count += 1
        else:
            inv /= math.factorial(run_count)
            run_value = idx
            run_count = 1
    inv /= math.factorial(run_count)
    return inv


def exp_taylor(logits: Tensor, degree: int) -> Tensor:
    """Compute sum_{l=0}^degree logits**l / l! elementwise."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    out = torch.ones_like(logits)
    term = torch.ones_like(logits)
    for l in range(1, degree + 1):
        term = term * logits / float(l)
        out = out + term
    return out


def safe_relative_l2(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    """Return ||a-b||_2 / max(||b||_2, eps) as a Python float."""
    return (torch.linalg.vector_norm(a - b) / torch.clamp(torch.linalg.vector_norm(b), min=eps)).item()


# -----------------------------------------------------------------------------
# Symmetric Taylor feature map and moment sketch
# -----------------------------------------------------------------------------


class SymmetricTaylorFeatureMap:
    """Symmetric monomial features for Taylor-expansion attention.

    For degree l, all combinations 0 <= i_1 <= ... <= i_l < d are used.
    The key feature includes the Taylor multinomial factor 1/prod_j alpha_j!,
    while the query feature is the corresponding monomial without the factor.

    Therefore

        <key_features_l(k), query_features_l(q)> = (k·q)^l / l!

    and summing over l = 0..t gives exp_{<=t}(k, q).
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        max_monomials: int = 5_000_000,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.dim = int(dim)
        self.degree = int(degree)
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.max_monomials = int(max_monomials)

        self.indices_by_degree: List[Optional[Tensor]] = []
        self.key_coeff_by_degree: List[Tensor] = []
        self.sizes: List[int] = []

        total = 0
        for l in range(degree + 1):
            m = _num_combinations_with_replacement(dim, l)
            total += m
            if total > self.max_monomials:
                raise ValueError(
                    f"Requested Taylor feature map has {total:,} monomials through degree {l}, "
                    f"exceeding max_monomials={self.max_monomials:,}. Lower --degree/--d or raise the limit."
                )
            self.sizes.append(m)
            if l == 0:
                self.indices_by_degree.append(None)
                self.key_coeff_by_degree.append(torch.ones(1, device=self.device, dtype=self.dtype))
                continue

            combos = list(itertools.combinations_with_replacement(range(dim), l))
            indices = torch.tensor(combos, device=self.device, dtype=torch.long)
            coeff = torch.tensor(
                [_product_factorial_inverse(tuple(c)) for c in combos],
                device=self.device,
                dtype=self.dtype,
            )
            self.indices_by_degree.append(indices)
            self.key_coeff_by_degree.append(coeff)

    @property
    def total_features(self) -> int:
        return int(sum(self.sizes))

    def key_features_by_degree(self, key: Tensor) -> List[Tensor]:
        """Return [phi_0(key), ..., phi_t(key)] with Taylor factors included."""
        key = key.to(device=self.device, dtype=self.dtype)
        if key.ndim != 1 or key.numel() != self.dim:
            raise ValueError(f"key must have shape [{self.dim}], got {tuple(key.shape)}")

        features: List[Tensor] = []
        for l in range(self.degree + 1):
            if l == 0:
                features.append(torch.ones(1, device=self.device, dtype=self.dtype))
            else:
                idx = self.indices_by_degree[l]
                assert idx is not None
                # key[idx] has shape [num_monomials_l, l].  Product handles repeated indices.
                monomials = key[idx].prod(dim=-1)
                features.append(monomials * self.key_coeff_by_degree[l])
        return features

    def query_features_by_degree(self, query: Tensor) -> List[Tensor]:
        """Return [psi_0(query), ..., psi_t(query)] without Taylor factors."""
        query = query.to(device=self.device, dtype=self.dtype)
        if query.ndim != 1 or query.numel() != self.dim:
            raise ValueError(f"query must have shape [{self.dim}], got {tuple(query.shape)}")

        features: List[Tensor] = []
        for l in range(self.degree + 1):
            if l == 0:
                features.append(torch.ones(1, device=self.device, dtype=self.dtype))
            else:
                idx = self.indices_by_degree[l]
                assert idx is not None
                features.append(query[idx].prod(dim=-1))
        return features

    def bytes(self, include_indices: bool = True) -> int:
        total = 0
        if include_indices:
            for idx in self.indices_by_degree:
                if idx is not None:
                    total += idx.numel() * idx.element_size()
        for coeff in self.key_coeff_by_degree:
            total += coeff.numel() * coeff.element_size()
        return int(total)


class TaylorMomentSketch:
    """Streaming low-degree Taylor moment sketch for attention numerator/denominator."""

    def __init__(
        self,
        dim: int,
        value_dim: int,
        degree: int,
        *,
        scale: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        max_monomials: int = 5_000_000,
    ) -> None:
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.degree = int(degree)
        self.scale = float(scale)
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.feature_map = SymmetricTaylorFeatureMap(
            dim,
            degree,
            device=self.device,
            dtype=self.dtype,
            max_monomials=max_monomials,
        )
        self.den_moments: List[Tensor] = [
            torch.zeros(m, device=self.device, dtype=self.dtype) for m in self.feature_map.sizes
        ]
        self.num_moments: List[Tensor] = [
            torch.zeros(m, self.value_dim, device=self.device, dtype=self.dtype)
            for m in self.feature_map.sizes
        ]
        self.length = 0

    @torch.no_grad()
    def update(self, key: Tensor, value: Tensor, weight: Union[float, Tensor] = 1.0) -> None:
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        if key.shape != (self.dim,):
            raise ValueError(f"key must have shape [{self.dim}], got {tuple(key.shape)}")
        if value.shape != (self.value_dim,):
            raise ValueError(f"value must have shape [{self.value_dim}], got {tuple(value.shape)}")
        w = torch.as_tensor(weight, device=self.device, dtype=self.dtype)
        feats = self.feature_map.key_features_by_degree(key)
        for l, feat in enumerate(feats):
            wf = w * feat
            self.den_moments[l].add_(wf)
            self.num_moments[l].add_(wf[:, None] * value[None, :])
        self.length += 1

    @torch.no_grad()
    def query(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (low_degree_numerator, low_degree_denominator)."""
        query = query.to(device=self.device, dtype=self.dtype)
        if query.shape != (self.dim,):
            raise ValueError(f"query must have shape [{self.dim}], got {tuple(query.shape)}")
        # The paper uses exp(<k,q>).  Standard transformer attention uses
        # exp(scale * <k,q>), which is exp(<k, scale*q>).
        q_eff = query * self.scale
        q_feats = self.feature_map.query_features_by_degree(q_eff)
        den = torch.zeros((), device=self.device, dtype=self.dtype)
        num = torch.zeros(self.value_dim, device=self.device, dtype=self.dtype)
        for l, q_feat in enumerate(q_feats):
            den = den + torch.dot(self.den_moments[l], q_feat)
            num = num + q_feat @ self.num_moments[l]
        return num, den

    def state_size_bytes(self, include_feature_map: bool = False) -> int:
        total = 0
        for x in self.den_moments:
            total += x.numel() * x.element_size()
        for x in self.num_moments:
            total += x.numel() * x.element_size()
        if include_feature_map:
            total += self.feature_map.bytes(include_indices=True)
        return int(total)


# -----------------------------------------------------------------------------
# Merge-and-reduce residual coreset
# -----------------------------------------------------------------------------


@dataclass
class CoresetBlock:
    keys: Tensor
    values: Tensor
    weights: Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> "CoresetBlock":
        return CoresetBlock(
            self.keys.to(device=device, dtype=dtype),
            self.values.to(device=device, dtype=dtype),
            self.weights.to(device=device, dtype=dtype),
        )

    @property
    def size(self) -> int:
        return int(self.weights.numel())


class MergeReduceCoreset:
    """Weighted merge-and-reduce coreset for the Taylor residual.

    `sorted_pair` compressor:
        Sort by a random projection, pair neighboring points, sample one point
        from each pair with probability proportional to its current weight, and
        assign the selected point the pair's total weight.  This is unbiased for
        any fixed test function f(k, v) and tends to reduce variance when nearby
        keys are paired.

    This is a practical executable replacement for the paper's discrepancy
    BaseCompress primitive.  It preserves the streaming merge-and-reduce shape
    of the construction but not the formal discrepancy guarantee.
    """

    def __init__(
        self,
        dim: int,
        value_dim: int,
        block_size: int,
        *,
        compressor: str = "sorted_pair",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        seed: int = 0,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.block_size = int(block_size)
        self.compressor = str(compressor)
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.seed = int(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self._buffer_keys: List[Tensor] = []
        self._buffer_values: List[Tensor] = []
        self.levels: List[Optional[CoresetBlock]] = []
        self.length = 0
        self.num_compressions = 0

    @torch.no_grad()
    def append(self, key: Tensor, value: Tensor) -> None:
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        if key.shape != (self.dim,):
            raise ValueError(f"key must have shape [{self.dim}], got {tuple(key.shape)}")
        if value.shape != (self.value_dim,):
            raise ValueError(f"value must have shape [{self.value_dim}], got {tuple(value.shape)}")
        self._buffer_keys.append(key.detach().clone())
        self._buffer_values.append(value.detach().clone())
        self.length += 1
        if len(self._buffer_keys) >= self.block_size:
            keys = torch.stack(self._buffer_keys, dim=0)
            values = torch.stack(self._buffer_values, dim=0)
            weights = torch.ones(keys.shape[0], device=self.device, dtype=self.dtype)
            self._buffer_keys.clear()
            self._buffer_values.clear()
            compressed = self._compress(CoresetBlock(keys, values, weights))
            self._insert_level(0, compressed)

    def _insert_level(self, level: int, block: CoresetBlock) -> None:
        while len(self.levels) <= level:
            self.levels.append(None)
        if self.levels[level] is None:
            self.levels[level] = block
            return
        old = self.levels[level]
        assert old is not None
        merged = CoresetBlock(
            torch.cat([old.keys, block.keys], dim=0),
            torch.cat([old.values, block.values], dim=0),
            torch.cat([old.weights, block.weights], dim=0),
        )
        self.levels[level] = None
        compressed = self._compress(merged)
        self._insert_level(level + 1, compressed)

    def _compress(self, block: CoresetBlock) -> CoresetBlock:
        if self.compressor == "none":
            return block
        if block.size <= 1:
            return block
        if self.compressor == "random":
            out = self._compress_random(block)
        elif self.compressor == "sorted_pair":
            out = self._compress_sorted_pair(block)
        else:
            raise ValueError("compressor must be one of {'sorted_pair', 'random', 'none'}")
        self.num_compressions += 1
        return out

    def _compress_random(self, block: CoresetBlock) -> CoresetBlock:
        n = block.size
        m = (n + 1) // 2
        perm = torch.randperm(n, generator=self.generator, device=self.device)[:m]
        # Uniform half-sampling with inverse-probability weight correction.
        factor = float(n) / float(m)
        return CoresetBlock(block.keys[perm], block.values[perm], block.weights[perm] * factor)

    def _compress_sorted_pair(self, block: CoresetBlock) -> CoresetBlock:
        n = block.size
        if n <= 2:
            # Pairing still works, but this branch avoids shape corner cases.
            return self._compress_random(block)

        direction = torch.randn(self.dim, generator=self.generator, device=self.device, dtype=self.dtype)
        direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-12)
        scores = block.keys @ direction
        order = torch.argsort(scores)
        keys = block.keys[order]
        values = block.values[order]
        weights = block.weights[order]

        pair_count = n // 2
        a = slice(0, 2 * pair_count, 2)
        b = slice(1, 2 * pair_count, 2)
        wa = weights[a]
        wb = weights[b]
        wsum = wa + wb
        prob_a = wa / torch.clamp(wsum, min=1e-30)
        u = torch.rand(pair_count, generator=self.generator, device=self.device, dtype=self.dtype)
        choose_a = u < prob_a

        new_keys = torch.where(choose_a[:, None], keys[a], keys[b])
        new_values = torch.where(choose_a[:, None], values[a], values[b])
        new_weights = wsum

        if n % 2 == 1:
            new_keys = torch.cat([new_keys, keys[-1:].clone()], dim=0)
            new_values = torch.cat([new_values, values[-1:].clone()], dim=0)
            new_weights = torch.cat([new_weights, weights[-1:].clone()], dim=0)
        return CoresetBlock(new_keys, new_values, new_weights)

    @torch.no_grad()
    def items(self) -> CoresetBlock:
        blocks: List[CoresetBlock] = []
        if self._buffer_keys:
            keys = torch.stack(self._buffer_keys, dim=0)
            values = torch.stack(self._buffer_values, dim=0)
            weights = torch.ones(len(self._buffer_keys), device=self.device, dtype=self.dtype)
            blocks.append(CoresetBlock(keys, values, weights))
        for block in self.levels:
            if block is not None and block.size > 0:
                blocks.append(block)
        if not blocks:
            return CoresetBlock(
                torch.empty(0, self.dim, device=self.device, dtype=self.dtype),
                torch.empty(0, self.value_dim, device=self.device, dtype=self.dtype),
                torch.empty(0, device=self.device, dtype=self.dtype),
            )
        return CoresetBlock(
            torch.cat([b.keys for b in blocks], dim=0),
            torch.cat([b.values for b in blocks], dim=0),
            torch.cat([b.weights for b in blocks], dim=0),
        )

    def num_items(self) -> int:
        return self.items().size

    def state_size_bytes(self) -> int:
        total = 0
        for x in self._buffer_keys:
            total += x.numel() * x.element_size()
        for x in self._buffer_values:
            total += x.numel() * x.element_size()
        for block in self.levels:
            if block is not None:
                total += block.keys.numel() * block.keys.element_size()
                total += block.values.numel() * block.values.element_size()
                total += block.weights.numel() * block.weights.element_size()
        return int(total)


# -----------------------------------------------------------------------------
# Article-inspired streaming attention
# -----------------------------------------------------------------------------


@dataclass
class AttentionQueryStats:
    denominator: float
    low_denominator: float
    tail_denominator: float
    coreset_items: int
    state_bytes: int


class ArticleStreamingAttention:
    """Streaming causal attention via Taylor moments + residual coreset.

    The state after seeing keys/values (k_1,v_1),...,(k_j,v_j) can answer any
    query q with an approximation to

        sum_i softmax_i(q) v_i,  softmax_i ∝ exp(scale * <q, k_i>).

    Call `update(k_t, v_t)` and then `query(q_t)` for causal self-attention that
    includes the current token, matching the streaming problem in the paper.
    """

    def __init__(
        self,
        dim: int,
        value_dim: int,
        *,
        degree: int = 2,
        block_size: int = 128,
        scale: Optional[float] = None,
        compressor: str = "sorted_pair",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        seed: int = 0,
        max_monomials: int = 5_000_000,
        denominator_eps: float = 1e-12,
    ) -> None:
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.degree = int(degree)
        self.scale = float(1.0 / math.sqrt(dim) if scale is None else scale)
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype)
        self.denominator_eps = float(denominator_eps)
        self.sketch = TaylorMomentSketch(
            dim,
            value_dim,
            degree,
            scale=self.scale,
            device=self.device,
            dtype=self.dtype,
            max_monomials=max_monomials,
        )
        self.coreset = MergeReduceCoreset(
            dim,
            value_dim,
            block_size,
            compressor=compressor,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )

    @torch.no_grad()
    def update(self, key: Tensor, value: Tensor) -> None:
        self.sketch.update(key, value)
        self.coreset.append(key, value)

    @torch.no_grad()
    def query(self, query: Tensor, *, return_stats: bool = False) -> Union[Tensor, Tuple[Tensor, AttentionQueryStats]]:
        query = query.to(device=self.device, dtype=self.dtype)
        low_num, low_den = self.sketch.query(query)
        residual_block = self.coreset.items()
        if residual_block.size == 0:
            tail_den = torch.zeros((), device=self.device, dtype=self.dtype)
            tail_num = torch.zeros(self.value_dim, device=self.device, dtype=self.dtype)
        else:
            logits = self.scale * (residual_block.keys @ query)
            tail = torch.exp(logits) - exp_taylor(logits, self.degree)
            weighted_tail = residual_block.weights * tail
            tail_den = weighted_tail.sum()
            tail_num = weighted_tail @ residual_block.values

        den = low_den + tail_den
        num = low_num + tail_num
        safe_den = torch.clamp(den, min=self.denominator_eps)
        out = num / safe_den
        if not return_stats:
            return out
        stats = AttentionQueryStats(
            denominator=float(den.detach().cpu()),
            low_denominator=float(low_den.detach().cpu()),
            tail_denominator=float(tail_den.detach().cpu()),
            coreset_items=residual_block.size,
            state_bytes=self.state_size_bytes(include_feature_map=False),
        )
        return out, stats

    def state_size_bytes(self, *, include_feature_map: bool = False) -> int:
        return self.sketch.state_size_bytes(include_feature_map=include_feature_map) + self.coreset.state_size_bytes()

    @property
    def length(self) -> int:
        return self.sketch.length


# -----------------------------------------------------------------------------
# Convenience exact/approximate attention functions
# -----------------------------------------------------------------------------


@torch.no_grad()
def classical_causal_attention(q: Tensor, k: Tensor, v: Tensor, *, scale: Optional[float] = None) -> Tensor:
    """Exact causal attention, supporting [N,D] or [B,H,N,D] tensors."""
    if q.shape[-2] != k.shape[-2] or q.shape[:-1] != k.shape[:-1]:
        raise ValueError("q and k must have matching leading dimensions and sequence length")
    if v.shape[:-1] != k.shape[:-1]:
        raise ValueError("v must have the same leading dimensions and sequence length as k")
    dim = q.shape[-1]
    s = float(1.0 / math.sqrt(dim) if scale is None else scale)
    logits = torch.matmul(q, k.transpose(-1, -2)) * s
    n = q.shape[-2]
    causal_mask = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    logits = logits.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, v)


@torch.no_grad()
def article_causal_attention_single(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    block_size: int = 128,
    scale: Optional[float] = None,
    compressor: str = "sorted_pair",
    seed: int = 0,
    max_monomials: int = 5_000_000,
    return_state_sizes: bool = False,
) -> Union[Tensor, Tuple[Tensor, List[int]]]:
    """Approximate causal attention for a single [N,D] sequence."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, v must have shapes [N,D], [N,D], [N,Dv]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if v.shape[0] != k.shape[0]:
        raise ValueError("v must have same sequence length as q/k")

    n, dim = k.shape
    value_dim = v.shape[-1]
    state = ArticleStreamingAttention(
        dim,
        value_dim,
        degree=degree,
        block_size=block_size,
        scale=scale,
        compressor=compressor,
        device=k.device,
        dtype=k.dtype,
        seed=seed,
        max_monomials=max_monomials,
    )
    outs: List[Tensor] = []
    sizes: List[int] = []
    for i in range(n):
        state.update(k[i], v[i])
        outs.append(state.query(q[i]))
        if return_state_sizes:
            sizes.append(state.state_size_bytes(include_feature_map=False))
    out = torch.stack(outs, dim=0)
    if return_state_sizes:
        return out, sizes
    return out


@torch.no_grad()
def article_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    degree: int = 2,
    block_size: int = 128,
    scale: Optional[float] = None,
    compressor: str = "sorted_pair",
    seed: int = 0,
    max_monomials: int = 5_000_000,
    return_state_sizes: bool = False,
) -> Union[Tensor, Tuple[Tensor, List[int]]]:
    """Approximate causal attention for [N,D] or [B,H,N,D] tensors.

    Batched/multi-head inputs are handled by looping over B and H, because this
    object represents a mutable streaming data structure.  That keeps the code
    simple and close to the article; for production kernels, fuse these loops.
    """
    if q.ndim == 2:
        return article_causal_attention_single(
            q,
            k,
            v,
            degree=degree,
            block_size=block_size,
            scale=scale,
            compressor=compressor,
            seed=seed,
            max_monomials=max_monomials,
            return_state_sizes=return_state_sizes,
        )
    if q.ndim != 4:
        raise ValueError("q/k/v must be [N,D] or [B,H,N,D]")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shape [B,H,N,D]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B,H,N,Dv]")

    bsz, heads, n, _ = q.shape
    out = torch.empty(*q.shape[:3], v.shape[-1], device=q.device, dtype=q.dtype)
    all_sizes: List[int] = []
    for b in range(bsz):
        for h in range(heads):
            result = article_causal_attention_single(
                q[b, h],
                k[b, h],
                v[b, h],
                degree=degree,
                block_size=block_size,
                scale=scale,
                compressor=compressor,
                seed=seed + b * heads + h,
                max_monomials=max_monomials,
                return_state_sizes=return_state_sizes,
            )
            if return_state_sizes:
                out[b, h], sizes = result  # type: ignore[misc]
                all_sizes.extend(sizes)
            else:
                out[b, h] = result  # type: ignore[assignment]
    if return_state_sizes:
        return out, all_sizes
    return out


# -----------------------------------------------------------------------------
# Small CLI sanity check
# -----------------------------------------------------------------------------


def _demo(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn(args.n, args.d, device=device, dtype=dtype)
    k = torch.randn(args.n, args.d, device=device, dtype=dtype)
    v = torch.randn(args.n, args.dv, device=device, dtype=dtype)
    # Keep logits in a friendly high-temperature regime for the low-degree split.
    q = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12) * args.radius
    k = k / torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12) * args.radius
    scale = args.scale if args.scale is not None else 1.0

    exact = classical_causal_attention(q, k, v, scale=scale)
    approx, sizes = article_causal_attention(
        q,
        k,
        v,
        degree=args.degree,
        block_size=args.block_size,
        scale=scale,
        compressor=args.compressor,
        seed=args.seed,
        return_state_sizes=True,
    )
    print(f"relative_l2={safe_relative_l2(approx, exact):.6g}")
    print(f"max_abs={(approx - exact).abs().max().item():.6g}")
    print(f"final_state_bytes={sizes[-1] if sizes else 0:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for streaming Taylor+coreset attention")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--dv", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--compressor", choices=["sorted_pair", "random", "none"], default="sorted_pair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    _demo(parser.parse_args())
