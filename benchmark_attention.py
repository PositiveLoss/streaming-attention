"""Benchmark article-style streaming attention against exact causal attention.

Example:
    python benchmark_attention.py --sweep-n 256 512 1024 --d 32 --dv 32 \
        --degree 2 --block-size 128 --device cpu

The exact baseline materializes the N x N causal score matrix.  The article
implementation keeps a low-degree Taylor moment sketch plus a merge-and-reduce
residual coreset.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

from streaming_taylor_coreset_attention import (
    ArticleStreamingAttention,
    article_causal_attention,
    classical_causal_attention,
    safe_relative_l2,
)


def resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = dtype.lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    return table[key]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, repeats: int, device: torch.device) -> Tuple[float, float]:
    # One warmup run keeps the measurement less noisy without hiding setup cost too much.
    fn()
    sync(device)
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(times), min(times)


def make_data(
    n: int,
    d: int,
    dv: int,
    *,
    radius: float,
    data: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    k = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    v = torch.randn(n, dv, device=device, dtype=dtype, generator=g)

    if data == "unit_ball":
        q_norm = torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12)
        k_norm = torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12)
        # Random radii in [0, radius] with the correct d-dimensional volume law.
        rq = torch.rand(n, 1, device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
        rk = torch.rand(n, 1, device=device, dtype=dtype, generator=g).pow(1.0 / d) * radius
        q = q / q_norm * rq
        k = k / k_norm * rk
    elif data == "unit_sphere":
        q = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-12) * radius
        k = k / torch.clamp(torch.linalg.vector_norm(k, dim=-1, keepdim=True), min=1e-12) * radius
    elif data == "normal":
        # Keep the usual transformer-style normal initialization.
        pass
    else:
        raise ValueError("data must be one of {'unit_ball', 'unit_sphere', 'normal'}")
    return q, k, v


def estimate_article_state_bytes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    degree: int,
    block_size: int,
    scale: float,
    compressor: str,
    seed: int,
) -> Tuple[int, int, int]:
    state = ArticleStreamingAttention(
        k.shape[-1],
        v.shape[-1],
        degree=degree,
        block_size=block_size,
        scale=scale,
        compressor=compressor,
        device=k.device,
        dtype=k.dtype,
        seed=seed,
    )
    max_bytes = 0
    max_items = 0
    for i in range(k.shape[0]):
        state.update(k[i], v[i])
        max_bytes = max(max_bytes, state.state_size_bytes(include_feature_map=False))
        max_items = max(max_items, state.coreset.num_items())
    return state.state_size_bytes(include_feature_map=False), max_bytes, max_items


def run_one(args: argparse.Namespace, n: int) -> dict:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    scale = args.scale if args.scale is not None else (1.0 / math.sqrt(args.d) if args.standard_scale else 1.0)

    q, k, v = make_data(
        n,
        args.d,
        args.dv,
        radius=args.radius,
        data=args.data,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    with torch.no_grad():
        exact = classical_causal_attention(q, k, v, scale=scale)
        approx = article_causal_attention(
            q,
            k,
            v,
            degree=args.degree,
            block_size=args.block_size,
            scale=scale,
            compressor=args.compressor,
            seed=args.seed,
        )
        rel_l2 = safe_relative_l2(approx, exact)
        max_abs = (approx - exact).abs().max().item()
        mean_abs = (approx - exact).abs().mean().item()

    exact_median_ms, exact_best_ms = time_call(
        lambda: classical_causal_attention(q, k, v, scale=scale), args.repeats, device
    )
    article_median_ms, article_best_ms = time_call(
        lambda: article_causal_attention(
            q,
            k,
            v,
            degree=args.degree,
            block_size=args.block_size,
            scale=scale,
            compressor=args.compressor,
            seed=args.seed,
        ),
        args.repeats,
        device,
    )

    final_state_bytes, max_state_bytes, max_coreset_items = estimate_article_state_bytes(
        q,
        k,
        v,
        degree=args.degree,
        block_size=args.block_size,
        scale=scale,
        compressor=args.compressor,
        seed=args.seed,
    )

    elem = torch.tensor([], dtype=dtype).element_size()
    classical_kv_bytes = n * (args.d + args.dv) * elem
    classical_naive_scores_bytes = n * n * elem

    return {
        "n": n,
        "d": args.d,
        "dv": args.dv,
        "degree": args.degree,
        "block_size": args.block_size,
        "compressor": args.compressor,
        "data": args.data,
        "radius": args.radius,
        "scale": scale,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "classical_median_ms": exact_median_ms,
        "classical_best_ms": exact_best_ms,
        "article_median_ms": article_median_ms,
        "article_best_ms": article_best_ms,
        "article_final_state_bytes": final_state_bytes,
        "article_max_state_bytes": max_state_bytes,
        "article_max_coreset_items": max_coreset_items,
        "classical_kv_bytes": classical_kv_bytes,
        "classical_naive_scores_bytes": classical_naive_scores_bytes,
    }


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} GiB"


def print_table(rows: List[dict]) -> None:
    headers = [
        "n",
        "rel_l2",
        "max_abs",
        "classical_ms",
        "article_ms",
        "article_state",
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
                    f"{r['classical_median_ms']:.3f}",
                    f"{r['article_median_ms']:.3f}",
                    human_bytes(int(r["article_max_state_bytes"])),
                    human_bytes(int(r["classical_naive_scores_bytes"])),
                    str(r["article_max_coreset_items"]),
                ]
            )
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-n", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--dv", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--compressor", choices=["sorted_pair", "random", "none"], default="sorted_pair")
    parser.add_argument("--data", choices=["unit_ball", "unit_sphere", "normal"], default="unit_ball")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--standard-scale", action="store_true", help="Use 1/sqrt(d) when --scale is omitted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = [run_one(args, n) for n in args.sweep_n]
    print_table(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
