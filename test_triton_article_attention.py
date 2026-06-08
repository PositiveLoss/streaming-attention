"""Smoke tests for triton_article_attention.py.

These tests are intentionally runnable on CPU-only machines.  Numerical Triton
checks are skipped unless both Triton and CUDA are available.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

MODULE_PATH = pathlib.Path(__file__).with_name("triton_article_attention.py")
spec = importlib.util.spec_from_file_location("triton_article_attention", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_import_and_cpu_guard() -> None:
    q = torch.randn(8, 4)
    k = torch.randn(8, 4)
    v = torch.randn(8, 4)
    try:
        mod.triton_article_causal_attention(q, k, v)
    except (ImportError, ValueError):
        # Expected without Triton/CUDA.
        return
    raise AssertionError("CPU tensors should not run through Triton CUDA kernels")


def test_segment_coreset_shapes_cpu() -> None:
    k = torch.randn(2, 17, 4)
    v = torch.randn(2, 17, 5)
    rep_k, rep_v, reps_per_block = mod.build_segment_mean_coreset(
        k, v, block_size=8, compress_stride=2
    )
    assert reps_per_block == 4
    # Blocks [0,8) and [8,16) have later queries; the partial block [16,17)
    # is current-only and is not included as a completed-block coreset.
    assert rep_k.shape == (2, 8, 4)
    assert rep_v.shape == (2, 8, 5)


def test_triton_numerics_when_available() -> None:
    if mod.triton is None or not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    n, d, dv = 64, 16, 16
    q = torch.randn(n, d, device="cuda", dtype=torch.float16) / d**0.5
    k = torch.randn(n, d, device="cuda", dtype=torch.float16) / d**0.5
    v = torch.randn(n, dv, device="cuda", dtype=torch.float16)

    exact = mod.torch_classical_causal_attention(q.float(), k.float(), v.float(), scale=1.0)
    flash = mod.triton_flash_causal_attention(q, k, v, scale=1.0, output_dtype=torch.float32)
    article_exact_resid = mod.triton_article_causal_attention(
        q,
        k,
        v,
        degree=2,
        block_size=16,
        compress_stride=1,
        scale=1.0,
        output_dtype=torch.float32,
    )
    assert mod.safe_relative_l2(flash, exact) < 2e-2
    assert mod.safe_relative_l2(article_exact_resid, exact) < 2e-2


if __name__ == "__main__":
    test_import_and_cpu_guard()
    test_segment_coreset_shapes_cpu()
    test_triton_numerics_when_available()
    print("ok")
