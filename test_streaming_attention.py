import math

import torch

from streaming_taylor_coreset_attention import (
    article_causal_attention,
    classical_causal_attention,
    exp_taylor,
)


def test_taylor_decomposition_is_exact_without_compression():
    torch.manual_seed(0)
    n, d, dv = 24, 6, 5
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    v = torch.randn(n, dv)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    exact = classical_causal_attention(q, k, v, scale=1.0)
    # block_size > n means all residual terms stay uncompressed in the live buffer.
    approx = article_causal_attention(q, k, v, degree=3, block_size=n + 1, scale=1.0)
    assert torch.allclose(approx, exact, atol=3e-6, rtol=3e-6)


def test_exp_taylor_known_values():
    x = torch.tensor([0.0, 1.0, -1.0])
    y = exp_taylor(x, 3)
    expected = torch.tensor([1.0, 1.0 + 1.0 + 0.5 + 1 / 6, 1.0 - 1.0 + 0.5 - 1 / 6])
    assert torch.allclose(y, expected)


if __name__ == "__main__":
    test_taylor_decomposition_is_exact_without_compression()
    test_exp_taylor_known_values()
    print("ok")
