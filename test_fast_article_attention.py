import torch

from fast_article_attention import (
    classical_causal_attention,
    fast_article_causal_attention,
    safe_relative_l2,
)


def test_exact_residual_reconstructs_classical() -> None:
    torch.manual_seed(0)
    n, d, dv = 64, 8, 7
    q = torch.randn(n, d) * 0.1
    k = torch.randn(n, d) * 0.1
    v = torch.randn(n, dv)

    exact = classical_causal_attention(q, k, v, scale=1.0)
    approx = fast_article_causal_attention(
        q,
        k,
        v,
        degree=2,
        block_size=16,
        scale=1.0,
        compressor="none",
        low_mode="stream",
    )
    assert safe_relative_l2(approx, exact) < 2e-6
    assert (approx - exact).abs().max().item() < 3e-6


def test_block_coreset_shapes_and_stats() -> None:
    torch.manual_seed(1)
    n, d, dv = 96, 12, 5
    q = torch.randn(n, d) * 0.1
    k = torch.randn(n, d) * 0.1
    v = torch.randn(n, dv)

    approx, stats = fast_article_causal_attention(
        q,
        k,
        v,
        degree=2,
        block_size=32,
        scale=1.0,
        compressor="sorted_pair",
        low_mode="stream",
        return_stats=True,
    )
    assert approx.shape == (n, dv)
    assert stats.feature_count == 1 + d + d * (d + 1) // 2
    assert stats.coreset_items > 0
    assert stats.approx_streaming_state_bytes > 0


if __name__ == "__main__":
    test_exact_residual_reconstructs_classical()
    test_block_coreset_shapes_and_stats()
    print("ok")
