# Streaming Taylor + Coreset Attention in PyTorch

This is an executable PyTorch implementation of the constructive high-temperature upper-bound idea from the uploaded TeX source for **"Towards Tight Bounds for Streaming Attention"** (`arXiv-2606.07205v1`).

The source defines the softmax kernel

```text
expker(k, q) = exp(<k, q>)
```

and decomposes it as

```text
exp(<k, q>) = exp_{<=t}(k, q) + exp_{>t}(k, q)
exp_{<=t}(k, q) = sum_{l=0}^t <k, q>^l / l!
```

The implemented streaming state stores:

1. **Exact low-degree moments** for `exp_{<=t}` using the symmetric monomial basis. Degree `l` uses `C(d + l - 1, l)` features, matching the compressed monomial count in the paper.
2. **A weighted merge-and-reduce residual coreset** for `exp_{>t}`.

For attention, the numerator is sketched with value-weighted Taylor moments, and the denominator with scalar Taylor moments:

```text
numerator   ≈ sum_i exp(scale * <q, k_i>) v_i
denominator ≈ sum_i exp(scale * <q, k_i>)
attention   = numerator / denominator
```

`scale=None` defaults to the transformer convention `1/sqrt(d)`. Passing `scale=1.0` matches the article's unscaled notation.

## Important implementation note

The paper's theoretical `BaseCompress` primitive is an abstract discrepancy-minimization routine cited from the literature. The TeX source does not provide code for it. This implementation keeps the same merge-and-reduce data-structure layout but uses practical unbiased compressors:

- `sorted_pair` default: sort by a random projection, pair neighboring keys, sample one per pair with probability proportional to weight, and assign the selected item the pair's total weight.
- `random`: uniform half-sampling with inverse-probability weight correction.
- `none`: disables compression; useful for checking that Taylor + exact residual reconstructs classical attention.

So the code is faithful to the constructive decomposition, but the default compressor is a practical stand-in rather than a proof-carrying discrepancy implementation.

## Files

- `streaming_taylor_coreset_attention.py` — implementation and small demo CLI.
- `benchmark_attention.py` — benchmark against exact classical causal attention.
- `test_streaming_attention.py` — sanity tests.
- `requirements.txt` — minimal dependency list.

## Quick sanity run

```bash
python streaming_taylor_coreset_attention.py \
  --n 128 --d 16 --dv 16 \
  --degree 2 --block-size 32 \
  --radius 1 --scale 1 \
  --compressor sorted_pair
```

When `--block-size` is larger than the sequence length, the residual is exact and the output should match classical causal attention up to floating-point error:

```bash
python streaming_taylor_coreset_attention.py \
  --n 64 --d 8 --dv 8 \
  --degree 2 --block-size 128 \
  --radius 1 --scale 1
```

## Benchmark

```bash
python benchmark_attention.py \
  --sweep-n 256 512 1024 \
  --d 32 --dv 32 \
  --degree 2 --block-size 128 \
  --data unit_ball --radius 1 \
  --scale 1 \
  --device cpu \
  --csv benchmark_results.csv
```

The benchmark reports:

- relative L2 error versus exact causal attention;
- classical exact-attention latency;
- article-style streaming latency;
- approximate streaming state size;
- naive classical score-matrix size `N*N*sizeof(dtype)`;
- maximum coreset items retained.

## Minimal Python usage

```python
import torch
from streaming_taylor_coreset_attention import (
    ArticleStreamingAttention,
    classical_causal_attention,
)

N, D, DV = 512, 32, 32
q = torch.randn(N, D)
k = torch.randn(N, D)
v = torch.randn(N, DV)

state = ArticleStreamingAttention(
    dim=D,
    value_dim=DV,
    degree=2,
    block_size=128,
    scale=1.0,
    compressor="sorted_pair",
)

outs = []
for i in range(N):
    state.update(k[i], v[i])       # stream key/value into the data structure
    outs.append(state.query(q[i])) # causal query over prefix 0..i
approx = torch.stack(outs)

exact = classical_causal_attention(q, k, v, scale=1.0)
print(torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact))
print(state.state_size_bytes())
```

## Practical parameter guidance

- Increase `degree` to shift more mass into exact Taylor moments and reduce coreset error. Memory grows as roughly `sum_l C(d+l-1,l) * value_dim`.
- Increase `block_size` to reduce compression variance. Memory grows roughly like `block_size * log(N)` plus the low-degree moment sketch.
- This construction is most accurate in the article's high-temperature/small-radius regime, where `scale * <q,k>` is not too large.
- For normal transformer activations, start with the standard scale `1/sqrt(d)` or normalize keys/queries during experiments.

## Fast implementation

`fast_article_attention.py` is a separate speed-oriented implementation.  It keeps the same Taylor/residual decomposition, but it avoids the slow per-token Python query path used in the reference module.

Main differences from the reference implementation:

- exact low-degree Taylor moments are computed with a tight streaming moment loop on CPU, or with prefix tensors on CUDA via `--low-mode auto`;
- completed residual blocks are compressed to pair-sampled block coresets;
- the current block residual is evaluated exactly with small causal block matrices;
- `--compressor none` evaluates the residual exactly and should match classical causal attention up to floating-point error.

CPU example:

```bash
python fast_article_attention.py \
  --sweep-n 256 512 1024 \
  --d 32 --dv 32 \
  --degree 2 --block-size 128 \
  --data unit_ball --radius 1 \
  --scale 1 \
  --device cpu \
  --threads 1
```

Exact-decomposition check:

```bash
python fast_article_attention.py \
  --sweep-n 256 \
  --d 32 --dv 32 \
  --degree 2 --block-size 128 \
  --compressor none \
  --data unit_ball --radius 1 \
  --scale 1 \
  --device cpu --threads 1
```

Python usage:

```python
from fast_article_attention import fast_article_causal_attention

approx = fast_article_causal_attention(
    q, k, v,
    degree=2,
    block_size=128,
    scale=1.0,
    compressor="sorted_pair",
    low_mode="auto",
)
```
