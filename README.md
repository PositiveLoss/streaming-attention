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
- `fast_article_attention.py` — vectorized PyTorch speed path.
- `test_fast_article_attention.py` — sanity tests for the speed path.
- `triton_article_attention.py` — CUDA/Triton residual kernels and benchmark CLI.
- `test_triton_article_attention.py` — CPU fallback test plus optional CUDA correctness check.

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

## Fast implementation

A separate optimized file is included as `fast_article_attention.py`. It specializes the Taylor feature map for degrees 0, 1, and 2, uses a tight streaming moment loop for the low-degree Taylor state, and evaluates the residual with chunked matrix multiplies over block coresets. It is designed for speed sweeps; the reference `streaming_taylor_coreset_attention.py` remains the closer literal merge-and-reduce implementation.

Example:

```bash
python fast_article_attention.py --sweep-n 256 512 1024 --d 32 --dv 32 --degree 2 --block-size 128 --data unit_ball --scale 1 --threads 1
```

Use `compressor none` to check that the Taylor + exact residual decomposition reconstructs classical attention up to roundoff.



## Triton/CUDA implementation

`triton_article_attention.py` adds GPU-first kernels for benchmarking:

- `triton_article_causal_attention(...)`: fused article-style approximation.
- `triton_flash_causal_attention(...)`: exact FlashAttention-style causal baseline that avoids the `N x N` score matrix.
- `torch_classical_causal_attention(...)`: PyTorch exact reference used by the benchmark.

The Triton article kernel is optimized for GPU throughput. It evaluates the low-degree Taylor term directly over causal tiles, keeps the residual exact inside the current article block, and uses deterministic segment-mean representatives for completed-block residuals. Set `--compress-stride 1` to disable residual compression; in that mode the residual path is exact and the result should match classical causal attention up to floating-point differences.

CUDA benchmark example:

```bash
python triton_article_attention.py \
  --sweep-n 512 1024 2048 4096 \
  --d 32 --dv 32 \
  --degree 2 \
  --block-size 128 \
  --compress-stride 2 \
  --data unit_ball --radius 1 \
  --scale 1 \
  --device cuda \
  --dtype float16 \
  --repeats 20
```

Exact-residual check on a CUDA + Triton machine:

```bash
python triton_article_attention.py \
  --sweep-n 256 512 \
  --d 32 --dv 32 \
  --degree 2 \
  --block-size 128 \
  --compress-stride 1 \
  --device cuda \
  --dtype float16
```

Python usage:

```python
from triton_article_attention import triton_article_causal_attention, triton_flash_causal_attention

article_out = triton_article_causal_attention(
    q.cuda(), k.cuda(), v.cuda(),
    degree=2,
    block_size=128,
    compress_stride=2,
    scale=1.0,
)

exact_triton_out = triton_flash_causal_attention(q.cuda(), k.cuda(), v.cuda(), scale=1.0)
```

Current kernel constraints:

- Requires a CUDA-enabled PyTorch build and `triton`.
- Supports `[N,D]` and `[B,H,N,D]` inputs.
- Supports `float16`, `bfloat16`, and `float32` inputs with fp32 accumulation.
- Supports Taylor degree `0`, `1`, or `2`.
- Compact kernel limits: `D <= 128`, `Dv <= 256`.
- `block_size` must be divisible by `compress_stride`.

Caveat: this file is a GPU-throughput implementation, not the literal online streaming data structure. The low-degree Taylor term is fused directly over causal tiles for speed. The reference file remains the closer implementation of the article’s streaming construction.

## Triton CUDA kernels

`triton_article_attention.py` adds GPU-first Triton kernels.  It is separate from both the reference implementation and the vectorized PyTorch fast path.

Provided APIs:

```python
from triton_article_attention import (
    triton_article_causal_attention,
    triton_flash_causal_attention,
)

# Approximate article-style Taylor + residual-coreset attention.
approx = triton_article_causal_attention(
    q, k, v,
    degree=2,
    block_size=128,
    compress_stride=2,   # 1 disables residual compression
    scale=1.0,
)

# Exact causal baseline using a FlashAttention-style online softmax kernel.
exact_triton = triton_flash_causal_attention(q, k, v, scale=1.0)
```

Benchmark example on a CUDA machine with Triton installed:

```bash
python triton_article_attention.py \
  --sweep-n 256 512 1024 2048 \
  --d 32 --dv 32 \
  --degree 2 \
  --block-size 128 \
  --compress-stride 2 \
  --data unit_ball \
  --scale 1 \
  --dtype float16 \
  --repeats 10
```

Exact-decomposition check:

```bash
python triton_article_attention.py \
  --sweep-n 256 \
  --d 32 --dv 32 \
  --degree 2 \
  --block-size 128 \
  --compress-stride 1 \
  --data unit_ball \
  --scale 1 \
  --dtype float16
```

Notes:

- `compress_stride=1` disables residual compression and should match exact causal softmax attention up to floating-point error in the high-temperature regime.
- `compress_stride=2` gives a half-sized deterministic segment-mean residual coreset for completed blocks.
- The Triton file uses a fused direct Taylor-prefix evaluation for speed experiments.  It avoids the `N x N` score matrix, but it is not the literal online streaming-state implementation from the paper.
- The current compact kernels target head dimensions `D <= 128` and value dimensions `Dv <= 256`.
