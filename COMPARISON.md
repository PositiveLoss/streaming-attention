# Comparisom

```
n | rel_l2 | max_abs | classical_ms | article_ms | article_state | score_matrix | coreset_items
--- | --- | --- | --- | --- | --- | --- | ---
256 | 0.0007019 | 0.001098 | 0.230 | 308.040 | 120.32 KiB | 256.00 KiB | 191
512 | 0.001339 | 0.001201 | 0.193 | 632.149 | 136.57 KiB | 1.00 MiB | 255
1024 | 0.002211 | 0.001588 | 0.360 | 1424.673 | 152.82 KiB | 4.00 MiB | 319
```

These results show the prototype is doing what it was mainly meant to demonstrate: **small approximation error and much better state/memory scaling**, but **not competitive runtime** against PyTorch’s highly optimized classical attention.

### Main interpretation

| Metric          | What it says                                                       |
| --------------- | ------------------------------------------------------------------ |
| `rel_l2`        | Approximation error versus exact classical causal attention.       |
| `max_abs`       | Worst absolute output-coordinate error.                            |
| `classical_ms`  | Time for vectorized exact causal attention.                        |
| `article_ms`    | Time for the streaming Taylor + coreset prototype.                 |
| `article_state` | Maximum streaming state size used by the article-style method.     |
| `score_matrix`  | Naive exact attention score matrix memory, `n × n`.                |
| `coreset_items` | Maximum number of residual coreset items stored during the stream. |

## 1. Accuracy looks good

The relative error stays very small:

```text
n=256   rel_l2 = 0.0007019  ≈ 0.070%
n=512   rel_l2 = 0.001339   ≈ 0.134%
n=1024  rel_l2 = 0.002211   ≈ 0.221%
```

The worst coordinate error also remains around `1e-3`:

```text
max_abs: 0.001098 → 0.001201 → 0.001588
```

So the approximation is close to classical attention on this benchmark. The error increases with `n`, which is expected: longer streams trigger more residual coreset compression, so more approximation noise accumulates. But the growth is mild, not explosive.

This is probably the best part of the result.

## 2. Memory scaling is the intended win

The classical score matrix grows quadratically:

```text
256    → 256 KiB
512    → 1 MiB
1024   → 4 MiB
```

That is exactly the `O(n²)` memory pattern for naive full causal attention scores.

The article-style state grows slowly:

```text
256    → 120.32 KiB
512    → 136.57 KiB
1024   → 152.82 KiB
```

So when `n` doubles, the naive score matrix grows by `4×`, while the article state only grows by about `13–16 KiB`.

The relative memory comparison improves quickly:

```text
n=256:   score matrix is ~2.1× larger than article state
n=512:   score matrix is ~7.5× larger
n=1024:  score matrix is ~26.8× larger
```

That is the clearest confirmation that the implementation captures the article’s intended streaming-memory behavior.

One nuance: `score_matrix` is the naive full-sequence exact-attention score matrix. For autoregressive inference, a strong classical baseline is often the KV cache, whose memory is `O(n(d + dv))`, not `O(n²)`. With the default `d=32`, `dv=32`, `float32`, the KV cache would be roughly:

```text
n=256:   64 KiB
n=512:   128 KiB
n=1024:  256 KiB
```

So versus KV cache, the article state is worse at `n=256`, similar around `n=512`, and better by `n=1024`. Against naive score-matrix attention, the article method wins strongly.

## 3. Coreset size is sublinear in practice

The coreset item count grows much more slowly than `n`:

```text
n=256:   191 items  ≈ 74.6% of n
n=512:   255 items  ≈ 49.8% of n
n=1024:  319 items  ≈ 31.2% of n
```

So the residual representation is becoming a smaller fraction of the sequence as `n` grows.

The pattern also matches the merge-and-reduce design. With a fixed block size, the implementation stores a buffer plus several compressed levels. The maximum state during the stream can occur just before a buffer flush, which is why the number is not just the final coreset size.

## 4. Runtime is currently bad for the article prototype

The article-style implementation is much slower:

```text
n=256:   308.040 ms vs 0.230 ms
n=512:   632.149 ms vs 0.193 ms
n=1024:  1424.673 ms vs 0.360 ms
```

That is roughly:

```text
n=256:   ~1,339× slower
n=512:   ~3,275× slower
n=1024:  ~3,957× slower
```

This does **not** mean the article’s idea is inherently that slow. It mainly reflects implementation mismatch:

Classical attention is one or two dense PyTorch kernels: matmul, mask, softmax, matmul. These are heavily optimized.

The article implementation is a literal streaming Python prototype. It loops over tokens, updates a mutable state, computes Taylor monomial features, queries a residual coreset, and repeatedly materializes coreset items. That is good for correctness and experimentation, but terrible for wall-clock speed.

The article time is roughly linear-ish:

```text
article ms/token:
256:   1.20 ms/token
512:   1.23 ms/token
1024:  1.39 ms/token
```

So the asymptotic shape is reasonable, but the constant factor is huge.

The classical timing is almost flat for these small `n` values because the benchmark is dominated by optimized-kernel overhead and cache effects. At larger `n`, naive exact attention should eventually show its quadratic cost more clearly, assuming memory does not become the bottleneck first.

## 5. Overall conclusion

The result is **positive for approximation and memory**, but **negative for speed**.

The implementation demonstrates:

```text
Good:
- Very low approximation error: ~0.07% to ~0.22% relative L2.
- Article state grows slowly with n.
- Naive score-matrix memory grows quadratically.
- Coreset fraction shrinks as sequence length increases.

Bad:
- Runtime is thousands of times slower than optimized classical attention.
- This implementation is a research prototype, not a production kernel.
```
