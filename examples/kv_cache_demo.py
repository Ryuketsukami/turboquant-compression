"""
TurboQuant — KV Cache Compression Demo
========================================

Simulates compressing a transformer's KV cache using TurboQuant
and measures attention score accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from turboquant import TurboQuantKVCache, TurboQuantConfig


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    rng = np.random.default_rng(42)

    # ── Configuration ──
    head_dim = 128       # typical attention head dimension
    seq_len = 256        # sequence length
    bits = 3             # bits per coordinate

    print(f"KV Cache Compression Demo")
    print(f"  Head dim:    {head_dim}")
    print(f"  Seq length:  {seq_len}")
    print(f"  Bits:        {bits} + 1 (QJL)")
    print("=" * 50)

    # ── Generate random KV pairs ──
    keys = rng.standard_normal((seq_len, head_dim))
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.standard_normal((seq_len, head_dim))
    values = values / np.linalg.norm(values, axis=1, keepdims=True)

    # ── Compress into TurboQuant KV cache ──
    config = TurboQuantConfig(dimension=head_dim, bits=bits, qjl_enabled=True)
    cache = TurboQuantKVCache(config)

    t0 = time.perf_counter()
    for t in range(seq_len):
        cache.append(keys[t], values[t])
    compress_time = time.perf_counter() - t0

    # ── Memory comparison ──
    original_bytes = 2 * seq_len * head_dim * 4  # keys + values, float32
    compressed_bytes = 2 * seq_len * cache.tq.compressed_size_bytes()

    print(f"\n[Memory]")
    print(f"  Original:   {original_bytes:>10,} bytes ({original_bytes / 1024:.1f} KB)")
    print(f"  Compressed: {compressed_bytes:>10,} bytes ({compressed_bytes / 1024:.1f} KB)")
    print(f"  Ratio:      {original_bytes / compressed_bytes:.1f}x")
    print(f"  Compress:   {compress_time * 1000:.1f} ms ({compress_time / seq_len * 1e6:.0f} us/token)")

    # ── Attention accuracy ──
    n_queries = 50
    corrs = []
    top1_matches = 0
    top5_matches = 0

    for _ in range(n_queries):
        query = rng.standard_normal(head_dim)
        query = query / np.linalg.norm(query)

        # True attention scores
        true_scores = keys @ query
        true_attn = softmax(true_scores)

        # Compressed attention scores
        est_scores = cache.attention_scores(query)
        est_attn = softmax(est_scores)

        # Correlation
        corrs.append(np.corrcoef(true_scores, est_scores)[0, 1])

        # Top-k accuracy
        true_top1 = np.argmax(true_scores)
        est_top1 = np.argmax(est_scores)
        if true_top1 == est_top1:
            top1_matches += 1

        true_top5 = set(np.argsort(true_scores)[-5:])
        est_top5 = set(np.argsort(est_scores)[-5:])
        if true_top1 in est_top5:
            top5_matches += 1

    print(f"\n[Attention Accuracy — {n_queries} queries]")
    print(f"  Score correlation:  {np.mean(corrs):.4f}")
    print(f"  Top-1 accuracy:     {top1_matches / n_queries * 100:.0f}%")
    print(f"  Top-5 recall:       {top5_matches / n_queries * 100:.0f}%")

    # ── Bit-width comparison ──
    print(f"\n[Bit-Width Comparison]")
    print(f"  {'Bits':>4} | {'Ratio':>6} | {'Score Corr':>10} | {'KB':>8}")
    print(f"  {'─' * 4}-+-{'─' * 6}-+-{'─' * 10}-+-{'─' * 8}")

    for b in [2, 3, 4]:
        cfg = TurboQuantConfig(dimension=head_dim, bits=b, qjl_enabled=True)
        c = TurboQuantKVCache(cfg)
        for t in range(seq_len):
            c.append(keys[t], values[t])

        query = rng.standard_normal(head_dim)
        query = query / np.linalg.norm(query)
        est = c.attention_scores(query)
        true = keys @ query
        cr = np.corrcoef(est, true)[0, 1]
        ratio = c.tq.compression_ratio()
        kb = 2 * seq_len * c.tq.compressed_size_bytes() / 1024

        print(f"  {b:>4} | {ratio:>5.1f}x | {cr:>10.4f} | {kb:>7.1f}")


if __name__ == "__main__":
    main()
