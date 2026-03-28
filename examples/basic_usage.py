"""
TurboQuant — Basic Usage Example
=================================

Demonstrates how to compress and decompress vectors,
and estimate inner products with TurboQuant.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from turboquant import TurboQuant, TurboQuantConfig


def main():
    rng = np.random.default_rng(42)

    # ── Configure ──
    d = 256         # vector dimension
    bits = 3        # bits per coordinate

    print(f"TurboQuant Basic Usage — d={d}, bits={bits}")
    print("=" * 50)

    # ── Stage 1 only (MSE variant) ──
    config_mse = TurboQuantConfig(dimension=d, bits=bits, qjl_enabled=False)
    tq_mse = TurboQuant(config_mse)

    x = rng.standard_normal(d)
    x = x / np.linalg.norm(x)

    compressed = tq_mse.compress(x)
    reconstructed = tq_mse.decompress(compressed)

    mse = np.mean((x - reconstructed) ** 2)
    cos_sim = np.dot(x, reconstructed) / (np.linalg.norm(x) * np.linalg.norm(reconstructed))

    print(f"\n[MSE Variant — PolarQuant only]")
    print(f"  MSE:              {mse:.6f}")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Compression ratio: {tq_mse.compression_ratio():.1f}x")
    print(f"  Compressed size:   {tq_mse.compressed_size_bytes()} bytes (vs {d * 4} bytes original)")

    # ── Stage 1 + Stage 2 (Prod variant) ──
    config_prod = TurboQuantConfig(dimension=d, bits=bits, qjl_enabled=True)
    tq_prod = TurboQuant(config_prod)

    key = rng.standard_normal(d)
    key = key / np.linalg.norm(key)
    compressed_key = tq_prod.compress(key)

    query = rng.standard_normal(d)
    query = query / np.linalg.norm(query)

    true_ip = np.dot(query, key)
    est_ip = tq_prod.inner_product(query, compressed_key)

    print(f"\n[Prod Variant — PolarQuant + QJL]")
    print(f"  True inner product:      {true_ip:.6f}")
    print(f"  Estimated inner product: {est_ip:.6f}")
    print(f"  Absolute error:          {abs(est_ip - true_ip):.6f}")
    print(f"  Compression ratio:       {tq_prod.compression_ratio():.1f}x")

    # ── Batch accuracy ──
    n = 100
    keys = rng.standard_normal((n, d))
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    compressed_keys = [tq_prod.compress(k) for k in keys]

    true_ips = []
    est_ips = []
    for i in range(n):
        q = rng.standard_normal(d)
        q = q / np.linalg.norm(q)
        ki = rng.integers(0, n)
        true_ips.append(np.dot(q, keys[ki]))
        est_ips.append(tq_prod.inner_product(q, compressed_keys[ki]))

    corr = np.corrcoef(true_ips, est_ips)[0, 1]
    rmse = np.sqrt(np.mean((np.array(est_ips) - np.array(true_ips)) ** 2))

    print(f"\n[Batch Accuracy — {n} random query-key pairs]")
    print(f"  IP correlation: {corr:.4f}")
    print(f"  IP RMSE:        {rmse:.6f}")


if __name__ == "__main__":
    main()
