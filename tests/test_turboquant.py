"""
Tests for TurboQuant compression.

Run with:
    pytest tests/test_turboquant.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from turboquant import (
    TurboQuant,
    TurboQuantConfig,
    TurboQuantKVCache,
    CompressedVector,
    build_lloyd_max_codebook,
    generate_rotation_matrix,
)


# ── Fixtures ──

@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def unit_vectors(rng):
    """64 random unit vectors in R^128."""
    X = rng.standard_normal((64, 128))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def tq_mse():
    """TurboQuant with QJL disabled (MSE variant)."""
    config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=False)
    return TurboQuant(config)


@pytest.fixture
def tq_prod():
    """TurboQuant with QJL enabled (Prod variant)."""
    config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
    return TurboQuant(config)


# ── Lloyd-Max Codebook Tests ──

class TestLloydMaxCodebook:
    def test_codebook_size(self):
        """Codebook should have 2^bits centroids."""
        for bits in [2, 3, 4]:
            centroids, boundaries = build_lloyd_max_codebook(64, bits)
            assert len(centroids) == 2 ** bits
            assert len(boundaries) == 2 ** bits + 1

    def test_boundaries_sorted(self):
        """Boundaries must be strictly sorted."""
        centroids, boundaries = build_lloyd_max_codebook(64, 3)
        assert np.all(np.diff(boundaries) > 0)

    def test_centroids_within_boundaries(self):
        """Each centroid must lie within its boundary interval."""
        centroids, boundaries = build_lloyd_max_codebook(64, 3)
        for i, c in enumerate(centroids):
            assert boundaries[i] <= c <= boundaries[i + 1]

    def test_centroids_in_range(self):
        """All centroids must be in [-1, 1]."""
        centroids, _ = build_lloyd_max_codebook(128, 4)
        assert np.all(centroids >= -1.0)
        assert np.all(centroids <= 1.0)

    def test_symmetry(self):
        """For Beta(a, a), the codebook should be approximately symmetric around 0."""
        centroids, _ = build_lloyd_max_codebook(64, 3)
        assert abs(np.mean(centroids)) < 0.01


# ── Rotation Matrix Tests ──

class TestRotationMatrix:
    def test_orthogonality(self):
        """Pi @ Pi.T should be identity."""
        Pi = generate_rotation_matrix(64)
        identity = Pi @ Pi.T
        np.testing.assert_allclose(identity, np.eye(64), atol=1e-10)

    def test_deterministic(self):
        """Same seed should produce same matrix."""
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=42)
        np.testing.assert_array_equal(Pi1, Pi2)

    def test_different_seeds(self):
        """Different seeds should produce different matrices."""
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=99)
        assert not np.allclose(Pi1, Pi2)


# ── Compression / Decompression Tests ──

class TestCompression:
    def test_compress_returns_indices(self, tq_mse, unit_vectors):
        """Compress should return valid indices."""
        compressed = tq_mse.compress(unit_vectors[0])
        assert isinstance(compressed, CompressedVector)
        assert compressed.indices.shape == (128,)
        assert compressed.indices.min() >= 0
        assert compressed.indices.max() < 2 ** 3

    def test_compress_with_qjl(self, tq_prod, unit_vectors):
        """With QJL enabled, compressed should include signs."""
        compressed = tq_prod.compress(unit_vectors[0])
        assert compressed.qjl_signs is not None
        assert set(np.unique(compressed.qjl_signs)).issubset({0, 1})

    def test_decompress_shape(self, tq_mse, unit_vectors):
        """Decompressed vector should have same shape as input."""
        compressed = tq_mse.compress(unit_vectors[0])
        reconstructed = tq_mse.decompress(compressed)
        assert reconstructed.shape == unit_vectors[0].shape

    def test_reconstruction_quality(self, tq_mse, unit_vectors):
        """At 3 bits, MSE should be well below 0.01 for unit vectors."""
        mses = []
        for x in unit_vectors:
            compressed = tq_mse.compress(x)
            x_hat = tq_mse.decompress(compressed)
            mses.append(np.mean((x - x_hat) ** 2))
        assert np.mean(mses) < 0.01

    def test_cosine_similarity(self, tq_mse, unit_vectors):
        """At 3 bits d=128, cosine similarity should be > 0.97."""
        cosines = []
        for x in unit_vectors:
            compressed = tq_mse.compress(x)
            x_hat = tq_mse.decompress(compressed)
            cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat) + 1e-15)
            cosines.append(cos)
        assert np.mean(cosines) > 0.97

    def test_higher_bits_lower_mse(self, unit_vectors):
        """More bits should produce lower reconstruction error."""
        mse_by_bits = {}
        for bits in [2, 3, 4]:
            config = TurboQuantConfig(dimension=128, bits=bits, qjl_enabled=False)
            tq = TurboQuant(config)
            mses = []
            for x in unit_vectors[:16]:
                c = tq.compress(x)
                x_hat = tq.decompress(c)
                mses.append(np.mean((x - x_hat) ** 2))
            mse_by_bits[bits] = np.mean(mses)
        assert mse_by_bits[4] < mse_by_bits[3] < mse_by_bits[2]

    def test_deterministic_compression(self, tq_mse, unit_vectors):
        """Same input should produce same compressed output."""
        c1 = tq_mse.compress(unit_vectors[0])
        c2 = tq_mse.compress(unit_vectors[0])
        np.testing.assert_array_equal(c1.indices, c2.indices)


# ── Inner Product Tests ──

class TestInnerProduct:
    def test_inner_product_accuracy(self, tq_prod, unit_vectors, rng):
        """Inner product estimates should correlate strongly with true values."""
        compressed = [tq_prod.compress(x) for x in unit_vectors]

        true_ips = []
        est_ips = []
        for _ in range(200):
            qi = rng.integers(0, len(unit_vectors))
            ki = rng.integers(0, len(unit_vectors))
            true_ips.append(np.dot(unit_vectors[qi], unit_vectors[ki]))
            est_ips.append(tq_prod.inner_product(unit_vectors[qi], compressed[ki]))

        corr = np.corrcoef(true_ips, est_ips)[0, 1]
        assert corr > 0.70, f"IP correlation {corr:.4f} is too low"

    def test_inner_product_low_rmse(self, tq_prod, unit_vectors, rng):
        """RMSE of inner product estimates should be small."""
        compressed = [tq_prod.compress(x) for x in unit_vectors]

        errors = []
        for _ in range(200):
            qi = rng.integers(0, len(unit_vectors))
            ki = rng.integers(0, len(unit_vectors))
            true_ip = np.dot(unit_vectors[qi], unit_vectors[ki])
            est_ip = tq_prod.inner_product(unit_vectors[qi], compressed[ki])
            errors.append((est_ip - true_ip) ** 2)

        rmse = np.sqrt(np.mean(errors))
        assert rmse < 0.15, f"IP RMSE {rmse:.4f} is too high"

    def test_mse_variant_inner_product(self, tq_mse, unit_vectors):
        """MSE variant (no QJL) should still give reasonable inner products."""
        x = unit_vectors[0]
        y = unit_vectors[1]
        compressed_y = tq_mse.compress(y)
        est_ip = tq_mse.inner_product(x, compressed_y)
        true_ip = np.dot(x, y)
        assert abs(est_ip - true_ip) < 0.1


# ── KV Cache Tests ──

class TestKVCache:
    def test_cache_append_and_length(self):
        """Cache length should match number of appended items."""
        config = TurboQuantConfig(dimension=64, bits=3, qjl_enabled=True)
        cache = TurboQuantKVCache(config)
        rng = np.random.default_rng(0)
        for _ in range(10):
            k = rng.standard_normal(64)
            v = rng.standard_normal(64)
            cache.append(k / np.linalg.norm(k), v / np.linalg.norm(v))
        assert len(cache) == 10

    def test_attention_scores_shape(self):
        """Attention scores should have length equal to cache size."""
        config = TurboQuantConfig(dimension=64, bits=3, qjl_enabled=True)
        cache = TurboQuantKVCache(config)
        rng = np.random.default_rng(0)
        for _ in range(16):
            k = rng.standard_normal(64)
            v = rng.standard_normal(64)
            cache.append(k / np.linalg.norm(k), v / np.linalg.norm(v))

        query = rng.standard_normal(64)
        query = query / np.linalg.norm(query)
        scores = cache.attention_scores(query)
        assert scores.shape == (16,)

    def test_attention_score_correlation(self):
        """Compressed attention scores should correlate with true scores."""
        d = 128
        seq_len = 32
        config = TurboQuantConfig(dimension=d, bits=3, qjl_enabled=True)
        cache = TurboQuantKVCache(config)
        rng = np.random.default_rng(0)

        keys = rng.standard_normal((seq_len, d))
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        values = rng.standard_normal((seq_len, d))
        values = values / np.linalg.norm(values, axis=1, keepdims=True)

        for t in range(seq_len):
            cache.append(keys[t], values[t])

        query = rng.standard_normal(d)
        query = query / np.linalg.norm(query)

        est_scores = cache.attention_scores(query)
        true_scores = keys @ query

        corr = np.corrcoef(est_scores, true_scores)[0, 1]
        assert corr > 0.70, f"Attention score correlation {corr:.4f} too low"


# ── Compression Ratio Tests ──

class TestCompressionRatio:
    def test_ratio_calculation(self):
        """Compression ratio should match expected formula."""
        config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=False)
        tq = TurboQuant(config)
        expected = (128 * 32) / (128 * 3)
        assert abs(tq.compression_ratio() - expected) < 0.01

    def test_ratio_with_qjl(self):
        """QJL adds 1 bit per dimension, lowering the ratio."""
        config_mse = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=False)
        config_prod = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
        tq_mse = TurboQuant(config_mse)
        tq_prod = TurboQuant(config_prod)
        assert tq_mse.compression_ratio() > tq_prod.compression_ratio()

    def test_compressed_size_bytes(self):
        """Compressed size in bytes should be correct."""
        config = TurboQuantConfig(dimension=128, bits=4, qjl_enabled=False)
        tq = TurboQuant(config)
        expected_bytes = (128 * 4 + 7) // 8  # 64 bytes
        assert tq.compressed_size_bytes() == expected_bytes


# ── Edge Cases ──

class TestEdgeCases:
    def test_zero_vector(self, tq_mse):
        """Should handle zero vector without error."""
        x = np.zeros(128)
        compressed = tq_mse.compress(x)
        reconstructed = tq_mse.decompress(compressed)
        assert reconstructed.shape == (128,)

    def test_single_dimension_large(self):
        """Should work with larger dimensions."""
        config = TurboQuantConfig(dimension=512, bits=3, qjl_enabled=True)
        tq = TurboQuant(config)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(512)
        x = x / np.linalg.norm(x)
        compressed = tq.compress(x)
        reconstructed = tq.decompress(compressed)
        mse = np.mean((x - reconstructed) ** 2)
        assert mse < 0.01

    def test_2bit_quantization(self):
        """2-bit quantization should work with higher MSE."""
        config = TurboQuantConfig(dimension=128, bits=2, qjl_enabled=False)
        tq = TurboQuant(config)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(128)
        x = x / np.linalg.norm(x)
        compressed = tq.compress(x)
        reconstructed = tq.decompress(compressed)
        cos = np.dot(x, reconstructed) / (np.linalg.norm(x) * np.linalg.norm(reconstructed) + 1e-15)
        assert cos > 0.93
