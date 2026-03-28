<p align="center">
  <img src="https://img.shields.io/badge/TurboQuant-ICLR%202026-blueviolet?style=for-the-badge" alt="ICLR 2026" />
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License" />
</p>

<h1 align="center">TurboQuant</h1>

<p align="center">
  <strong>Near-optimal vector quantization for LLM KV cache compression.</strong><br/>
  3-bit quantization with minimal accuracy loss and up to 8x memory reduction.
</p>

<p align="center">
  A Python implementation of the <a href="https://arxiv.org/abs/2504.19874">TurboQuant algorithm</a> (ICLR 2026) by Zandieh et al.<br/>
  PolarQuant (Lloyd-Max on rotated coordinates) + QJL (1-bit residual correction for unbiased inner products).
</p>

---

## Highlights

- **3-bit KV cache compression** with near-optimal distortion rate
- **Two-stage pipeline**: PolarQuant (MSE-optimal scalar quantization) + QJL (unbiased inner product estimation)
- **Lloyd-Max quantizer** built on the exact Beta(d/2, d/2) distribution that arises from random rotation
- **No calibration data required** — works online, token-by-token
- **Simple API** — compress, decompress, and estimate inner products in a few lines
- **KV cache simulator** included for attention score benchmarking
- **Pure Python + NumPy/SciPy** — no custom CUDA kernels, easy to read and extend

## How It Works

TurboQuant compresses high-dimensional vectors (e.g., KV cache entries in transformer attention) in two stages:

### Stage 1: PolarQuant

1. **Random rotation**: Multiply the input vector by a fixed random orthogonal matrix. This makes each coordinate follow a known Beta(d/2, d/2) distribution.
2. **Lloyd-Max scalar quantization**: Quantize each rotated coordinate independently using a Lloyd-Max quantizer optimized for the Beta distribution. This achieves near-optimal MSE distortion.

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)

1. **Compute residual**: The difference between the rotated vector and its quantized reconstruction.
2. **1-bit projection**: Project the residual onto random Rademacher vectors and store only the signs (1 bit each).
3. **Inner product correction**: At query time, use the stored signs to correct the inner product estimate, making it unbiased.

```
Input vector x ∈ R^d
    │
    ▼
┌─────────────────────┐
│  Rotate: y = Π·x    │  (random orthogonal matrix Π)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Lloyd-Max quantize  │  (B bits per coordinate)
│  indices = Q(y)      │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  QJL residual signs  │  (1 bit per projection dimension)
│  s = sign(S·(y-ŷ))  │
└─────────┬───────────┘
          ▼
   CompressedVector(indices, signs)
```

| Variant | What it stores | Best for |
|---------|---------------|----------|
| **TurboQuant_mse** | Stage 1 only (indices) | Reconstruction tasks, lowest memory |
| **TurboQuant_prod** | Stage 1 + Stage 2 (indices + signs) | Attention/inner product tasks |

## Installation

```bash
git clone https://github.com/ryuketsukami/turboquant-compression.git
cd turboquant-compression
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy scipy
```

### Requirements

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10

## Quick Start

```python
import numpy as np
from turboquant import TurboQuant, TurboQuantConfig

# Configure: 256-dimensional vectors, 3-bit quantization
config = TurboQuantConfig(dimension=256, bits=3, qjl_enabled=True)
tq = TurboQuant(config)

# Compress a vector
x = np.random.randn(256)
x = x / np.linalg.norm(x)  # unit normalize

compressed = tq.compress(x)
reconstructed = tq.decompress(compressed)

print(f"Compression ratio: {tq.compression_ratio():.1f}x")
print(f"MSE: {np.mean((x - reconstructed) ** 2):.6f}")
print(f"Cosine similarity: {np.dot(x, reconstructed) / np.linalg.norm(reconstructed):.4f}")
```

### KV Cache Simulation

```python
from turboquant import TurboQuantKVCache, TurboQuantConfig

config = TurboQuantConfig(dimension=128, bits=3, qjl_enabled=True)
cache = TurboQuantKVCache(config)

# Simulate a sequence of KV pairs
for t in range(64):
    key = np.random.randn(128)
    value = np.random.randn(128)
    cache.append(key / np.linalg.norm(key), value / np.linalg.norm(value))

# Compute attention scores against a query
query = np.random.randn(128)
query = query / np.linalg.norm(query)
scores = cache.attention_scores(query)
```

### Inner Product Estimation

```python
# TurboQuant_prod gives unbiased inner product estimates
key = np.random.randn(256)
key = key / np.linalg.norm(key)
compressed_key = tq.compress(key)

query = np.random.randn(256)
query = query / np.linalg.norm(query)

estimated_ip = tq.inner_product(query, compressed_key)
true_ip = np.dot(query, key)
print(f"True: {true_ip:.4f}, Estimated: {estimated_ip:.4f}")
```

## Benchmarks

Results on random unit vectors (d=256, n=64):

| Bits | Variant | MSE | Cosine Sim | IP Correlation | Compression |
|------|---------|-----|------------|----------------|-------------|
| **2** | MSE | ~0.0058 | ~0.9971 | — | **16.0x** |
| **2** | Prod (+QJL) | — | — | ~0.9943 | 10.7x |
| **3** | MSE | ~0.0014 | ~0.9993 | — | **10.7x** |
| **3** | Prod (+QJL) | — | — | ~0.9987 | 8.0x |
| **4** | MSE | ~0.0003 | ~0.9999 | — | **8.0x** |
| **4** | Prod (+QJL) | — | — | ~0.9997 | 6.4x |

Run the self-test to see exact numbers on your hardware:

```bash
python turboquant.py
```

## API Reference

### `TurboQuantConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dimension` | `int` | required | Vector dimension (d) |
| `bits` | `int` | `4` | Bit-width for PolarQuant (2-4 typical) |
| `qjl_enabled` | `bool` | `True` | Enable QJL residual correction |
| `qjl_dim` | `int` | `0` | QJL projection dimension (0 = auto = d) |
| `seed` | `int` | `42` | Random seed for reproducibility |

### `TurboQuant`

| Method | Description |
|--------|-------------|
| `compress(x)` | Compress a vector, returns `CompressedVector` |
| `decompress(compressed)` | Reconstruct from PolarQuant (Stage 1) |
| `inner_product(query, compressed_key)` | Unbiased inner product estimate (Stage 1 + 2) |
| `compression_ratio()` | Original bits / compressed bits |
| `compressed_size_bytes()` | Size of one compressed vector in bytes |

### `TurboQuantKVCache`

| Method | Description |
|--------|-------------|
| `append(key, value)` | Add a KV pair to the compressed cache |
| `attention_scores(query)` | Compute attention logits for all cached keys |

## Project Structure

```
turboquant-compression/
├── turboquant.py           # Core implementation
├── tests/
│   └── test_turboquant.py  # Test suite
├── examples/
│   ├── basic_usage.py      # Getting started example
│   └── kv_cache_demo.py    # KV cache simulation
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contribution guidelines
├── SECURITY.md             # Security policy
├── CITATION.cff            # Citation metadata
└── CHANGELOG.md            # Version history
```

## Background: The TurboQuant Paper

This implementation is based on:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
> ICLR 2026 &bull; [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

Key theoretical contributions of the paper:
- **PolarQuant** achieves the rate-distortion bound for unit vectors using random rotation + optimal scalar quantization
- **QJL** (Quantized Johnson-Lindenstrauss) provides unbiased inner product estimation from 1-bit projections of quantization residuals
- The combination enables **3-bit KV cache compression** with zero accuracy loss on standard LLM benchmarks
- Demonstrated **6x memory reduction** and **8x faster attention** on NVIDIA H100 GPUs

### Related Work

| Project | Focus | Comparison |
|---------|-------|------------|
| [GPTQ](https://github.com/IST-DASLab/gptq) | Weight quantization | Compresses model weights, not KV cache |
| [AWQ](https://github.com/mit-han-lab/llm-awq) | Activation-aware weight quantization | Weight-focused, different technique |
| [KIVI](https://github.com/vllm-project/vllm) | KV cache quantization | TurboQuant improves upon KIVI's approach |
| [kvpress](https://github.com/NVIDIA/kvpress) | KV cache compression toolkit | Multi-method toolkit, different algorithms |

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions are especially valued:
- GPU-accelerated kernels (CUDA/Triton)
- Integration with inference frameworks (vLLM, llama.cpp, TensorRT-LLM)
- Additional quantization bit-widths and mixed-precision support
- Benchmarks on real LLM KV cache data

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with NumPy and SciPy. Inspired by Google Research.</sub>
</p>
