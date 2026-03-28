# Agent Instructions — TurboQuant Compression

This file provides instructions for AI coding agents (Claude Code, GitHub Copilot, Codex CLI, Cursor, Windsurf, etc.) working with this repository.

## What This Project Is

TurboQuant is a Python implementation of the TurboQuant algorithm from the ICLR 2026 paper by Zandieh et al. (arXiv:2504.19874). It compresses high-dimensional vectors — specifically LLM key-value cache entries — using a two-stage pipeline:

1. **PolarQuant** (Stage 1): Random orthogonal rotation + Lloyd-Max scalar quantization on the resulting Beta(d/2, d/2) distribution
2. **QJL** (Stage 2): 1-bit Quantized Johnson-Lindenstrauss residual correction for unbiased inner product estimation

## Project Structure

```
turboquant/              # Python package
  __init__.py            # Core implementation: TurboQuant, TurboQuantConfig, TurboQuantKVCache, CompressedVector
  __main__.py            # Entry point for python -m turboquant
turboquant.py            # Standalone self-test script
tests/test_turboquant.py # 27 pytest tests
examples/                # Usage examples
```

## Key Classes and Functions

- `TurboQuantConfig(dimension, bits, qjl_enabled, qjl_dim, seed)` — Configuration dataclass
- `TurboQuant(config)` — Main compressor. Methods: `compress(x)`, `decompress(compressed)`, `inner_product(query, compressed_key)`
- `TurboQuantKVCache(config)` — KV cache simulator. Methods: `append(key, value)`, `attention_scores(query)`
- `CompressedVector` — Dataclass holding quantization indices and optional QJL signs
- `build_lloyd_max_codebook(dimension, bits)` — Builds the offline codebook
- `generate_rotation_matrix(dimension, seed)` — Random orthogonal matrix via QR

## Coding Conventions

- Pure Python with NumPy and SciPy only — no PyTorch, no CUDA
- Type hints on all public function signatures
- NumPy-style docstrings
- Tests use pytest with fixed random seeds for determinism
- Package importable as `from turboquant import TurboQuant, TurboQuantConfig`

## How to Run Tests

```bash
pip install -e .
pytest tests/ -v          # 27 tests
python -m turboquant      # self-test with benchmarks
```

## Common Tasks

- **Add a new quantization variant**: Extend `TurboQuant` class, add tests in `tests/test_turboquant.py`
- **Change bit-width support**: Modify `build_lloyd_max_codebook()` — the Lloyd-Max iteration is generic
- **Add GPU acceleration**: Replace NumPy operations with CuPy or PyTorch equivalents
- **Integrate with vLLM/HuggingFace**: Use `TurboQuantKVCache` as a reference for the compression API
