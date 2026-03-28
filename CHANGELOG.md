# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-28

### Added

- Core TurboQuant implementation with PolarQuant (Stage 1) and QJL (Stage 2)
- Lloyd-Max quantizer for Beta(d/2, d/2) distribution
- Random orthogonal rotation matrix generation
- `TurboQuant` class with compress, decompress, and inner product estimation
- `TurboQuantKVCache` class for KV cache simulation
- Self-test and demo suite (`python turboquant.py`)
- Test suite with compression, inner product, and KV cache tests
- Basic usage and KV cache demo examples
- Full documentation and API reference
