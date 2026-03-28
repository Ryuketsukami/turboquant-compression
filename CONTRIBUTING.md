# Contributing to TurboQuant

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/turboquant-compression.git
   cd turboquant-compression
   ```
3. **Install** dependencies:
   ```bash
   pip install -e ".[dev]"
   # or
   pip install numpy scipy pytest
   ```
4. **Run tests** to make sure everything works:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add or update tests as needed
4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
5. Run the self-test to verify correctness:
   ```bash
   python turboquant.py
   ```
6. Commit your changes with a clear message:
   ```bash
   git commit -m "Add: description of what you added"
   ```
7. Push and open a Pull Request

## What We're Looking For

### High-Impact Contributions

- **GPU kernels** — CUDA or Triton implementations of PolarQuant and QJL stages
- **Framework integrations** — vLLM, llama.cpp, TensorRT-LLM, SGLang
- **Mixed-precision support** — adaptive bit-width per layer or per head
- **Benchmarks** — results on real LLM KV cache data (LLaMA, Mistral, etc.)
- **Streaming/online mode** — incremental codebook updates for non-stationary data

### Always Welcome

- Bug fixes with a test that reproduces the issue
- Performance improvements with benchmark evidence
- Documentation improvements
- New examples demonstrating use cases

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Keep functions focused — one function, one responsibility
- Add docstrings to public functions and classes
- Use NumPy-style docstrings for complex functions

## Testing

- Every new feature needs tests
- Every bug fix needs a regression test
- Tests should be deterministic (use fixed random seeds)
- Run the full suite before submitting:
  ```bash
  pytest tests/ -v
  ```

## Pull Request Guidelines

- **One feature per PR** — keep changes focused
- **Descriptive title** — e.g., "Add Triton kernel for PolarQuant rotation"
- **Link related issues** — use "Fixes #123" or "Relates to #456"
- **Include benchmarks** if your change affects performance
- **Update documentation** if your change affects the public API

## Reporting Issues

When filing an issue, please include:

- Python version and OS
- NumPy and SciPy versions
- Steps to reproduce
- Expected vs actual behavior
- Error traceback (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
