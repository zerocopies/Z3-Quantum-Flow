# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-06-23

### Added
- ✨ Initial release of Z3 Quantum-Flow
- Zero-copy weight loading with memory mapping
- Quantum-KV cache with view-based writes
- Batched prefill for efficient prompt processing
- Per-token autoregressive decode
- Sliding window session manager with context overflow handling
- Support for Llama 3.1 8B (Q4_K_M quantized)
- Interactive CLI interface (`qflow`)
- HTTP inference server (`qflow-server`)
- Browser-based chat UI (`z1-web.html`)
- BPE tokenizer from GGUF metadata
- Sampling algorithms: temperature, top-p, repetition penalty

### Performance
- Llama 3.1 8B: **1.49 tok/s** decode on 12-year-old ThinkPad X240
- **3x speedup** in prefill over baseline (32,911ms → 11,309ms)
- **1.8x speedup** in decode (0.83 → 1.49 tok/s)

### Documentation
- Comprehensive README with architecture diagrams
- Contributing guidelines
- Code of Conduct
- Security policy
- Project structure overview

---

## [Unreleased]

### In Progress
- Phi-3-mini support (fused QKV split)
- Qwen2.5-Coder support (QKV bias + GQA broadcasting)
- macOS support
- Windows support
- ARM platform optimizations (Raspberry Pi, mobile)

### Planned
- Multi-model routing layer (Buzz Router)
- Multi-agent coordination (NEXUS)
- Batched decode (multiple sequences)
- Used-context attention optimization
- Context size CLI argument
- Additional benchmarks and performance reports

---

## Versioning

Z3 Quantum-Flow follows Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes or architecture redesigns
- **MINOR**: New features, model support, or platform additions
- **PATCH**: Bug fixes, performance improvements, documentation

---

*For more details, see [GitHub Releases](https://github.com/zerocopies/Z3-Quantum-Flow/releases).*
