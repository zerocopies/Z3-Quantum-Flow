# Contributing to Z3 Quantum-Flow

First off, thanks for considering contributing to Z3 Quantum-Flow! It's people like you that make this project such a great tool.

## 🎯 Ways to contribute

### 🐛 Report bugs
- Use the [GitHub Issues](https://github.com/zerocopies/Z3-Quantum-Flow/issues) tracker
- Include your hardware specs (CPU, RAM, OS)
- Provide reproduction steps and error logs
- Attach model details (name, quantization format)

### 💡 Suggest enhancements
- Check [existing discussions](https://github.com/zerocopies/Z3-Quantum-Flow/discussions) first
- Describe the use case and expected behavior
- Link to relevant papers or implementations if applicable

### 📝 Improve documentation
- Fix typos or unclear explanations
- Add examples or use cases
- Improve code comments
- Update architecture diagrams

### 💻 Submit code

**Areas of particular interest:**

- **Model support:** New quantization formats (Q3_K, Q6_K), attention variants (GQA, MQA), model architectures
- **Performance:** SIMD optimizations, platform-specific kernels, memory layout improvements
- **Platforms:** macOS, Windows, ARM (Raspberry Pi, mobile)
- **Testing:** Unit tests, benchmark suites, regression tests
- **Tooling:** Integration examples, build system improvements

---

## 🚀 Getting started with development

### Prerequisites
- Rust 1.75+ ([install](https://rustup.rs/))
- Linux environment (or WSL2)
- Basic understanding of LLM inference

### Setup

```bash
git clone https://github.com/zerocopies/Z3-Quantum-Flow
cd Z3-Quantum-Flow
cd z1-core

# Build in debug mode for faster compilation
cargo build

# Run tests
cargo test

# Run with a model
./target/debug/qflow /path/to/model.gguf
```

### Project structure reference

- `src/graph.rs` — Core inference engine (compute graph building, execution)
- `src/generate.rs` — Autoregressive generation loop, session management
- `src/loader.rs` — GGUF model loading, zero-copy weight handling
- `src/mapper.rs` — Memory mapping utilities
- `src/tokenizer.rs` — BPE tokenizer implementation
- `src/logits.rs` — Sampling algorithms (temperature, top-p, penalties)
- `src/gguf.rs` — GGUF format parsing
- `src/ggml_ffi.rs` — Raw ggml C bindings

---

## 📋 Pull Request Process

1. **Fork** the repository
2. **Create a branch** with a descriptive name: `feature/model-support` or `fix/kv-cache-bug`
3. **Make your changes** with clear commit messages
4. **Add tests** if applicable
5. **Run `cargo fmt` and `cargo clippy`** to ensure code quality
6. **Submit a PR** with a detailed description of what and why

### PR checklist

- [ ] Code follows Rust style guidelines (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Tests pass (`cargo test`)
- [ ] Documentation updated (README, code comments)
- [ ] Commit messages are clear and descriptive
- [ ] Performance impact assessed (if applicable)

---

## 🔍 Code guidelines

### Rust idioms
- Use `?` operator for error propagation
- Prefer `match` over nested `if-let`
- Avoid `unwrap()` — use `Result` and `Option` types
- Keep functions focused and readable

### Memory safety
- Zero-copy operations are critical — document memory ownership carefully
- Use `unsafe` sparingly and document why it's necessary
- Consider edge cases (buffer boundaries, alignment)

### Performance
- Profile before optimizing — use `cargo flamegraph` or similar tools
- Document performance trade-offs
- Test on resource-constrained hardware (the test bench is a 12-year-old laptop!)

### Comments & documentation
- Document *why*, not just *what*
- Explain non-obvious memory or compute patterns
- Add examples for public APIs

---

## 🧪 Testing

### Run all tests
```bash
cargo test
```

### Run specific test
```bash
cargo test kv_cache
```

### Test with a specific model
```bash
./target/debug/qflow /path/to/llama-8b-q4_k_m.gguf
```

---

## 📊 Performance benchmarking

We benchmark on a ThinkPad X240 (i5-4300U, 8GB RAM) to ensure the engine runs efficiently on constrained hardware.

### Quick benchmark
```bash
time ./target/release/qflow /path/to/model.gguf <<< "What is Rust?"
```

### Profile with flamegraph
```bash
cargo flamegraph --release --bin qflow -- /path/to/model.gguf
```

---

## 🤔 Questions?

- **GitHub Discussions:** [Ask questions here](https://github.com/zerocopies/Z3-Quantum-Flow/discussions)
- **GitHub Issues:** [Report bugs or feature requests](https://github.com/zerocopies/Z3-Quantum-Flow/issues)

---

## 📜 License

By contributing to Z3 Quantum-Flow, you agree that your contributions will be licensed under its Apache 2.0 license.

---

*Thank you for helping make Z3 Quantum-Flow better! 🚀*
