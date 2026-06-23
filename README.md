# Z3 Quantum-Flow

[![Rust](https://img.shields.io/badge/rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/zerocopies/Z3-Quantum-Flow?style=flat-square)](https://github.com/zerocopies/Z3-Quantum-Flow)

**A literally zero-copy local LLM inference engine — built from scratch in Rust.**

Z3 Quantum-Flow is a custom inference engine for running quantized large language models locally, with **no dependency on llama.cpp's high-level API**. It implements its own compute graph, KV cache management, batched prefill, and autoregressive decode loop directly on top of ggml primitives.

Built and maintained by [Zero Copies](https://github.com/zerocopies) — engineered for resource-constrained hardware without compromising on correctness.

> **Perfect for:** Laptops, servers, edge devices, or any system where memory bandwidth matters. Runs Llama 3.1 8B at **1.49 tok/s on a 12-year-old ThinkPad X240**.

---

## 🚀 What makes it different

Most local inference tools are **wrappers around llama.cpp**. Z3 Quantum-Flow is **not**. It owns its entire forward pass — from GGUF weight loading through memory-mapped tensors, to the compute graph, KV cache, and token sampling. Every component was designed, reviewed, and hardened through multiple iterations.

### Key design decisions:

- **Zero-copy weight loading** — model weights are memory-mapped directly from disk. No heap allocation for weights, ever. The engine wraps the mmap base pointer in a ggml backend buffer.
- **Quantum-KV cache** — a single contiguous backend-allocated buffer for all layers. K and V tensors for each layer are views into this buffer at fixed offsets. Zero-copy writes via `ggml_cpy` into view slices — no round-trip through host memory during decode.
- **Batched prefill** — all prompt tokens processed in a single graph pass with a causal mask. One gallocr plan regardless of prompt length.
- **Per-token decode** — single-token autoregressive decode with graph rebuild per step. View offsets for KV writes are baked in at build time.
- **Sliding window session manager** — when context fills, drops oldest turns and re-prefills from the system prompt, seamlessly.

---

## ⚡ Performance

Tested on a **ThinkPad X240** (Intel Core i5-4300U, 8GB RAM, SSD) — 12-year-old hardware.

| Model | Prefill | Decode | Context |
|-------|---------|--------|---------|
| **Llama 3.1 8B Q4_K_M** | ~11s / 28 tokens | **1.49 tok/s** | 512 |

**Improvement over initial Z1 baseline:**
- Prefill: 32,911ms → 11,309ms (**3x faster**) after batched prefill
- Decode: 0.83 → 1.49 tok/s (**1.8x faster**) after graph correctness fixes

> Metrics table will be expanded with Phi-3-mini and Qwen2.5-Coder results.

---

## 🏗️ Architecture

```
GGUF file (mmap)
     │
     ▼
MappedModel (zero-copy weight tensors)
     │
     ▼
ForwardPass (Z3 Quantum-Flow Engine)
  ├── ModelDNA        — hyperparameters from GGUF metadata
  ├── QuantumKV       — contiguous KV cache, view-based writes
  ├── build_prefill_graph()  — batched N-token graph
  ├── build_graph()          — single-token decode graph
  └── cleanup_graph_resources() — single-point teardown
     │
     ▼
generate.rs (sliding window session + sampling)
     │
     ▼
qflow binary / qflow-server HTTP API
```

---

## 📦 Models supported

| Model | Status | Notes |
|-------|--------|-------|
| **Llama 3.1 8B (Q4_K_M)** | ✅ Working | Fully tested, production-ready |
| Qwen2.5-Coder 1.5B / 3B | 🔧 In progress | QKV bias + GQA fix needed |
| Phi-3-mini | 🔧 In progress | Fused QKV split needed |

---

## 🎯 Quick start

### Requirements
- Rust 1.75+
- Linux (tested on Linux Mint)
- A GGUF model file (e.g., from [Hugging Face](https://huggingface.co/models?other=gguf))

### Build

```bash
git clone https://github.com/zerocopies/Z3-Quantum-Flow
cd Z3-Quantum-Flow/z1-core
cargo build --release --bin qflow
```

### Run

```bash
# Default 512 context
./target/release/qflow /path/to/model.gguf

# Custom context size
Z1_CTX_SIZE=2048 ./target/release/qflow /path/to/model.gguf
```

### Commands in chat

```
/reset   — clear conversation memory and KV cache
/exit    — quit
```

---

## 🌐 HTTP Server

Z3 Quantum-Flow ships a standalone HTTP inference server for integration with external applications:

```bash
cargo build --release --bin qflow-server
./target/release/qflow-server
```

### API Endpoints

```
GET  /health          — liveness check
POST /load_model      — load a GGUF file
POST /chat            — run inference, returns text + stats
```

Compatible with the included `z1-web.html` browser UI.

### Example request

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Rust?"}'
```

---

## 📂 Project structure

```
Z3-Quantum-Flow/
├── z1-core/
│   ├── src/
│   │   ├── graph.rs       — Z3 Quantum-Flow engine (ForwardPass)
│   │   ├── generate.rs    — autoregressive loop + sliding window session
│   │   ├── loader.rs      — GGUF loader + zero-copy mmap
│   │   ├── mapper.rs      — memory mapper
│   │   ├── tokenizer.rs   — BPE tokenizer from GGUF metadata
│   │   ├── logits.rs      — sampling (temperature, top-p, repetition penalty)
│   │   ├── gguf.rs        — GGUF format parser
│   │   ├── ggml_ffi.rs    — raw ggml bindings
│   │   └── bin/
│   │       └── z1-server.rs  — HTTP inference server
│   └── Cargo.toml
├── ZeroCopies/            — Tauri desktop UI (in development)
├── z1-web.html            — browser chat UI
├── LICENSE
└── README.md
```

---

## 🗓️ Roadmap

- [ ] Phi-3 fused QKV support
- [ ] Qwen2.5 QKV bias + GQA broadcasting
- [ ] Buzz Router — governance and multi-model routing layer
- [ ] NEXUS — multi-agent coordination system
- [ ] Batched decode (multiple sequences)
- [ ] Used-context attention optimization (attend to head, not full n_ctx)
- [ ] Context size CLI argument
- [ ] macOS & Windows support
- [ ] Performance optimizations for ARM (Pi, mobile)

---

## 🤝 Contributing

Contributions are welcome! Areas of particular interest:

- Model architecture support (new quantization formats, attention variants)
- Performance optimization (SIMD, platform-specific kernels)
- Documentation improvements
- Test coverage

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📚 Further reading

- [Zero-Copy Memory Mapping in Rust](https://docs.rust-embedded.org/book/)
- [GGML Overview](https://github.com/ggerganov/ggml)
- [LLM Inference Optimization](https://arxiv.org/abs/2206.04615)
- [Quantization for LLMs](https://arxiv.org/abs/2210.17323)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---

## 🔗 Community

- **GitHub Issues:** [Report bugs or request features](https://github.com/zerocopies/Z3-Quantum-Flow/issues)
- **GitHub Discussions:** [Ask questions and share ideas](https://github.com/zerocopies/Z3-Quantum-Flow/discussions)

---

*Part of the [Zero Copies](https://github.com/zerocopies) product family — building AI infrastructure for the real world.*
