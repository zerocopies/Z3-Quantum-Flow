```markdown
# Inference Z1

A zero-copy inference engine for Llama 3 models, built in pure Rust from scratch.

Unlike mainstream wrappers (like Ollama) or massive frameworks (like Candle), Inference Z1 does not hide behind bloated dependencies. It is a deeply optimized, systems-level engine where the physical execution graph, the KV cache logic, and the memory mapping are governed by hand. It was built to extract absolute maximum performance from older, heavily constrained hardware.

## ✨ Core Features

* **Zero-Copy Memory Mapping:** Uses `memmap2` to map massive neural network weights directly into virtual memory. The engine allocates **0 bytes** of heap memory for the model weights, allowing it to run 4.5GB models on machines with limited RAM (e.g., dual-core laptops from 2014) without crashing the OS.
* **Sliding-Window KV Cache:** A persistent, custom-built conversation memory manager. When the engine reaches its context limit, it drops the oldest tokens while preserving the system prompt, seamlessly re-prefilling without ever throwing a "Context Full" panic.
* **Manual Thread Governance:** Hardcoded hardware thread locks to prevent aggressive BPE (Byte-Pair Encoding) and CPU cache contention on older hyperthreaded architectures.
* **Cross-Platform:** Safely abstracts POSIX and Windows file handling. CI pipelines guarantee it compiles and runs natively on Windows, macOS, and Linux.

## 📥 Prerequisites & Model Download

To run Z1, you need the Rust toolchain installed and a quantized Llama 3.1 `.gguf` file. 

**⚠️ Compatibility Note:** While the engine handles standard GGUF parsing, **it has currently only been rigorously tested with the Llama 3.1 8B Instruct model** (specifically the Q4_K_M quantization). Other architectures or quantizations may require slight adjustments to the tensor math.

* **Download Link:** [Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (via bartowski on Hugging Face)](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)

## 🚀 Usage

1. Clone this repository.
2. Download the model file using the link above.
3. Boot the engine by passing the model path as a command-line argument.

```bash
cargo run --release -- /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

```

### Quick Run (Alias)

Typing the full cargo command every time can get tedious. You can set up a quick terminal alias to make booting the engine frictionless:

```bash
alias z1="RUST_LOG=info cargo run --release --"

# Now you can just run:
z1 /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

```

Alternatively, you can just build the binary once and run it directly:

```bash
cargo build --release
./target/release/z1 /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

```

## 🏗️ Architecture

The engine is strictly decoupled into two components:

1. **`z1` (Library Crate):** Contains the zero-copy loader, tokenizer, mathematical graph pass, and sliding window session manager.
2. **`main.rs` (Binary):** A lightweight executable that wires the components together into a terminal chat loop.

*(Note: Because of this decoupled architecture, the `z1` core is fully prepped to be wrapped in a Tauri application for a native desktop GUI in the future).*

## 💬 Try It Out & Share Your Thoughts!

I highly encourage everyone to clone this repository, download the model, and spin it up on your own hardware! Whether you are running it on a beefy modern desktop or pushing a decade-old laptop to its absolute limits, I want to hear how it performs for you.

All types of reviews, thoughts, bug reports, and pull requests are warmly welcomed. Let's push the limits of local, low-resource AI together!

```

```
