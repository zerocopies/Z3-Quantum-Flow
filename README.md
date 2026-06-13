# Inference Z1

**A from-scratch, zero-copy LLM inference engine in Rust.**

Built and benchmarked on a 2014 ThinkPad X240 (8 GB RAM, no GPU). Runs Llama 3.1 8B (Q4_K_M quantized) with a persistent decode graph and a hand-rolled KV cache.

Part of the **Zero Copies** project.

---

## What this is

Inference Z1 loads a GGUF model file by memory-mapping it directly into ggml tensors — the 4.5 GiB of model weights are **never copied onto the heap**. The forward pass is a hand-built ggml compute graph (Q/K/V projections, RoPE, GQA attention, SwiGLU FFN, lm_head) with no dependency on `llama_decode` or the high-level llama.cpp API.

On top of that:

- **A persistent decode graph** — built once, reused for every generated token. No per-token graph rebuild or reallocation.
- **A real KV cache** — F32, allocated in a backend buffer, written via `ggml_cpy` on prefill and `ggml_backend_tensor_set` on decode. Attention reads the full cached history through `ggml_view_2d`.
- **Multi-turn chat** — the KV cache persists across conversation turns. Follow-up messages append to existing context instead of re-running the whole conversation from scratch.
- **A correctness-gated dev harness** — `--bench` runs a regression check ("does it actually say Paris?") before reporting any speed numbers.

---

## Performance

Measured on a ThinkPad X240 (Intel i5-4300U, 8 GB RAM, 2 physical cores), Llama 3.1 8B Instruct, Q4_K_M quantization, 2 threads:

| Stage | Result |
|---|---|
| Correctness | "The capital of France is" → **Paris** ✓ |
| Decode speed | **~1.6–1.75 tok/s**, sustained across 100+ tokens |
| Context window | 512 tokens |

The journey to get here:

```
No KV cache, full re-prefill every token:        ~0.05 tok/s
+ KV cache (cache history, no graph reuse):       0.13 tok/s   (2.6x)
+ Persistent decode graph (no per-token rebuild): 1.6  tok/s   (12x)
+ 2-thread tuning (vs 1 or 4):                     1.75 tok/s   (best on this CPU)
```

That's a **~32x speedup** from architecture alone — no hardware change, no different quantization.

Why 2 threads beats 4 on this CPU: the workload is memory-bandwidth bound, not compute bound. The X240 has 2 physical cores with hyperthreading; the extra hyperthreads contend for the same memory bus and make things slightly *worse*. Your mileage will vary by CPU — run `--bench` to find your machine's sweet spot.

---

## Building

Requires Rust (stable) and a C/C++ toolchain (the vendored ggml/llama.cpp C sources are compiled as part of the build).

```bash
git clone <this repo>
cd inference-z1
cargo build --release
```

First build compiles the vendored ggml C sources and can take several minutes. Subsequent builds (Rust-only changes) are fast (~10-30s).

You'll need a GGUF model file. This project was built and tested against:

```
Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Place it anywhere and point `-m` at it.

---

## Usage

### Single-shot prompt
```bash
./target/release/z1 -m model.gguf -p "Explain how zero-copy memory mapping works." -n 100
```

### Interactive chat (multi-turn, persistent context)
```bash
./target/release/z1 -m model.gguf --chat
```

Conversation history is held in the KV cache across turns — follow-up questions can reference earlier messages. The cache holds 512 tokens; use `/reset` to start a new conversation, or `/quit` / `/exit` to leave.

### Dev harness / benchmark
```bash
./target/release/z1 -m model.gguf --bench
```

Runs a correctness regression (checks the model says "Paris" to a factual prompt at low temperature) followed by a 3-run decode-speed benchmark. If the regression fails, the benchmark is skipped — there's no point measuring the speed of a broken forward pass.

### Options
```
-m, --model <path>       Path to GGUF model file
-p, --prompt <text>      Single-shot prompt (non-interactive)
-c, --chat                Interactive multi-turn chat
-b, --bench               Run dev harness (regression + speed)
-n, --max-tokens <N>      Maximum tokens to generate  [default: 512]
-t, --temperature <f>     Sampling temperature        [default: 0.7]
    --top-p <f>           Nucleus sampling threshold  [default: 0.9]
    --no-template         Skip the Llama 3.1 chat template wrapping
-h, --help                Print help
```

### Debugging
Set `Z1_TRACE=1` to print per-token diagnostics (KV cache head position, mask state, argmax logits at each step):

```bash
Z1_TRACE=1 ./target/release/z1 -m model.gguf -p "hi" -n 5
```

---

## Architecture

```
gguf.rs      → parse GGUF header (metadata, tensor descriptors), no weight loading
loader.rs    → memory-map the model file, wrap as ggml CPU backend buffer,
               point tensors directly into the mmap (zero heap copies for weights)
graph.rs     → the forward pass: hyperparameters, KV cache, persistent decode
               graph, prefill (full-prompt) and decode_one (single-token)
logits.rs    → final RMS-norm, lm_head projection, temperature/top-p/top-k sampling
tokenizer.rs → BPE tokenizer built from GGUF vocab tables
generate.rs  → autoregressive generation loop, chat templating, multi-turn sessions
main.rs      → CLI entry point: --prompt, --chat, --bench
```

### Why "zero-copy"

A typical loader reads the GGUF file into a `Vec<u8>` (or several), then copies tensor data out of that buffer into the format the inference library expects — doubling (or more) the memory footprint of the model.

Inference Z1 instead:

1. `mmap`s the GGUF file directly (`MAP_PRIVATE`)
2. Wraps that mapping as a ggml CPU backend buffer (`ggml_backend_cpu_buffer_from_ptr`)
3. Creates ggml tensor descriptors whose `data` pointers point **directly into the mmap**

The OS page cache does the heavy lifting — pages are loaded from disk on first access and can be evicted under memory pressure without Inference Z1 ever having made its own copy.

### Why the persistent decode graph matters

Naively, each generated token requires: build a ggml compute graph for the whole model → allocate working memory for every intermediate tensor (`gallocr`) → run the graph → free everything → repeat. For a 32-layer model, that's a lot of repeated setup for a single token.

Inference Z1 builds the decode graph **once**, with the KV cache and a fixed-size attention mask wired in as persistent input tensors. Each subsequent token just updates three small tensors (the new token ID, its position, and the attention mask) and re-runs the same graph. This eliminated the dominant per-token overhead and was the single largest performance win in this project (0.13 → 1.6 tok/s).

---

## Known limitations

- **Llama 3.1 architecture only.** Other architectures (Mistral, etc.) will load but produce poor output — the tokenizer and RoPE/attention parameters are tuned for Llama 3.1's GQA layout.
- **512-token context window.** Chosen to fit comfortably in 8 GB RAM alongside the model weights. Increasing this is a one-line change (`KVCache::new(..., n_ctx)`) at the cost of more RAM for the cache.
- **CPU-only.** No GPU backend. This is by design — the goal is good performance on modest, GPU-less hardware.
- Single conversation thread — the KV cache holds one conversation at a time.

---

## License

[Choose and confirm — see note below]

This project vendors ggml/llama.cpp sources (MIT licensed) for the underlying tensor operations and C build. Inference Z1's own Rust code (loader, graph construction, KV cache, tokenizer, generation loop, CLI) is original work.

> **Note:** confirm your chosen license here before publishing, and ensure attribution to the vendored llama.cpp/ggml project (MIT) is correct and complete.

---

## Project context

Inference Z1 is the reference engine of **Zero Copies** — built incrementally, with every architectural decision measured against real numbers on real (modest) hardware. The philosophy: understand every layer, measure everything, and let the numbers tell the story.
