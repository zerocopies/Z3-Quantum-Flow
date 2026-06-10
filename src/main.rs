// main.rs — Z.1 inference engine entry point
//
// CLI modes:
//   z1 --prompt "your question"          — single-shot generation (B1 path)
//   z1 --chat                            — interactive multi-turn REPL (B1 path)
//   z1 --bench                           — benchmark harness (reports tok/s)
//   z1 --model <path>                    — override default model path
//
// The Option A (llama.cpp loader) path is retired. All inference now goes
// through the B1 zero-copy mmap path: gguf.rs → loader.rs → graph.rs →
// logits.rs + tokenizer.rs → generate.rs.

mod gguf;
mod ggml_ffi;
mod llama_ffi;
mod loader;
mod graph;
mod mapper;
mod logits;
mod tokenizer;
mod generate;
mod engine;    // kept for the benchmark harness; can be removed later
mod harness;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process;

use gguf::{GgufHeader, GgufValue};
use graph::ForwardPass;
use generate::{generate, GenerateConfig, Session};
use loader::MappedModel;
use tokenizer::Tokenizer;
use logits::SamplingConfig;

// ── Default model path ────────────────────────────────────────────────────────

const DEFAULT_MODEL: &str = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

// ── CLI argument parsing (no external crate) ──────────────────────────────────

#[derive(Debug, Default)]
struct Args {
    model_path: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    bench: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    no_chat_template: bool,
    help: bool,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                a.model_path = raw.get(i).map(PathBuf::from);
            }
            "--prompt" | "-p" => {
                i += 1;
                a.prompt = raw.get(i).cloned();
            }
            "--chat" | "-c"      => a.chat = true,
            "--bench" | "-b"     => a.bench = true,
            "--no-template"      => a.no_chat_template = true,
            "--help" | "-h"      => a.help = true,
            "--max-tokens" | "-n" => {
                i += 1;
                a.max_tokens = raw.get(i).and_then(|s| s.parse().ok());
            }
            "--temperature" | "-t" => {
                i += 1;
                a.temperature = raw.get(i).and_then(|s| s.parse().ok());
            }
            "--top-p" => {
                i += 1;
                a.top_p = raw.get(i).and_then(|s| s.parse().ok());
            }
            other => eprintln!("[Z.1] warning: unknown argument '{other}'"),
        }
        i += 1;
    }
    a
}

fn print_help() {
    println!(r#"
Z.1 — Local LLM inference engine (Llama 3.1 8B, zero-copy B1 path)

USAGE:
    z1 [OPTIONS]

OPTIONS:
    -m, --model <path>       Path to GGUF model file
                             [default: {DEFAULT_MODEL}]
    -p, --prompt <text>      Single-shot prompt (non-interactive)
    -c, --chat               Interactive multi-turn chat REPL
    -b, --bench              Run benchmark harness and print tok/s
    -n, --max-tokens <N>     Maximum tokens to generate  [default: 512]
    -t, --temperature <f>    Sampling temperature        [default: 0.7]
        --top-p <f>          Nucleus sampling threshold  [default: 0.9]
        --no-template        Skip Llama 3.1 chat template wrapping
    -h, --help               Print this help

EXAMPLES:
    z1 --prompt "Explain quantum entanglement in simple terms"
    z1 --chat
    z1 --bench
    z1 --model /data/models/llama-3.1-8b.gguf --chat
"#);
}

/// Load the GGUF file, build the tokenizer from vocab tables, prepare the
/// B1 forward pass. All weight data stays in the mmap — zero heap copies.
fn load_model(path: &PathBuf, args: &Args) -> Result<(GgufHeader, MappedModel, GenerateConfig), Box<dyn std::error::Error>> {
    eprint!("[Z.1] Loading model from {} … ", path.display());
    let _ = io::stderr().flush();

    let header = GgufHeader::from_file(path)?;
    let model  = MappedModel::load(path)?;
    eprintln!("OK");

    let mut sampling = SamplingConfig::default();
    if let Some(t) = args.temperature { sampling.temperature = t; }
    if let Some(p) = args.top_p      { sampling.top_p = p; }

    let cfg = GenerateConfig {
        max_new_tokens: args.max_tokens.unwrap_or(512),
        sampling,
        context_len: 4096,
        print_timing: true,
        add_bos: true,
        chat_template: !args.no_chat_template,
    };

    Ok((header, model, cfg))
}

/// Extract tokenizer vocab tables from a parsed GgufHeader.
fn build_tokenizer(header: &GgufHeader) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    eprint!("[Z.1] Building tokenizer … ");
    let _ = io::stderr().flush();

    // Extract vocab arrays from metadata
    let tokens: Vec<String> = extract_string_array(&header.metadata, "tokenizer.ggml.tokens")?;
    let scores: Vec<f32> = extract_f32_array(&header.metadata, "tokenizer.ggml.scores").unwrap_or_default();
    let types: Vec<u32> = extract_u32_array(&header.metadata, "tokenizer.ggml.token_type").unwrap_or_default();
    let merges: Vec<String> = extract_string_array(&header.metadata, "tokenizer.ggml.merges").unwrap_or_default();

    let tok = Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)?;
    eprintln!("OK ({} tokens, {} merges)", tok.vocab_size(), merges.len());
    Ok(tok)
}

/// Helper: extract string array from GGUF metadata
fn extract_string_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::String(s) = v {
                    result.push(s.clone());
                }
            }
            Ok(result)
        }
        None => Err(format!("missing {}", key).into()),
        _ => Err(format!("{} is not a string array", key).into()),
    }
}

/// Helper: extract f32 array from GGUF metadata
fn extract_f32_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::F32(f) = v {
                    result.push(*f);
                }
            }
            Ok(result)
        }
        None => Ok(Vec::new()),
        _ => Err(format!("{} is not an f32 array", key).into()),
    }
}

/// Helper: extract u32 array from GGUF metadata
fn extract_u32_array(meta: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    match meta.get(key) {
        Some(GgufValue::Array(arr)) => {
            let mut result = Vec::new();
            for v in arr {
                if let GgufValue::U32(u) = v {
                    result.push(*u);
                }
            }
            Ok(result)
        }
        None => Ok(Vec::new()),
        _ => Err(format!("{} is not a u32 array", key).into()),
    }
}

// ── Single-shot generation ────────────────────────────────────────────────────

fn run_prompt(
    prompt: &str,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) {
    match generate(prompt, fwd, model, tok, cfg) {
        Ok(_) => println!(),
        Err(e) => { eprintln!("\n[Z.1] generation error: {e}"); process::exit(1); }
    }
}

// ── Interactive REPL ──────────────────────────────────────────────────────────

fn run_chat(
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) {
    println!("[Z.1] Interactive chat — type your message, Enter to send, Ctrl-C to exit.\n");
    let stdin = io::stdin();
    let mut _session = Session::new(cfg.context_len);
    loop {
        print!("You: ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => { println!("\n[Z.1] Goodbye."); break; }
            Ok(_) => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if trimmed == "/reset" { _session = Session::new(cfg.context_len); eprintln!("[Z.1] Context reset."); continue; }
        if trimmed == "/quit" || trimmed == "/exit" { println!("[Z.1] Goodbye."); break; }
        print!("Z.1:  ");
        let _ = io::stdout().flush();
        match generate(trimmed, fwd, model, tok, cfg) {
            Ok(_) => println!(),
            Err(e) => eprintln!("\n[Z.1] error: {e}"),
        }
    }
}

// ── Benchmark harness ─────────────────────────────────────────────────────────

fn run_bench(
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) {
    const BENCH_PROMPT: &str = "Write a detailed explanation of how transformer attention works.";
    const BENCH_RUNS: usize = 3;
    eprintln!("[Z.1] Benchmark: {BENCH_RUNS} runs");
    let mut all_tps: Vec<f64> = Vec::new();
    for run in 1..=BENCH_RUNS {
        eprint!("  Run {run}/{BENCH_RUNS} … ");
        let _ = io::stderr().flush();
        let bench_cfg = GenerateConfig { max_new_tokens: 128, print_timing: false, ..cfg.clone() };
        match generate(BENCH_PROMPT, fwd, model, tok, &bench_cfg) {
            Ok(stats) => {
                let tps = stats.tokens_per_second();
                eprintln!("{tps:.2} tok/s ({} tokens in {:.0}ms)", stats.generated_tokens, stats.generate_ms);
                all_tps.push(tps);
            }
            Err(e) => eprintln!("ERROR: {e}"),
        }
    }
    if !all_tps.is_empty() {
        let mean = all_tps.iter().sum::<f64>() / all_tps.len() as f64;
        let min  = all_tps.iter().cloned().fold(f64::MAX, f64::min);
        let max  = all_tps.iter().cloned().fold(f64::MIN, f64::max);
        println!("\n[Z.1] Benchmark — Mean: {mean:.2} tok/s  Min: {min:.2}  Max: {max:.2}");
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    if args.help || (!args.chat && !args.bench && args.prompt.is_none()) {
        print_help();
        process::exit(0);
    }

    let model_path = args.model_path.clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));

    // ── Load GGUF header and MappedModel ──────────────────────────────────────
    let (header, model, cfg) = match load_model(&model_path, &args) {
        Ok(v) => v,
        Err(e) => { eprintln!("[Z.1] Failed to load model: {e}"); process::exit(1); }
    };

    // ── Build tokenizer from GGUF vocab tables ────────────────────────────────
    let tok = match build_tokenizer(&header) {
        Ok(t) => t,
        Err(e) => { eprintln!("[Z.1] Failed to build tokenizer: {e}"); process::exit(1); }
    };

    // ── Build B1 forward pass from mmap tensors ───────────────────────────────
    eprint!("[Z.1] Initialising B1 forward pass … ");
    let _ = io::stderr().flush();

    let mut fwd = match ForwardPass::new(&model) {
        Ok(f) => { eprintln!("OK"); f }
        Err(e) => { eprintln!("FAILED: {e}"); process::exit(1); }
    };

    let hidden_size = header.metadata.get("llama.embedding_length")
        .and_then(|v| v.as_u32()).unwrap_or(4096) as usize;
    let vocab_size = header.metadata.get("llama.vocab_size")
        .and_then(|v| v.as_u32())
        .unwrap_or_else(|| tok.vocab_size() as u32) as usize;

    eprintln!("[Z.1] Ready. hidden={hidden_size}, vocab={vocab_size}");

    // ── Dispatch ──────────────────────────────────────────────────────────────
    if args.bench {
        run_bench(&mut fwd, &model, &tok, &cfg);
    } else if args.chat {
        run_chat(&mut fwd, &model, &tok, &cfg);
    } else if let Some(ref prompt) = args.prompt {
        run_prompt(prompt, &mut fwd, &model, &tok, &cfg);
    }
}
