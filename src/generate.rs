// generate.rs — autoregressive generation loop for the B1 inference path

use std::io::{self, Write};
use std::time::Instant;

use crate::graph::{ForwardPass, ForwardError};
use crate::loader::MappedModel;
use crate::logits::{sample_token, SamplingConfig, rng_seed_from_time, LogitError};
use crate::tokenizer::{Tokenizer, TOKEN_EOS};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GenerateError {
    Forward(ForwardError),
    Logit(LogitError),
    Io(io::Error),
    EmptyPrompt,
    ContextLengthExceeded { max: usize },
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward(e)  => write!(f, "forward pass error: {e}"),
            Self::Logit(e)    => write!(f, "logit error: {e}"),
            Self::Io(e)       => write!(f, "I/O error: {e}"),
            Self::EmptyPrompt => write!(f, "prompt is empty"),
            Self::ContextLengthExceeded { max } =>
                write!(f, "context length exceeded (max {max} tokens)"),
        }
    }
}

impl std::error::Error for GenerateError {}
impl From<ForwardError> for GenerateError { fn from(e: ForwardError) -> Self { Self::Forward(e) } }
impl From<LogitError>   for GenerateError { fn from(e: LogitError)   -> Self { Self::Logit(e) } }
impl From<io::Error>    for GenerateError { fn from(e: io::Error)    -> Self { Self::Io(e) } }

// ── Generation config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub context_len: usize,
    pub print_timing: bool,
    pub add_bos: bool,
    pub chat_template: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            sampling: SamplingConfig::default(),
            context_len: 4096,
            print_timing: true,
            add_bos: true,
            chat_template: true,
        }
    }
}

// ── Generation statistics ─────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_ms: f64,
    pub generate_ms: f64,
}

impl GenerateStats {
    pub fn tokens_per_second(&self) -> f64 {
        if self.generate_ms < 1.0 { return 0.0; }
        self.generated_tokens as f64 / (self.generate_ms / 1000.0)
    }
}

impl std::fmt::Display for GenerateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n\n[Z.1] prompt: {} tokens ({:.0}ms) | generated: {} tokens ({:.0}ms, {:.2} tok/s)",
            self.prompt_tokens, self.prompt_ms,
            self.generated_tokens, self.generate_ms,
            self.tokens_per_second(),
        )
    }
}

// ── Core generation function ──────────────────────────────────────────────────

pub fn generate(
    prompt: &str,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<GenerateStats, GenerateError> {

    if prompt.trim().is_empty() { return Err(GenerateError::EmptyPrompt); }

    // Tokenise — build token sequence directly for correct special token IDs
    let prompt_t0 = Instant::now();
    let prompt_ids: Vec<u32> = if cfg.chat_template {
        build_chat_tokens(prompt, tok)
    } else {
        tok.encode(prompt, cfg.add_bos)
    };
    if prompt_ids.is_empty() { return Err(GenerateError::EmptyPrompt); }
    if prompt_ids.len() >= cfg.context_len {
        return Err(GenerateError::ContextLengthExceeded { max: cfg.context_len });
    }

    // DEBUG: print all token IDs to stderr
    eprint!("[Z.1 DEBUG] prompt tokens ({}): ", prompt_ids.len());
    for id in prompt_ids.iter() { eprint!("{} ", id); }
    eprintln!();
    let prompt_token_count = prompt_ids.len();

    // Prefill — runs full prompt, writes K/V into cache, returns logits for last position
    let mut logits = fwd.prefill(&prompt_ids, model)?;
    let prompt_ms = prompt_t0.elapsed().as_secs_f64() * 1000.0;

    // Prefill wrote all prompt K/V into the cache. Do NOT reset here —
    // kv.head now points to the next free slot (= prompt_len).

    // Sample first token — recent_tokens starts empty (repeat penalty is for generation only)
    let mut rng = rng_seed_from_time();
    let mut recent_tokens: Vec<u32> = Vec::new();
    let mut next_token = sample_token(&mut logits, &cfg.sampling, &recent_tokens, &mut rng)?;
    eprintln!("[Z.1 DEBUG] first token id: {} decode: {:?}", next_token, tok.decode_one(next_token));

    // Autoregressive loop — decode_one() passes only the new token each step.
    // The KV cache in graph.rs holds all past K/V, so attention over the full
    // history is O(1) per step instead of O(n).
    let gen_t0 = Instant::now();
    let mut generated = 0usize;
    let stdout = io::stdout();
    println!();

    loop {
        if tok.is_eos(next_token) { break; }
        if generated >= cfg.max_new_tokens { break; }

        // Stream token to stdout immediately
        if let Some(piece) = tok.decode_one(next_token) {
            let mut handle = stdout.lock();
            handle.write_all(piece.as_bytes())?;
            handle.flush()?;
        }

        // Update repeat-penalty window (last 64 tokens)
        recent_tokens.push(next_token);
        if recent_tokens.len() > 64 { recent_tokens.remove(0); }

        // Single-token decode — cache supplies all past context
        logits = fwd.decode_one(next_token, model)?;
        generated += 1;

        next_token = sample_token(&mut logits, &cfg.sampling, &recent_tokens, &mut rng)?;
    }

    // Reset cache after generation so the next call starts clean
    fwd.reset_kv();

    let generate_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
    let stats = GenerateStats { prompt_tokens: prompt_token_count, generated_tokens: generated,
        prompt_ms, generate_ms };

    if cfg.print_timing { eprintln!("{stats}"); }
    Ok(stats)
}

// ── Llama 3.1 chat template ───────────────────────────────────────────────────

// Special token IDs for Llama 3.1
const T_BOS:              u32 = 128_000; // <|begin_of_text|>
const T_START_HEADER:     u32 = 128_006; // <|start_header_id|>
const T_END_HEADER:       u32 = 128_007; // <|end_header_id|>
const T_EOT:              u32 = 128_009; // <|eot_id|>

/// Build the Llama 3.1 instruct token sequence directly from token IDs,
/// bypassing text encoding for special tokens.
///
/// Format:
///   BOS system_header \n\n {system} EOT user_header \n\n {user} EOT assistant_header \n\n
pub fn build_chat_tokens(user_message: &str, tok: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();

    // BOS
    ids.push(T_BOS);

    // System turn — role names use their vocab token IDs directly
    // In Llama 3.1 vocab these are stored as Ġsystem, Ġuser, Ġassistant (with leading space)
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("system"));
    ids.push(T_END_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("\n\nYou are a helpful AI assistant."));
    ids.push(T_EOT);

    // User turn
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("user"));
    ids.push(T_END_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos(&format!("\n\n{user_message}")));
    ids.push(T_EOT);

    // Assistant header (model generates the response after this)
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("assistant"));
    ids.push(T_END_HEADER);
    ids.push(271);   // "\n\n"

    ids
}

pub fn llama3_chat_template(user_message: &str) -> String {
    // Kept for compatibility but build_chat_tokens is preferred
    user_message.to_string()
}

// ── Session (multi-turn) ──────────────────────────────────────────────────────

pub struct Session {
    pub history: Vec<u32>,
    pub context_len: usize,
}

impl Session {
    pub fn new(context_len: usize) -> Self { Self { history: Vec::new(), context_len } }
    pub fn is_empty(&self) -> bool { self.history.is_empty() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_format() {
        let out = llama3_chat_template("Hello");
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("Hello"));
        assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn generate_stats_display() {
        let stats = GenerateStats { prompt_tokens: 20, generated_tokens: 100,
            prompt_ms: 500.0, generate_ms: 71_000.0 };
        assert!((stats.tokens_per_second() - 100.0 / 71.0).abs() < 0.1);
    }
}
