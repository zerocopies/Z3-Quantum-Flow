// generate.rs — autoregressive generation loop for the B1 inference path
//
// Two entry points:
//   generate()       — single-shot. Resets KV cache before and after.
//   generate_turn()  — multi-turn chat. Does NOT reset KV cache; each turn's
//                       tokens are appended via prefill() starting at the
//                       current kv.head, so conversation history persists
//                       across turns for free (the persistent decode graph
//                       already supports this).

use std::io::{self, Write};
use std::sync::OnceLock;

/// Returns true if Z1_TRACE=1 is set in the environment. Cached after first call.
fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("Z1_TRACE").map(|v| v == "1").unwrap_or(false))
}
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
    ContextFull { used: i64, max: i64 },
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
            Self::ContextFull { used, max } =>
                write!(f, "conversation context full ({used}/{max} tokens) — use /reset to start a new conversation"),
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

// ── Core generation loop (shared by both entry points) ────────────────────────
//
// Runs prefill on `turn_ids`, then decodes until EOS or max_new_tokens.
// Does NOT reset the KV cache — caller decides whether to reset.

fn run_generation(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<GenerateStats, GenerateError> {
    let (stats, _text) = run_generation_inner(turn_ids, fwd, model, tok, cfg, false)?;
    Ok(stats)
}

/// Like `run_generation`, but also returns the decoded text of the generated
/// tokens and suppresses `[Z.1 DEBUG]` lines + streaming to stdout. Used by
/// the bench/dev-harness for correctness checks (e.g. "does it say Paris?").
pub fn run_generation_captured(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<(GenerateStats, String), GenerateError> {
    run_generation_inner(turn_ids, fwd, model, tok, cfg, true)
}

fn run_generation_inner(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
    quiet: bool,
) -> Result<(GenerateStats, String), GenerateError> {

    if turn_ids.is_empty() { return Err(GenerateError::EmptyPrompt); }

    // ── Capacity check against the actual KV cache, not cfg.context_len ────────
    let n_ctx  = fwd.kv.n_ctx;
    let used   = fwd.kv.head;
    let needed = turn_ids.len() as i64;
    if used + needed > n_ctx {
        return Err(GenerateError::ContextFull { used: used + needed, max: n_ctx });
    }

    let prompt_t0 = Instant::now();

    if !quiet && trace_enabled() {
        eprint!("[Z.1 DEBUG] turn tokens ({}): ", turn_ids.len());
        for id in turn_ids.iter() { eprint!("{} ", id); }
        eprintln!();
    }
    let prompt_token_count = turn_ids.len();

    // Prefill appends at fwd.kv.head — existing history is preserved.
    let mut logits = fwd.prefill(turn_ids, model)?;
    let prompt_ms = prompt_t0.elapsed().as_secs_f64() * 1000.0;

    let mut rng = rng_seed_from_time();
    let mut recent_tokens: Vec<u32> = Vec::new();
    let mut next_token = sample_token(&mut logits, &cfg.sampling, &recent_tokens, &mut rng)?;
    if !quiet && trace_enabled() {
        eprintln!("[Z.1 DEBUG] first token id: {} decode: {:?}", next_token, tok.decode_one(next_token));
    }

    let gen_t0 = Instant::now();
    let mut generated = 0usize;
    let stdout = io::stdout();
    let mut text = String::new();
    if !quiet { println!(); }

    loop {
        if tok.is_eos(next_token) { break; }
        if generated >= cfg.max_new_tokens { break; }
        if fwd.kv.head >= n_ctx { break; } // cache full — stop gracefully

        if let Some(piece) = tok.decode_one(next_token) {
            if quiet {
                text.push_str(&piece);
            } else {
                let mut handle = stdout.lock();
                handle.write_all(piece.as_bytes())?;
                handle.flush()?;
            }
        }

        recent_tokens.push(next_token);
        if recent_tokens.len() > 64 { recent_tokens.remove(0); }

        logits = fwd.decode_one(next_token, model)?;
        generated += 1;

        next_token = sample_token(&mut logits, &cfg.sampling, &recent_tokens, &mut rng)?;
    }

    let generate_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
    let stats = GenerateStats { prompt_tokens: prompt_token_count, generated_tokens: generated,
        prompt_ms, generate_ms };

    if cfg.print_timing && !quiet { eprintln!("{stats}"); }
    Ok((stats, text))
}

// ── Single-shot generation ──────────────────────────────────────────────────
//
// Resets the KV cache before AND after, so each call starts from a clean
// context. Used by --prompt and --bench.

pub fn generate(
    prompt: &str,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<GenerateStats, GenerateError> {

    if prompt.trim().is_empty() { return Err(GenerateError::EmptyPrompt); }

    // Ensure a clean slate regardless of prior state
    fwd.reset_kv();

    let prompt_ids: Vec<u32> = if cfg.chat_template {
        build_chat_tokens(prompt, tok)
    } else {
        tok.encode(prompt, cfg.add_bos)
    };
    if prompt_ids.is_empty() { return Err(GenerateError::EmptyPrompt); }
    if prompt_ids.len() >= cfg.context_len {
        return Err(GenerateError::ContextLengthExceeded { max: cfg.context_len });
    }

    let stats = run_generation(&prompt_ids, fwd, model, tok, cfg)?;

    // Reset after so the next single-shot call also starts clean
    fwd.reset_kv();
    Ok(stats)
}

// ── Multi-turn chat generation ────────────────────────────────────────────────
//
// `turn_number` is 0 for the first message in a conversation, 1+ for
// follow-ups. The first turn includes BOS + system prompt; follow-ups are
// just the user message + assistant header, appended to the existing cache.
//
// Does NOT reset the KV cache — conversation history persists until /reset
// (caller calls fwd.reset_kv() explicitly) or the cache fills up
// (ContextFull error).

pub fn generate_turn(
    user_message: &str,
    turn_number: usize,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<GenerateStats, GenerateError> {

    if user_message.trim().is_empty() { return Err(GenerateError::EmptyPrompt); }

    let turn_ids: Vec<u32> = if turn_number == 0 {
        build_chat_tokens(user_message, tok)
    } else {
        build_followup_chat_tokens(user_message, tok)
    };

    run_generation(&turn_ids, fwd, model, tok, cfg)
}

// ── Llama 3.1 chat template ───────────────────────────────────────────────────

// Special token IDs for Llama 3.1
const T_BOS:          u32 = 128_000; // <|begin_of_text|>
const T_START_HEADER: u32 = 128_006; // <|start_header_id|>
const T_END_HEADER:   u32 = 128_007; // <|end_header_id|>
const T_EOT:          u32 = 128_009; // <|eot_id|>
const T_NEWLINES:     u32 = 271;     // "\n\n"

/// Build the Llama 3.1 instruct token sequence for the FIRST turn of a
/// conversation: BOS + system prompt + user message + assistant header.
pub fn build_chat_tokens(user_message: &str, tok: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();

    ids.push(T_BOS);

    // System turn
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

    // Assistant header — model generates the response after this
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("assistant"));
    ids.push(T_END_HEADER);
    ids.push(T_NEWLINES);

    ids
}

/// Build the token sequence for a FOLLOW-UP turn: just the new user message
/// wrapped in user/assistant headers, with NO BOS and NO system prompt.
/// Appended to the existing KV cache, which already holds everything before
/// this point (including the model's previous EOT from its last reply).
pub fn build_followup_chat_tokens(user_message: &str, tok: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();

    // User turn
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("user"));
    ids.push(T_END_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos(&format!("\n\n{user_message}")));
    ids.push(T_EOT);

    // Assistant header
    ids.push(T_START_HEADER);
    ids.extend_from_slice(&tok.encode_no_bos("assistant"));
    ids.push(T_END_HEADER);
    ids.push(T_NEWLINES);

    ids
}

pub fn llama3_chat_template(user_message: &str) -> String {
    // Kept for compatibility but build_chat_tokens is preferred
    user_message.to_string()
}

// ── Session (multi-turn) ──────────────────────────────────────────────────────
//
// Tracks turn count so generate_turn() knows whether to include BOS+system.
// The KV cache itself (in ForwardPass) holds the actual conversation state.

pub struct Session {
    pub turn_count:  usize,
    pub context_len: usize,
}

impl Session {
    pub fn new(context_len: usize) -> Self { Self { turn_count: 0, context_len } }
    pub fn is_empty(&self) -> bool { self.turn_count == 0 }
    pub fn record_turn(&mut self) { self.turn_count += 1; }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_format() {
        let out = llama3_chat_template("Hello");
        assert!(out.contains("Hello"));
    }

    #[test]
    fn generate_stats_display() {
        let stats = GenerateStats { prompt_tokens: 20, generated_tokens: 100,
            prompt_ms: 500.0, generate_ms: 71_000.0 };
        assert!((stats.tokens_per_second() - 100.0 / 71.0).abs() < 0.1);
    }

    #[test]
    fn session_turn_tracking() {
        let mut s = Session::new(512);
        assert!(s.is_empty());
        s.record_turn();
        assert!(!s.is_empty());
        assert_eq!(s.turn_count, 1);
    }
}
