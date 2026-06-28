// generate.rs — autoregressive generation loop
//
// Multi-architecture chat template support:
//   Llama3 — uses token IDs 128000/128006/128007/128009
//   Phi3   — uses text tokens <|system|>, <|user|>, <|assistant|>, <|end|>
//   Qwen2  — uses text tokens <|im_start|>, <|im_end|>
//   Raw    — BOS + raw text, no template (fallback)

use std::io::{self, Write};
use std::sync::OnceLock;
use std::time::Instant;

use crate::graph::{ForwardPass, ForwardError};
use crate::loader::MappedModel;
use crate::logits::{sample_token, SamplingConfig, rng_seed_from_time, LogitError};
use crate::tokenizer::Tokenizer;

fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("Z1_TRACE").map(|v| v == "1").unwrap_or(false))
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum GenerateError {
    #[error("Forward pass error: {0}")]
    Forward(#[from] ForwardError),
    #[error("Logit error: {0}")]
    Logit(#[from] LogitError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Prompt is empty")]
    EmptyPrompt,
    #[error("Context length exceeded (max {max} tokens)")]
    ContextLengthExceeded { max: usize },
    #[error("Conversation context full ({used}/{max} tokens)")]
    ContextFull { used: i64, max: i64 },
}

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
            max_new_tokens: 256,
            sampling: SamplingConfig::default(),
            context_len: 512,
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

// ── Chat template ─────────────────────────────────────────────────────────────

/// Arch-aware chat template builder.
/// Returns (system_tokens, user_turn_tokens, eos_token_id)
struct ChatTemplate {
    arch: String,
}

impl ChatTemplate {
    fn new(arch: &str) -> Self {
        Self { arch: arch.to_string() }
    }

    // Llama 3.1 special token IDs
    const LLAMA_BOS:          u32 = 128_000;
    const LLAMA_START_HEADER: u32 = 128_006;
    const LLAMA_END_HEADER:   u32 = 128_007;
    const LLAMA_EOT:          u32 = 128_009;
    const LLAMA_NEWLINES:     u32 = 271;

    fn build_system_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        match self.arch.as_str() {
            "llama" => {
                let mut sys = vec![Self::LLAMA_BOS, Self::LLAMA_START_HEADER];
                sys.extend_from_slice(&tok.encode_no_bos("system"));
                sys.push(Self::LLAMA_END_HEADER);
                sys.push(Self::LLAMA_NEWLINES);
                sys.extend_from_slice(&tok.encode_no_bos("You are a highly capable AI assistant."));
                sys.push(Self::LLAMA_EOT);
                sys
            }
            "phi3" => {
                // Phi-3 template: <|system|>\n{content}<|end|>\n
                let mut sys = Vec::new();
                sys.push(1u32); // BOS <s>
                sys.extend_from_slice(&tok.encode_no_bos("<|system|>\n"));
                sys.extend_from_slice(&tok.encode_no_bos("You are a helpful AI assistant."));
                sys.extend_from_slice(&tok.encode_no_bos("<|end|>\n"));
                sys
            }
            "qwen2" | "qwen" => {
                // Qwen2 template: <|im_start|>system\n{content}<|im_end|>\n
                let mut sys = Vec::new();
                sys.extend_from_slice(&tok.encode_no_bos("<|im_start|>system\n"));
                sys.extend_from_slice(&tok.encode_no_bos("You are Z3, a helpful AI assistant built by Zero Copies. You are running on Z3-Quantum-Flow, a custom inference engine."));
                sys.extend_from_slice(&tok.encode_no_bos("<|im_end|>\n"));
                sys
            }
            _ => {
                // Raw: just BOS
                vec![1u32]
            }
        }
    }

    fn build_user_turn(&self, tok: &Tokenizer, user_message: &str) -> Vec<u32> {
        match self.arch.as_str() {
            "llama" => {
                let mut turn = vec![Self::LLAMA_START_HEADER];
                turn.extend_from_slice(&tok.encode_no_bos("user"));
                turn.push(Self::LLAMA_END_HEADER);
                turn.push(Self::LLAMA_NEWLINES);
                turn.extend_from_slice(&tok.encode_no_bos(user_message));
                turn.push(Self::LLAMA_EOT);
                turn.push(Self::LLAMA_START_HEADER);
                turn.extend_from_slice(&tok.encode_no_bos("assistant"));
                turn.push(Self::LLAMA_END_HEADER);
                turn.push(Self::LLAMA_NEWLINES);
                turn
            }
            "phi3" => {
                // <|user|>\n{msg}<|end|>\n<|assistant|>\n
                let mut turn = Vec::new();
                turn.extend_from_slice(&tok.encode_no_bos("<|user|>\n"));
                turn.extend_from_slice(&tok.encode_no_bos(user_message));
                turn.extend_from_slice(&tok.encode_no_bos("<|end|>\n"));
                turn.extend_from_slice(&tok.encode_no_bos("<|assistant|>\n"));
                turn
            }
            "qwen2" | "qwen" => {
                // <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n
                let mut turn = Vec::new();
                turn.extend_from_slice(&tok.encode_no_bos("<|im_start|>user\n"));
                turn.extend_from_slice(&tok.encode_no_bos(user_message));
                turn.extend_from_slice(&tok.encode_no_bos("<|im_end|>\n"));
                turn.extend_from_slice(&tok.encode_no_bos("<|im_start|>assistant\n"));
                turn
            }
            _ => {
                // Raw: just tokenize the message
                tok.encode_no_bos(user_message)
            }
        }
    }

    fn eot_token(&self, tok: &Tokenizer) -> u32 {
        match self.arch.as_str() {
            "llama" => Self::LLAMA_EOT,
            "phi3"  => {
                // <|end|> token — encode and take first token
                let ids = tok.encode_no_bos("<|end|>");
                ids.first().copied().unwrap_or(32007)
            }
            "qwen2" | "qwen" => {
                let ids = tok.encode_no_bos("<|im_end|>");
                ids.first().copied().unwrap_or(151645)
            }
            _ => 2, // generic EOS
        }
    }
}

// ── Core generation loop ──────────────────────────────────────────────────────

fn run_generation(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
    eot: u32,
) -> Result<(GenerateStats, Vec<u32>), GenerateError> {
    let (stats, _text, generated_ids) = run_generation_inner(turn_ids, fwd, model, tok, cfg, false, eot)?;
    Ok((stats, generated_ids))
}

pub fn run_generation_captured(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<(GenerateStats, String, Vec<u32>), GenerateError> {
    // For captured mode we use a generic EOS — caller should use generate_turn_captured instead
    run_generation_inner(turn_ids, fwd, model, tok, cfg, true, 2)
}

fn run_generation_inner(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
    quiet: bool,
    eot: u32,
) -> Result<(GenerateStats, String, Vec<u32>), GenerateError> {

    if turn_ids.is_empty() { return Err(GenerateError::EmptyPrompt); }

    let n_ctx  = fwd.kv.n_ctx;
    let used   = fwd.kv.head;
    let needed = turn_ids.len() as i64;

    if used + needed > n_ctx {
        return Err(GenerateError::ContextFull { used: used + needed, max: n_ctx });
    }

    let prompt_t0 = Instant::now();
    let prompt_token_count = turn_ids.len();

    let mut logits = fwd.prefill(turn_ids, model)?;
    let prompt_ms = prompt_t0.elapsed().as_secs_f64() * 1000.0;

    let mut rng = rng_seed_from_time();
    let mut recent_tokens: Vec<u32> = Vec::new();
    let mut generated_ids: Vec<u32> = Vec::new();

    let mut next_token = sample_token(&mut logits, &cfg.sampling, &recent_tokens, &mut rng)?;

    let gen_t0 = Instant::now();
    let mut generated = 0usize;
    let stdout = io::stdout();
    let mut text = String::new();

    if !quiet { println!(); }

    loop {
        if tok.is_eos(next_token) || next_token == eot { break; }
        if generated >= cfg.max_new_tokens { break; }
        if fwd.kv.head >= n_ctx { break; }

        generated_ids.push(next_token);

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
    let stats = GenerateStats {
        prompt_tokens: prompt_token_count,
        generated_tokens: generated,
        prompt_ms,
        generate_ms,
    };

    if cfg.print_timing && !quiet { eprintln!("{stats}"); }
    Ok((stats, text, generated_ids))
}

// ── Session (Sliding Window Manager) ──────────────────────────────────────────

pub struct Session {
    pub turn_count: usize,
    pub context_len: usize,
    pub system_tokens: Vec<u32>,
    pub history_tokens: Vec<u32>,
    arch: String,
    eot: u32,
}

impl Session {
    /// arch — pass fwd.dna().arch.as_str() from main.rs
    pub fn new(context_len: usize, tok: &Tokenizer, arch: &str) -> Self {
        let tmpl = ChatTemplate::new(arch);
        let system_tokens = tmpl.build_system_tokens(tok);
        let eot = tmpl.eot_token(tok);

        log::info!("[Z.3] Chat template: arch={} system_tokens={} eot={}",
            arch, system_tokens.len(), eot);

        Self {
            turn_count: 0,
            context_len,
            system_tokens,
            history_tokens: Vec::new(),
            arch: arch.to_string(),
            eot,
        }
    }

    pub fn is_empty(&self) -> bool { self.turn_count == 0 }

    pub fn reset(&mut self) {
        self.turn_count = 0;
        self.history_tokens.clear();
    }
}

// ── Multi-turn chat generation (With Sliding Window) ──────────────────────────

pub fn generate_turn(
    user_message: &str,
    session: &mut Session,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<GenerateStats, GenerateError> {

    if user_message.trim().is_empty() { return Err(GenerateError::EmptyPrompt); }

    let tmpl = ChatTemplate::new(&session.arch);
    let new_turn = tmpl.build_user_turn(tok, user_message);
    let eot = session.eot;

    let needed_space = new_turn.len() + cfg.max_new_tokens;
    let available_space = session.context_len.saturating_sub(session.system_tokens.len());

    if needed_space > available_space {
        return Err(GenerateError::ContextLengthExceeded { max: session.context_len });
    }

    let mut requires_reprefill = false;

    while session.system_tokens.len() + session.history_tokens.len() + needed_space > session.context_len {
        let drop_amount = 128.min(session.history_tokens.len());
        session.history_tokens.drain(0..drop_amount);
        requires_reprefill = true;
    }

    session.history_tokens.extend_from_slice(&new_turn);

    let mut tokens_to_process = Vec::new();

    if requires_reprefill || session.turn_count == 0 {
        fwd.reset_kv();
        tokens_to_process.extend_from_slice(&session.system_tokens);
        tokens_to_process.extend_from_slice(&session.history_tokens);
        if trace_enabled() { eprintln!("[Z.1] Sliding window activated. Re-prefilling context."); }
    } else {
        tokens_to_process.extend_from_slice(&new_turn);
    }

    session.turn_count += 1;

    let (stats, generated_ids) = run_generation(&tokens_to_process, fwd, model, tok, cfg, eot)?;

    session.history_tokens.extend_from_slice(&generated_ids);
    session.history_tokens.push(eot);

    Ok(stats)
}

// ── Captured variant for Desktop UI environments ──────────────────────────────

pub fn generate_turn_captured(
    user_message: &str,
    session: &mut Session,
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<(GenerateStats, String), GenerateError> {
    if user_message.trim().is_empty() { return Err(GenerateError::EmptyPrompt); }

    let tmpl = ChatTemplate::new(&session.arch);
    let new_turn = tmpl.build_user_turn(tok, user_message);
    let eot = session.eot;

    let needed_space = new_turn.len() + cfg.max_new_tokens;
    let available_space = session.context_len.saturating_sub(session.system_tokens.len());

    if needed_space > available_space {
        return Err(GenerateError::ContextLengthExceeded { max: session.context_len });
    }

    let mut requires_reprefill = false;

    while session.system_tokens.len() + session.history_tokens.len() + needed_space > session.context_len {
        let drop_amount = 128.min(session.history_tokens.len());
        session.history_tokens.drain(0..drop_amount);
        requires_reprefill = true;
    }

    session.history_tokens.extend_from_slice(&new_turn);

    let mut tokens_to_process = Vec::new();
    if requires_reprefill || session.turn_count == 0 {
        fwd.reset_kv();
        tokens_to_process.extend_from_slice(&session.system_tokens);
        tokens_to_process.extend_from_slice(&session.history_tokens);
    } else {
        tokens_to_process.extend_from_slice(&new_turn);
    }

    session.turn_count += 1;

    let (stats, text, generated_ids) = run_generation_inner(
        &tokens_to_process, fwd, model, tok, cfg, true, eot)?;

    session.history_tokens.extend_from_slice(&generated_ids);
    session.history_tokens.push(eot);

    Ok((stats, text))
}
