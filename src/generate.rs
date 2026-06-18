// generate.rs — autoregressive generation loop for the B1 inference path
//
// Features a Sliding-Window KV Cache manager. When the conversation exceeds
// the context limit, it drops the oldest conversation turns while preserving 
// the system prompt, resets the KV cache, and seamlessly re-prefills.

use std::io::{self, Write};
use std::sync::OnceLock;
use std::time::Instant;

use crate::graph::{ForwardPass, ForwardError};
use crate::loader::MappedModel;
use crate::logits::{sample_token, SamplingConfig, rng_seed_from_time, LogitError};
use crate::tokenizer::Tokenizer;

/// Returns true if Z1_TRACE=1 is set in the environment. Cached after first call.
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

// ── Core generation loop ──────────────────────────────────────────────────────

fn run_generation(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<(GenerateStats, Vec<u32>), GenerateError> {
    let (stats, _text, generated_ids) = run_generation_inner(turn_ids, fwd, model, tok, cfg, false)?;
    Ok((stats, generated_ids))
}

pub fn run_generation_captured(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
) -> Result<(GenerateStats, String, Vec<u32>), GenerateError> {
    run_generation_inner(turn_ids, fwd, model, tok, cfg, true)
}

fn run_generation_inner(
    turn_ids: &[u32],
    fwd: &mut ForwardPass,
    model: &MappedModel,
    tok: &Tokenizer,
    cfg: &GenerateConfig,
    quiet: bool,
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
        if tok.is_eos(next_token) { break; }
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
    let stats = GenerateStats { prompt_tokens: prompt_token_count, generated_tokens: generated, prompt_ms, generate_ms };

    if cfg.print_timing && !quiet { eprintln!("{stats}"); }
    Ok((stats, text, generated_ids))
}

// ── Llama 3.1 chat template ───────────────────────────────────────────────────

const T_BOS:          u32 = 128_000; // <|begin_of_text|>
const T_START_HEADER: u32 = 128_006; // <|start_header_id|>
const T_END_HEADER:   u32 = 128_007; // <|end_header_id|>
const T_EOT:          u32 = 128_009; // <|eot_id|>
const T_NEWLINES:     u32 = 271;     // "\n\n"

// ── Session (Sliding Window Manager) ──────────────────────────────────────────

pub struct Session {
    pub turn_count: usize,
    pub context_len: usize,
    pub system_tokens: Vec<u32>,
    pub history_tokens: Vec<u32>,
}

impl Session {
    pub fn new(context_len: usize, tok: &Tokenizer) -> Self {
        let mut sys = vec![T_BOS, T_START_HEADER];
        sys.extend_from_slice(&tok.encode_no_bos("system"));
        sys.push(T_END_HEADER);
        sys.push(T_NEWLINES); // FIX: Manually push the newline token boundary
        sys.extend_from_slice(&tok.encode_no_bos("You are a highly capable AI assistant."));
        sys.push(T_EOT);

        Self {
            turn_count: 0,
            context_len,
            system_tokens: sys,
            history_tokens: Vec::new(),
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

    // 1. Format the new user message
    let mut new_turn = vec![T_START_HEADER];
    new_turn.extend_from_slice(&tok.encode_no_bos("user"));
    new_turn.push(T_END_HEADER);
    new_turn.push(T_NEWLINES); // FIX: Manually push the newline token boundary
    new_turn.extend_from_slice(&tok.encode_no_bos(user_message));
    new_turn.push(T_EOT);

    // Add assistant prompt header
    new_turn.push(T_START_HEADER);
    new_turn.extend_from_slice(&tok.encode_no_bos("assistant"));
    new_turn.push(T_END_HEADER);
    new_turn.push(T_NEWLINES);

    // 2. Sliding Window Logic: Check if we are going to exceed RAM
    let needed_space = new_turn.len() + cfg.max_new_tokens;
    let available_space = session.context_len.saturating_sub(session.system_tokens.len());

    if needed_space > available_space {
        return Err(GenerateError::ContextLengthExceeded { max: session.context_len });
    }

    let mut requires_reprefill = false;

    // Slide the window: Drop oldest tokens until we have enough space
    while session.system_tokens.len() + session.history_tokens.len() + needed_space > session.context_len {
        let drop_amount = 128.min(session.history_tokens.len());
        session.history_tokens.drain(0..drop_amount);
        requires_reprefill = true;
    }

    // 3. Update session memory with the new user message
    session.history_tokens.extend_from_slice(&new_turn);

    // 4. Inject to KV Cache
    let mut tokens_to_process = Vec::new();
    
    if requires_reprefill || session.turn_count == 0 {
        // We had to drop old memory (or it's turn 1), so wipe the KV cache and reload
        fwd.reset_kv();
        tokens_to_process.extend_from_slice(&session.system_tokens);
        tokens_to_process.extend_from_slice(&session.history_tokens);
        if trace_enabled() { eprintln!("[Z.1] Sliding window activated. Re-prefilling context."); }
    } else {
        // No memory dropped! We can seamlessly append this turn to the existing KV cache.
        tokens_to_process.extend_from_slice(&new_turn);
    }

    session.turn_count += 1;

    // 5. Run Generation
    let (stats, generated_ids) = run_generation(&tokens_to_process, fwd, model, tok, cfg)?;

    // 6. Save the AI's generated response into our session history so it remembers it next time
    session.history_tokens.extend_from_slice(&generated_ids);
    session.history_tokens.push(T_EOT);

    Ok(stats)
}
