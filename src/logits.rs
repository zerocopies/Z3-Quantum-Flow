// logits.rs — output norm and sampling for the B1 inference path
//
// Responsibilities:
//   1. Apply final RMS-norm to the last hidden state (the step skipped in the
//      raw forward pass that caused the scaling bug).
//   2. Multiply by the lm_head weight matrix to project to vocab logits.
//   3. Expose temperature scaling + top-p (nucleus) sampling so generate.rs
//      can draw the next token without pulling in any external dependency.
//
// All maths run on the CPU in plain Rust — no ggml calls here, keeping the
// output stage fully auditable and dependency-free.

use std::fmt;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LogitError {
    ShapeMismatch { expected: usize, got: usize },
    EmptyLogits,
    InvalidTemperature(f32),
    InvalidTopP(f32),
}

impl fmt::Display for LogitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } =>
                write!(f, "logit shape mismatch: expected {expected}, got {got}"),
            Self::EmptyLogits =>
                write!(f, "logit slice is empty"),
            Self::InvalidTemperature(t) =>
                write!(f, "temperature must be > 0.0, got {t}"),
            Self::InvalidTopP(p) =>
                write!(f, "top_p must be in (0.0, 1.0], got {p}"),
        }
    }
}

impl std::error::Error for LogitError {}

// ── RMS-norm ──────────────────────────────────────────────────────────────────

/// Apply RMS-norm in place: x = x / rms(x) * weight
///
/// This is the step that was missing from the raw B1 forward pass output,
/// causing logit values to be unscaled and distribution-distorting.
///
/// `hidden`  — mutable slice of the final hidden state  [hidden_size]
/// `weight`  — the learned norm weight vector            [hidden_size]
/// `eps`     — small constant for numerical stability (typically 1e-5)
pub fn rms_norm_inplace(hidden: &mut [f32], weight: &[f32], eps: f32) -> Result<(), LogitError> {
    let n = hidden.len();
    if weight.len() != n {
        return Err(LogitError::ShapeMismatch { expected: n, got: weight.len() });
    }
    if n == 0 {
        return Err(LogitError::EmptyLogits);
    }

    // rms = sqrt( mean(x^2) + eps )
    let mean_sq: f32 = hidden.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let rms_inv = 1.0 / (mean_sq + eps).sqrt();

    for (h, w) in hidden.iter_mut().zip(weight.iter()) {
        *h = *h * rms_inv * w;
    }
    Ok(())
}

// ── lm_head projection ────────────────────────────────────────────────────────

/// Project the final hidden state to vocab logits via the unembedding matrix.
///
/// Llama 3.1 ties the lm_head weight to the token embedding table, so
/// `lm_head` is the same pointer as `embed_tokens` — shape [vocab_size × hidden_size].
///
/// Returns a freshly allocated Vec<f32> of length `vocab_size`.
/// For performance on repeated calls, prefer `project_into` which reuses a buffer.
pub fn project_to_logits(
    hidden: &[f32],
    lm_head: &[f32],         // row-major [vocab_size × hidden_size]
    vocab_size: usize,
) -> Result<Vec<f32>, LogitError> {
    let hidden_size = hidden.len();
    if lm_head.len() != vocab_size * hidden_size {
        return Err(LogitError::ShapeMismatch {
            expected: vocab_size * hidden_size,
            got: lm_head.len(),
        });
    }

    let mut logits = vec![0.0f32; vocab_size];
    project_into(hidden, lm_head, vocab_size, &mut logits)?;
    Ok(logits)
}

/// Same as `project_to_logits` but writes into a pre-allocated buffer.
/// `out` must have length == `vocab_size`.
pub fn project_into(
    hidden: &[f32],
    lm_head: &[f32],
    vocab_size: usize,
    out: &mut [f32],
) -> Result<(), LogitError> {
    let hidden_size = hidden.len();
    if out.len() != vocab_size {
        return Err(LogitError::ShapeMismatch { expected: vocab_size, got: out.len() });
    }

    // dot product of hidden with each row of lm_head
    for (v, row) in out.iter_mut().zip(lm_head.chunks_exact(hidden_size)) {
        *v = row.iter().zip(hidden.iter()).map(|(a, b)| a * b).sum();
    }
    Ok(())
}

// ── Sampling ──────────────────────────────────────────────────────────────────

/// Sampling configuration passed to `sample_token`.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. 1.0 = neutral. Lower → more deterministic.
    pub temperature: f32,
    /// Nucleus (top-p) threshold in (0.0, 1.0]. 1.0 disables top-p filtering.
    pub top_p: f32,
    /// If Some(k), restrict to top-k tokens before top-p.
    pub top_k: Option<usize>,
    /// Repeat penalty applied to recently seen tokens. 1.0 = no penalty.
    pub repeat_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            repeat_penalty: 1.1,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<(), LogitError> {
        if self.temperature <= 0.0 {
            return Err(LogitError::InvalidTemperature(self.temperature));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(LogitError::InvalidTopP(self.top_p));
        }
        Ok(())
    }
}

/// Apply repeat penalty: divide logits of recently seen token ids by `penalty`.
/// `recent` should be the last N token ids (N ≤ 64 is typical).
pub fn apply_repeat_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 { return; }
    for &tok in recent {
        let idx = tok as usize;
        if idx < logits.len() {
            // Penalise in the direction that reduces probability
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Draw the next token id from `logits` using the provided config and an rng seed.
///
/// Algorithm:
///   1. Apply repeat penalty
///   2. Scale by 1/temperature
///   3. Optionally filter to top-k
///   4. Softmax
///   5. Filter to top-p nucleus
///   6. Sample from the remaining distribution
///
/// `rng_state` is a mutable u64 used as a simple xorshift64 RNG — no dependency needed.
pub fn sample_token(
    logits: &mut [f32],
    cfg: &SamplingConfig,
    recent: &[u32],
    rng_state: &mut u64,
) -> Result<u32, LogitError> {
    cfg.validate()?;
    if logits.is_empty() { return Err(LogitError::EmptyLogits); }

    // 1. Repeat penalty
    apply_repeat_penalty(logits, recent, cfg.repeat_penalty);

    // 2. Temperature scaling
    let inv_temp = 1.0 / cfg.temperature;
    for l in logits.iter_mut() { *l *= inv_temp; }

    // 3. Build index array for top-k / sorting
    let vocab = logits.len();
    let mut indices: Vec<usize> = (0..vocab).collect();

    // Sort descending by logit value (only need top-k if specified)
    let k = cfg.top_k.unwrap_or(vocab).min(vocab);
    // Partial sort: bring top-k to the front
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    // Sort the top-k descending so softmax + nucleus scan is correct
    indices.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Softmax over the top-k subset (numerically stable)
    let max_l = logits[indices[0]];
    let mut exps: Vec<f32> = indices.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() { *e /= sum; }

    // 5. Top-p nucleus: keep tokens until cumulative prob ≥ top_p
    let mut cum = 0.0f32;
    let mut cutoff = exps.len();
    for (idx, &p) in exps.iter().enumerate() {
        cum += p;
        if cum >= cfg.top_p {
            cutoff = idx + 1;
            break;
        }
    }
    exps.truncate(cutoff);
    indices.truncate(cutoff);

    // Re-normalise after truncation
    let nucleus_sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() { *e /= nucleus_sum; }

    // 6. Sample with xorshift64
    let r = xorshift64(rng_state);
    let threshold = (r as f32) / (u64::MAX as f32);
    let mut cum2 = 0.0f32;
    for (&idx, &p) in indices.iter().zip(exps.iter()) {
        cum2 += p;
        if threshold <= cum2 {
            return Ok(idx as u32);
        }
    }

    // Fallback: return the top token (should never reach here)
    Ok(indices[0] as u32)
}

// ── xorshift64 RNG ────────────────────────────────────────────────────────────

/// Minimal xorshift64. Seed must not be 0.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Seed from system time (nanoseconds). Falls back to a fixed seed if time unavailable.
pub fn rng_seed_from_time() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 + d.as_secs() * 1_000_000_000)
        .unwrap_or(0x517cc1b727220a95);
    if ns == 0 { 0x517cc1b727220a95 } else { ns }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight() {
        let mut h = vec![3.0f32, 4.0];
        let w = vec![1.0f32, 1.0];
        rms_norm_inplace(&mut h, &w, 1e-5).unwrap();
        // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((h[0] - 3.0 / rms).abs() < 1e-5);
        assert!((h[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn project_identity() {
        // hidden = [1, 0], lm_head rows = [[1,0],[0,1]], expect [1,0]
        let hidden = vec![1.0f32, 0.0];
        let lm_head = vec![1.0f32, 0.0, 0.0, 1.0];
        let logits = project_to_logits(&hidden, &lm_head, 2).unwrap();
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sample_token_greedy() {
        // With very low temperature the highest-logit token should almost always win
        let mut logits = vec![0.0f32; 10];
        logits[7] = 100.0; // overwhelmingly dominant
        let cfg = SamplingConfig { temperature: 0.01, top_p: 1.0, top_k: None, repeat_penalty: 1.0 };
        let mut rng = 12345u64;
        let tok = sample_token(&mut logits, &cfg, &[], &mut rng).unwrap();
        assert_eq!(tok, 7);
    }

    #[test]
    fn repeat_penalty_reduces_logit() {
        let mut logits = vec![1.0f32; 5];
        logits[2] = 2.0;
        apply_repeat_penalty(&mut logits, &[2], 2.0);
        assert!((logits[2] - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
    }
}
