/// Z.1 — harness.rs
///
/// Performance and correctness harness.
///
/// Implements all improvements suggested in the original code review:
///   1. Proper error-result propagation (no panics on bad input).
///   2. Ceiling-division layer counts.
///   3. Atomic stop flag (AtomicBool) — no Mutex<bool>.
///   4. Condvar-based prefetch wakeup — no busy sleep.
///   5. Structured logging via log:: macros.
///   6. Timing at every stage (prompt ingestion + generation separately).
///   7. Repetition-penalty sampler.
///   8. Multi-run statistics with mean, min, max, and std-dev.
///   9. Warm-up run (excluded from statistics).
///  10. Correctness smoke-test: verifies the engine can round-trip a token.

use std::time::Instant;

use anyhow::{bail, Result};

use crate::engine::{GenStats, LlamaEngine};

// ── Harness prompts ───────────────────────────────────────────────────────────

const BENCH_PROMPTS: &[&str] = &[
    "Explain the difference between stack and heap memory allocation in C.",
    "Write a Rust function that merges two sorted vectors into one.",
    "What are the main advantages of memory-mapped I/O for large files?",
    "Describe the transformer attention mechanism in plain English.",
    "List five best practices for writing safe concurrent Rust code.",
];

// ── Public entry point ────────────────────────────────────────────────────────

/// Run `n_runs` timed inference passes and print a statistics summary.
pub fn run(engine: &mut LlamaEngine, n_runs: usize) -> Result<()> {
    if n_runs == 0 {
        bail!("[Z.1 Harness] n_runs must be > 0");
    }

    log::info!("────────────────────────────────────────────────");
    log::info!("[Z.1 Harness] Starting benchmark ({} runs + 1 warm-up)", n_runs);
    log::info!("[Z.1 Harness] Context size : {} tokens", engine.ctx_size());
    log::info!("[Z.1 Harness] Vocab size   : {} tokens", engine.vocab_size());
    log::info!("────────────────────────────────────────────────");

    // ── Correctness smoke-test ────────────────────────────────────────────────
    smoke_test(engine)?;

    // ── Warm-up (results discarded) ───────────────────────────────────────────
    log::info!("[Z.1 Harness] Warm-up run …");
    let prompt = BENCH_PROMPTS[0];
    let _ = engine.generate(prompt, |_| {})?;
    log::info!("[Z.1 Harness] Warm-up done. Starting timed runs.");

    // ── Timed runs ────────────────────────────────────────────────────────────
    let mut all_stats: Vec<GenStats> = Vec::with_capacity(n_runs);

    for run in 0..n_runs {
        let prompt = BENCH_PROMPTS[run % BENCH_PROMPTS.len()];
        log::info!("[Z.1 Harness] Run {}/{} — prompt: {:?}…", run + 1, n_runs, &prompt[..40.min(prompt.len())]);

        let t_wall = Instant::now();
        let (_, stats) = engine.generate(prompt, |_| {})?;
        let wall_ms = t_wall.elapsed().as_secs_f64() * 1000.0;

        stats.print();
        log::info!(
            "[Z.1 Harness]   Wall time : {:.1} ms | Gen speed: {:.2} tok/s",
            wall_ms,
            stats.tokens_per_sec()
        );

        all_stats.push(stats);
    }

    // ── Summary statistics ────────────────────────────────────────────────────
    print_summary(&all_stats);
    Ok(())
}

// ── Smoke test ────────────────────────────────────────────────────────────────

/// Verifies that tokenise → detokenise round-trips cleanly.
fn smoke_test(engine: &mut LlamaEngine) -> Result<()> {
    log::info!("[Z.1 Harness] Running smoke test …");

    let sample = "Hello, Z.1!";
    let tokens = engine.tokenize(sample, false)?;
    if tokens.is_empty() {
        bail!("[Z.1 Harness] Smoke test FAILED: tokenizer returned 0 tokens for {:?}", sample);
    }

    let reconstructed: String = tokens
        .iter()
        .map(|&t| engine.token_to_piece(t))
        .collect();

    // We don't require exact equality (BPE can add leading spaces), just
    // that the key words survive the round-trip.
    if !reconstructed.contains("Hello") || !reconstructed.contains("Z.1") {
        bail!(
            "[Z.1 Harness] Smoke test FAILED: round-trip mismatch.\n  In : {:?}\n  Out: {:?}",
            sample,
            reconstructed
        );
    }

    log::info!(
        "[Z.1 Harness] Smoke test PASSED ({} → {} tokens → {:?})",
        sample, tokens.len(), reconstructed.trim()
    );
    Ok(())
}

// ── Statistics helpers ────────────────────────────────────────────────────────

fn print_summary(stats: &[GenStats]) {
    if stats.is_empty() { return; }

    let tps: Vec<f64>  = stats.iter().map(|s| s.tokens_per_sec()).collect();
    let ptps: Vec<f64> = stats.iter().map(|s| s.prompt_tokens_per_sec()).collect();
    let gms: Vec<f64>  = stats.iter().map(|s| s.generate_ms).collect();

    log::info!("────────────────────────────────────────────────");
    log::info!("[Z.1 Harness] ── Summary ({} runs) ──", stats.len());
    log::info!(
        "[Z.1 Harness]   Gen speed   : mean {:.2}  min {:.2}  max {:.2}  σ {:.2}  tok/s",
        mean(&tps), min_f(&tps), max_f(&tps), std_dev(&tps)
    );
    log::info!(
        "[Z.1 Harness]   Prompt speed: mean {:.2}  min {:.2}  max {:.2}  tok/s",
        mean(&ptps), min_f(&ptps), max_f(&ptps)
    );
    log::info!(
        "[Z.1 Harness]   Gen latency : mean {:.1}  min {:.1}  max {:.1}  ms",
        mean(&gms), min_f(&gms), max_f(&gms)
    );
    log::info!(
        "[Z.1 Harness]   Avg tokens  : {:.1} prompt + {:.1} generated",
        stats.iter().map(|s| s.n_prompt_tokens as f64).sum::<f64>() / stats.len() as f64,
        stats.iter().map(|s| s.n_generated_tokens as f64).sum::<f64>() / stats.len() as f64,
    );
    log::info!("────────────────────────────────────────────────");
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
}

fn min_f(xs: &[f64]) -> f64 { xs.iter().cloned().fold(f64::INFINITY, f64::min) }
fn max_f(xs: &[f64]) -> f64 { xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
