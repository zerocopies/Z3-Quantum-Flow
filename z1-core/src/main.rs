// --- Imports ---
use std::env;
use std::io::{self, Write};
use std::path::Path;
use anyhow::{Result, bail};

// Import everything from your Z1 library
use z3_quantum_flow::loader::MappedModel;
use z3_quantum_flow::tokenizer::Tokenizer;
use z3_quantum_flow::graph::ForwardPass;
use z3_quantum_flow::generate::{generate_turn, Session, GenerateConfig};
use z3_quantum_flow::gguf::GgufValue;

// --- Helper Functions (MUST BE OUTSIDE main) ---

fn get_str_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<String> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
    } else { 
        Vec::new() 
    }
}

fn get_f32_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<f32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::F32(f) = v { Some(*f) } else { None }).collect()
    } else { 
        Vec::new() 
    }
}

fn get_u32_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<u32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::U32(u) = v { Some(*u) } else { None }).collect()
    } else { 
        Vec::new() 
    }
}

// ------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    println!("========================================");
    println!(" Inference Z1 - Zero-Copy Engine Active");
    println!("========================================\n");

    // 1. Parse Arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        bail!("Usage: cargo run --release --bin z1 -- <path_to_model.gguf>");
    }
    let model_path = Path::new(&args[1]);

    if !model_path.exists() {
        bail!("Model file not found at: {}", model_path.display());
    }

    println!("[System] Loading model from {}...", model_path.display());

    // 2. Load Model & Tokenizer
    let model = MappedModel::load(model_path)?;
    
    let tokens = get_str_arr(&model.header.metadata, "tokenizer.ggml.tokens");
    let scores = get_f32_arr(&model.header.metadata, "tokenizer.ggml.scores");
    let types  = get_u32_arr(&model.header.metadata, "tokenizer.ggml.token_type");
    let merges = get_str_arr(&model.header.metadata, "tokenizer.ggml.merges");
    
    let tokenizer = Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)?;

    // ── 🚀 DYNAMIC CONTEXT SIZE LOGIC ───────────────────────
    // Check for Z1_CTX_SIZE env var, default to 2048 if not set
    let ctx_size = std::env::var("Z1_CTX_SIZE")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(2048); 

    println!("[System] Context window configured via Env Var: {} tokens", ctx_size);
    // ─────────────────────────────────────────────────────────

    // 3. Initialize ForwardPass with DYNAMIC ctx_size
    // Ensure ForwardPass::new accepts (model, n_ctx)
    let mut fwd = ForwardPass::new(&model, ctx_size)?;
    
    // Update GenerateConfig to match the actual context size
    let mut cfg = GenerateConfig::default();
    cfg.context_len = ctx_size as usize;

    // 4. Initialize the Sliding-Window Session
    let mut session = Session::new(cfg.context_len, &tokenizer);

    println!("[System] Engine ready. Context window: {} tokens.", cfg.context_len);
    println!("         Type '/exit' to quit or '/reset' to clear context.\n");

    // 5. Interactive Chat Loop
    loop {
        print!("You: ");
        io::stdout().flush()?;
        
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        
        let trimmed = user_input.trim();
        if trimmed.is_empty() { continue; }
        
        if trimmed == "/exit" || trimmed == "/quit" { 
            println!("Shutting down engine. Goodbye!");
            break; 
        }
        if trimmed == "/reset" {
            session.reset();
            fwd.reset_kv();
            println!("[System] Conversation memory cleared.\n");
            continue;
        }

        print!("Z1: ");
        io::stdout().flush()?;

        let _stats = generate_turn(trimmed, &mut session, &mut fwd, &model, &tokenizer, &cfg)?;
    }

    Ok(())
}