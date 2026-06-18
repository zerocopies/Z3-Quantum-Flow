use std::env;
use std::io::{self, Write};
use std::path::Path;
use anyhow::{Result, bail};

// Import everything from your Z1 library
use z1::loader::MappedModel;
use z1::tokenizer::Tokenizer;
use z1::graph::ForwardPass;
use z1::generate::{generate_turn, Session, GenerateConfig};
use z1::gguf::GgufValue;

// --- Helpers to extract Tokenizer data from GGUF Metadata ---
fn get_str_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<String> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
    } else { Vec::new() }
}

fn get_f32_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<f32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::F32(f) = v { Some(*f) } else { None }).collect()
    } else { Vec::new() }
}

fn get_u32_arr(metadata: &std::collections::HashMap<String, GgufValue>, key: &str) -> Vec<u32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::U32(u) = v { Some(*u) } else { None }).collect()
    } else { Vec::new() }
}
// ------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();

    println!("========================================");
    println!(" Inference Z1 - Zero-Copy Engine Active");
    println!("========================================\n");

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        bail!("Usage: cargo run --release -- <path_to_model.gguf>");
    }
    let model_path = Path::new(&args[1]);

    if !model_path.exists() {
        bail!("Model file not found at: {}", model_path.display());
    }

    println!("[System] Loading model from {}...", model_path.display());

    // 1. Corrected MappedModel Init
    let model = MappedModel::load(model_path)?;
    
    // 2. Corrected Tokenizer Init (Extracting parts from the GGUF header)
    let tokens = get_str_arr(&model.header.metadata, "tokenizer.ggml.tokens");
    let scores = get_f32_arr(&model.header.metadata, "tokenizer.ggml.scores");
    let types  = get_u32_arr(&model.header.metadata, "tokenizer.ggml.token_type");
    let merges = get_str_arr(&model.header.metadata, "tokenizer.ggml.merges");
    let tokenizer = Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)?;

    // 3. Corrected ForwardPass Init
    let mut fwd = ForwardPass::new(&model)?;
    let cfg = GenerateConfig::default();

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