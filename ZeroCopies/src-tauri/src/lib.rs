// z1-desktop/src-tauri/src/lib.rs
//
// Wires the Tauri IPC bridge to real Z1 inference (z1-core).
// No HTTP server involved — Portal calls Rust directly through
// Tauri's invoke(), and Rust calls z1-core's run_generation_captured()
// directly.
//
// HONEST NOTE: chat_inference() below duplicates the chat-template
// token-building logic that already lives inside generate_turn() in
// generate.rs, because that logic isn't exposed as a standalone
// function and generate_turn() itself only streams to stdout rather
// than returning text. This works, but it's fragile — if generate.rs's
// internal template logic changes, this file goes stale silently.
// CLEANER FIX (do this before shipping publicly): add a
// `generate_turn_captured()` variant inside generate.rs itself that
// shares the same turn-building code and calls run_generation_captured
// instead of run_generation. Then this file just calls that one
// function instead of reimplementing it.
//
// This REPLACES the existing chat_inference stub. Back up your
// current lib.rs before overwriting, in case there's logic in there
// you want to keep (e.g. the `greet` command, if still used).

use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;

use z3_quantum_flow::tokenizer::Tokenizer;
use z3_quantum_flow::loader::MappedModel;
use z3_quantum_flow::graph::ForwardPass;
use z3_quantum_flow::generate::{Session, GenerateConfig, run_generation_captured};
use z3_quantum_flow::gguf::GgufValue;
use std::collections::HashMap;

// Same metadata-extraction helpers as main.rs, duplicated here because
// they're free functions in the binary crate, not exposed by z1-core
// as a library. If these ever move into z1-core proper, this block
// can be deleted and the z1:: versions imported instead.
fn get_str_arr(metadata: &HashMap<String, GgufValue>, key: &str) -> Vec<String> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
    } else { Vec::new() }
}
fn get_f32_arr(metadata: &HashMap<String, GgufValue>, key: &str) -> Vec<f32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::F32(f) = v { Some(*f) } else { None }).collect()
    } else { Vec::new() }
}
fn get_u32_arr(metadata: &HashMap<String, GgufValue>, key: &str) -> Vec<u32> {
    if let Some(GgufValue::Array(arr)) = metadata.get(key) {
        arr.iter().filter_map(|v| if let GgufValue::U32(u) = v { Some(*u) } else { None }).collect()
    } else { Vec::new() }
}

/// Holds the currently loaded model + session, if any.
/// Mutex because Tauri commands can be called from multiple threads;
/// only one inference call should run at a time on a single CPU engine.
pub struct EngineState {
    pub loaded: Mutex<Option<LoadedModel>>,
}

pub struct LoadedModel {
    model: MappedModel,
    tokenizer: Tokenizer,
    fwd: ForwardPass,
    session: Session,
    cfg: GenerateConfig,
    #[allow(dead_code)]
    model_name: String,
}

impl Default for EngineState {
    fn default() -> Self {
        Self { loaded: Mutex::new(None) }
    }
}

/// Frontend calls this when the user clicks "Load a model" and picks a
/// single .gguf file via native file dialog. Mirrors main.rs exactly:
/// tokenizer is extracted from the GGUF header's own metadata, not a
/// separate file.
#[tauri::command]
async fn load_model(
    model_path: String,
    state: State<'_, EngineState>,
) -> Result<String, String> {
    let model_path = PathBuf::from(&model_path);

    let model = MappedModel::load(&model_path).map_err(|e| e.to_string())?;

    let tokens = get_str_arr(&model.header.metadata, "tokenizer.ggml.tokens");
    let scores = get_f32_arr(&model.header.metadata, "tokenizer.ggml.scores");
    let types  = get_u32_arr(&model.header.metadata, "tokenizer.ggml.token_type");
    let merges = get_str_arr(&model.header.metadata, "tokenizer.ggml.merges");
    let tokenizer = Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)
        .map_err(|e| e.to_string())?;

    let fwd = ForwardPass::new(&model).map_err(|e| e.to_string())?;

    let cfg = GenerateConfig::default();
    let session = Session::new(cfg.context_len, &tokenizer);

    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown model".to_string());

    let mut guard = state.loaded.lock().map_err(|_| "engine lock poisoned".to_string())?;
    *guard = Some(LoadedModel { model, tokenizer, fwd, session, cfg, model_name: model_name.clone() });

    Ok(model_name)
}

/// Chat-template constants, matching generate.rs exactly (Llama 3.1 format).
const T_START_HEADER: u32 = 128_006;
const T_END_HEADER: u32 = 128_007;
const T_EOT: u32 = 128_009;
const T_NEWLINES: u32 = 271;

#[tauri::command]
async fn chat_inference(
    prompt: String,
    state: State<'_, EngineState>,
) -> Result<serde_json::Value, String> {
    let mut guard = state.loaded.lock().map_err(|_| "engine lock poisoned".to_string())?;
    let loaded = guard.as_mut().ok_or_else(|| "no model loaded".to_string())?;

    if prompt.trim().is_empty() {
        return Err("empty prompt".to_string());
    }

    // Build this turn's tokens the same way generate_turn() does internally.
    let mut new_turn = vec![T_START_HEADER];
    new_turn.extend_from_slice(&loaded.tokenizer.encode_no_bos("user"));
    new_turn.push(T_END_HEADER);
    new_turn.extend_from_slice(&loaded.tokenizer.encode_no_bos(&format!("\n\n{prompt}")));
    new_turn.push(T_EOT);
    new_turn.push(T_START_HEADER);
    new_turn.extend_from_slice(&loaded.tokenizer.encode_no_bos("assistant"));
    new_turn.push(T_END_HEADER);
    new_turn.push(T_NEWLINES);

    let needed_space = new_turn.len() + loaded.cfg.max_new_tokens;
    let available_space = loaded.session.context_len.saturating_sub(loaded.session.system_tokens.len());
    if needed_space > available_space {
        return Err(format!("context length exceeded (max {} tokens)", loaded.session.context_len));
    }

    let mut requires_reprefill = false;
    while loaded.session.system_tokens.len() + loaded.session.history_tokens.len() + needed_space
        > loaded.session.context_len
    {
        let drop_amount = 128.min(loaded.session.history_tokens.len());
        loaded.session.history_tokens.drain(0..drop_amount);
        requires_reprefill = true;
    }

    loaded.session.history_tokens.extend_from_slice(&new_turn);

    let mut tokens_to_process = Vec::new();
    if requires_reprefill || loaded.session.turn_count == 0 {
        loaded.fwd.reset_kv();
        tokens_to_process.extend_from_slice(&loaded.session.system_tokens);
        tokens_to_process.extend_from_slice(&loaded.session.history_tokens);
    } else {
        tokens_to_process.extend_from_slice(&new_turn);
    }
    loaded.session.turn_count += 1;

    let (stats, text, generated_ids) = run_generation_captured(
        &tokens_to_process,
        &mut loaded.fwd,
        &loaded.model,
        &loaded.tokenizer,
        &loaded.cfg,
    )
    .map_err(|e| e.to_string())?;

    loaded.session.history_tokens.extend_from_slice(&generated_ids);
    loaded.session.history_tokens.push(T_EOT);

    Ok(serde_json::json!({
        "text": text,
        "stats": format!("{} tokens · {:.2} tok/s", stats.generated_tokens, stats.tokens_per_second())
    }))
}

#[tauri::command]
fn is_model_loaded(state: State<'_, EngineState>) -> bool {
    state.loaded.lock().map(|g| g.is_some()).unwrap_or(false)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(EngineState::default())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            load_model,
            chat_inference,
            is_model_loaded
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
