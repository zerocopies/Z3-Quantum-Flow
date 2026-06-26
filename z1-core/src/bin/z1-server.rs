// z1-core/src/bin/z1-server.rs
//
// Standalone HTTP server for Z1 — the "web version" front door.
// Same inference logic already proven working in the Tauri app's
// lib.rs, just exposed over plain HTTP instead of Tauri's IPC bridge.
// This means: any browser tab, no Tauri, no native app shell required.
//
// Run with:
//   cargo run --release --bin z1-server
//
// Then open the matching z1-web.html file (served separately, or via
// `python3 -m http.server` from the same folder) in any browser.
//
// Endpoints:
//   GET  /health                 -> "ok" if server is alive
//   POST /load_model  {path}     -> loads a .gguf file, returns model name
//   POST /chat         {prompt}  -> runs inference, returns text + stats

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tiny_http::{Server, Response, Header, Method};

use z3_quantum_flow::tokenizer::Tokenizer;
use z3_quantum_flow::loader::MappedModel;
use z3_quantum_flow::graph::ForwardPass;
use z3_quantum_flow::generate::{Session, GenerateConfig, run_generation_captured};
use z3_quantum_flow::gguf::GgufValue;

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

struct LoadedModel {
    model: MappedModel,
    tokenizer: Tokenizer,
    fwd: ForwardPass,
    session: Session,
    cfg: GenerateConfig,
    model_name: String,
}

const T_START_HEADER: u32 = 128_006;
const T_END_HEADER: u32 = 128_007;
const T_EOT: u32 = 128_009;
const T_NEWLINES: u32 = 271;

#[derive(Deserialize)]
struct LoadRequest { path: String }

#[derive(Serialize)]
struct LoadResponse { model_name: String }

#[derive(Deserialize)]
struct ChatRequest { prompt: String }

#[derive(Serialize)]
struct ChatResponse { text: String, stats: String }

#[derive(Serialize)]
struct ErrorResponse { error: String }

fn cors_header() -> Header {
    Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap()
}
fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

fn json_response(body: String, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(body)
        .with_status_code(status)
        .with_header(json_header())
        .with_header(cors_header())
}

fn main() {
    env_logger::init();
    let state: Mutex<Option<LoadedModel>> = Mutex::new(None);

    let addr = "127.0.0.1:7474";
    let server = Server::http(addr).expect("failed to bind — is something already running on 7474?");
    println!("========================================");
    println!(" Inference Z1 — HTTP server");
    println!("========================================");
    println!("[System] Listening on http://{addr}");
    println!("[System] Endpoints: GET /health, GET /status, POST /load_model, POST /chat, POST /new_session\n");

    for mut request in server.incoming_requests() {
        // Handle CORS preflight for browser fetch() calls
        if request.method() == &Method::Options {
            let _ = request.respond(
                Response::empty(204)
                    .with_header(cors_header())
                    .with_header(Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"POST, GET, OPTIONS"[..]).unwrap())
                    .with_header(Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"Content-Type"[..]).unwrap())
            );
            continue;
        }

        let url = request.url().to_string();

        match (request.method(), url.as_str()) {
            (Method::Get, "/status") => {
                let guard = state.lock().unwrap();
                let body = match guard.as_ref() {
                    Some(loaded) => format!(r#"{{"loaded":true,"model_name":"{}"}}"#, loaded.model_name),
                    None => r#"{"loaded":false,"model_name":null}"#.to_string(),
                };
                let _ = request.respond(json_response(body, 200));
            }

            (Method::Get, "/health") => {
                let _ = request.respond(json_response(r#"{"status":"ok"}"#.to_string(), 200));
            }

            (Method::Post, "/load_model") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);

                let parsed: Result<LoadRequest, _> = serde_json::from_str(&body);
                let path = match parsed {
                    Ok(p) => p.path,
                    Err(e) => {
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{error: format!("bad request: {e}")}).unwrap(), 400));
                        continue;
                    }
                };

                let model_path_raw = PathBuf::from(&path);

                // Resolve symlinks first — this is exactly what bit us during
                // testing: a symlinked model path made loader.rs's file-open
                // behave inconsistently and fail with a misleading error.
                // canonicalize() also fails clearly if the path doesn't exist
                // at all, which lets us give a much better message up front.
                let model_path = match std::fs::canonicalize(&model_path_raw) {
                    Ok(p) => p,
                    Err(_) => {
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{
                                error: format!("Couldn't find a file at that path. Double-check it's correct and the file exists: {}", model_path_raw.display())
                            }).unwrap(), 400));
                        continue;
                    }
                };

                if model_path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                    let _ = request.respond(json_response(
                        serde_json::to_string(&ErrorResponse{
                            error: "That file doesn't look like a .gguf model file. Z1 currently only loads GGUF format models.".to_string()
                        }).unwrap(), 400));
                    continue;
                }

                let result = (|| -> anyhow::Result<String> {
                    let model = MappedModel::load(&model_path)
                        .map_err(|_| anyhow::anyhow!("Found the file, but couldn't read it as a valid model. It may be corrupted, incomplete, or not actually a GGUF file despite the extension."))?;
                    let tokens = get_str_arr(&model.header.metadata, "tokenizer.ggml.tokens");
                    let scores = get_f32_arr(&model.header.metadata, "tokenizer.ggml.scores");
                    let types  = get_u32_arr(&model.header.metadata, "tokenizer.ggml.token_type");
                    let merges = get_str_arr(&model.header.metadata, "tokenizer.ggml.merges");
                    let tokenizer = Tokenizer::from_gguf_parts(&tokens, &scores, &types, &merges)
                        .map_err(|_| anyhow::anyhow!("This model's tokenizer data looks malformed. The model file may be incomplete or use an unsupported tokenizer format."))?;
                    let n_ctx = std::env::var("Z1_CTX_SIZE").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(2048);
                    let fwd = ForwardPass::new(&model, n_ctx)
                        .map_err(|_| anyhow::anyhow!("Couldn't build the compute graph for this model. It may use an unsupported architecture."))?;
                    let cfg = GenerateConfig::default();
                    let session = Session::new(cfg.context_len, &tokenizer, fwd.dna().arch.as_str());
                    let model_name = model_path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "unknown model".to_string());

                    let mut guard = state.lock().unwrap();
                    *guard = Some(LoadedModel { model, tokenizer, fwd, session, cfg, model_name: model_name.clone() });
                    Ok(model_name)
                })();

                match result {
                    Ok(model_name) => {
                        println!("[System] Model loaded: {model_name}");
                        let _ = request.respond(json_response(
                            serde_json::to_string(&LoadResponse{model_name}).unwrap(), 200));
                    }
                    Err(e) => {
                        eprintln!("[Error] load_model failed: {:#}", e);
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{error: format!("{:#}", e)}).unwrap(), 500));
                    }
                }
            }

            (Method::Post, "/new_session") => {
                let mut guard = state.lock().unwrap();
                match guard.as_mut() {
                    Some(loaded) => {
                        loaded.session = Session::new(loaded.cfg.context_len, &loaded.tokenizer, loaded.fwd.dna().arch.as_str());
                        loaded.fwd.reset_kv();
                        let _ = request.respond(json_response(r#"{"status":"ok"}"#.to_string(), 200));
                    }
                    None => {
                        let _ = request.respond(json_response(r#"{"status":"ok"}"#.to_string(), 200));
                    }
                }
            }

            (Method::Post, "/chat") => {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);

                let parsed: Result<ChatRequest, _> = serde_json::from_str(&body);
                let prompt = match parsed {
                    Ok(p) => p.prompt,
                    Err(e) => {
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{error: format!("bad request: {e}")}).unwrap(), 400));
                        continue;
                    }
                };

                let mut guard = state.lock().unwrap();
                let loaded = match guard.as_mut() {
                    Some(l) => l,
                    None => {
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{error: "No model is loaded yet — load one first.".to_string()}).unwrap(), 400));
                        continue;
                    }
                };

                if prompt.trim().is_empty() {
                    let _ = request.respond(json_response(
                        serde_json::to_string(&ErrorResponse{error: "empty prompt".to_string()}).unwrap(), 400));
                    continue;
                }

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
                    let _ = request.respond(json_response(
                        serde_json::to_string(&ErrorResponse{error: format!("This conversation has gotten too long for the current context window ({} tokens). Try starting a new chat.", loaded.session.context_len)}).unwrap(), 400));
                    continue;
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

                let gen_result = run_generation_captured(
                    &tokens_to_process, &mut loaded.fwd, &loaded.model, &loaded.tokenizer, &loaded.cfg);

                match gen_result {
                    Ok((stats, text, generated_ids)) => {
                        loaded.session.history_tokens.extend_from_slice(&generated_ids);
                        loaded.session.history_tokens.push(T_EOT);

                        let resp = ChatResponse {
                            text,
                            stats: format!("{} tokens · {:.2} tok/s", stats.generated_tokens, stats.tokens_per_second()),
                        };
                        let _ = request.respond(json_response(serde_json::to_string(&resp).unwrap(), 200));
                    }
                    Err(e) => {
                        let _ = request.respond(json_response(
                            serde_json::to_string(&ErrorResponse{error: e.to_string()}).unwrap(), 500));
                    }
                }
            }

            _ => {
                let _ = request.respond(json_response(
                    serde_json::to_string(&ErrorResponse{error: "not found".to_string()}).unwrap(), 404));
            }
        }
    }
}
