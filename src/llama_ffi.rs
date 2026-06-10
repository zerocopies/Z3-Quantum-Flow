/// Z.1 — llama_ffi.rs
///
/// Raw FFI bindings matching llama.cpp b3534 exactly.
/// Structs verified against vendor/llama.cpp/include/llama.h at that commit.

#[allow(non_camel_case_types, non_snake_case, dead_code)]

use libc::{c_char, c_float, c_int, c_void, size_t};

// ── Opaque handles ────────────────────────────────────────────────────────────
#[repr(C)] pub struct llama_model   { _p: [u8; 0] }
#[repr(C)] pub struct llama_context { _p: [u8; 0] }

pub type llama_token  = i32;
pub type llama_pos    = i32;
pub type llama_seq_id = i32;

// ── llama_model_params — verified against b3534 include/llama.h ──────────────
// Field order must match C exactly (copy-by-value ABI).
//
// struct llama_model_params {
//     int32_t n_gpu_layers;
//     enum llama_split_mode split_mode;   // i32
//     int32_t main_gpu;
//     const float * tensor_split;
//     const char  * rpc_servers;          // added in b3534
//     llama_progress_callback progress_callback;  // fn ptr
//     void * progress_callback_user_data;
//     const struct llama_model_kv_override * kv_overrides;
//     bool vocab_only;
//     bool use_mmap;
//     bool use_mlock;
//     bool check_tensors;                 // added in b3534
// };
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_model_params {
    pub n_gpu_layers:                c_int,
    pub split_mode:                  c_int,
    pub main_gpu:                    c_int,
    pub tensor_split:                *const c_float,
    pub rpc_servers:                 *const c_char,
    pub progress_callback:           *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides:                *const c_void,
    pub vocab_only:                  bool,
    pub use_mmap:                    bool,
    pub use_mlock:                   bool,
    pub check_tensors:               bool,
}

// ── llama_context_params — verified against b3534 include/llama.h ────────────
// struct llama_context_params {
//     uint32_t seed;
//     uint32_t n_ctx;
//     uint32_t n_batch;
//     uint32_t n_ubatch;
//     uint32_t n_seq_max;
//     uint32_t n_threads;
//     uint32_t n_threads_batch;
//     enum llama_rope_scaling_type rope_scaling_type; // i32
//     enum llama_pooling_type      pooling_type;      // i32
//     enum llama_attention_type    attention_type;    // i32
//     float rope_freq_base;
//     float rope_freq_scale;
//     float yarn_ext_factor;
//     float yarn_attn_factor;
//     float yarn_beta_fast;
//     float yarn_beta_slow;
//     uint32_t yarn_orig_ctx;
//     float defrag_thold;
//     ggml_backend_sched_eval_callback cb_eval;  // fn ptr
//     void * cb_eval_user_data;
//     enum ggml_type type_k; // i32
//     enum ggml_type type_v; // i32
//     bool logits_all;
//     bool embeddings;
//     bool offload_kqv;
//     bool flash_attn;
//     ggml_abort_callback abort_callback; // fn ptr
//     void * abort_callback_data;
// };
#[repr(C)]
#[derive(Debug, Clone)]
pub struct llama_context_params {
    pub seed:                u32,
    pub n_ctx:               u32,
    pub n_batch:             u32,
    pub n_ubatch:            u32,
    pub n_seq_max:           u32,
    pub n_threads:           u32,
    pub n_threads_batch:     u32,
    pub rope_scaling_type:   c_int,
    pub pooling_type:        c_int,
    pub attention_type:      c_int,
    pub rope_freq_base:      c_float,
    pub rope_freq_scale:     c_float,
    pub yarn_ext_factor:     c_float,
    pub yarn_attn_factor:    c_float,
    pub yarn_beta_fast:      c_float,
    pub yarn_beta_slow:      c_float,
    pub yarn_orig_ctx:       u32,
    pub defrag_thold:        c_float,
    pub cb_eval:             *mut c_void,
    pub cb_eval_user_data:   *mut c_void,
    pub type_k:              c_int,
    pub type_v:              c_int,
    pub logits_all:          bool,
    pub embeddings:          bool,
    pub offload_kqv:         bool,
    pub flash_attn:          bool,
    pub abort_callback:      *mut c_void,
    pub abort_callback_data: *mut c_void,
}

// ── Token data ────────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_token_data {
    pub id:    llama_token,
    pub logit: c_float,
    pub p:     c_float,
}

// ── Batch ─────────────────────────────────────────────────────────────────────
#[repr(C)]
pub struct llama_batch {
    pub n_tokens:   c_int,
    pub token:      *mut llama_token,
    pub embd:       *mut c_float,
    pub pos:        *mut llama_pos,
    pub n_seq_id:   *mut c_int,
    pub seq_id:     *mut *mut llama_seq_id,
    pub logits:     *mut i8,
    pub all_pos_0:  llama_pos,
    pub all_pos_1:  llama_pos,
    pub all_seq_id: llama_seq_id,
}

// ── C API ─────────────────────────────────────────────────────────────────────
extern "C" {
    // Backend — no argument in b3534
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    // Default params
    pub fn llama_model_default_params()   -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;

    // Model lifecycle
    pub fn llama_load_model_from_file(
        path_model: *const c_char,
        params:     llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_free_model(model: *mut llama_model);

    // Context lifecycle
    pub fn llama_new_context_with_model(
        model:  *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);

    // Tokenisation
    pub fn llama_tokenize(
        model:        *const llama_model,
        text:         *const c_char,
        text_len:     c_int,
        tokens:       *mut llama_token,
        n_max_tokens: c_int,
        add_bos:      bool,
        special:      bool,
    ) -> c_int;

    // token_to_piece gained an `lstrip` int arg in b3534
    pub fn llama_token_to_piece(
        model:   *const llama_model,
        token:   llama_token,
        buf:     *mut c_char,
        length:  c_int,
        lstrip:  c_int,
        special: bool,
    ) -> c_int;

    // Batch
    pub fn llama_batch_init(n_tokens: i32, embd: c_int, n_seq_max: c_int) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);

    // Decode
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;

    // Logits — index is the position within the batch that had logits=1
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: c_int) -> *mut c_float;

    // Vocab
    pub fn llama_n_vocab(model: *const llama_model) -> c_int;
    pub fn llama_token_eos(model: *const llama_model) -> llama_token;
    pub fn llama_token_bos(model: *const llama_model) -> llama_token;

    // KV cache
    pub fn llama_kv_cache_clear(ctx: *mut llama_context);

    // Context info
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;

    // Model info
    pub fn llama_model_desc(
        model:    *const llama_model,
        buf:      *mut c_char,
        buf_size: size_t,
    ) -> c_int;
    pub fn llama_model_size(model: *const llama_model) -> u64;
    pub fn llama_model_n_params(model: *const llama_model) -> u64;
}
