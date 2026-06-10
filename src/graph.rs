/// Z.1 — graph.rs
///
/// Llama 3.1 forward pass with ggml ops.
/// Zero-copy weights from mmap. ggml_gallocr allocates intermediate tensors.
/// KV cache: F32 cache tensors per layer; new K/V written via ggml_cpy into
/// cache views, full history read back for attention.

use std::ffi::c_void;
use std::fmt;
use libc::c_int;

use anyhow::{bail, Result};

use crate::ggml_ffi::{self as ffi, GgmlInitParams};
use crate::loader::MappedModel;

// ── Error types ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ForwardError {
    ShapeMismatch { expected: usize, got: usize },
    MissingTensor(String),
    ComputeFailed(i32),
    AllocationFailed(String),
}

impl fmt::Display for ForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } =>
                write!(f, "shape mismatch: expected {expected}, got {got}"),
            Self::MissingTensor(name) =>
                write!(f, "missing tensor: {name}"),
            Self::ComputeFailed(status) =>
                write!(f, "compute failed with status {status}"),
            Self::AllocationFailed(msg) =>
                write!(f, "allocation failed: {msg}"),
        }
    }
}

impl std::error::Error for ForwardError {}

// ── Tensor shape diagnostics ──────────────────────────────────────────────────
//
// ggml_tensor is opaque in our FFI, but the b3534 struct layout is fixed:
//   enum ggml_type type;             // 4 bytes
//   enum ggml_backend_type backend;  // 4 bytes
//   struct ggml_backend_buffer *buf; // 8 bytes
//   int64_t ne[4];                   // <-- byte offset 16
// So we can read the dims with raw pointer arithmetic.

unsafe fn tensor_dims(t: *const ffi::ggml_tensor) -> [i64; 4] {
    let p = (t as *const u8).add(16) as *const i64;
    [*p.add(0), *p.add(1), *p.add(2), *p.add(3)]
}

pub struct LlamaHparams {
    pub n_vocab:      i64,
    pub n_embd:       i64,
    pub n_head:       i64,
    pub n_head_kv:    i64,
    pub n_layer:      i64,
    pub n_ff:         i64,
    pub n_rot:        i64,
    pub freq_base:    f32,
    pub rms_norm_eps: f32,
}

impl LlamaHparams {
    pub fn from_model(model: &MappedModel) -> Self {
        let arch = model.header.architecture().unwrap_or("llama");
        let meta = &model.header.metadata;
        let get_u32 = |key: &str| -> i64 {
            meta.get(key).and_then(|v| v.as_u32()).unwrap_or(0) as i64
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            meta.get(key)
                .and_then(|v| if let crate::gguf::GgufValue::F32(f) = v { Some(*f) } else { None })
                .unwrap_or(default)
        };
        LlamaHparams {
            n_vocab:      get_u32("llama.vocab_size"),
            n_embd:       get_u32(&format!("{}.embedding_length", arch)),
            n_head:       get_u32(&format!("{}.attention.head_count", arch)),
            n_head_kv:    get_u32(&format!("{}.attention.head_count_kv", arch)),
            n_layer:      get_u32(&format!("{}.block_count", arch)),
            n_ff:         get_u32(&format!("{}.feed_forward_length", arch)),
            n_rot:        get_u32(&format!("{}.rope.dimension_count", arch)),
            freq_base:    get_f32(&format!("{}.rope.freq_base", arch), 500000.0),
            rms_norm_eps: get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-5),
        }
    }
    pub fn head_dim(&self) -> i64 { self.n_embd / self.n_head }
}

// ── KV cache ──────────────────────────────────────────────────────────────────
//
// One K and one V tensor per layer, F32, shape [n_head_kv * head_dim, n_ctx].
// Row i holds token i's K (or V) for all KV heads. `head` is the next free
// row (== number of valid cached tokens).

pub struct KVCache {
    pub ctx:   *mut ffi::ggml_context,
    pub k:     Vec<*mut ffi::ggml_tensor>,
    pub v:     Vec<*mut ffi::ggml_tensor>,
    pub n_ctx: i64,
    pub head:  i64,
}
unsafe impl Send for KVCache {}

impl KVCache {
    pub fn new(n_layers: usize, n_head_kv: i64, head_dim: i64, n_ctx: i64) -> Self {
        // F32 = 4 bytes per element
        let bytes_per = (n_head_kv * head_dim * n_ctx * 4) as usize;
        let total     = bytes_per * n_layers * 2 + 4 * 1024 * 1024;
        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: total, mem_buffer: std::ptr::null_mut(), no_alloc: false,
            })
        };
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            unsafe {
                // dtype 0 = GGML_TYPE_F32
                k.push(ffi::ggml_new_tensor_2d(ctx, 0, n_head_kv * head_dim, n_ctx));
                v.push(ffi::ggml_new_tensor_2d(ctx, 0, n_head_kv * head_dim, n_ctx));
            }
        }
        KVCache { ctx, k, v, n_ctx, head: 0 }
    }
    pub fn clear(&mut self) { self.head = 0; }
}
impl Drop for KVCache {
    fn drop(&mut self) { unsafe { ffi::ggml_free(self.ctx); } }
}

pub struct LlamaGraph {
    pub hp:      LlamaHparams,
    pub backend: ffi::ggml_backend_t,
    pub kv:      KVCache,
}
unsafe impl Send for LlamaGraph {}

impl LlamaGraph {
    pub fn new(model: &MappedModel) -> Result<Self> {
        let hp      = LlamaHparams::from_model(model);
        let backend = unsafe { ffi::ggml_backend_cpu_init() };
        if backend.is_null() { bail!("[Z.1 Graph] ggml_backend_cpu_init failed"); }
        log::info!("[Z.1 Graph] layers={} embd={} heads={}/{} ff={} rot={} freq_base={}",
            hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_ff, hp.n_rot, hp.freq_base);
        let kv = KVCache::new(hp.n_layer as usize, hp.n_head_kv, hp.head_dim(), 512);
        Ok(LlamaGraph { hp, backend, kv })
    }

    pub fn forward(
        &mut self,
        model:  &MappedModel,
        tokens: &[i32],
        pos:    i32,
    ) -> Result<Vec<f32>> {
        let n_tokens  = tokens.len() as i64;
        let hp        = &self.hp;
        let tok_bytes = n_tokens as usize * 4;

        // ── Input backend buffer ──────────────────────────────────────────────
        let inp_buf_size = tok_bytes * 2 + 256;
        let inp_buf = unsafe { ffi::ggml_backend_alloc_buffer(self.backend, inp_buf_size) };
        if inp_buf.is_null() { bail!("[Z.1 Graph] ggml_backend_alloc_buffer failed"); }
        let inp_base = unsafe { ffi::ggml_backend_buffer_get_base(inp_buf) } as usize;

        let inp_ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 4096, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if inp_ctx.is_null() {
            unsafe { ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] ggml_init for inp_ctx failed");
        }

        // Token ids at base of input buffer
        let inp_tokens = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26 /*I32*/, n_tokens) };
        unsafe {
            ffi::ggml_backend_tensor_alloc(inp_buf, inp_tokens, inp_base as *mut c_void);
            ffi::ggml_backend_tensor_set(
                inp_tokens, tokens.as_ptr() as *const c_void, 0, tok_bytes,
            );
        }

        // Position ids after token ids (64-byte aligned)
        let pos_offset = (tok_bytes + 63) & !63;
        let positions: Vec<i32> = (pos..pos + n_tokens as i32).collect();
        let inp_pos = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26 /*I32*/, n_tokens) };
        unsafe {
            ffi::ggml_backend_tensor_alloc(inp_buf, inp_pos, (inp_base + pos_offset) as *mut c_void);
            ffi::ggml_backend_tensor_set(
                inp_pos, positions.as_ptr() as *const c_void, 0, tok_bytes,
            );
        }

        // ── Compute graph context ─────────────────────────────────────────────
        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 64 * 1024 * 1024, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] ggml_init for graph ctx failed");
        }

        // ── Graph object created BEFORE the layer loop ────────────────────────
        // The KV-cache write ops (ggml_cpy) are not on the logits dependency
        // chain, so they must be explicitly expanded into the graph as we go.
        // Expansion order guarantees each layer's cache write executes before
        // the next layer's nodes, and the read views see fresh data.
        let graph = unsafe { ffi::ggml_new_graph_custom(ctx, 8192, false) };

        // ── Token embeddings ──────────────────────────────────────────────────
        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(ctx, embd_w, inp_tokens) };

        // ── KV bookkeeping ────────────────────────────────────────────────────
        let hd      = hp.head_dim();
        let kv_head = self.kv.head;            // row we write into
        let kv_len  = kv_head + n_tokens;      // valid rows after this call
        // row stride in bytes: n_head_kv * head_dim * sizeof(f32)
        let kv_row_stride = (hp.n_head_kv * hd * 4) as usize;

        if kv_len > self.kv.n_ctx {
            unsafe { ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] KV cache overflow: {} > {}", kv_len, self.kv.n_ctx);
        }

        // ── Transformer layers ────────────────────────────────────────────────
        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} not found", layer, name))
            };

            // ── Attention norm + QKV projections ──────────────────────────────
            let normed = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("attn_norm.weight")?) };
            let q = unsafe { ffi::ggml_mul_mat(ctx, t("attn_q.weight")?, normed) };
            let k = unsafe { ffi::ggml_mul_mat(ctx, t("attn_k.weight")?, normed) };
            let v = unsafe { ffi::ggml_mul_mat(ctx, t("attn_v.weight")?, normed) };

            // Reshape into per-head tensors
            let q = unsafe { ffi::ggml_reshape_3d(ctx, q, hd, hp.n_head,    n_tokens) };
            let k = unsafe { ffi::ggml_reshape_3d(ctx, k, hd, hp.n_head_kv, n_tokens) };
            let v = unsafe { ffi::ggml_reshape_3d(ctx, v, hd, hp.n_head_kv, n_tokens) };

            // RoPE on Q and K
            let q = unsafe { ffi::ggml_rope_ext(ctx, q, inp_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };
            let k = unsafe { ffi::ggml_rope_ext(ctx, k, inp_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };

            // ── Write new K and V into the cache ──────────────────────────────
            let k_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, k), hp.n_head_kv * hd, n_tokens) };
            let v_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, v), hp.n_head_kv * hd, n_tokens) };

            let k_cache = self.kv.k[layer];
            let v_cache = self.kv.v[layer];

            let k_write_view = unsafe { ffi::ggml_view_2d(ctx, k_cache,
                hp.n_head_kv * hd, n_tokens,
                kv_row_stride,
                kv_head as usize * kv_row_stride) };
            let v_write_view = unsafe { ffi::ggml_view_2d(ctx, v_cache,
                hp.n_head_kv * hd, n_tokens,
                kv_row_stride,
                kv_head as usize * kv_row_stride) };

            let k_cpy = unsafe { ffi::ggml_cpy(ctx, k_flat, k_write_view) };
            let v_cpy = unsafe { ffi::ggml_cpy(ctx, v_flat, v_write_view) };

            // CRITICAL: expand the copy nodes into the graph or they never run.
            unsafe {
                ffi::ggml_build_forward_expand(graph, k_cpy);
                ffi::ggml_build_forward_expand(graph, v_cpy);
            }

            // ── Read the full K/V history [0..kv_len] from cache ──────────────
            let k_read = unsafe { ffi::ggml_view_2d(ctx, k_cache,
                hp.n_head_kv * hd, kv_len,
                kv_row_stride, 0) };
            let v_read = unsafe { ffi::ggml_view_2d(ctx, v_cache,
                hp.n_head_kv * hd, kv_len,
                kv_row_stride, 0) };

            let k_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, k_read), hd, hp.n_head_kv, kv_len) };
            let v_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, v_read), hd, hp.n_head_kv, kv_len) };

            // ── Attention (same permute pattern as the Paris-verified code) ───
            let scale = 1.0 / (hd as f32).sqrt();
            let q  = unsafe { ffi::ggml_permute(ctx, q,    0, 2, 1, 3) };
            let k  = unsafe { ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, k_3d, 0, 2, 1, 3)) };
            let v  = unsafe { ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, v_3d, 1, 2, 0, 3)) };

            // Diagnostic: print shapes once (layer 0) so any mul_mat assert is
            // immediately explainable. Remove once stable.
            if layer == 0 {
                unsafe {
                    eprintln!("[Z.1 DIAG] L0 q={:?} k={:?} v={:?} (kv_head={} kv_len={} n_tokens={})",
                        tensor_dims(q), tensor_dims(k), tensor_dims(v),
                        kv_head, kv_len, n_tokens);
                }
            }

            let kq = unsafe { ffi::ggml_scale(ctx, ffi::ggml_mul_mat(ctx, k, q), scale) };
            let kq = unsafe { ffi::ggml_soft_max_ext(ctx, kq, std::ptr::null_mut(), 1.0, 0.0) };

            if layer == 0 {
                unsafe {
                    eprintln!("[Z.1 DIAG] L0 kq={:?}", tensor_dims(kq));
                }
            }

            let av = unsafe { ffi::ggml_mul_mat(ctx, v, kq) };
            let av = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, av, 0, 2, 1, 3)),
                hp.n_embd, n_tokens) };

            let attn = unsafe { ffi::ggml_mul_mat(ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(ctx, cur, attn) };

            // ── FFN SwiGLU ────────────────────────────────────────────────────
            let ffn_in = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("ffn_norm.weight")?) };
            let gate = unsafe { ffi::ggml_silu(ctx,
                ffi::ggml_mul_mat(ctx, t("ffn_gate.weight")?, ffn_in)) };
            let up   = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_up.weight")?, ffn_in) };
            let ffn  = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_down.weight")?,
                ffi::ggml_mul(ctx, gate, up)) };
            cur = unsafe { ffi::ggml_add(ctx, cur, ffn) };
        }

        // ── Final norm + lm_head ──────────────────────────────────────────────
        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight not found"))?;
        cur = unsafe { ffi::ggml_mul(ctx,
            ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), out_norm) };

        let last = unsafe { ffi::ggml_view_1d(ctx, cur, hp.n_embd,
            ((n_tokens - 1) * hp.n_embd * 4) as usize) };
        let lm_head = model.tensor("output.weight")
            .ok_or_else(|| anyhow::anyhow!("output.weight not found"))?;
        let logits = unsafe { ffi::ggml_mul_mat(ctx, lm_head, last) };
        unsafe { ffi::ggml_set_output(logits); }

        // ── Finish graph ──────────────────────────────────────────────────────
        unsafe { ffi::ggml_build_forward_expand(graph, logits) };

        // ── Allocate intermediate tensors via gallocr ─────────────────────────
        let buft   = unsafe { ffi::ggml_backend_cpu_buffer_type() };
        let galloc = unsafe { ffi::ggml_gallocr_new(buft) };
        if galloc.is_null() {
            unsafe { ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] ggml_gallocr_new failed");
        }

        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(galloc, graph) };
        if !ok {
            unsafe {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx);
                ffi::ggml_backend_buffer_free(inp_buf);
            }
            bail!("[Z.1 Graph] ggml_gallocr_alloc_graph failed");
        }
        log::debug!("[Z.1 Graph] Compute graph allocated via gallocr.");

        // ── Execute ───────────────────────────────────────────────────────────
        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, graph) };
        if status != 0 {
            unsafe {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx);
                ffi::ggml_backend_buffer_free(inp_buf);
            }
            bail!("[Z.1 Graph] compute failed status={}", status);
        }

        self.kv.head = kv_len.min(self.kv.n_ctx);

        // ── Read logits ───────────────────────────────────────────────────────
        let n_vocab = hp.n_vocab as usize;
        let mut out = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(
                logits, out.as_mut_ptr() as *mut c_void, 0, n_vocab * 4,
            );
        }

        unsafe {
            ffi::ggml_gallocr_free(galloc);
            ffi::ggml_free(ctx);
            ffi::ggml_free(inp_ctx);
            ffi::ggml_backend_buffer_free(inp_buf);
        }
        Ok(out)
    }

    pub fn reset_kv(&mut self) { self.kv.clear(); }

    // ── B1 Path: Prefill and single-token decode ──────────────────────────────

    /// Run the forward pass over the entire prompt and return logits for the
    /// last token position. Writes all prompt K/V into the cache.
    pub fn prefill(
        &mut self,
        token_ids: &[u32],
        model: &MappedModel,
    ) -> Result<Vec<f32>, ForwardError> {
        if token_ids.is_empty() {
            return Err(ForwardError::AllocationFailed("empty token sequence".into()));
        }
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        self.forward(model, &tokens, 0)
            .map_err(|e| ForwardError::AllocationFailed(e.to_string()))
    }

    /// Run the forward pass for a single new token and return its logits.
    /// Position is tracked automatically via self.kv.head.
    pub fn decode_one(
        &mut self,
        token_id: u32,
        model: &MappedModel,
    ) -> Result<Vec<f32>, ForwardError> {
        let pos = self.kv.head as i32;
        let tokens = vec![token_id as i32];
        self.forward(model, &tokens, pos)
            .map_err(|e| ForwardError::AllocationFailed(e.to_string()))
    }
}

impl Drop for LlamaGraph {
    fn drop(&mut self) {
        unsafe { ffi::ggml_backend_free(self.backend); }
        log::info!("[Z.1 Graph] dropped.");
    }
}

// ── Type alias for clarity ────────────────────────────────────────────────────

/// ForwardPass is an alias for LlamaGraph — the forward-inference engine.
pub type ForwardPass = LlamaGraph;
