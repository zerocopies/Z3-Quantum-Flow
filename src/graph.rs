/// Z.1 — graph.rs  [Graph-reuse edition]
///
/// Two execution paths:
///   prefill()    — full prompt; local graph rebuilt each call (once per generation)
///   decode_one() — single token; persistent pre-built graph, reused every step
///
/// KV cache tensors are allocated in a proper backend buffer so both
/// ggml_cpy (prefill) and ggml_backend_tensor_set (decode) can write to them.

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

// ── Tensor dims (b3534 layout: ne[4] at byte offset 16) ──────────────────────

unsafe fn tensor_dims(t: *const ffi::ggml_tensor) -> [i64; 4] {
    let p = (t as *const u8).add(16) as *const i64;
    [*p, *p.add(1), *p.add(2), *p.add(3)]
}

// ── Hyperparameters ───────────────────────────────────────────────────────────

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
// Tensors are allocated in a proper backend buffer (not ggml's internal pool).
// This is required so that both ggml_cpy (prefill, inside graph) and
// ggml_backend_tensor_set (decode, outside graph) can write to them.

pub struct KVCache {
    pub ctx:      *mut ffi::ggml_context,         // tensor descriptors
    pub data_buf: ffi::ggml_backend_buffer_t,     // actual data storage
    pub k:        Vec<*mut ffi::ggml_tensor>,
    pub v:        Vec<*mut ffi::ggml_tensor>,
    pub n_ctx:    i64,
    pub head:     i64,
}
unsafe impl Send for KVCache {}

impl KVCache {
    /// n_layers  — number of transformer layers
    /// n_head_kv — number of KV heads (e.g. 8 for GQA)
    /// head_dim  — dimension per head (e.g. 128)
    /// n_ctx     — maximum context length
    /// backend   — must be the same backend used by the inference graph
    pub fn new(
        n_layers:  usize,
        n_head_kv: i64,
        head_dim:  i64,
        n_ctx:     i64,
        backend:   ffi::ggml_backend_t,
    ) -> Self {
        let bytes_per  = (n_head_kv * head_dim * n_ctx * 4) as usize; // F32
        let total_data = bytes_per * n_layers * 2 + 1024;

        // Allocate data in a backend-registered buffer so buffer pointer is non-NULL
        let data_buf = unsafe { ffi::ggml_backend_alloc_buffer(backend, total_data) };
        let buf_base = unsafe { ffi::ggml_backend_buffer_get_base(data_buf) } as usize;

        // Small context for tensor descriptors only (no data in this context)
        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size:   (n_layers * 2 + 16) * 512,
                mem_buffer: std::ptr::null_mut(),
                no_alloc:   true,
            })
        };

        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            unsafe {
                let kt = ffi::ggml_new_tensor_2d(ctx, 0, n_head_kv * head_dim, n_ctx);
                ffi::ggml_backend_tensor_alloc(
                    data_buf, kt,
                    (buf_base + i * bytes_per) as *mut c_void,
                );
                k.push(kt);

                let vt = ffi::ggml_new_tensor_2d(ctx, 0, n_head_kv * head_dim, n_ctx);
                ffi::ggml_backend_tensor_alloc(
                    data_buf, vt,
                    (buf_base + n_layers * bytes_per + i * bytes_per) as *mut c_void,
                );
                v.push(vt);
            }
        }

        // Zero-initialise so unused positions are benign
        let zeros = vec![0.0f32; (n_head_kv * head_dim * n_ctx) as usize];
        for i in 0..n_layers {
            unsafe {
                ffi::ggml_backend_tensor_set(k[i],
                    zeros.as_ptr() as *const c_void, 0, bytes_per);
                ffi::ggml_backend_tensor_set(v[i],
                    zeros.as_ptr() as *const c_void, 0, bytes_per);
            }
        }

        KVCache { ctx, data_buf, k, v, n_ctx, head: 0 }
    }

    pub fn clear(&mut self) { self.head = 0; }
}

impl Drop for KVCache {
    fn drop(&mut self) {
        unsafe {
            ffi::ggml_free(self.ctx);
            ffi::ggml_backend_buffer_free(self.data_buf);
        }
    }
}

// ── Main graph struct ─────────────────────────────────────────────────────────

pub struct LlamaGraph {
    pub hp:      LlamaHparams,
    pub backend: ffi::ggml_backend_t,
    pub kv:      KVCache,

    // ── Persistent decode graph (built once on first decode_one call) ─────────
    d_built:   bool,
    d_inp_ctx: *mut ffi::ggml_context,        // keeps input tensor descriptors alive
    d_ctx:     *mut ffi::ggml_context,        // compute graph context
    d_graph:   *mut ffi::ggml_cgraph,
    d_galloc:  ffi::ggml_gallocr_t,
    d_inp_buf: ffi::ggml_backend_buffer_t,
    d_token:   *mut ffi::ggml_tensor,         // [1] i32
    d_pos:     *mut ffi::ggml_tensor,         // [1] i32
    d_mask:    *mut ffi::ggml_tensor,         // [n_ctx, 1] f32
    d_logits:  *mut ffi::ggml_tensor,         // [n_vocab] f32
    d_k_out:   Vec<*mut ffi::ggml_tensor>,    // per-layer K for cache write
    d_v_out:   Vec<*mut ffi::ggml_tensor>,    // per-layer V for cache write
}
unsafe impl Send for LlamaGraph {}

impl LlamaGraph {
    pub fn new(model: &MappedModel) -> Result<Self> {
        let hp      = LlamaHparams::from_model(model);
        let backend = unsafe { ffi::ggml_backend_cpu_init() };
        if backend.is_null() { bail!("[Z.1 Graph] ggml_backend_cpu_init failed"); }
        log::info!("[Z.1 Graph] layers={} embd={} heads={}/{} ff={} rot={} freq_base={}",
            hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_ff, hp.n_rot, hp.freq_base);
        // Pass backend so KV tensors land in a backend-registered buffer
        let kv = KVCache::new(
            hp.n_layer as usize, hp.n_head_kv, hp.head_dim(), 512, backend,
        );
        Ok(LlamaGraph {
            hp, backend, kv,
            d_built:   false,
            d_inp_ctx: std::ptr::null_mut(),
            d_ctx:     std::ptr::null_mut(),
            d_graph:   std::ptr::null_mut(),
            d_galloc:  std::ptr::null_mut(),
            d_inp_buf: std::ptr::null_mut(),
            d_token:   std::ptr::null_mut(),
            d_pos:     std::ptr::null_mut(),
            d_mask:    std::ptr::null_mut(),
            d_logits:  std::ptr::null_mut(),
            d_k_out:   Vec::new(),
            d_v_out:   Vec::new(),
        })
    }

    // ── Build the persistent single-token decode graph ────────────────────────
    //
    // n_tokens=1, kv_len=n_ctx (fixed). d_mask gates which cache positions
    // are attended to. K/V for the new token are output tensors — written to
    // the KV cache by Rust after each execution step.

    fn build_decode_graph(&mut self, model: &MappedModel) -> Result<()> {
        let hp            = &self.hp;
        let n_ctx         = self.kv.n_ctx;
        let hd            = hp.head_dim();
        let n_tokens      = 1i64;
        let kv_row_stride = (hp.n_head_kv * hd * 4) as usize;

        // Input buffer: token(4B) + gap + pos(4B) + gap + mask(n_ctx×4B)
        let mask_bytes   = (n_ctx * 4) as usize;
        let inp_buf_size = 128 + mask_bytes + 256;
        let inp_buf = unsafe { ffi::ggml_backend_alloc_buffer(self.backend, inp_buf_size) };
        if inp_buf.is_null() { bail!("[Z.1 Decode] input buffer alloc failed"); }
        let inp_base = unsafe { ffi::ggml_backend_buffer_get_base(inp_buf) } as usize;

        let inp_ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 8192, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if inp_ctx.is_null() {
            unsafe { ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Decode] inp_ctx alloc failed");
        }

        let d_token = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(inp_buf, d_token, inp_base as *mut c_void); }

        let d_pos = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(inp_buf, d_pos, (inp_base + 64) as *mut c_void); }

        // mask [n_ctx, 1] f32 — added to kq inside soft_max_ext
        let d_mask = unsafe { ffi::ggml_new_tensor_2d(inp_ctx, 0, n_ctx, 1) };
        unsafe { ffi::ggml_backend_tensor_alloc(inp_buf, d_mask, (inp_base + 128) as *mut c_void); }

        // Initialise mask: all positions invalid
        let neg_inf = vec![-10000.0f32; n_ctx as usize];
        unsafe {
            ffi::ggml_backend_tensor_set(d_mask,
                neg_inf.as_ptr() as *const c_void, 0, mask_bytes);
        }

        // Compute graph context — persistent for the session
        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 128 * 1024 * 1024, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Decode] graph ctx alloc failed");
        }

        let graph = unsafe { ffi::ggml_new_graph_custom(ctx, 16384, false) };

        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(ctx, embd_w, d_token) };

        let mut d_k_out = Vec::with_capacity(hp.n_layer as usize);
        let mut d_v_out = Vec::with_capacity(hp.n_layer as usize);

        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} not found", layer, name))
            };

            // Attention norm + QKV projections
            let normed = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("attn_norm.weight")?) };
            let q = unsafe { ffi::ggml_mul_mat(ctx, t("attn_q.weight")?, normed) };
            let k = unsafe { ffi::ggml_mul_mat(ctx, t("attn_k.weight")?, normed) };
            let v = unsafe { ffi::ggml_mul_mat(ctx, t("attn_v.weight")?, normed) };

            let q = unsafe { ffi::ggml_reshape_3d(ctx, q, hd, hp.n_head,    n_tokens) };
            let k = unsafe { ffi::ggml_reshape_3d(ctx, k, hd, hp.n_head_kv, n_tokens) };
            let v = unsafe { ffi::ggml_reshape_3d(ctx, v, hd, hp.n_head_kv, n_tokens) };

            // RoPE
            let q = unsafe { ffi::ggml_rope_ext(ctx, q, d_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };
            let k = unsafe { ffi::ggml_rope_ext(ctx, k, d_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };

            // K/V for current token — output tensors written to cache after execution
            let k_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, k), hp.n_head_kv * hd, n_tokens) };
            let v_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, v), hp.n_head_kv * hd, n_tokens) };
            unsafe {
                ffi::ggml_set_output(k_flat);
                ffi::ggml_set_output(v_flat);
                ffi::ggml_build_forward_expand(graph, k_flat);
                ffi::ggml_build_forward_expand(graph, v_flat);
            }
            d_k_out.push(k_flat);
            d_v_out.push(v_flat);

            // Read full KV cache [n_head_kv*hd, n_ctx] — fixed shape every step
            let k_cache = self.kv.k[layer];
            let v_cache = self.kv.v[layer];
            let k_full = unsafe { ffi::ggml_view_2d(ctx, k_cache,
                hp.n_head_kv * hd, n_ctx, kv_row_stride, 0) };
            let v_full = unsafe { ffi::ggml_view_2d(ctx, v_cache,
                hp.n_head_kv * hd, n_ctx, kv_row_stride, 0) };

            let k_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, k_full), hd, hp.n_head_kv, n_ctx) };
            let v_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, v_full), hd, hp.n_head_kv, n_ctx) };

            // Attention — mask applied inside soft_max_ext
            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe { ffi::ggml_cont(ctx,
                ffi::ggml_permute(ctx, k_3d, 0, 2, 1, 3)) };
            let v_perm = unsafe { ffi::ggml_cont(ctx,
                ffi::ggml_permute(ctx, v_3d, 1, 2, 0, 3)) };

            let kq = unsafe { ffi::ggml_scale(ctx,
                ffi::ggml_mul_mat(ctx, k_perm, q_perm), scale) };
            let kq = unsafe { ffi::ggml_soft_max_ext(ctx, kq, d_mask, 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(ctx, v_perm, kq) };
            let av = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, av, 0, 2, 1, 3)),
                hp.n_embd, n_tokens) };

            let attn = unsafe { ffi::ggml_mul_mat(ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(ctx, cur, attn) };

            // FFN SwiGLU
            let ffn_in = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("ffn_norm.weight")?) };
            let gate = unsafe { ffi::ggml_silu(ctx,
                ffi::ggml_mul_mat(ctx, t("ffn_gate.weight")?, ffn_in)) };
            let up   = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_up.weight")?, ffn_in) };
            let ffn  = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_down.weight")?,
                ffi::ggml_mul(ctx, gate, up)) };
            cur = unsafe { ffi::ggml_add(ctx, cur, ffn) };
        }

        // Final norm + lm_head
        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight not found"))?;
        cur = unsafe { ffi::ggml_mul(ctx,
            ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), out_norm) };
        let lm_head = model.tensor("output.weight")
            .ok_or_else(|| anyhow::anyhow!("output.weight not found"))?;
        let d_logits = unsafe { ffi::ggml_mul_mat(ctx, lm_head, cur) };
        unsafe {
            ffi::ggml_set_output(d_logits);
            ffi::ggml_build_forward_expand(graph, d_logits);
        }

        // Allocate intermediates — one-time cost
        let buft   = unsafe { ffi::ggml_backend_cpu_buffer_type() };
        let galloc = unsafe { ffi::ggml_gallocr_new(buft) };
        if galloc.is_null() {
            unsafe { ffi::ggml_free(inp_ctx); ffi::ggml_free(ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Decode] gallocr_new failed");
        }
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(galloc, graph) };
        if !ok {
            unsafe {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(inp_ctx); ffi::ggml_free(ctx);
                ffi::ggml_backend_buffer_free(inp_buf);
            }
            bail!("[Z.1 Decode] gallocr_alloc_graph failed");
        }

        self.d_inp_ctx = inp_ctx;
        self.d_ctx     = ctx;
        self.d_graph   = graph;
        self.d_galloc  = galloc;
        self.d_inp_buf = inp_buf;
        self.d_token   = d_token;
        self.d_pos     = d_pos;
        self.d_mask    = d_mask;
        self.d_logits  = d_logits;
        self.d_k_out   = d_k_out;
        self.d_v_out   = d_v_out;
        self.d_built   = true;

        log::info!("[Z.1 Decode] persistent decode graph built (n_ctx={}, layers={})",
            n_ctx, hp.n_layer);
        Ok(())
    }

    // ── Execute the persistent decode graph for one token ─────────────────────

    fn execute_decode(&mut self, token_id: u32, model: &MappedModel) -> Result<Vec<f32>> {
        if !self.d_built {
            self.build_decode_graph(model)?;
        }

        let hp            = &self.hp;
        let hd            = hp.head_dim();
        let n_ctx         = self.kv.n_ctx;
        let kv_head       = self.kv.head;
        let kv_row_stride = (hp.n_head_kv * hd * 4) as usize;

        if kv_head >= n_ctx {
            bail!("[Z.1 Decode] KV cache full ({}/{})", kv_head, n_ctx);
        }

        // Update inputs
        let tok = token_id as i32;
        unsafe { ffi::ggml_backend_tensor_set(self.d_token,
            &tok as *const i32 as *const c_void, 0, 4); }

        let pos = kv_head as i32;
        unsafe { ffi::ggml_backend_tensor_set(self.d_pos,
            &pos as *const i32 as *const c_void, 0, 4); }

        // Update mask: first kv_head positions valid (0.0), rest masked (-1e4)
        let mask_bytes = (n_ctx * 4) as usize;
        let mut mask = vec![-10000.0f32; n_ctx as usize];
        for i in 0..kv_head as usize { mask[i] = 0.0; }
        unsafe { ffi::ggml_backend_tensor_set(self.d_mask,
            mask.as_ptr() as *const c_void, 0, mask_bytes); }

        // Execute — no graph rebuild, no realloc
        let status = unsafe {
            ffi::ggml_backend_graph_compute(self.backend, self.d_graph)
        };
        if status != 0 { bail!("[Z.1 Decode] compute failed status={}", status); }

        // Write K/V for this token into the cache at position kv_head
        let kv_elem = (hp.n_head_kv * hd) as usize;
        let mut buf = vec![0.0f32; kv_elem];
        for layer in 0..hp.n_layer as usize {
            unsafe {
                ffi::ggml_backend_tensor_get(self.d_k_out[layer],
                    buf.as_mut_ptr() as *mut c_void, 0, kv_row_stride);
                ffi::ggml_backend_tensor_set(self.kv.k[layer],
                    buf.as_ptr() as *const c_void,
                    kv_head as usize * kv_row_stride, kv_row_stride);

                ffi::ggml_backend_tensor_get(self.d_v_out[layer],
                    buf.as_mut_ptr() as *mut c_void, 0, kv_row_stride);
                ffi::ggml_backend_tensor_set(self.kv.v[layer],
                    buf.as_ptr() as *const c_void,
                    kv_head as usize * kv_row_stride, kv_row_stride);
            }
        }

        self.kv.head = (kv_head + 1).min(n_ctx);

        // Read logits
        let n_vocab = hp.n_vocab as usize;
        let mut out = vec![0.0f32; n_vocab];
        unsafe { ffi::ggml_backend_tensor_get(self.d_logits,
            out.as_mut_ptr() as *mut c_void, 0, n_vocab * 4); }

        // ── DIAGNOSTIC: decode step state + argmax ────────────────────────────
        {
            let (amax, mval) = out.iter().enumerate()
                .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
            // First few elements of mask, to verify it's not all -10000
            let mut mask_sample = vec![0.0f32; 4];
            unsafe { ffi::ggml_backend_tensor_get(self.d_mask,
                mask_sample.as_mut_ptr() as *mut c_void, 0, 16); }
            eprintln!("[Z.1 DIAG] decode: token_in={} kv_head={} pos={} new_kv_head={} mask[0..4]={:?} argmax={} val={:.3}",
                tok, kv_head, pos, self.kv.head, mask_sample, amax, mval);
        }

        Ok(out)
    }

    // ── Prefill: full prompt, local graph (rebuilt each call) ─────────────────

    pub fn forward(&mut self, model: &MappedModel, tokens: &[i32], pos: i32)
        -> Result<Vec<f32>>
    {
        let hp        = &self.hp;
        let n_tokens  = tokens.len() as i64;
        let tok_bytes = n_tokens as usize * 4;

        let inp_buf = unsafe {
            ffi::ggml_backend_alloc_buffer(self.backend, tok_bytes * 2 + 256)
        };
        if inp_buf.is_null() { bail!("[Z.1 Graph] ggml_backend_alloc_buffer failed"); }
        let inp_base = unsafe { ffi::ggml_backend_buffer_get_base(inp_buf) } as usize;

        let inp_ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 4096, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if inp_ctx.is_null() {
            unsafe { ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] inp_ctx alloc failed");
        }

        let inp_tokens = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26, n_tokens) };
        unsafe {
            ffi::ggml_backend_tensor_alloc(inp_buf, inp_tokens, inp_base as *mut c_void);
            ffi::ggml_backend_tensor_set(inp_tokens,
                tokens.as_ptr() as *const c_void, 0, tok_bytes);
        }

        let pos_offset = (tok_bytes + 63) & !63;
        let positions: Vec<i32> = (pos..pos + n_tokens as i32).collect();
        let inp_pos = unsafe { ffi::ggml_new_tensor_1d(inp_ctx, 26, n_tokens) };
        unsafe {
            ffi::ggml_backend_tensor_alloc(inp_buf, inp_pos,
                (inp_base + pos_offset) as *mut c_void);
            ffi::ggml_backend_tensor_set(inp_pos,
                positions.as_ptr() as *const c_void, 0, tok_bytes);
        }

        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size: 64 * 1024 * 1024, mem_buffer: std::ptr::null_mut(), no_alloc: true,
            })
        };
        if ctx.is_null() {
            unsafe { ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] graph ctx alloc failed");
        }

        let graph = unsafe { ffi::ggml_new_graph_custom(ctx, 8192, false) };

        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(ctx, embd_w, inp_tokens) };

        let hd            = hp.head_dim();
        let kv_head       = self.kv.head;
        let kv_len        = kv_head + n_tokens;
        let kv_row_stride = (hp.n_head_kv * hd * 4) as usize;

        if kv_len > self.kv.n_ctx {
            unsafe {
                ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx);
                ffi::ggml_backend_buffer_free(inp_buf);
            }
            bail!("[Z.1 Graph] KV cache overflow: {} > {}", kv_len, self.kv.n_ctx);
        }

        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} not found", layer, name))
            };

            let normed = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("attn_norm.weight")?) };
            let q = unsafe { ffi::ggml_mul_mat(ctx, t("attn_q.weight")?, normed) };
            let k = unsafe { ffi::ggml_mul_mat(ctx, t("attn_k.weight")?, normed) };
            let v = unsafe { ffi::ggml_mul_mat(ctx, t("attn_v.weight")?, normed) };

            let q = unsafe { ffi::ggml_reshape_3d(ctx, q, hd, hp.n_head,    n_tokens) };
            let k = unsafe { ffi::ggml_reshape_3d(ctx, k, hd, hp.n_head_kv, n_tokens) };
            let v = unsafe { ffi::ggml_reshape_3d(ctx, v, hd, hp.n_head_kv, n_tokens) };

            let q = unsafe { ffi::ggml_rope_ext(ctx, q, inp_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };
            let k = unsafe { ffi::ggml_rope_ext(ctx, k, inp_pos, std::ptr::null_mut(),
                hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0) };

            // Write K/V into cache via ggml_cpy
            let k_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, k), hp.n_head_kv * hd, n_tokens) };
            let v_flat = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, v), hp.n_head_kv * hd, n_tokens) };

            let k_write = unsafe { ffi::ggml_view_2d(ctx, self.kv.k[layer],
                hp.n_head_kv * hd, n_tokens, kv_row_stride,
                kv_head as usize * kv_row_stride) };
            let v_write = unsafe { ffi::ggml_view_2d(ctx, self.kv.v[layer],
                hp.n_head_kv * hd, n_tokens, kv_row_stride,
                kv_head as usize * kv_row_stride) };

            let k_cpy = unsafe { ffi::ggml_cpy(ctx, k_flat, k_write) };
            let v_cpy = unsafe { ffi::ggml_cpy(ctx, v_flat, v_write) };
            unsafe {
                ffi::ggml_build_forward_expand(graph, k_cpy);
                ffi::ggml_build_forward_expand(graph, v_cpy);
            }

            // Read K/V history [0..kv_len]
            let k_read = unsafe { ffi::ggml_view_2d(ctx, self.kv.k[layer],
                hp.n_head_kv * hd, kv_len, kv_row_stride, 0) };
            let v_read = unsafe { ffi::ggml_view_2d(ctx, self.kv.v[layer],
                hp.n_head_kv * hd, kv_len, kv_row_stride, 0) };

            let k_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, k_read), hd, hp.n_head_kv, kv_len) };
            let v_3d = unsafe { ffi::ggml_reshape_3d(ctx,
                ffi::ggml_cont(ctx, v_read), hd, hp.n_head_kv, kv_len) };

            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe { ffi::ggml_cont(ctx,
                ffi::ggml_permute(ctx, k_3d, 0, 2, 1, 3)) };
            let v_perm = unsafe { ffi::ggml_cont(ctx,
                ffi::ggml_permute(ctx, v_3d, 1, 2, 0, 3)) };

            let kq = unsafe { ffi::ggml_scale(ctx,
                ffi::ggml_mul_mat(ctx, k_perm, q_perm), scale) };
            let kq = unsafe { ffi::ggml_soft_max_ext(ctx, kq,
                std::ptr::null_mut(), 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(ctx, v_perm, kq) };
            let av = unsafe { ffi::ggml_reshape_2d(ctx,
                ffi::ggml_cont(ctx, ffi::ggml_permute(ctx, av, 0, 2, 1, 3)),
                hp.n_embd, n_tokens) };

            let attn = unsafe { ffi::ggml_mul_mat(ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(ctx, cur, attn) };

            let ffn_in = unsafe { ffi::ggml_mul(ctx,
                ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), t("ffn_norm.weight")?) };
            let gate = unsafe { ffi::ggml_silu(ctx,
                ffi::ggml_mul_mat(ctx, t("ffn_gate.weight")?, ffn_in)) };
            let up   = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_up.weight")?, ffn_in) };
            let ffn  = unsafe { ffi::ggml_mul_mat(ctx, t("ffn_down.weight")?,
                ffi::ggml_mul(ctx, gate, up)) };
            cur = unsafe { ffi::ggml_add(ctx, cur, ffn) };
        }

        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight not found"))?;
        cur = unsafe { ffi::ggml_mul(ctx,
            ffi::ggml_rms_norm(ctx, cur, hp.rms_norm_eps), out_norm) };

        let last = unsafe { ffi::ggml_view_1d(ctx, cur, hp.n_embd,
            ((n_tokens - 1) * hp.n_embd * 4) as usize) };
        let lm_head = model.tensor("output.weight")
            .ok_or_else(|| anyhow::anyhow!("output.weight not found"))?;
        let logits = unsafe { ffi::ggml_mul_mat(ctx, lm_head, last) };
        unsafe {
            ffi::ggml_set_output(logits);
            ffi::ggml_build_forward_expand(graph, logits);
        }

        let buft   = unsafe { ffi::ggml_backend_cpu_buffer_type() };
        let galloc = unsafe { ffi::ggml_gallocr_new(buft) };
        if galloc.is_null() {
            unsafe { ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx); ffi::ggml_backend_buffer_free(inp_buf); }
            bail!("[Z.1 Graph] gallocr_new failed");
        }
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(galloc, graph) };
        if !ok {
            unsafe {
                ffi::ggml_gallocr_free(galloc);
                ffi::ggml_free(ctx); ffi::ggml_free(inp_ctx);
                ffi::ggml_backend_buffer_free(inp_buf);
            }
            bail!("[Z.1 Graph] gallocr_alloc_graph failed");
        }

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

        let n_vocab = hp.n_vocab as usize;
        let mut out = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(logits,
                out.as_mut_ptr() as *mut c_void, 0, n_vocab * 4);
        }

        // ── DIAGNOSTIC: argmax of prefill output ──────────────────────────────
        {
            let (amax, mval) = out.iter().enumerate()
                .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
            eprintln!("[Z.1 DIAG] prefill done: kv_head_before={} kv_len={} new_kv_head={} argmax={} val={:.3}",
                kv_head, kv_len, self.kv.head, amax, mval);
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

    pub fn prefill(&mut self, token_ids: &[u32], model: &MappedModel)
        -> Result<Vec<f32>, ForwardError>
    {
        if token_ids.is_empty() {
            return Err(ForwardError::AllocationFailed("empty token sequence".into()));
        }
        let tokens: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        self.forward(model, &tokens, 0)
            .map_err(|e| ForwardError::AllocationFailed(e.to_string()))
    }

    pub fn decode_one(&mut self, token_id: u32, model: &MappedModel)
        -> Result<Vec<f32>, ForwardError>
    {
        self.execute_decode(token_id, model)
            .map_err(|e| ForwardError::AllocationFailed(e.to_string()))
    }
}

impl Drop for LlamaGraph {
    fn drop(&mut self) {
        if self.d_built {
            unsafe {
                ffi::ggml_gallocr_free(self.d_galloc);
                ffi::ggml_free(self.d_ctx);
                ffi::ggml_free(self.d_inp_ctx);
                ffi::ggml_backend_buffer_free(self.d_inp_buf);
            }
        }
        unsafe { ffi::ggml_backend_free(self.backend); }
        log::info!("[Z.1 Graph] dropped.");
    }
}

pub type ForwardPass = LlamaGraph;
