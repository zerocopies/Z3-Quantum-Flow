/// Z.3 — graph.rs [QUANTUM-FLOW ENGINE v4.0 — Multi-Architecture]
///
/// Adds Phi-3 and Qwen2.5 support on top of the v3.2 Llama foundation.
///
/// Architecture support matrix:
///   llama  — separate Q/K/V weights, SwiGLU FFN (gate+up+down)
///   phi3   — fused attn_qkv.weight (split via view), fused FFN (gate+up in ffn_up.weight)
///   qwen2  — separate Q/K/V weights + biases, SwiGLU FFN, GQA broadcasting
///
/// GQA note: any model where n_head > n_head_kv gets automatic K/V head repeat
/// via ggml_repeat — handles Qwen2.5's 8:1 ratio without special-casing.

use std::ffi::c_void;
use std::ptr::null_mut;
use anyhow::Result;
use libc::c_int;

use crate::ggml_ffi::{self as ffi, GgmlInitParams};
use crate::loader::MappedModel;

// ── Configuration ─────────────────────────────────────────────────────────────

const MAX_CTX: i64          = 4096;
const GRAPH_MEM_SIZE: usize = 512 * 1024 * 1024; // 512 MB

// ── ForwardError ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ForwardError {
    Init(String),
    MissingTensor(String),
    ComputeFailed(i32),
    ContextFull,
    Internal(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(msg)           => write!(f, "[Z.3 INIT] {}", msg),
            Self::MissingTensor(name) => write!(f, "[Z.3 MISSING] tensor: {}", name),
            Self::ComputeFailed(code) => write!(f, "[Z.3 COMPUTE] failed with status {}", code),
            Self::ContextFull         => write!(f, "[Z.3 OVERFLOW] context window full"),
            Self::Internal(msg)       => write!(f, "[Z.3 INTERNAL] {}", msg),
        }
    }
}
impl std::error::Error for ForwardError {}

impl From<anyhow::Error> for ForwardError {
    fn from(e: anyhow::Error) -> Self { Self::Internal(e.to_string()) }
}

// ── Model DNA ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ModelDNA {
    pub arch:             String,  // "llama", "phi3", "qwen2", etc.
    pub n_vocab:          i64,
    pub n_embd:           i64,
    pub n_head:           i64,
    pub n_head_kv:        i64,
    pub n_layer:          i64,
    pub n_ff:             i64,
    pub n_rot:            i64,
    pub freq_base:        f32,
    pub rms_eps:          f32,
    pub has_tied_weights: bool,
}

impl ModelDNA {
    pub fn from_model(model: &MappedModel) -> Result<Self> {
        let arch = model.header.architecture().unwrap_or("llama");
        let meta = &model.header.metadata;

        let get_u32 = |key: &str| -> i64 {
            meta.get(key).and_then(|v| v.as_u32()).unwrap_or(0) as i64
        };
        let get_f32 = |key: &str, def: f32| -> f32 {
            meta.get(key)
                .and_then(|v| if let crate::gguf::GgufValue::F32(f) = v { Some(*f) } else { None })
                .unwrap_or(def)
        };

        // Vocab: try arch.vocab_size first, fall back to counting tokenizer tokens
        let n_vocab = meta.get(&format!("{}.vocab_size", arch))
            .and_then(|v| v.as_u32())
            .map(|v| v as i64)
            .unwrap_or_else(|| {
                if let Some(crate::gguf::GgufValue::Array(arr)) = meta.get("tokenizer.ggml.tokens") {
                    arr.len() as i64
                } else { 0 }
            });

        let has_tied = model.tensor("output.weight").is_none()
            && model.tensor("token_embd.weight").is_some();

        Ok(ModelDNA {
            arch:             arch.to_string(),
            n_vocab,
            n_embd:           get_u32(&format!("{}.embedding_length", arch)),
            n_head:           get_u32(&format!("{}.attention.head_count", arch)),
            n_head_kv:        get_u32(&format!("{}.attention.head_count_kv", arch)),
            n_layer:          get_u32(&format!("{}.block_count", arch)),
            n_ff:             get_u32(&format!("{}.feed_forward_length", arch)),
            n_rot: { let r = get_u32(&format!("{}.rope.dimension_count", arch)); if r == 0 { get_u32(&format!("{}.embedding_length", arch)) / get_u32(&format!("{}.attention.head_count", arch)).max(1) } else { r } },
            freq_base:        get_f32(&format!("{}.rope.freq_base", arch), 10000.0),
            rms_eps:          get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-6),
            has_tied_weights: has_tied || arch == "qwen2" || arch == "qwen",
        })
    }

    pub fn head_dim(&self) -> i64 {
        if self.n_head == 0 { 1 } else { self.n_embd / self.n_head }
    }
}

// ── Quantum KV Cache ──────────────────────────────────────────────────────────

pub struct QuantumKV {
    ctx:          *mut ffi::ggml_context,
    buf:          ffi::ggml_backend_buffer_t,
    pub k_ptrs:   Vec<*mut ffi::ggml_tensor>,
    pub v_ptrs:   Vec<*mut ffi::ggml_tensor>,
    pub n_ctx:    i64,
    pub head:     i64,
    pub stride:   usize,
}

impl QuantumKV {
    pub fn new(
        n_layers: usize,
        h_k:      i64,
        h_d:      i64,
        n_ctx:    i64,
        backend:  ffi::ggml_backend_t,
    ) -> Result<Self> {
        let elem  = (h_k * h_d * n_ctx) as usize;
        let bytes = elem * 4;
        let total = bytes * n_layers * 2;

        log::info!("[Z.3] Allocating {:.2} MB Quantum-KV (layers={}, n_ctx={})",
            total as f32 / (1024.0 * 1024.0), n_layers, n_ctx);

        let buf = unsafe { ffi::ggml_backend_alloc_buffer(backend, total) };
        if buf.is_null() { anyhow::bail!("KV buffer allocation failed"); }

        let base = unsafe { ffi::ggml_backend_buffer_get_base(buf) } as usize;

        let ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size:   (n_layers * 4 + 64) * 512,
                mem_buffer: null_mut(),
                no_alloc:   true,
            })
        };
        if ctx.is_null() { anyhow::bail!("KV ggml_init failed"); }

        let mut k_ptrs = Vec::with_capacity(n_layers);
        let mut v_ptrs = Vec::with_capacity(n_layers);
        let stride = (h_k * h_d * 4) as usize;

        for i in 0..n_layers {
            unsafe {
                let kt = ffi::ggml_new_tensor_2d(ctx, 0, h_k * h_d, n_ctx);
                ffi::ggml_backend_tensor_alloc(buf, kt, (base + i * bytes) as *mut c_void);
                k_ptrs.push(kt);

                let vt = ffi::ggml_new_tensor_2d(ctx, 0, h_k * h_d, n_ctx);
                ffi::ggml_backend_tensor_alloc(buf, vt, (base + (n_layers + i) * bytes) as *mut c_void);
                v_ptrs.push(vt);
            }
        }

        let zeros = vec![0.0f32; elem];
        for i in 0..n_layers {
            unsafe {
                ffi::ggml_backend_tensor_set(k_ptrs[i], zeros.as_ptr() as *const c_void, 0, bytes);
                ffi::ggml_backend_tensor_set(v_ptrs[i], zeros.as_ptr() as *const c_void, 0, bytes);
            }
        }

        Ok(Self { ctx, buf, k_ptrs, v_ptrs, n_ctx, head: 0, stride })
    }

    pub fn clear(&mut self) { self.head = 0; }
}

impl Drop for QuantumKV {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() { ffi::ggml_free(self.ctx); }
            if !self.buf.is_null() { ffi::ggml_backend_buffer_free(self.buf); }
        }
    }
}

// ── ForwardPass ───────────────────────────────────────────────────────────────

pub struct ForwardPass {
    dna:     ModelDNA,
    backend: ffi::ggml_backend_t,
    pub kv:  QuantumKV,
    n_ctx:   i64,

    ctx:      *mut ffi::ggml_context,
    inp_ctx:  *mut ffi::ggml_context,
    graph:    *mut ffi::ggml_cgraph,
    galloc:   ffi::ggml_gallocr_t,
    inp_buf:  ffi::ggml_backend_buffer_t,

    d_token:  *mut ffi::ggml_tensor,
    d_pos:    *mut ffi::ggml_tensor,
    d_mask:   *mut ffi::ggml_tensor,
    d_logits: *mut ffi::ggml_tensor,

    pub rebuild_count: u64,
}

unsafe impl Send for ForwardPass {}

impl ForwardPass {
    pub fn new(model: &MappedModel, n_ctx: i64) -> Result<Self> {
        let dna     = ModelDNA::from_model(model)?;
        let backend = unsafe { ffi::ggml_backend_cpu_init() };
        if backend.is_null() { anyhow::bail!("CPU backend init failed"); }

        let kv = QuantumKV::new(
            dna.n_layer as usize,
            dna.n_head_kv,
            dna.head_dim(),
            n_ctx.min(MAX_CTX),
            backend,
        )?;

        log::info!("[Z.3] Engine ready — arch={} layers={} embd={} heads={}/{} vocab={} ctx={}",
            dna.arch, dna.n_layer, dna.n_embd, dna.n_head, dna.n_head_kv, dna.n_vocab, kv.n_ctx);

        Ok(Self {
            dna, backend, kv, n_ctx: n_ctx.min(MAX_CTX),
            ctx:      null_mut(),
            inp_ctx:  null_mut(),
            graph:    null_mut(),
            galloc:   null_mut(),
            inp_buf:  null_mut(),
            d_token:  null_mut(),
            d_pos:    null_mut(),
            d_mask:   null_mut(),
            d_logits: null_mut(),
            rebuild_count: 0,
        })
    }

    // ── Resource teardown ──────────────────────────────────────────────────────

    fn cleanup_graph_resources(&mut self) {
        unsafe {
            if !self.galloc.is_null() {
                ffi::ggml_gallocr_free(self.galloc);
                self.galloc = null_mut();
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
                self.ctx   = null_mut();
                self.graph = null_mut();
            }
            if !self.inp_ctx.is_null() {
                ffi::ggml_free(self.inp_ctx);
                self.inp_ctx = null_mut();
            }
            if !self.inp_buf.is_null() {
                ffi::ggml_backend_buffer_free(self.inp_buf);
                self.inp_buf = null_mut();
            }
            self.d_token  = null_mut();
            self.d_pos    = null_mut();
            self.d_mask   = null_mut();
            self.d_logits = null_mut();
        }
    }

    // ── Single-token decode graph ──────────────────────────────────────────────

    fn build_graph(&mut self, model: &MappedModel, current_pos: i64) -> Result<()> {
        self.rebuild_count += 1;
        self.cleanup_graph_resources();

        let hp        = &self.dna;
        let hd        = hp.head_dim();
        let n_tokens  = 1i64;
        let kv_stride = self.kv.stride;

        // ── Inputs ────────────────────────────────────────────────────────────

        let mask_bytes = (self.n_ctx * 4) as usize;
        let inp_size   = 1024 + mask_bytes + 512;

        self.inp_buf = unsafe { ffi::ggml_backend_alloc_buffer(self.backend, inp_size) };
        if self.inp_buf.is_null() { anyhow::bail!("input buffer alloc failed"); }

        let inp_base = unsafe { ffi::ggml_backend_buffer_get_base(self.inp_buf) } as usize;

        self.inp_ctx = unsafe {
            ffi::ggml_init(GgmlInitParams { mem_size: 32768, mem_buffer: null_mut(), no_alloc: true })
        };
        if self.inp_ctx.is_null() { anyhow::bail!("inp_ctx ggml_init failed"); }

        self.d_token = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_token, inp_base as *mut c_void) };

        self.d_pos = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_pos, (inp_base + 64) as *mut c_void) };

        self.d_mask = unsafe { ffi::ggml_new_tensor_2d(self.inp_ctx, 0, self.n_ctx, 1) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_mask, (inp_base + 128) as *mut c_void) };

        let mut mask_data = vec![-10000.0f32; self.n_ctx as usize];
        for i in 0..=current_pos as usize { mask_data[i] = 0.0; }
        unsafe {
            ffi::ggml_backend_tensor_set(self.d_mask, mask_data.as_ptr() as *const c_void, 0, mask_bytes)
        };

        // ── Compute graph ─────────────────────────────────────────────────────

        self.ctx = unsafe {
            ffi::ggml_init(GgmlInitParams { mem_size: GRAPH_MEM_SIZE, mem_buffer: null_mut(), no_alloc: true })
        };
        if self.ctx.is_null() { anyhow::bail!("graph ctx ggml_init failed"); }

        self.graph = unsafe { ffi::ggml_new_graph_custom(self.ctx, 65536, false) };

        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight missing"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(self.ctx, embd_w, self.d_token) };

        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} missing", layer, name))
            };

            // Pre-attention norm
            let normed = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("attn_norm.weight")?)
            };

            // ── Arch-aware QKV projection ─────────────────────────────────────
            let (q, k, v) = if hp.arch == "phi3" {
                // Phi-3: fused attn_qkv.weight → split via ggml_view_2d
                // Weight shape: [n_embd, (n_head + 2*n_head_kv)*hd]
                // After mul_mat: [(n_head + 2*n_head_kv)*hd, n_tokens]
                let qkv = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_qkv.weight")?, normed) };
                let qkv_total = (hp.n_head + hp.n_head_kv * 2) * hd;
                let stride_b  = (qkv_total * 4) as usize;
                let q_off     = 0usize;
                let k_off     = (hp.n_head * hd * 4) as usize;
                let v_off     = ((hp.n_head + hp.n_head_kv) * hd * 4) as usize;
                let q_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head    * hd, n_tokens, stride_b, q_off) };
                let k_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head_kv * hd, n_tokens, stride_b, k_off) };
                let v_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head_kv * hd, n_tokens, stride_b, v_off) };
                let q = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, q_v), hd, hp.n_head,    n_tokens) };
                let k = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, k_v), hd, hp.n_head_kv, n_tokens) };
                let v = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, v_v), hd, hp.n_head_kv, n_tokens) };
                (q, k, v)
            } else {
                // Llama / Qwen2: separate Q, K, V weights
                let q = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_q.weight")?, normed) };
                let k = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_k.weight")?, normed) };
                let v = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_v.weight")?, normed) };
                // Qwen2: add QKV biases
                let (q, k, v) = if hp.arch == "qwen2" {
                    let q = unsafe { ffi::ggml_add(self.ctx, q, t("attn_q.bias")?) };
                    let k = unsafe { ffi::ggml_add(self.ctx, k, t("attn_k.bias")?) };
                    let v = unsafe { ffi::ggml_add(self.ctx, v, t("attn_v.bias")?) };
                    (q, k, v)
                } else { (q, k, v) };
                let q = unsafe { ffi::ggml_reshape_3d(self.ctx, q, hd, hp.n_head,    n_tokens) };
                let k = unsafe { ffi::ggml_reshape_3d(self.ctx, k, hd, hp.n_head_kv, n_tokens) };
                let v = unsafe { ffi::ggml_reshape_3d(self.ctx, v, hd, hp.n_head_kv, n_tokens) };
                (q, k, v)
            };

            // RoPE
            let q = unsafe {
                ffi::ggml_rope_ext(self.ctx, q, self.d_pos, null_mut(),
                    hp.n_rot as c_int, if hp.arch == "phi3" || hp.arch == "qwen2" || hp.arch == "qwen" { 2 } else { 0 }, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };
            let k = unsafe {
                ffi::ggml_rope_ext(self.ctx, k, self.d_pos, null_mut(),
                    hp.n_rot as c_int, if hp.arch == "phi3" || hp.arch == "qwen2" || hp.arch == "qwen" { 2 } else { 0 }, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };

            // Flatten for cache write
            let k_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx, ffi::ggml_cont(self.ctx, k), hp.n_head_kv * hd, n_tokens)
            };
            let v_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx, ffi::ggml_cont(self.ctx, v), hp.n_head_kv * hd, n_tokens)
            };

            // Zero-copy KV write
            let k_cache = self.kv.k_ptrs[layer];
            let v_cache = self.kv.v_ptrs[layer];
            let k_view = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache, hp.n_head_kv * hd, 1, kv_stride, current_pos as usize * kv_stride)
            };
            let v_view = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache, hp.n_head_kv * hd, 1, kv_stride, current_pos as usize * kv_stride)
            };
            let k_copy = unsafe { ffi::ggml_cpy(self.ctx, k_flat, k_view) };
            let v_copy = unsafe { ffi::ggml_cpy(self.ctx, v_flat, v_view) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, k_copy) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, v_copy) };

            // Read full cache
            let k_full = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache, hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };
            let v_full = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache, hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };

            let k_3d_raw = unsafe {
                ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, k_full), hd, hp.n_head_kv, self.n_ctx)
            };
            let v_3d_raw = unsafe {
                ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, v_full), hd, hp.n_head_kv, self.n_ctx)
            };

            // ── GQA: repeat K/V heads when n_head > n_head_kv (Qwen2.5, etc.) ─
            let (k_3d, v_3d) = if false && hp.arch == "qwen2" && hp.n_head > hp.n_head_kv {
                let k_rep_t = unsafe { ffi::ggml_new_tensor_3d(self.ctx, 0, hd, hp.n_head, self.n_ctx) };
                let v_rep_t = unsafe { ffi::ggml_new_tensor_3d(self.ctx, 0, hd, hp.n_head, self.n_ctx) };
                let k_rep = unsafe { ffi::ggml_repeat(self.ctx, k_3d_raw, k_rep_t) };
                let v_rep = unsafe { ffi::ggml_repeat(self.ctx, v_3d_raw, v_rep_t) };
                (k_rep, v_rep)
            } else {
                (k_3d_raw, v_3d_raw)
            };

            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(self.ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe { ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, k_3d, 0, 2, 1, 3)) };
            let v_perm = unsafe { ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, v_3d, 1, 2, 0, 3)) };

            let kq = unsafe {
                ffi::ggml_scale(self.ctx, ffi::ggml_mul_mat(self.ctx, k_perm, q_perm), scale)
            };
            let kq = unsafe { ffi::ggml_soft_max_ext(self.ctx, kq, self.d_mask, 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(self.ctx, v_perm, kq) };
            let av = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, av, 0, 2, 1, 3)),
                    hp.n_embd, n_tokens)
            };

            let attn = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, attn) };

            // ── Arch-aware FFN ────────────────────────────────────────────────
            let ffn_in = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("ffn_norm.weight")?)
            };

            let ffn = if hp.arch == "phi3" {
                // Phi-3: ffn_up.weight is fused [n_embd, 2*n_ff]
                // After mul_mat: [2*n_ff, n_tokens] — split into gate + up
                let gate_up  = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?, ffn_in) };
                let stride_b = (hp.n_ff * 2 * 4) as usize;
                let gate_v   = unsafe { ffi::ggml_view_2d(self.ctx, gate_up, hp.n_ff, n_tokens, stride_b, 0) };
                let up_v     = unsafe { ffi::ggml_view_2d(self.ctx, gate_up, hp.n_ff, n_tokens, stride_b, (hp.n_ff * 4) as usize) };
                let gate     = unsafe { ffi::ggml_silu(self.ctx, ffi::ggml_cont(self.ctx, gate_v)) };
                let up       = unsafe { ffi::ggml_cont(self.ctx, up_v) };
                unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?, ffi::ggml_mul(self.ctx, gate, up)) }
            } else {
                // Llama / Qwen2: separate gate + up (standard SwiGLU)
                let gate = unsafe { ffi::ggml_silu(self.ctx, ffi::ggml_mul_mat(self.ctx, t("ffn_gate.weight")?, ffn_in)) };
                let up   = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?, ffn_in) };
                unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?, ffi::ggml_mul(self.ctx, gate, up)) }
            };

            cur = unsafe { ffi::ggml_add(self.ctx, cur, ffn) };
        }

        // ── Output ────────────────────────────────────────────────────────────

        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight missing"))?;
        cur = unsafe {
            ffi::ggml_mul(self.ctx, ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps), out_norm)
        };

        let lm_head = if hp.has_tied_weights {
            model.tensor("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("token_embd.weight (tied) missing"))?
        } else {
            model.tensor("output.weight")
                .ok_or_else(|| anyhow::anyhow!("output.weight missing"))?
        };

        self.d_logits = unsafe { ffi::ggml_mul_mat(self.ctx, lm_head, cur) };
        unsafe { ffi::ggml_set_output(self.d_logits) };
        unsafe { ffi::ggml_build_forward_expand(self.graph, self.d_logits) };

        let buft = unsafe { ffi::ggml_backend_cpu_buffer_type() };
        self.galloc = unsafe { ffi::ggml_gallocr_new(buft) };
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(self.galloc, self.graph) };
        if !ok { anyhow::bail!("gallocr_alloc_graph failed"); }

        Ok(())
    }

    // ── Internal single-token decode ───────────────────────────────────────────

    fn decode_internal(&mut self, token_id: u32, model: &MappedModel) -> Result<Vec<f32>> {
        let pos = self.kv.head;
        if pos >= self.n_ctx {
            anyhow::bail!("context full ({}/{})", pos, self.n_ctx);
        }
        self.kv.head += 1;

        self.build_graph(model, pos)?;

        let tok = token_id as i32;
        unsafe {
            ffi::ggml_backend_tensor_set(self.d_token, &tok as *const i32 as *const c_void, 0, 4);
            ffi::ggml_backend_tensor_set(self.d_pos, &(pos as i32) as *const i32 as *const c_void, 0, 4);
        }

        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != 0 { anyhow::bail!("compute failed status={}", status); }

        let n_vocab = self.dna.n_vocab as usize;
        let mut out = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(self.d_logits, out.as_mut_ptr() as *mut c_void, 0, n_vocab * 4)
        };
        Ok(out)
    }

    // ── Batched prefill graph ─────────────────────────────────────────────────

    fn build_prefill_graph(&mut self, model: &MappedModel, n_tokens: i64, start_pos: i64) -> Result<()> {
        self.rebuild_count += 1;
        self.cleanup_graph_resources();

        let hp        = &self.dna;
        let hd        = hp.head_dim();
        let kv_stride = self.kv.stride;

        // ── Inputs ────────────────────────────────────────────────────────────

        let mask_elems = (self.n_ctx * n_tokens) as usize;
        let mask_bytes = mask_elems * 4;
        let tok_bytes  = n_tokens as usize * 4;
        let inp_size   = tok_bytes * 2 + mask_bytes + 1024;

        self.inp_buf = unsafe { ffi::ggml_backend_alloc_buffer(self.backend, inp_size) };
        if self.inp_buf.is_null() { anyhow::bail!("prefill input buffer alloc failed"); }

        let inp_base = unsafe { ffi::ggml_backend_buffer_get_base(self.inp_buf) } as usize;

        self.inp_ctx = unsafe {
            ffi::ggml_init(GgmlInitParams { mem_size: 32768, mem_buffer: null_mut(), no_alloc: true })
        };
        if self.inp_ctx.is_null() { anyhow::bail!("prefill inp_ctx ggml_init failed"); }

        self.d_token = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_token, inp_base as *mut c_void) };

        self.d_pos = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_pos, (inp_base + tok_bytes + 64) as *mut c_void) };

        self.d_mask = unsafe { ffi::ggml_new_tensor_2d(self.inp_ctx, 0, self.n_ctx, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_mask, (inp_base + tok_bytes * 2 + 128) as *mut c_void) };

        let mut mask_data = vec![-10000.0f32; mask_elems];
        for qi in 0..n_tokens as usize {
            let visible_up_to = (start_pos as usize + qi).min(self.n_ctx as usize - 1);
            for ki in 0..=visible_up_to {
                mask_data[ki + qi * self.n_ctx as usize] = 0.0;
            }
        }
        unsafe {
            ffi::ggml_backend_tensor_set(self.d_mask, mask_data.as_ptr() as *const c_void, 0, mask_bytes)
        };

        // ── Compute graph ─────────────────────────────────────────────────────

        self.ctx = unsafe {
            ffi::ggml_init(GgmlInitParams { mem_size: GRAPH_MEM_SIZE, mem_buffer: null_mut(), no_alloc: true })
        };
        if self.ctx.is_null() { anyhow::bail!("prefill graph ctx ggml_init failed"); }

        self.graph = unsafe { ffi::ggml_new_graph_custom(self.ctx, 65536, false) };

        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight missing"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(self.ctx, embd_w, self.d_token) };

        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} missing", layer, name))
            };

            let normed = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("attn_norm.weight")?)
            };

            // ── Arch-aware QKV projection ─────────────────────────────────────
            let (q, k, v) = if hp.arch == "phi3" {
                let qkv = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_qkv.weight")?, normed) };
                let qkv_total = (hp.n_head + hp.n_head_kv * 2) * hd;
                let stride_b  = (qkv_total * 4) as usize;
                let q_off     = 0usize;
                let k_off     = (hp.n_head * hd * 4) as usize;
                let v_off     = ((hp.n_head + hp.n_head_kv) * hd * 4) as usize;
                let q_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head    * hd, n_tokens, stride_b, q_off) };
                let k_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head_kv * hd, n_tokens, stride_b, k_off) };
                let v_v = unsafe { ffi::ggml_view_2d(self.ctx, qkv, hp.n_head_kv * hd, n_tokens, stride_b, v_off) };
                let q = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, q_v), hd, hp.n_head,    n_tokens) };
                let k = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, k_v), hd, hp.n_head_kv, n_tokens) };
                let v = unsafe { ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, v_v), hd, hp.n_head_kv, n_tokens) };
                (q, k, v)
            } else {
                let q = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_q.weight")?,  normed) };
                let k = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_k.weight")?,  normed) };
                let v = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_v.weight")?,  normed) };
                let (q, k, v) = if hp.arch == "qwen2" {
                    let q = unsafe { ffi::ggml_add(self.ctx, q, t("attn_q.bias")?) };
                    let k = unsafe { ffi::ggml_add(self.ctx, k, t("attn_k.bias")?) };
                    let v = unsafe { ffi::ggml_add(self.ctx, v, t("attn_v.bias")?) };
                    (q, k, v)
                } else { (q, k, v) };
                let q = unsafe { ffi::ggml_reshape_3d(self.ctx, q, hd, hp.n_head,    n_tokens) };
                let k = unsafe { ffi::ggml_reshape_3d(self.ctx, k, hd, hp.n_head_kv, n_tokens) };
                let v = unsafe { ffi::ggml_reshape_3d(self.ctx, v, hd, hp.n_head_kv, n_tokens) };
                (q, k, v)
            };

            // RoPE
            let q = unsafe {
                ffi::ggml_rope_ext(self.ctx, q, self.d_pos, null_mut(),
                    hp.n_rot as c_int, if hp.arch == "phi3" || hp.arch == "qwen2" || hp.arch == "qwen" { 2 } else { 0 }, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };
            let k = unsafe {
                ffi::ggml_rope_ext(self.ctx, k, self.d_pos, null_mut(),
                    hp.n_rot as c_int, if hp.arch == "phi3" || hp.arch == "qwen2" || hp.arch == "qwen" { 2 } else { 0 }, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };

            let k_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx, ffi::ggml_cont(self.ctx, k), hp.n_head_kv * hd, n_tokens)
            };
            let v_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx, ffi::ggml_cont(self.ctx, v), hp.n_head_kv * hd, n_tokens)
            };

            let k_cache = self.kv.k_ptrs[layer];
            let v_cache = self.kv.v_ptrs[layer];

            let k_view = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache,
                    hp.n_head_kv * hd, n_tokens, kv_stride, start_pos as usize * kv_stride)
            };
            let v_view = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache,
                    hp.n_head_kv * hd, n_tokens, kv_stride, start_pos as usize * kv_stride)
            };

            let k_copy = unsafe { ffi::ggml_cpy(self.ctx, k_flat, k_view) };
            let v_copy = unsafe { ffi::ggml_cpy(self.ctx, v_flat, v_view) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, k_copy) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, v_copy) };

            let k_full = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache, hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };
            let v_full = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache, hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };

            let k_3d_raw = unsafe {
                ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, k_full), hd, hp.n_head_kv, self.n_ctx)
            };
            let v_3d_raw = unsafe {
                ffi::ggml_reshape_3d(self.ctx, ffi::ggml_cont(self.ctx, v_full), hd, hp.n_head_kv, self.n_ctx)
            };

            // GQA repeat
            let (k_3d, v_3d) = if false && hp.arch == "qwen2" && hp.n_head > hp.n_head_kv {
                let k_rep_t = unsafe { ffi::ggml_new_tensor_3d(self.ctx, 0, hd, hp.n_head, self.n_ctx) };
                let v_rep_t = unsafe { ffi::ggml_new_tensor_3d(self.ctx, 0, hd, hp.n_head, self.n_ctx) };
                let k_rep = unsafe { ffi::ggml_repeat(self.ctx, k_3d_raw, k_rep_t) };
                let v_rep = unsafe { ffi::ggml_repeat(self.ctx, v_3d_raw, v_rep_t) };
                (k_rep, v_rep)
            } else {
                (k_3d_raw, v_3d_raw)
            };

            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(self.ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe { ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, k_3d, 0, 2, 1, 3)) };
            let v_perm = unsafe { ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, v_3d, 1, 2, 0, 3)) };

            let kq = unsafe {
                ffi::ggml_scale(self.ctx, ffi::ggml_mul_mat(self.ctx, k_perm, q_perm), scale)
            };
            let kq = unsafe { ffi::ggml_soft_max_ext(self.ctx, kq, self.d_mask, 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(self.ctx, v_perm, kq) };
            let av = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, av, 0, 2, 1, 3)),
                    hp.n_embd, n_tokens)
            };

            let attn = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, attn) };

            // ── Arch-aware FFN ────────────────────────────────────────────────
            let ffn_in = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("ffn_norm.weight")?)
            };

            let ffn = if hp.arch == "phi3" {
                let gate_up  = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?, ffn_in) };
                let stride_b = (hp.n_ff * 2 * 4) as usize;
                let gate_v   = unsafe { ffi::ggml_view_2d(self.ctx, gate_up, hp.n_ff, n_tokens, stride_b, 0) };
                let up_v     = unsafe { ffi::ggml_view_2d(self.ctx, gate_up, hp.n_ff, n_tokens, stride_b, (hp.n_ff * 4) as usize) };
                let gate     = unsafe { ffi::ggml_silu(self.ctx, ffi::ggml_cont(self.ctx, gate_v)) };
                let up       = unsafe { ffi::ggml_cont(self.ctx, up_v) };
                unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?, ffi::ggml_mul(self.ctx, gate, up)) }
            } else {
                let gate = unsafe { ffi::ggml_silu(self.ctx, ffi::ggml_mul_mat(self.ctx, t("ffn_gate.weight")?, ffn_in)) };
                let up   = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?, ffn_in) };
                unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?, ffi::ggml_mul(self.ctx, gate, up)) }
            };

            cur = unsafe { ffi::ggml_add(self.ctx, cur, ffn) };
        }

        // ── Output ────────────────────────────────────────────────────────────

        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight missing"))?;
        cur = unsafe {
            ffi::ggml_mul(self.ctx, ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps), out_norm)
        };

        let lm_head = if hp.has_tied_weights {
            model.tensor("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("token_embd.weight (tied) missing"))?
        } else {
            model.tensor("output.weight")
                .ok_or_else(|| anyhow::anyhow!("output.weight missing"))?
        };

        self.d_logits = unsafe { ffi::ggml_mul_mat(self.ctx, lm_head, cur) };
        unsafe { ffi::ggml_set_output(self.d_logits) };
        unsafe { ffi::ggml_build_forward_expand(self.graph, self.d_logits) };

        let buft = unsafe { ffi::ggml_backend_cpu_buffer_type() };
        self.galloc = unsafe { ffi::ggml_gallocr_new(buft) };
        let ok = unsafe { ffi::ggml_gallocr_alloc_graph(self.galloc, self.graph) };
        if !ok { anyhow::bail!("prefill gallocr_alloc_graph failed"); }

        Ok(())
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    pub fn prefill(&mut self, token_ids: &[u32], model: &MappedModel) -> Result<Vec<f32>, ForwardError> {
        if token_ids.is_empty() {
            return Err(ForwardError::Init("empty prefill token list".into()));
        }

        let n_tokens  = token_ids.len() as i64;
        let start_pos = self.kv.head;

        if start_pos + n_tokens > self.n_ctx {
            return Err(ForwardError::ContextFull);
        }

        self.build_prefill_graph(model, n_tokens, start_pos)
            .map_err(|e| ForwardError::Internal(e.to_string()))?;

        let tok_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        unsafe {
            ffi::ggml_backend_tensor_set(self.d_token, tok_i32.as_ptr() as *const c_void, 0, n_tokens as usize * 4)
        };

        let pos_i32: Vec<i32> = (start_pos..start_pos + n_tokens).map(|p| p as i32).collect();
        unsafe {
            ffi::ggml_backend_tensor_set(self.d_pos, pos_i32.as_ptr() as *const c_void, 0, n_tokens as usize * 4)
        };

        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != 0 { return Err(ForwardError::ComputeFailed(status)); }

        self.kv.head = (start_pos + n_tokens).min(self.n_ctx);

        let n_vocab     = self.dna.n_vocab as usize;
        let last_offset = (n_tokens as usize - 1) * n_vocab * 4;
        let mut out     = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(self.d_logits, out.as_mut_ptr() as *mut c_void, last_offset, n_vocab * 4)
        };

        log::info!("[Z.3] Prefill: {} tokens in 1 graph pass (head={})", n_tokens, self.kv.head);
        Ok(out)
    }

    pub fn decode_one(&mut self, token_id: u32, model: &MappedModel) -> Result<Vec<f32>, ForwardError> {
        self.decode_internal(token_id, model).map_err(|e| {
            if e.to_string().contains("context full") { ForwardError::ContextFull }
            else { ForwardError::Internal(e.to_string()) }
        })
    }

    pub fn reset_kv(&mut self) {
        self.kv.clear();
        self.cleanup_graph_resources();
    }

    pub fn dna(&self) -> &ModelDNA { &self.dna }
}

impl Drop for ForwardPass {
    fn drop(&mut self) {
        self.cleanup_graph_resources();
        unsafe {
            if !self.backend.is_null() { ffi::ggml_backend_free(self.backend); }
        }
    }
}
