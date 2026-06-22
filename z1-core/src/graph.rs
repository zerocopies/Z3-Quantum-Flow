/// Z.3 — graph.rs [QUANTUM-FLOW ENGINE v3.2 — ForwardPass Compatible]
///
/// Public API matches the existing generate.rs / main.rs interface exactly:
///   ForwardPass::new(&model)
///   fwd.prefill(token_ids, model)  -> Result<Vec<f32>, ForwardError>
///   fwd.decode_one(token_id, model) -> Result<Vec<f32>, ForwardError>
///   fwd.reset_kv()
///   fwd.kv.n_ctx  (pub field)
///   fwd.kv.head   (pub field)
///
/// Internal fixes over v3.2:
///   1. Metadata keys have dot separator ("{}.vocab_size" not "{}vocab_size")
///   2. Overflow checked before advance — no corrupted KV state
///   3. ggml_init null-checked in QuantumKV::new and build_graph
///   4. cleanup_graph_resources() — single teardown, correct order
///   5. Graph freed with ctx (not separately) — no double-free
///   6. galloc freed before ctx
///   7. rebuild_count for performance profiling
///
/// prefill() note: currently sequential decode_one per token (correct, not batched).
/// A proper batched prefill is a future optimisation.

use std::ffi::c_void;
use std::ptr::null_mut;
use anyhow::Result;
use libc::c_int;

use crate::ggml_ffi::{self as ffi, GgmlInitParams};
use crate::loader::MappedModel;

// ── Configuration ─────────────────────────────────────────────────────────────


const MAX_CTX: i64         = 4096;
const GRAPH_MEM_SIZE: usize = 512 * 1024 * 1024; // 512 MB

// ── ForwardError (matches generate.rs import) ─────────────────────────────────

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

// Bridge anyhow → ForwardError for internal calls
impl From<anyhow::Error> for ForwardError {
    fn from(e: anyhow::Error) -> Self { Self::Internal(e.to_string()) }
}

// ── Model DNA ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ModelDNA {
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

        let has_tied = model.tensor("output.weight").is_none()
            && model.tensor("token_embd.weight").is_some();

        Ok(ModelDNA {
            n_vocab: meta.get(&format!("{}.vocab_size", arch)).and_then(|v| v.as_u32()).map(|v| v as i64).unwrap_or_else(|| { if let Some(crate::gguf::GgufValue::Array(arr)) = meta.get("tokenizer.ggml.tokens") { arr.len() as i64 } else { 0 } }),
            n_embd:           get_u32(&format!("{}.embedding_length", arch)),
            n_head:           get_u32(&format!("{}.attention.head_count", arch)),
            n_head_kv:        get_u32(&format!("{}.attention.head_count_kv", arch)),
            n_layer:          get_u32(&format!("{}.block_count", arch)),
            n_ff:             get_u32(&format!("{}.feed_forward_length", arch)),
            n_rot:            get_u32(&format!("{}.rope.dimension_count", arch)),
            freq_base:        get_f32(&format!("{}.rope.freq_base", arch), 10000.0),
            rms_eps:          get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch), 1e-6),
            has_tied_weights: has_tied || arch == "qwen2",
        })
    }

    pub fn head_dim(&self) -> i64 {
        if self.n_head == 0 { 1 } else { self.n_embd / self.n_head }
    }
}

// ── Quantum KV Cache ──────────────────────────────────────────────────────────
///
/// Fields are pub so generate.rs can read fwd.kv.n_ctx and fwd.kv.head directly.

pub struct QuantumKV {
    ctx:          *mut ffi::ggml_context,
    buf:          ffi::ggml_backend_buffer_t,
    pub k_ptrs:   Vec<*mut ffi::ggml_tensor>,
    pub v_ptrs:   Vec<*mut ffi::ggml_tensor>,
    pub n_ctx:    i64,
    pub head:     i64,   // current write position; matches generate.rs fwd.kv.head
    pub stride:   usize, // bytes per row (one position)
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
        let bytes = elem * 4; // f32
        let total = bytes * n_layers * 2;

        log::info!("[Z.3] Allocating {:.2} MB Quantum-KV (layers={}, n_ctx={})",
            total as f32 / (1024.0 * 1024.0), n_layers, n_ctx);

        let buf = unsafe { ffi::ggml_backend_alloc_buffer(backend, total) };
        if buf.is_null() {
            anyhow::bail!("KV buffer allocation failed");
        }

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

        // Zero-init the entire cache
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
/// Public name matches generate.rs / main.rs imports.

pub struct ForwardPass {
    dna:     ModelDNA,
    backend: ffi::ggml_backend_t,
    pub kv:  QuantumKV,  // pub so generate.rs can access kv.head and kv.n_ctx
    n_ctx:   i64,

    // Graph resources — all managed via cleanup_graph_resources()
    ctx:      *mut ffi::ggml_context,
    inp_ctx:  *mut ffi::ggml_context,
    graph:    *mut ffi::ggml_cgraph,  // lives inside ctx arena; freed with ctx
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
    /// Matches main.rs: ForwardPass::new(&model)
    /// Uses DEFAULT_N_CTX (512) — matches existing GenerateConfig::default()
    pub fn new(model: &MappedModel, n_ctx: i64) -> Result<Self> {
        // n_ctx is now passed by caller - removed DEFAULT_N_CTX hardcode
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

        log::info!("[Z.3] Engine ready — layers={} embd={} heads={}/{} vocab={} ctx={}",
            dna.n_layer, dna.n_embd, dna.n_head, dna.n_head_kv, dna.n_vocab, kv.n_ctx);

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
            // ctx owns the graph — freeing ctx frees graph implicitly
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

    // ── Graph construction ─────────────────────────────────────────────────────
    ///
    /// Rebuilds for every token — view offsets are baked in at build time.
    /// Copy ops are added to the graph before the logits path, guaranteeing
    /// write-before-read on the CPU's topological executor.

    fn build_graph(&mut self, model: &MappedModel, current_pos: i64) -> Result<()> {
        self.rebuild_count += 1;
        self.cleanup_graph_resources();

        let hp        = &self.dna;
        let hd        = hp.head_dim();
        let n_tokens  = 1i64;
        let kv_stride = self.kv.stride;

        // ── Inputs ─────────────────────────────────────────────────────────────

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

        // Positions 0..=current_pos visible, rest masked
        let mut mask_data = vec![-10000.0f32; self.n_ctx as usize];
        for i in 0..=current_pos as usize { mask_data[i] = 0.0; }
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.d_mask, mask_data.as_ptr() as *const c_void, 0, mask_bytes,
            )
        };

        // ── Compute graph ──────────────────────────────────────────────────────

        self.ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size:   GRAPH_MEM_SIZE,
                mem_buffer: null_mut(),
                no_alloc:   true,
            })
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

            // Pre-attention norm + projections
            let normed = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("attn_norm.weight")?)
            };
            let q = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_q.weight")?, normed) };
            let k = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_k.weight")?, normed) };
            let v = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_v.weight")?, normed) };

            let q = unsafe { ffi::ggml_reshape_3d(self.ctx, q, hd, hp.n_head,    n_tokens) };
            let k = unsafe { ffi::ggml_reshape_3d(self.ctx, k, hd, hp.n_head_kv, n_tokens) };
            let v = unsafe { ffi::ggml_reshape_3d(self.ctx, v, hd, hp.n_head_kv, n_tokens) };

            // RoPE
            let q = unsafe {
                ffi::ggml_rope_ext(self.ctx, q, self.d_pos, null_mut(),
                    hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };
            let k = unsafe {
                ffi::ggml_rope_ext(self.ctx, k, self.d_pos, null_mut(),
                    hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };

            // Flatten for cache write
            let k_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, k), hp.n_head_kv * hd, n_tokens)
            };
            let v_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, v), hp.n_head_kv * hd, n_tokens)
            };

            // ── Zero-copy KV write ─────────────────────────────────────────────
            let k_cache = self.kv.k_ptrs[layer];
            let v_cache = self.kv.v_ptrs[layer];

            let k_offset = (current_pos as usize) * kv_stride;
            let v_offset = (current_pos as usize) * kv_stride;

            let k_view = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache,
                    hp.n_head_kv * hd, 1, kv_stride, k_offset)
            };
            let v_view = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache,
                    hp.n_head_kv * hd, 1, kv_stride, v_offset)
            };

            let k_copy = unsafe { ffi::ggml_cpy(self.ctx, k_flat, k_view) };
            let v_copy = unsafe { ffi::ggml_cpy(self.ctx, v_flat, v_view) };

            // Copy ops first — write-before-read guaranteed on CPU topological executor
            unsafe { ffi::ggml_build_forward_expand(self.graph, k_copy) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, v_copy) };

            // ── Attention (reads full cache, including current_pos) ─────────────
            let k_full = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache,
                    hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };
            let v_full = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache,
                    hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };

            let k_3d = unsafe {
                ffi::ggml_reshape_3d(self.ctx,
                    ffi::ggml_cont(self.ctx, k_full), hd, hp.n_head_kv, self.n_ctx)
            };
            let v_3d = unsafe {
                ffi::ggml_reshape_3d(self.ctx,
                    ffi::ggml_cont(self.ctx, v_full), hd, hp.n_head_kv, self.n_ctx)
            };

            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(self.ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe {
                ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, k_3d, 0, 2, 1, 3))
            };
            let v_perm = unsafe {
                ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, v_3d, 1, 2, 0, 3))
            };

            let kq = unsafe {
                ffi::ggml_scale(self.ctx,
                    ffi::ggml_mul_mat(self.ctx, k_perm, q_perm), scale)
            };
            let kq = unsafe { ffi::ggml_soft_max_ext(self.ctx, kq, self.d_mask, 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(self.ctx, v_perm, kq) };
            let av = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx,
                        ffi::ggml_permute(self.ctx, av, 0, 2, 1, 3)),
                    hp.n_embd, n_tokens)
            };

            let attn = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, attn) };

            // ── FFN (SwiGLU) ───────────────────────────────────────────────────
            let ffn_in = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("ffn_norm.weight")?)
            };
            let gate = unsafe {
                ffi::ggml_silu(self.ctx,
                    ffi::ggml_mul_mat(self.ctx, t("ffn_gate.weight")?, ffn_in))
            };
            let up  = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?,   ffn_in) };
            let ffn = unsafe {
                ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?,
                    ffi::ggml_mul(self.ctx, gate, up))
            };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, ffn) };
        }

        // ── Output ─────────────────────────────────────────────────────────────

        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight missing"))?;
        cur = unsafe {
            ffi::ggml_mul(self.ctx,
                ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps), out_norm)
        };

        let lm_head = if hp.has_tied_weights {
            model.tensor("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("token_embd.weight (tied lm_head) missing"))?
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
        // Check overflow BEFORE advancing — keeps KV state clean on error
        let pos = self.kv.head;
        if pos >= self.n_ctx {
            anyhow::bail!("context full ({}/{})", pos, self.n_ctx);
        }
        self.kv.head += 1;

        self.build_graph(model, pos)?;

        let tok = token_id as i32;
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.d_token, &tok as *const i32 as *const c_void, 0, 4);
            ffi::ggml_backend_tensor_set(
                self.d_pos, &(pos as i32) as *const i32 as *const c_void, 0, 4);
        }

        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != 0 { anyhow::bail!("compute failed status={}", status); }

        let n_vocab = self.dna.n_vocab as usize;
        let mut out = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(
                self.d_logits, out.as_mut_ptr() as *mut c_void, 0, n_vocab * 4)
        };

        Ok(out)
    }

    // ── Batched prefill graph ─────────────────────────────────────────────────
    /// Builds a single graph that processes all N prompt tokens in one pass.
    /// Key differences from build_graph (single token):
    ///   - d_token/d_pos are [n_tokens] not [1]
    ///   - Causal mask is [n_ctx, n_tokens] — each column has its own visible range
    ///   - KV view covers N rows at once — one ggml_cpy for all positions
    ///   - Output logits are [n_vocab, n_tokens] — we extract last token only

    fn build_prefill_graph(
        &mut self,
        model: &MappedModel,
        n_tokens: i64,
        start_pos: i64,
    ) -> Result<()> {
        self.rebuild_count += 1;
        self.cleanup_graph_resources();

        let hp        = &self.dna;
        let hd        = hp.head_dim();
        let kv_stride = self.kv.stride;

        // ── Inputs ─────────────────────────────────────────────────────────────

        // d_token [n_tokens], d_pos [n_tokens], d_mask [n_ctx, n_tokens]
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

        // token ids: [n_tokens]
        self.d_token = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(self.inp_buf, self.d_token, inp_base as *mut c_void) };

        // positions: [n_tokens]
        self.d_pos = unsafe { ffi::ggml_new_tensor_1d(self.inp_ctx, 26, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(
            self.inp_buf, self.d_pos, (inp_base + tok_bytes + 64) as *mut c_void) };

        // causal mask: [n_ctx, n_tokens]
        // ggml layout: ne[0]=n_ctx, ne[1]=n_tokens
        // element [ki, qi] is at ki + qi*n_ctx
        // query at position start_pos+qi can attend to keys 0..=start_pos+qi
        self.d_mask = unsafe { ffi::ggml_new_tensor_2d(self.inp_ctx, 0, self.n_ctx, n_tokens) };
        unsafe { ffi::ggml_backend_tensor_alloc(
            self.inp_buf, self.d_mask, (inp_base + tok_bytes * 2 + 128) as *mut c_void) };

        let mut mask_data = vec![-10000.0f32; mask_elems];
        for qi in 0..n_tokens as usize {
            let visible_up_to = (start_pos as usize + qi).min(self.n_ctx as usize - 1);
            for ki in 0..=visible_up_to {
                mask_data[ki + qi * self.n_ctx as usize] = 0.0;
            }
        }
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.d_mask, mask_data.as_ptr() as *const c_void, 0, mask_bytes)
        };

        // ── Compute graph ──────────────────────────────────────────────────────

        self.ctx = unsafe {
            ffi::ggml_init(GgmlInitParams {
                mem_size:   GRAPH_MEM_SIZE,
                mem_buffer: null_mut(),
                no_alloc:   true,
            })
        };
        if self.ctx.is_null() { anyhow::bail!("prefill graph ctx ggml_init failed"); }

        self.graph = unsafe { ffi::ggml_new_graph_custom(self.ctx, 65536, false) };

        // Embedding lookup → [n_embd, n_tokens]
        let embd_w = model.tensor("token_embd.weight")
            .ok_or_else(|| anyhow::anyhow!("token_embd.weight missing"))?;
        let mut cur = unsafe { ffi::ggml_get_rows(self.ctx, embd_w, self.d_token) };

        for layer in 0..hp.n_layer as usize {
            let t = |name: &str| -> Result<*mut ffi::ggml_tensor> {
                model.layer_tensor(layer, name)
                    .ok_or_else(|| anyhow::anyhow!("blk.{}.{} missing", layer, name))
            };

            // Pre-attention norm + projections → [n_embd, n_tokens]
            let normed = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("attn_norm.weight")?)
            };
            let q = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_q.weight")?,  normed) };
            let k = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_k.weight")?,  normed) };
            let v = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_v.weight")?,  normed) };

            // Reshape → [hd, n_heads, n_tokens]
            let q = unsafe { ffi::ggml_reshape_3d(self.ctx, q, hd, hp.n_head,    n_tokens) };
            let k = unsafe { ffi::ggml_reshape_3d(self.ctx, k, hd, hp.n_head_kv, n_tokens) };
            let v = unsafe { ffi::ggml_reshape_3d(self.ctx, v, hd, hp.n_head_kv, n_tokens) };

            // RoPE — d_pos [n_tokens] applies per-token positions
            let q = unsafe {
                ffi::ggml_rope_ext(self.ctx, q, self.d_pos, null_mut(),
                    hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };
            let k = unsafe {
                ffi::ggml_rope_ext(self.ctx, k, self.d_pos, null_mut(),
                    hp.n_rot as c_int, 0, 131072, hp.freq_base, 1.0, 0.0, 1.0, 32.0, 1.0)
            };

            // Flatten → [n_head_kv * hd, n_tokens]
            let k_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, k), hp.n_head_kv * hd, n_tokens)
            };
            let v_flat = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx, v), hp.n_head_kv * hd, n_tokens)
            };

            // ── KV write: all N rows in one shot ──────────────────────────────
            let k_cache = self.kv.k_ptrs[layer];
            let v_cache = self.kv.v_ptrs[layer];

            // View of N rows starting at start_pos
            let k_view = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache,
                    hp.n_head_kv * hd, n_tokens,
                    kv_stride,
                    start_pos as usize * kv_stride)
            };
            let v_view = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache,
                    hp.n_head_kv * hd, n_tokens,
                    kv_stride,
                    start_pos as usize * kv_stride)
            };

            let k_copy = unsafe { ffi::ggml_cpy(self.ctx, k_flat, k_view) };
            let v_copy = unsafe { ffi::ggml_cpy(self.ctx, v_flat, v_view) };

            // Copy ops first — write-before-read on CPU topological executor
            unsafe { ffi::ggml_build_forward_expand(self.graph, k_copy) };
            unsafe { ffi::ggml_build_forward_expand(self.graph, v_copy) };

            // ── Attention (reads full cache) ───────────────────────────────────
            let k_full = unsafe {
                ffi::ggml_view_2d(self.ctx, k_cache,
                    hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };
            let v_full = unsafe {
                ffi::ggml_view_2d(self.ctx, v_cache,
                    hp.n_head_kv * hd, self.n_ctx, kv_stride, 0)
            };

            let k_3d = unsafe {
                ffi::ggml_reshape_3d(self.ctx,
                    ffi::ggml_cont(self.ctx, k_full), hd, hp.n_head_kv, self.n_ctx)
            };
            let v_3d = unsafe {
                ffi::ggml_reshape_3d(self.ctx,
                    ffi::ggml_cont(self.ctx, v_full), hd, hp.n_head_kv, self.n_ctx)
            };

            let scale  = 1.0 / (hd as f32).sqrt();
            let q_perm = unsafe { ffi::ggml_permute(self.ctx, q, 0, 2, 1, 3) };
            let k_perm = unsafe {
                ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, k_3d, 0, 2, 1, 3))
            };
            let v_perm = unsafe {
                ffi::ggml_cont(self.ctx, ffi::ggml_permute(self.ctx, v_3d, 1, 2, 0, 3))
            };

            // kq: [n_ctx, n_tokens, n_head, 1]
            let kq = unsafe {
                ffi::ggml_scale(self.ctx,
                    ffi::ggml_mul_mat(self.ctx, k_perm, q_perm), scale)
            };
            // d_mask [n_ctx, n_tokens] applied across all heads
            let kq = unsafe { ffi::ggml_soft_max_ext(self.ctx, kq, self.d_mask, 1.0, 0.0) };
            let av = unsafe { ffi::ggml_mul_mat(self.ctx, v_perm, kq) };
            let av = unsafe {
                ffi::ggml_reshape_2d(self.ctx,
                    ffi::ggml_cont(self.ctx,
                        ffi::ggml_permute(self.ctx, av, 0, 2, 1, 3)),
                    hp.n_embd, n_tokens)
            };

            let attn = unsafe { ffi::ggml_mul_mat(self.ctx, t("attn_output.weight")?, av) };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, attn) };

            // ── FFN ───────────────────────────────────────────────────────────
            let ffn_in = unsafe {
                ffi::ggml_mul(self.ctx,
                    ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps),
                    t("ffn_norm.weight")?)
            };
            let gate = unsafe {
                ffi::ggml_silu(self.ctx,
                    ffi::ggml_mul_mat(self.ctx, t("ffn_gate.weight")?, ffn_in))
            };
            let up  = unsafe { ffi::ggml_mul_mat(self.ctx, t("ffn_up.weight")?,   ffn_in) };
            let ffn = unsafe {
                ffi::ggml_mul_mat(self.ctx, t("ffn_down.weight")?,
                    ffi::ggml_mul(self.ctx, gate, up))
            };
            cur = unsafe { ffi::ggml_add(self.ctx, cur, ffn) };
        }

        // ── Output: [n_vocab, n_tokens] ────────────────────────────────────────
        let out_norm = model.tensor("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("output_norm.weight missing"))?;
        cur = unsafe {
            ffi::ggml_mul(self.ctx,
                ffi::ggml_rms_norm(self.ctx, cur, hp.rms_eps), out_norm)
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

    /// Prefill: process all N prompt tokens in ONE graph pass.
    /// Builds once, computes once — O(1) graph builds regardless of prompt length.
    pub fn prefill(
        &mut self,
        token_ids: &[u32],
        model: &MappedModel,
    ) -> Result<Vec<f32>, ForwardError> {
        if token_ids.is_empty() {
            return Err(ForwardError::Init("empty prefill token list".into()));
        }

        let n_tokens  = token_ids.len() as i64;
        let start_pos = self.kv.head;

        if start_pos + n_tokens > self.n_ctx {
            return Err(ForwardError::ContextFull);
        }

        // Build one graph for all N tokens
        self.build_prefill_graph(model, n_tokens, start_pos)
            .map_err(|e| ForwardError::Internal(e.to_string()))?;

        // Set token ids: [n_tokens]
        let tok_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.d_token,
                tok_i32.as_ptr() as *const c_void,
                0,
                n_tokens as usize * 4,
            )
        };

        // Set positions: [start_pos, start_pos+1, ..., start_pos+n_tokens-1]
        let pos_i32: Vec<i32> = (start_pos..start_pos + n_tokens)
            .map(|p| p as i32)
            .collect();
        unsafe {
            ffi::ggml_backend_tensor_set(
                self.d_pos,
                pos_i32.as_ptr() as *const c_void,
                0,
                n_tokens as usize * 4,
            )
        };

        // Execute — one compute pass for all tokens
        let status = unsafe { ffi::ggml_backend_graph_compute(self.backend, self.graph) };
        if status != 0 {
            return Err(ForwardError::ComputeFailed(status));
        }

        // Advance KV head by N
        self.kv.head = (start_pos + n_tokens).min(self.n_ctx);

        // Extract last token's logits from [n_vocab, n_tokens]
        // Last token is at column (n_tokens-1): byte offset = (n_tokens-1) * n_vocab * 4
        let n_vocab    = self.dna.n_vocab as usize;
        let last_offset = (n_tokens as usize - 1) * n_vocab * 4;
        let mut out    = vec![0.0f32; n_vocab];
        unsafe {
            ffi::ggml_backend_tensor_get(
                self.d_logits,
                out.as_mut_ptr() as *mut c_void,
                last_offset,
                n_vocab * 4,
            )
        };

        log::info!("[Z.3] Prefill: {} tokens in 1 graph pass (head={})",
            n_tokens, self.kv.head);

        Ok(out)
    }

    /// Decode a single token, return logits.
    pub fn decode_one(
        &mut self,
        token_id: u32,
        model: &MappedModel,
    ) -> Result<Vec<f32>, ForwardError> {
        self.decode_internal(token_id, model).map_err(|e| {
            if e.to_string().contains("context full") {
                ForwardError::ContextFull
            } else {
                ForwardError::Internal(e.to_string())
            }
        })
    }

    /// Reset KV cache — matches generate.rs fwd.reset_kv()
    pub fn reset_kv(&mut self) {
        self.kv.clear();
        self.cleanup_graph_resources();
    }

    /// Expose DNA for callers that need model metadata
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
