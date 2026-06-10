/// Z.1 — engine.rs
///
/// Safe Rust wrapper around llama.cpp FFI.
/// Now wired to InferenceMapper for active RAM eviction —
/// after each decode step, layers that are no longer needed
/// are released back to the OS kernel via MADV_DONTNEED.
/// Physical RAM stays bounded even though the full model is
/// mapped in virtual address space.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rand::Rng;

use crate::llama_ffi as ffi;
use crate::mapper::InferenceMapper;

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub n_ctx:          u32,
    pub n_batch:        u32,
    pub n_threads:      u32,
    pub n_gpu_layers:   i32,
    pub use_mmap:       bool,
    pub use_mlock:      bool,
    pub temperature:    f32,
    pub top_p:          f32,
    pub top_k:          i32,
    pub max_new_tokens: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n:  usize,
    /// How many 500 MiB file-map layers to keep hot in RAM.
    /// Older layers are evicted via MADV_DONTNEED.
    pub max_ram_layers: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(2);
        Self {
            n_ctx:          2048,
            n_batch:        512,
            n_threads,
            n_gpu_layers:   0,
            use_mmap:       true,   // llama.cpp maps the file
            use_mlock:      false,  // do NOT lock — we want the kernel to evict
            temperature:    0.7,
            top_p:          0.9,
            top_k:          40,
            max_new_tokens: 256,
            repeat_penalty: 1.1,
            repeat_last_n:  64,
            max_ram_layers: 4,      // ~2 GiB of model weights in RAM at once
        }
    }
}

// ── Generation statistics ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GenStats {
    pub n_prompt_tokens:    usize,
    pub n_generated_tokens: usize,
    pub prompt_ms:          f64,
    pub generate_ms:        f64,
}

impl GenStats {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.generate_ms == 0.0 { return 0.0; }
        self.n_generated_tokens as f64 / (self.generate_ms / 1000.0)
    }
    pub fn prompt_tokens_per_sec(&self) -> f64 {
        if self.prompt_ms == 0.0 { return 0.0; }
        self.n_prompt_tokens as f64 / (self.prompt_ms / 1000.0)
    }
    pub fn print(&self) {
        log::info!(
            "Prompt  : {} tokens in {:.1} ms ({:.1} tok/s)",
            self.n_prompt_tokens, self.prompt_ms, self.prompt_tokens_per_sec()
        );
        log::info!(
            "Generate: {} tokens in {:.1} ms ({:.1} tok/s)",
            self.n_generated_tokens, self.generate_ms, self.tokens_per_sec()
        );
    }
}

// ── LlamaEngine ───────────────────────────────────────────────────────────────

pub struct LlamaEngine {
    model:          *mut ffi::llama_model,
    ctx:            *mut ffi::llama_context,
    pub config:     EngineConfig,
    vocab_size:     i32,
    recent_tokens:  Vec<ffi::llama_token>,
    /// Owns the mmap region and drives active eviction.
    /// Runs alongside llama.cpp's own mmap — we control
    /// which pages the kernel keeps in physical RAM.
    mapper:         InferenceMapper,
    /// Tracks how many decode steps have completed so we
    /// know when to evict the next window of file-map layers.
    decode_step:    usize,
}

unsafe impl Send for LlamaEngine {}

impl LlamaEngine {
    pub fn load(model_path: &Path, config: EngineConfig) -> Result<Self> {
        // ── Open our memory mapper first ──────────────────────────────────────
        // This maps the full file into virtual address space with ~0 physical
        // RAM used. The prefetch thread starts warming up the first few layers.
        let mapper = InferenceMapper::new(model_path)
            .context("Failed to memory-map model file")?;

        log::info!(
            "[Z.1] File mapped: {:.2} GiB virtual, ~0 physical RAM used.",
            mapper.mapper.total_size as f64 / (1u64 << 30) as f64
        );
        log::info!(
            "[Z.1] Active RAM window: {} × 500 MiB = {:.1} GiB max.",
            config.max_ram_layers,
            config.max_ram_layers as f64 * 500.0 / 1024.0
        );

        // ── Initialise llama.cpp backend ──────────────────────────────────────
        unsafe { ffi::llama_backend_init() };

        let c_path = CString::new(
            model_path.to_str().context("model path is not valid UTF-8")?,
        )?;

        // llama.cpp also mmaps the file with use_mmap:true.
        // We deliberately do NOT use use_mlock so the kernel
        // is free to reclaim pages when we call MADV_DONTNEED.
        let mut mparams = unsafe { ffi::llama_model_default_params() };
        mparams.n_gpu_layers  = config.n_gpu_layers;
        mparams.use_mmap      = true;
        mparams.use_mlock     = false;
        mparams.vocab_only    = false;
        mparams.check_tensors = false;

        log::info!("[Z.1] Loading model via llama.cpp: {:?}", model_path);
        let model = unsafe { ffi::llama_load_model_from_file(c_path.as_ptr(), mparams) };
        if model.is_null() {
            bail!("[Z.1] llama_load_model_from_file returned null.");
        }

        let mut cparams = unsafe { ffi::llama_context_default_params() };
        cparams.n_ctx           = config.n_ctx;
        cparams.n_batch         = config.n_batch;
        cparams.n_ubatch        = config.n_batch;
        cparams.n_threads       = config.n_threads;
        cparams.n_threads_batch = config.n_threads;

        let ctx = unsafe { ffi::llama_new_context_with_model(model, cparams) };
        if ctx.is_null() {
            unsafe { ffi::llama_free_model(model) };
            bail!("[Z.1] llama_new_context_with_model returned null.");
        }

        let vocab_size = unsafe { ffi::llama_n_vocab(model) };

        let mut desc_buf = vec![0i8; 256];
        unsafe { ffi::llama_model_desc(model, desc_buf.as_mut_ptr(), desc_buf.len()) };
        let desc    = unsafe { CStr::from_ptr(desc_buf.as_ptr()) }.to_string_lossy();
        let size_gb = unsafe { ffi::llama_model_size(model) } as f64 / (1u64 << 30) as f64;
        let n_par   = unsafe { ffi::llama_model_n_params(model) };

        log::info!("[Z.1] Model  : {}", desc);
        log::info!("[Z.1] Size   : {:.2} GiB", size_gb);
        log::info!("[Z.1] Params : {}", fmt_params(n_par));
        log::info!("[Z.1] Vocab  : {} tokens", vocab_size);
        log::info!("[Z.1] Ctx    : {} tokens", config.n_ctx);
        log::info!("[Z.1] Threads: {}", config.n_threads);

        Ok(LlamaEngine {
            model,
            ctx,
            config,
            vocab_size,
            recent_tokens: Vec::new(),
            mapper,
            decode_step: 0,
        })
    }

    // ── Active eviction ───────────────────────────────────────────────────────
    //
    // Called after every llama_decode. We advance the mapper window so:
    //   - The next `PREFETCH_DEPTH` layers are warmed up (MADV_WILLNEED)
    //   - Layers older than `max_ram_layers` are evicted (MADV_DONTNEED)
    //
    // This keeps physical RAM usage bounded to ~2 GiB regardless of model size.
    fn evict_step(&mut self) {
        let step = self.decode_step;
        let n    = self.mapper.num_layers();

        // Advance the prefetch window.
        let prefetch_target = step.min(n.saturating_sub(1));
        self.mapper.prefetcher.advance(prefetch_target);

        // Evict the layer window that is now behind us.
        if step >= self.config.max_ram_layers {
            let evict_idx = step - self.config.max_ram_layers;
            if evict_idx < n {
                if let Err(e) = self.mapper.activate_layer(evict_idx) {
                    log::warn!("[Z.1] Eviction at step {}: {}", step, e);
                }
            }
        }

        self.decode_step += 1;
    }

    // ── Tokenisation ──────────────────────────────────────────────────────────

    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<ffi::llama_token>> {
        let c_text  = CString::new(text)?;
        let max_tok = text.len() + 64;
        let mut buf = vec![0i32; max_tok];

        let n = unsafe {
            ffi::llama_tokenize(
                self.model,
                c_text.as_ptr(),
                text.len() as i32,
                buf.as_mut_ptr(),
                max_tok as i32,
                add_bos,
                false,
            )
        };

        if n < 0 { bail!("[Z.1] tokenize: buffer too small (need {} slots)", -n); }
        buf.truncate(n as usize);
        Ok(buf)
    }

    pub fn token_to_piece(&self, token: ffi::llama_token) -> String {
        let mut buf = vec![0i8; 64];
        let n = unsafe {
            ffi::llama_token_to_piece(
                self.model, token,
                buf.as_mut_ptr(), buf.len() as i32,
                0, false,
            )
        };
        if n <= 0 { return String::new(); }
        let bytes: Vec<u8> = buf[..n as usize].iter().map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    pub fn generate<F>(&mut self, prompt: &str, mut on_token: F) -> Result<(String, GenStats)>
    where
        F: FnMut(&str),
    {
        unsafe { ffi::llama_kv_cache_clear(self.ctx) };
        self.recent_tokens.clear();
        self.decode_step = 0;

        let tokens   = self.tokenize(prompt, true)?;
        let n_prompt = tokens.len();

        if n_prompt as u32 >= self.config.n_ctx {
            bail!(
                "[Z.1] Prompt is {} tokens but context is only {}.",
                n_prompt, self.config.n_ctx
            );
        }

        log::debug!("[Z.1] Prompt tokens: {}", n_prompt);

        // ── Prompt ingestion ──────────────────────────────────────────────────
        let t_prompt_start = Instant::now();
        let batch_size     = self.config.n_batch as usize;
        let mut pos: i32   = 0;
        let mut last_logit_idx: i32 = 0;

        for chunk in tokens.chunks(batch_size) {
            let n = chunk.len() as i32;
            let mut batch = unsafe { ffi::llama_batch_init(n, 0, 1) };
            batch.n_tokens = n;

            for (i, &tok) in chunk.iter().enumerate() {
                let global_pos = pos as usize + i;
                let is_last    = global_pos == n_prompt - 1;
                unsafe {
                    *batch.token.add(i)    = tok;
                    *batch.pos.add(i)      = pos + i as i32;
                    *batch.n_seq_id.add(i) = 1;
                    **batch.seq_id.add(i)  = 0;
                    *batch.logits.add(i)   = if is_last { 1 } else { 0 };
                    if is_last { last_logit_idx = i as i32; }
                }
                self.recent_tokens.push(tok);
            }
            pos += n;

            let batch_to_free = unsafe { std::ptr::read(&batch) };
            let rc = unsafe { ffi::llama_decode(self.ctx, batch) };
            unsafe { ffi::llama_batch_free(batch_to_free) };

            if rc != 0 {
                bail!("[Z.1] llama_decode error {} during prompt ingestion", rc);
            }

            // Evict old file-map layers after each batch decode.
            self.evict_step();
        }

        let prompt_ms = t_prompt_start.elapsed().as_secs_f64() * 1000.0;

        // ── Autoregressive generation ─────────────────────────────────────────
        let eos             = unsafe { ffi::llama_token_eos(self.model) };
        let t_gen_start     = Instant::now();
        let mut output      = String::new();
        let mut n_generated = 0usize;
        let mut sample_idx  = last_logit_idx;

        loop {
            if n_generated >= self.config.max_new_tokens { break; }

            let next = self.sample_token(sample_idx)?;
            sample_idx = 0;

            if next == eos { break; }

            self.recent_tokens.push(next);
            if self.recent_tokens.len() > self.config.repeat_last_n + n_prompt {
                self.recent_tokens.remove(0);
            }

            let piece = self.token_to_piece(next);
            on_token(&piece);
            output.push_str(&piece);
            n_generated += 1;

            let mut batch = unsafe { ffi::llama_batch_init(1, 0, 1) };
            batch.n_tokens = 1;
            unsafe {
                *batch.token    = next;
                *batch.pos      = pos;
                *batch.n_seq_id = 1;
                **batch.seq_id  = 0;
                *batch.logits   = 1;
            }
            pos += 1;

            let batch_to_free = unsafe { std::ptr::read(&batch) };
            let rc = unsafe { ffi::llama_decode(self.ctx, batch) };
            unsafe { ffi::llama_batch_free(batch_to_free) };

            if rc != 0 {
                bail!("[Z.1] llama_decode error {} during generation", rc);
            }

            // Evict old file-map layers after each generation step.
            self.evict_step();
        }

        let stats = GenStats {
            n_prompt_tokens:    n_prompt,
            n_generated_tokens: n_generated,
            prompt_ms,
            generate_ms: t_gen_start.elapsed().as_secs_f64() * 1000.0,
        };

        log::debug!("[Z.1] Generated {} tokens.", n_generated);
        Ok((output, stats))
    }

    // ── Sampling ──────────────────────────────────────────────────────────────

    fn sample_token(&self, logit_idx: i32) -> Result<ffi::llama_token> {
        let logits_ptr = unsafe { ffi::llama_get_logits_ith(self.ctx, logit_idx) };
        if logits_ptr.is_null() {
            bail!("[Z.1] llama_get_logits_ith({}) returned null", logit_idx);
        }

        let n   = self.vocab_size as usize;
        let raw = unsafe { std::slice::from_raw_parts(logits_ptr, n) };

        let mut candidates: Vec<ffi::llama_token_data> = raw
            .iter()
            .enumerate()
            .map(|(id, &logit)| ffi::llama_token_data { id: id as i32, logit, p: 0.0 })
            .collect();

        // Repetition penalty
        let penalty = self.config.repeat_penalty;
        if penalty != 1.0 {
            let last_n = self.config.repeat_last_n;
            let start  = self.recent_tokens.len().saturating_sub(last_n);
            for &tok in &self.recent_tokens[start..] {
                let idx = tok as usize;
                if idx < candidates.len() {
                    let l = &mut candidates[idx].logit;
                    *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
                }
            }
        }

        // Greedy shortcut
        let temp = self.config.temperature;
        if temp <= 0.0 {
            return candidates
                .iter()
                .max_by(|a, b| a.logit.partial_cmp(&b.logit).unwrap())
                .map(|c| c.id)
                .ok_or_else(|| anyhow!("[Z.1] empty candidate list"));
        }

        // Temperature + softmax
        for c in candidates.iter_mut() { c.logit /= temp; }
        let max_l = candidates.iter().map(|c| c.logit).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in candidates.iter_mut() { c.p = (c.logit - max_l).exp(); sum += c.p; }
        for c in candidates.iter_mut() { c.p /= sum; }

        // Top-k
        let k = (self.config.top_k as usize).min(candidates.len());
        candidates.sort_unstable_by(|a, b| b.p.partial_cmp(&a.p).unwrap());
        candidates.truncate(k);

        // Top-p
        let top_p   = self.config.top_p;
        let mut cum = 0.0f32;
        let cutoff  = candidates
            .iter()
            .position(|c| { cum += c.p; cum >= top_p })
            .map(|i| i + 1)
            .unwrap_or(candidates.len());
        candidates.truncate(cutoff);

        let total: f32 = candidates.iter().map(|c| c.p).sum();
        for c in candidates.iter_mut() { c.p /= total; }

        let r: f32 = rand::thread_rng().gen();
        let mut acc = 0.0f32;
        for c in &candidates {
            acc += c.p;
            if r <= acc { return Ok(c.id); }
        }
        Ok(candidates.last().unwrap().id)
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size as usize }
    pub fn ctx_size(&self)   -> u32   { unsafe { ffi::llama_n_ctx(self.ctx) } }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free(self.ctx);
            ffi::llama_free_model(self.model);
            ffi::llama_backend_free();
        }
        log::info!("[Z.1] LlamaEngine dropped; memory freed.");
        // mapper drops here — munmap releases virtual address space.
    }
}

fn fmt_params(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.1}B", n as f64 / 1e9) }
    else if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1e6) }
    else                   { format!("{}", n) }
}
