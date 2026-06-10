/// Z.1 — loader.rs
///
/// Option B1: Zero-copy model loader.
/// Wraps the mmap region as a ggml CPU backend buffer and points
/// each tensor's data pointer directly into the mmap.
/// No heap allocation for weights.

use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::ggml_ffi::{
    self as ffi,
    ggml_backend_buffer, ggml_backend_buffer_free,
    ggml_backend_cpu_buffer_from_ptr, ggml_backend_tensor_alloc,
    ggml_context, ggml_free, ggml_init, ggml_new_tensor, ggml_set_name,
    GgmlInitParams, GgmlType,
};
use crate::gguf::GgufHeader;
use crate::mapper::FileMapper;

pub struct MappedModel {
    pub file_mapper: FileMapper,
    pub ggml_ctx:    *mut ggml_context,
    pub weight_buf:  *mut ggml_backend_buffer,
    pub tensors:     HashMap<String, *mut ffi::ggml_tensor>,
    pub header:      GgufHeader,
}

unsafe impl Send for MappedModel {}
unsafe impl Sync for MappedModel {}

impl MappedModel {
    pub fn load(path: &Path) -> Result<Self> {
        // 1. Parse header (no weights)
        let header = GgufHeader::from_file(path)
            .context("Failed to parse GGUF header")?;

        log::info!("[Z.1 Loader] GGUF v{}: {} tensors, data offset={}",
            header.version, header.n_tensors, header.data_offset);

        // 2. Memory-map the full file
        let file_mapper = FileMapper::open(path)
            .context("Failed to mmap model file")?;

        let mmap_base = file_mapper.ptr.as_ptr() as usize;
        let file_size = file_mapper.total_size;

        log::info!("[Z.1 Loader] mmap base=0x{:x}, size={:.2} GiB",
            mmap_base, file_size as f64 / (1u64 << 30) as f64);

        // 3. Wrap mmap as ggml CPU buffer — zero copy, zero heap
        let weight_buf = unsafe {
            ggml_backend_cpu_buffer_from_ptr(file_mapper.ptr.as_ptr(), file_size)
        };
        if weight_buf.is_null() {
            bail!("[Z.1 Loader] ggml_backend_cpu_buffer_from_ptr returned null");
        }
        log::info!("[Z.1 Loader] ggml weight buffer wraps mmap — 0 heap bytes for weights.");

        // 4. ggml context for tensor descriptors only (~100 KiB)
        let desc_mem = (header.n_tensors as usize + 16) * 512;
        let ggml_ctx = unsafe {
            ggml_init(GgmlInitParams {
                mem_size:   desc_mem,
                mem_buffer: std::ptr::null_mut(),
                no_alloc:   true,
            })
        };
        if ggml_ctx.is_null() {
            bail!("[Z.1 Loader] ggml_init returned null");
        }

        // 5. Create tensors and point them into the mmap
        let data_base = mmap_base + header.data_offset as usize;
        let mut tensors = HashMap::with_capacity(header.tensors.len());
        let mut skipped = 0usize;

        for ti in &header.tensors {
            let tensor_addr = data_base + ti.offset as usize;
            let ggml_type   = GgmlType::from(ti.ggml_type);

            if ggml_type == GgmlType::Unknown {
                log::warn!("[Z.1 Loader] Unknown type {} for '{}', skipping.", ti.ggml_type, ti.name);
                skipped += 1;
                continue;
            }
            if tensor_addr >= mmap_base + file_size {
                log::warn!("[Z.1 Loader] '{}' offset out of bounds, skipping.", ti.name);
                skipped += 1;
                continue;
            }

            let mut ne = [1i64; 4];
            for (i, &d) in ti.dims.iter().enumerate().take(4) {
                ne[i] = d as i64;
            }
            let ndim = ti.dims.len().max(1) as libc::c_int;

            let tensor = unsafe { ggml_new_tensor(ggml_ctx, ggml_type as i32, ndim, ne.as_ptr()) };
            if tensor.is_null() {
                log::warn!("[Z.1 Loader] ggml_new_tensor null for '{}'.", ti.name);
                skipped += 1;
                continue;
            }

            let c_name = CString::new(ti.name.as_str()).unwrap_or_default();
            unsafe { ggml_set_name(tensor, c_name.as_ptr()) };

            // Zero-copy injection: tensor->data = mmap_base + data_offset + tensor_offset
            unsafe {
                ggml_backend_tensor_alloc(
                    weight_buf,
                    tensor,
                    tensor_addr as *mut std::ffi::c_void,
                );
            }

            tensors.insert(ti.name.clone(), tensor);
        }

        log::info!("[Z.1 Loader] {} tensors loaded ({} skipped). Heap for weights: 0 bytes.",
            tensors.len(), skipped);

        Ok(MappedModel { file_mapper, ggml_ctx, weight_buf, tensors, header })
    }

    pub fn tensor(&self, name: &str) -> Option<*mut ffi::ggml_tensor> {
        self.tensors.get(name).copied()
    }

    pub fn layer_tensor(&self, layer: usize, suffix: &str) -> Option<*mut ffi::ggml_tensor> {
        self.tensor(&format!("blk.{}.{}", layer, suffix))
    }

    pub fn n_layers(&self) -> usize {
        self.header.layer_count().unwrap_or(32) as usize
    }

    pub fn n_embd(&self) -> usize {
        self.header.embedding_length().unwrap_or(4096) as usize
    }

    pub fn print_tensor_summary(&self) {
        log::info!("[Z.1 Loader] {} tensors loaded.", self.tensors.len());
        let mut names: Vec<&String> = self.tensors.keys().collect();
        names.sort();
        for name in names.iter().take(10) {
            log::debug!("[Z.1 Loader]   {}", name);
        }
        if names.len() > 10 {
            log::debug!("[Z.1 Loader]   ... and {} more.", names.len() - 10);
        }
    }
}

impl Drop for MappedModel {
    fn drop(&mut self) {
        unsafe {
            ggml_free(self.ggml_ctx);
            ggml_backend_buffer_free(self.weight_buf);
        }
        log::info!("[Z.1 Loader] MappedModel dropped. All weight memory released.");
    }
}
