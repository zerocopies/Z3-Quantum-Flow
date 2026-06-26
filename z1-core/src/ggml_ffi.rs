#![allow(non_camel_case_types, dead_code)]

// Z.1 — ggml_ffi.rs
// Single source of truth for all ggml opaque types and FFI declarations.
// Both loader.rs and graph.rs import from here.

use std::ffi::c_void;
use libc::c_int;

// ── Opaque handle types ───────────────────────────────────────────────────────
#[repr(C)] pub struct ggml_context           { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_tensor            { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_cgraph            { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_backend           { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_backend_buffer    { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_backend_buffer_type { pub _p: [u8; 0] }
#[repr(C)] pub struct ggml_gallocr           { pub _p: [u8; 0] }

pub type ggml_backend_t           = *mut ggml_backend;
pub type ggml_backend_buffer_t    = *mut ggml_backend_buffer;
pub type ggml_backend_buffer_type_t = *mut ggml_backend_buffer_type;
pub type ggml_gallocr_t           = *mut ggml_gallocr;

// ── Init params ───────────────────────────────────────────────────────────────
#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size:   usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc:   bool,
}

// ── ggml type ids ─────────────────────────────────────────────────────────────
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgmlType {
    F32=0, F16=1, Q4_0=2, Q4_1=3,
    Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
    Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15,
    BF16=30, I32=26, Unknown=999,
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0=>Self::F32, 1=>Self::F16, 2=>Self::Q4_0, 3=>Self::Q4_1,
            6=>Self::Q5_0, 7=>Self::Q5_1, 8=>Self::Q8_0, 9=>Self::Q8_1,
            10=>Self::Q2_K, 11=>Self::Q3_K, 12=>Self::Q4_K, 13=>Self::Q5_K,
            14=>Self::Q6_K, 15=>Self::Q8_K, 30=>Self::BF16, 26=>Self::I32,
            _=>Self::Unknown,
        }
    }
}

/// Convenience alias so graph.rs can write `ffi::GGML_TYPE_F32` as a c_int.
pub const GGML_TYPE_F32: c_int = 0;

extern "C" {
    // Context
    pub fn ggml_init(params: GgmlInitParams) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);

    // Tensor creation
    pub fn ggml_new_tensor(ctx: *mut ggml_context, dtype: c_int, ndim: c_int, ne: *const i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_1d(ctx: *mut ggml_context, dtype: c_int, ne0: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_2d(ctx: *mut ggml_context, dtype: c_int, ne0: i64, ne1: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_3d(ctx: *mut ggml_context, dtype: c_int, ne0: i64, ne1: i64, ne2: i64) -> *mut ggml_tensor;
    pub fn ggml_set_name(tensor: *mut ggml_tensor, name: *const libc::c_char);
    pub fn ggml_set_input(tensor: *mut ggml_tensor);
    pub fn ggml_set_output(tensor: *mut ggml_tensor);
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;
    pub fn ggml_nelements(tensor: *const ggml_tensor) -> i64;

    // Backend buffer (for mmap zero-copy weight injection)
    pub fn ggml_backend_cpu_buffer_from_ptr(ptr: *mut c_void, size: usize) -> ggml_backend_buffer_t;
    pub fn ggml_backend_buffer_free(buffer: ggml_backend_buffer_t);
    pub fn ggml_backend_buffer_get_base(buffer: ggml_backend_buffer_t) -> *mut c_void;
    pub fn ggml_backend_tensor_alloc(buffer: ggml_backend_buffer_t, tensor: *mut ggml_tensor, addr: *mut c_void);

    // Tensor data read/write (via tensor metadata)
    pub fn ggml_backend_tensor_set(tensor: *mut ggml_tensor, data: *const c_void, offset: usize, size: usize);
    pub fn ggml_backend_tensor_get(tensor: *const ggml_tensor, data: *mut c_void, offset: usize, size: usize);

    // Ops
    pub fn ggml_get_rows(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor;
    pub fn ggml_mul(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_add(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_mul_mat(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_scale(ctx: *mut ggml_context, a: *mut ggml_tensor, s: f32) -> *mut ggml_tensor;
    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_cpy(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_soft_max_ext(ctx: *mut ggml_context, a: *mut ggml_tensor, mask: *mut ggml_tensor, scale: f32, max_bias: f32) -> *mut ggml_tensor;
    pub fn ggml_rope_ext(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor, c: *mut ggml_tensor, n_dims: c_int, mode: c_int, n_ctx_orig: c_int, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> *mut ggml_tensor;
    pub fn ggml_reshape_2d(ctx: *mut ggml_context, a: *mut ggml_tensor, ne0: i64, ne1: i64) -> *mut ggml_tensor;
    pub fn ggml_reshape_3d(ctx: *mut ggml_context, a: *mut ggml_tensor, ne0: i64, ne1: i64, ne2: i64) -> *mut ggml_tensor;
    pub fn ggml_permute(ctx: *mut ggml_context, a: *mut ggml_tensor, ax0: c_int, ax1: c_int, ax2: c_int, ax3: c_int) -> *mut ggml_tensor;
    pub fn ggml_repeat(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_view_1d(ctx: *mut ggml_context, a: *mut ggml_tensor, ne0: i64, offset: usize) -> *mut ggml_tensor;
    pub fn ggml_view_2d(ctx: *mut ggml_context, a: *mut ggml_tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> *mut ggml_tensor;

    // Graph
    pub fn ggml_new_graph_custom(ctx: *mut ggml_context, size: usize, grads: bool) -> *mut ggml_cgraph;
    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);

    // Backend
    pub fn ggml_backend_cpu_init() -> ggml_backend_t;
    pub fn ggml_backend_cpu_set_n_threads(backend: ggml_backend_t, n_threads: c_int);
    pub fn ggml_backend_free(backend: ggml_backend_t);
    pub fn ggml_backend_graph_compute(backend: ggml_backend_t, cgraph: *mut ggml_cgraph) -> c_int;
    pub fn ggml_backend_alloc_buffer(backend: ggml_backend_t, size: usize) -> ggml_backend_buffer_t;
    pub fn ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t;

    // Graph allocator — allocates memory for all intermediate compute tensors
    pub fn ggml_gallocr_new(buft: ggml_backend_buffer_type_t) -> ggml_gallocr_t;
    pub fn ggml_gallocr_free(galloc: ggml_gallocr_t);
    pub fn ggml_gallocr_alloc_graph(galloc: ggml_gallocr_t, graph: *mut ggml_cgraph) -> bool;
}

// Additional ops used by graph.rs KV cache path
extern "C" {
}