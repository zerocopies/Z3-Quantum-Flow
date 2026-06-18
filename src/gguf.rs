/// Z.1 — gguf.rs
///
/// Parses the GGUF v1/v2/v3 file header to extract metadata and tensor
/// descriptors *without* loading any weight data into RAM.
///
/// Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

// ── Magic & version ───────────────────────────────────────────────────────────
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const SUPPORTED_VERSIONS: [u32; 3] = [1, 2, 3];

// ── Value types as defined in the GGUF spec ───────────────────────────────────
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(s) = self { Some(s) } else { None }
    }
    pub fn as_u32(&self) -> Option<u32> {
        if let GgufValue::U32(v) = self { Some(*v) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let GgufValue::U64(v) = self { Some(*v) } else { None }
    }
}

// ── Tensor descriptor (no data, just shape + offset) ─────────────────────────
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name:        String,
    /// Shape in each dimension.
    pub dims:        Vec<u64>,
    /// GGML quantization type id.
    pub ggml_type:   u32,
    /// Byte offset from the start of the tensor-data section.
    pub offset:      u64,
}

impl TensorInfo {
    /// Total number of elements (product of all dimensions).
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

// ── Top-level header ──────────────────────────────────────────────────────────
#[derive(Debug)]
pub struct GgufHeader {
    pub version:    u32,
    pub n_tensors:  u64,
    pub metadata:   HashMap<String, GgufValue>,
    pub tensors:    Vec<TensorInfo>,
    /// Byte offset in the file where tensor data begins (after the header).
    pub data_offset: u64,
}

impl GgufHeader {
    /// Parse the GGUF header from `path`.  Weights are *not* loaded.
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        Self::parse(&mut r)
    }

    fn parse<R: Read + Seek>(r: &mut R) -> io::Result<Self> {
        // Magic
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("not a GGUF file (magic = 0x{:08X})", magic),
            ));
        }

        // Version
        let version = read_u32(r)?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {}", version),
            ));
        }

        let n_tensors  = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };
        let n_kv       = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };

        // Metadata key-value pairs
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key   = read_gguf_string(r)?;
            let value = read_gguf_value(r, version)?;
            metadata.insert(key, value);
        }

        // Tensor descriptors
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name      = read_gguf_string(r)?;
            let n_dims    = read_u32(r)? as usize;
            let mut dims  = vec![0u64; n_dims];
            for d in dims.iter_mut() {
                *d = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };
            }
            let ggml_type = read_u32(r)?;
            let offset    = read_u64(r)?;
            tensors.push(TensorInfo { name, dims, ggml_type, offset });
        }

        // Align to 32 bytes to find the start of tensor data.
        let pos = r.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufHeader { version, n_tensors, metadata, tensors, data_offset })
    }

    // ── Convenience accessors ─────────────────────────────────────────────────

    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn context_length(&self) -> Option<u32> {
        // Key varies by architecture, e.g. "llama.context_length"
        let arch = self.architecture().unwrap_or("llama");
        self.metadata
            .get(&format!("{}.context_length", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture().unwrap_or("llama");
        self.metadata
            .get(&format!("{}.embedding_length", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn layer_count(&self) -> Option<u32> {
        let arch = self.architecture().unwrap_or("llama");
        self.metadata
            .get(&format!("{}.block_count", arch))
            .and_then(|v| v.as_u32())
    }

    pub fn print_summary(&self) {
        log::info!("── GGUF Header ──────────────────────────────");
        log::info!("  Version    : {}", self.version);
        log::info!("  Model      : {}", self.model_name().unwrap_or("unknown"));
        log::info!("  Arch       : {}", self.architecture().unwrap_or("unknown"));
        log::info!("  Layers     : {}", self.layer_count().unwrap_or(0));
        log::info!("  Context    : {}", self.context_length().unwrap_or(0));
        log::info!("  Tensors    : {}", self.n_tensors);
        log::info!("  Data offset: {} bytes", self.data_offset);
        log::info!("─────────────────────────────────────────────");
    }
}

// ── Low-level readers ─────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_u32(r)?.to_le_bytes()))
}
fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_u64(r)?.to_le_bytes()))
}

fn read_gguf_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_gguf_value<R: Read + Seek>(r: &mut R, version: u32) -> io::Result<GgufValue> {
    let type_id = read_u32(r)?;
    match type_id {
        0  => Ok(GgufValue::U8(read_u8(r)?)),
        1  => Ok(GgufValue::I8(read_u8(r)? as i8)),
        2  => Ok(GgufValue::U16(read_u16(r)?)),
        3  => Ok(GgufValue::I16(read_u16(r)? as i16)),
        4  => Ok(GgufValue::U32(read_u32(r)?)),
        5  => Ok(GgufValue::I32(read_u32(r)? as i32)),
        6  => Ok(GgufValue::F32(read_f32(r)?)),
        7  => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8  => Ok(GgufValue::String(read_gguf_string(r)?)),
        9  => {
            // Array: element type + count + elements
            let elem_type = read_u32(r)?;
            let count     = if version == 1 { read_u32(r)? as u64 } else { read_u64(r)? };
            let mut arr   = Vec::with_capacity(count as usize);
            for _ in 0..count {
                // Temporarily wrap type_id back so we can recurse.
                // We do this by writing the elem_type into a local buffer.
                let elem = read_gguf_value_by_type(r, elem_type)?;
                arr.push(elem);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_u64(r)? as i64)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {}", other),
        )),
    }
}

fn read_gguf_value_by_type<R: Read + Seek>(r: &mut R, type_id: u32) -> io::Result<GgufValue> {
    match type_id {
        0  => Ok(GgufValue::U8(read_u8(r)?)),
        1  => Ok(GgufValue::I8(read_u8(r)? as i8)),
        2  => Ok(GgufValue::U16(read_u16(r)?)),
        3  => Ok(GgufValue::I16(read_u16(r)? as i16)),
        4  => Ok(GgufValue::U32(read_u32(r)?)),
        5  => Ok(GgufValue::I32(read_u32(r)? as i32)),
        6  => Ok(GgufValue::F32(read_f32(r)?)),
        7  => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8  => Ok(GgufValue::String(read_gguf_string(r)?)),
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_u64(r)? as i64)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported array element type {}", other),
        )),
    }
}
