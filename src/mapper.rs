/// Z.1 — mapper.rs
///
/// Memory-mapped file streamer with background prefetch and active eviction.
/// Fully cross-platform using `memmap2` (supports Linux, macOS, and Windows).
///
/// How it works:
///   1. FileMapper mmaps the entire model file into virtual address space.
///   2. Prefetcher runs a background thread that issues MADV_WILLNEED (on Unix)
///      on upcoming 500 MiB windows.
///   3. evict_layer() issues MADV_DONTNEED (on Unix) on old windows so the 
///      kernel reclaims that physical RAM. On Windows, the OS memory manager
///      handles paging natively.

use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use std::ffi::c_void;

use anyhow::{bail, Result};
use memmap2::{Mmap, MmapOptions};

/// Size of each eviction/prefetch window (500 MiB).
pub const LAYER_SIZE_BYTES: usize = 500 * 1024 * 1024;
/// How many windows ahead to prefetch.
pub const PREFETCH_DEPTH:   usize = 2;
/// How many windows to keep hot before evicting.
pub const MAX_RAM_LAYERS:   usize = 4;

// ── Cross-Platform Memory Advice ─────────────────────────────────────────────

#[derive(Copy, Clone, Debug)]
pub enum Advice {
    WillNeed,
    DontNeed,
}

#[cfg(unix)]
fn safe_madvise(
    addr: NonNull<c_void>,
    len: usize,
    advice: Advice,
) -> std::io::Result<()> {
    let c_advice = match advice {
        Advice::WillNeed => libc::MADV_WILLNEED,
        Advice::DontNeed => libc::MADV_DONTNEED,
    };
    if unsafe { libc::madvise(addr.as_ptr(), len, c_advice) } == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
fn safe_madvise(
    _addr: NonNull<c_void>,
    _len: usize,
    _advice: Advice,
) -> std::io::Result<()> {
    // Windows memory manager natively handles demand paging aggressively.
    // We fall back safely so compilation succeeds and the app runs normally.
    Ok(())
}

/// Ceiling division: number of LAYER_SIZE_BYTES windows that cover `total` bytes.
#[inline]
pub fn num_layers(total: usize) -> usize {
    (total + LAYER_SIZE_BYTES - 1) / LAYER_SIZE_BYTES
}

// ── FileMapper ────────────────────────────────────────────────────────────────

/// Owns the full virtual mmap region.
/// Physical pages are loaded lazily by the kernel/OS on first access.
pub struct FileMapper {
    _mmap: Mmap, // Keeps the memory mapping alive, automatically unmaps on drop
    pub ptr: NonNull<c_void>,
    pub total_size: usize,
}

unsafe impl Send for FileMapper {}
unsafe impl Sync for FileMapper {}

impl FileMapper {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let sz = file.metadata()?.len() as usize;
        
        if sz == 0 {
            bail!("model file is empty");
        }

        // Cross-platform safe mmap using memmap2
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // ggml requires a mutable void pointer for its backend injection
        let ptr = unsafe { NonNull::new_unchecked(mmap.as_ptr() as *mut c_void) };

        log::info!(
            "[Z.1] FileMapper: {:.2} GiB mapped to virtual memory via memmap2.",
            sz as f64 / (1u64 << 30) as f64
        );

        Ok(FileMapper {
            _mmap: mmap,
            ptr,
            total_size: sz,
        })
    }

    /// Raw pointer to the start of window `idx`.
    ///
    /// # Safety
    /// Caller must ensure `idx < num_layers(self.total_size)`.
    pub unsafe fn layer_ptr(&self, idx: usize) -> NonNull<u8> {
        NonNull::new_unchecked(
            (self.ptr.as_ptr() as usize + idx * LAYER_SIZE_BYTES) as *mut u8
        )
    }

    /// Evict window `idx` from physical RAM.
    pub fn evict_layer(&self, idx: usize) -> std::io::Result<()> {
        let offset = idx * LAYER_SIZE_BYTES;
        if offset >= self.total_size { return Ok(()); }
        
        let len = LAYER_SIZE_BYTES.min(self.total_size - offset);
        let addr = unsafe {
            NonNull::new_unchecked((self.ptr.as_ptr() as usize + offset) as *mut c_void)
        };
        safe_madvise(addr, len, Advice::DontNeed)
    }

    /// Warm up window `idx` — ask the OS to start loading it from disk.
    pub fn prefetch_layer(&self, idx: usize) -> std::io::Result<()> {
        let offset = idx * LAYER_SIZE_BYTES;
        if offset >= self.total_size { return Ok(()); }
        
        let len = LAYER_SIZE_BYTES.min(self.total_size - offset);
        let addr = unsafe {
            NonNull::new_unchecked((self.ptr.as_ptr() as usize + offset) as *mut c_void)
        };
        safe_madvise(addr, len, Advice::WillNeed)
    }
}

// ── Prefetcher ────────────────────────────────────────────────────────────────

/// Background thread that issues WILLNEED ahead of the current window.
pub struct Prefetcher {
    handle:        Option<std::thread::JoinHandle<()>>,
    stop:          Arc<AtomicBool>,
    notify:        Arc<(Mutex<()>, Condvar)>,
    pub current_layer: Arc<AtomicUsize>,
}

impl Prefetcher {
    pub fn spawn(base: NonNull<c_void>, total_size: usize) -> Self {
        let stop          = Arc::new(AtomicBool::new(false));
        let notify        = Arc::new((Mutex::new(()), Condvar::new()));
        let current_layer = Arc::new(AtomicUsize::new(0));
        let base_usize    = base.as_ptr() as usize;
        let nlayers       = num_layers(total_size);

        let (stop2, notify2, cur2) = (
            stop.clone(), notify.clone(), current_layer.clone(),
        );

        let handle = std::thread::spawn(move || {
            let (lock, cvar) = &*notify2;
            loop {
                let g = lock.lock().unwrap();
                let _ = cvar.wait_timeout(g, Duration::from_millis(100));

                if stop2.load(Ordering::Acquire) { break; }

                let cur = cur2.load(Ordering::Acquire);
                for i in 1..=PREFETCH_DEPTH {
                    let t = cur + i;
                    if t >= nlayers { break; }
                    
                    let offset = t * LAYER_SIZE_BYTES;
                    let len    = LAYER_SIZE_BYTES.min(total_size - offset);
                    let addr = unsafe {
                        NonNull::new_unchecked(
                            (base_usize + offset) as *mut c_void
                        )
                    };
                    
                    if let Err(e) = safe_madvise(addr, len, Advice::WillNeed) {
                        log::warn!("[Z.1] Prefetch layer {}: {}", t, e);
                    } else {
                        log::debug!("[Z.1] Prefetched window {}.", t);
                    }
                }
            }
            log::debug!("[Z.1] Prefetch thread exiting.");
        });

        Prefetcher { handle: Some(handle), stop, notify, current_layer }
    }

    pub fn advance(&self, layer: usize) {
        self.current_layer.store(layer, Ordering::Release);
        self.notify.1.notify_one();
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.notify.1.notify_all();
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
    }
}

// ── InferenceMapper ───────────────────────────────────────────────────────────

pub struct InferenceMapper {
    pub mapper:     FileMapper,
    pub prefetcher: Prefetcher,
}

impl InferenceMapper {
    pub fn new(path: &Path) -> Result<Self> {
        let mapper     = FileMapper::open(path)?;
        let prefetcher = Prefetcher::spawn(mapper.ptr, mapper.total_size);

        // Warm up the first windows immediately.
        for i in 0..PREFETCH_DEPTH.min(num_layers(mapper.total_size)) {
            if let Err(e) = mapper.prefetch_layer(i) {
                log::warn!("[Z.1] Initial prefetch window {}: {}", i, e);
            }
        }

        Ok(InferenceMapper { mapper, prefetcher })
    }

    pub fn num_layers(&self) -> usize {
        num_layers(self.mapper.total_size)
    }

    pub fn activate_layer(&mut self, evict_idx: usize) -> Result<()> {
        let n = self.num_layers();
        if evict_idx >= n {
            return Ok(());
        }
        if let Err(e) = self.mapper.evict_layer(evict_idx) {
            log::warn!("[Z.1] DONTNEED window {}: {}", evict_idx, e);
        } else {
            log::debug!("[Z.1] Evicted window {} from physical RAM.", evict_idx);
        }
        Ok(())
    }

    pub fn shutdown(mut self) {
        self.prefetcher.shutdown();
        // mapper drops cleanly via RAII
    }
}

impl Drop for InferenceMapper {
    fn drop(&mut self) {
        self.prefetcher.shutdown();
    }
}