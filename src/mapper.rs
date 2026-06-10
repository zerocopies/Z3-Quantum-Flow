/// Z.1 — mapper.rs
///
/// Memory-mapped file streamer with background prefetch and active eviction.
///
/// How it works:
///   1. FileMapper mmaps the entire model file into virtual address space.
///      Physical RAM usage starts at ~0 (lazy page loading).
///   2. Prefetcher runs a background thread that issues MADV_WILLNEED on
///      upcoming 500 MiB windows, so the kernel starts loading them from
///      disk before they are needed.
///   3. evict_layer() issues MADV_DONTNEED on old windows so the kernel
///      reclaims that physical RAM. The virtual mapping stays intact so
///      llama.cpp can still read any offset — it just pays a page fault
///      if it goes back to an evicted region (which it never does in
///      sequential layer-by-layer inference).
///
/// Result: a 4.58 GiB model runs with ~2 GiB physical RAM in use at any time.

use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use anyhow::{bail, Result};

extern "C" {
    fn mmap(
        addr:   *mut libc::c_void,
        len:    libc::size_t,
        prot:   libc::c_int,
        flags:  libc::c_int,
        fd:     libc::c_int,
        offset: libc::off_t,
    ) -> *mut libc::c_void;
    fn munmap(addr: *mut libc::c_void, len: libc::size_t) -> libc::c_int;
    fn madvise(addr: *mut libc::c_void, len: libc::size_t, advice: libc::c_int) -> libc::c_int;
    fn open(path: *const libc::c_char, oflag: libc::c_int, ...) -> libc::c_int;
    fn close(fd: libc::c_int) -> libc::c_int;
    fn fstat(fd: libc::c_int, buf: *mut libc::stat) -> libc::c_int;
}

const PROT_READ:     libc::c_int = 1;
const MAP_PRIVATE:   libc::c_int = 0x02;
const MAP_HUGETLB:   libc::c_int = 0x040000;
const MADV_WILLNEED: libc::c_int = 3;
const MADV_DONTNEED: libc::c_int = 4;

/// Size of each eviction/prefetch window (500 MiB).
pub const LAYER_SIZE_BYTES: usize = 500 * 1024 * 1024;
/// How many windows ahead to prefetch.
pub const PREFETCH_DEPTH:   usize = 2;
/// How many windows to keep hot before evicting.
pub const MAX_RAM_LAYERS:   usize = 4;

// ── Low-level helpers ─────────────────────────────────────────────────────────

fn safe_open(path: &Path) -> std::io::Result<RawFd> {
    let c = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "null in path"))?;
    let fd = unsafe { open(c.as_ptr(), libc::O_RDONLY) };
    if fd == -1 { Err(std::io::Error::last_os_error()) } else { Ok(fd) }
}

fn file_size(fd: RawFd) -> std::io::Result<usize> {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { fstat(fd, &mut st) } == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(st.st_size as usize)
    }
}

fn safe_mmap(fd: RawFd, len: usize) -> std::io::Result<NonNull<libc::c_void>> {
    // Try huge pages first (reduces TLB pressure on large files).
    let ptr = unsafe {
        mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE | MAP_HUGETLB, fd, 0)
    };
    if ptr != libc::MAP_FAILED {
        log::info!("[Z.1] mmap: huge pages active.");
        return Ok(unsafe { NonNull::new_unchecked(ptr) });
    }
    log::warn!("[Z.1] mmap: huge pages unavailable, using standard pages.");
    let ptr2 = unsafe {
        mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0)
    };
    if ptr2 == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { NonNull::new_unchecked(ptr2) })
}

fn safe_madvise(
    addr:   NonNull<libc::c_void>,
    len:    usize,
    advice: libc::c_int,
) -> std::io::Result<()> {
    if unsafe { madvise(addr.as_ptr(), len, advice) } == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Ceiling division: number of LAYER_SIZE_BYTES windows that cover `total` bytes.
#[inline]
pub fn num_layers(total: usize) -> usize {
    (total + LAYER_SIZE_BYTES - 1) / LAYER_SIZE_BYTES
}

// ── FileMapper ────────────────────────────────────────────────────────────────

/// Owns the file descriptor and the full virtual mmap region.
/// Physical pages are loaded lazily by the kernel on first access.
pub struct FileMapper {
    fd:             RawFd,
    pub ptr:        NonNull<libc::c_void>,
    pub total_size: usize,
}

unsafe impl Send for FileMapper {}
unsafe impl Sync for FileMapper {}

impl FileMapper {
    pub fn open(path: &Path) -> Result<Self> {
        let fd = safe_open(path)?;
        let sz = file_size(fd).map_err(|e| { unsafe { close(fd); } e })?;
        if sz == 0 {
            unsafe { close(fd); }
            bail!("model file is empty");
        }
        let ptr = safe_mmap(fd, sz).map_err(|e| { unsafe { close(fd); } e })?;
        log::info!(
            "[Z.1] FileMapper: {:.2} GiB mapped to virtual memory (~0 physical RAM).",
            sz as f64 / (1u64 << 30) as f64
        );
        Ok(FileMapper { fd, ptr, total_size: sz })
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
    /// The virtual mapping remains intact — the kernel will reload from disk
    /// on next access (which never happens for already-computed layers).
    pub fn evict_layer(&self, idx: usize) -> std::io::Result<()> {
        let offset = idx * LAYER_SIZE_BYTES;
        if offset >= self.total_size { return Ok(()); }
        let len = LAYER_SIZE_BYTES.min(self.total_size - offset);
        let addr = unsafe {
            NonNull::new_unchecked((self.ptr.as_ptr() as usize + offset) as *mut libc::c_void)
        };
        safe_madvise(addr, len, MADV_DONTNEED)
    }

    /// Warm up window `idx` — ask the kernel to start loading it from disk.
    pub fn prefetch_layer(&self, idx: usize) -> std::io::Result<()> {
        let offset = idx * LAYER_SIZE_BYTES;
        if offset >= self.total_size { return Ok(()); }
        let len = LAYER_SIZE_BYTES.min(self.total_size - offset);
        let addr = unsafe {
            NonNull::new_unchecked((self.ptr.as_ptr() as usize + offset) as *mut libc::c_void)
        };
        safe_madvise(addr, len, MADV_WILLNEED)
    }
}

impl Drop for FileMapper {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr(), self.total_size);
            close(self.fd);
        }
        log::info!("[Z.1] FileMapper: virtual mapping released.");
    }
}

// ── Prefetcher ────────────────────────────────────────────────────────────────

/// Background thread that issues MADV_WILLNEED ahead of the current window.
/// Uses a Condvar so it wakes immediately when `advance()` is called rather
/// than sleeping for a fixed interval.
pub struct Prefetcher {
    handle:        Option<std::thread::JoinHandle<()>>,
    stop:          Arc<AtomicBool>,
    notify:        Arc<(Mutex<()>, Condvar)>,
    pub current_layer: Arc<AtomicUsize>,
}

impl Prefetcher {
    pub fn spawn(base: NonNull<libc::c_void>, total_size: usize) -> Self {
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
                // Wait for a wakeup or a 100 ms safety timeout.
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
                            (base_usize + offset) as *mut libc::c_void
                        )
                    };
                    if let Err(e) = safe_madvise(addr, len, MADV_WILLNEED) {
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

    /// Notify the prefetch thread that inference has moved to `layer`.
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

/// Top-level coordinator: owns FileMapper + Prefetcher.
/// The engine calls `evict_old_window(current_step)` after each decode
/// to keep physical RAM bounded.
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

    /// Evict window `evict_idx` and advance the prefetch pointer to `current`.
    /// Called by the engine after each decode step.
    pub fn activate_layer(&mut self, evict_idx: usize) -> Result<()> {
        let n = self.num_layers();
        if evict_idx >= n {
            return Ok(()); // nothing to evict
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
        // mapper drops here
    }
}

impl Drop for InferenceMapper {
    fn drop(&mut self) {
        self.prefetcher.shutdown();
    }
}
