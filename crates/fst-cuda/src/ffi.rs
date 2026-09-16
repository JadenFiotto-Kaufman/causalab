//! The crate's only `unsafe`: `dlopen` of `libcudart` and `libcufile`, the C
//! signatures transcribed from their headers, and a thin safe wrapper around
//! each call that turns the C status into a [`CudaError`].
//!
//! Nothing here allocates device memory or decides anything. Device buffers
//! arrive as plain addresses because the caller owns them — the crate-wide
//! contract that a [`crate::DevicePtr`] describes live device memory of the
//! stated size, bounds-checked by the caller before it gets here. That
//! contract is what makes the wrappers safe functions; every `unsafe` block
//! below states the invariant it relies on.
//!
//! Function pointers are copied out of their `libloading::Symbol` and stored
//! next to the `Library` that owns them, so a pointer never outlives its
//! `dlclose`. Objects that must call back into a library on drop (pinned
//! buffers, streams) hold an `Arc` to it.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;

use libloading::Library;

use crate::CudaError;

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Where `libcudart` is looked for, in order: sonames first so the dynamic
/// loader's own search path (`LD_LIBRARY_PATH`, `ldconfig`) wins, then the
/// default toolkit prefix.
pub const CUDART_CANDIDATES: &[&str] = &[
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so",
    "/usr/local/cuda/lib64/libcudart.so.13",
    "/usr/local/cuda/lib64/libcudart.so.12",
    "/usr/local/cuda/lib64/libcudart.so",
];

/// Where `libcufile` is looked for, in order: the sonames, then the same
/// absolute paths `fst_core::env::Env::probe` checks (kept in step by hand;
/// `fst-core` depends on this crate, not the other way round), so the planner
/// and the loader agree on what "present" means.
pub const CUFILE_CANDIDATES: &[&str] = &[
    "libcufile.so.0",
    "libcufile.so",
    "/usr/local/cuda/lib64/libcufile.so",
    "/usr/local/cuda/lib64/libcufile.so.0",
    "/usr/lib/x86_64-linux-gnu/libcufile.so.0",
    "/usr/lib/x86_64-linux-gnu/libcufile.so",
];

/// `dlopen` the first candidate that loads. Failing all of them, an
/// [`CudaError::Unavailable`] naming every candidate and what the loader
/// said about each.
fn open_first(library: &'static str, candidates: &[&str]) -> Result<(Library, String), CudaError> {
    let mut reasons = Vec::with_capacity(candidates.len());
    for name in candidates {
        // SAFETY: loading a shared library runs its initialisers. libcudart
        // and libcufile are NVIDIA's runtime libraries and running their
        // constructors in this process is the purpose of this crate; a
        // failed attempt leaves the process untouched.
        match unsafe { Library::new(*name) } {
            Ok(lib) => return Ok((lib, (*name).to_owned())),
            Err(e) => reasons.push(format!("{name}: {e}")),
        }
    }
    if reasons.is_empty() {
        reasons.push("no candidate library names given".to_owned());
    }
    Err(CudaError::Unavailable {
        library,
        reason: reasons.join("; "),
    })
}

/// Resolve `name` in `lib` to a function pointer of type `T`.
///
/// `T` must be the `unsafe extern "C" fn` alias declared in this module for
/// `name`, transcribed from the library's header; every call site pairs the
/// two on one line so the pairing can be checked by eye.
fn symbol<T: Copy>(lib: &Library, library: &'static str, name: &str) -> Result<T, CudaError> {
    // SAFETY: `T` is the function-pointer type transcribed from the header
    // for `name`, and the pointer is only ever called through that type. It
    // is copied out of the `Symbol` into a struct that also owns `lib`, so
    // it is never used after the library is unloaded.
    let sym = unsafe { lib.get::<T>(name) }.map_err(|e| CudaError::Unavailable {
        library,
        reason: format!("{name}: {e}"),
    })?;
    Ok(*sym)
}

// ---------------------------------------------------------------------------
// cudart: types from cuda_runtime_api.h / driver_types.h
// ---------------------------------------------------------------------------

/// `cudaError_t`: an `int`-sized enum; 0 is `cudaSuccess`.
type CudaErrorT = c_int;
/// `cudaStream_t`: an opaque handle.
type CudaStreamT = *mut c_void;

/// `cudaErrorMemoryAllocation`: what a null result from `cudaHostAlloc` is
/// reported as, and what an unrepresentable size maps to.
pub const CUDA_ERROR_MEMORY_ALLOCATION: i32 = 2;
/// `cudaErrorInvalidDevice`: what a device ordinal that does not fit a C
/// `int` maps to, before any call is made.
pub const CUDA_ERROR_INVALID_DEVICE: i32 = 101;
/// `cudaMemcpyKind::cudaMemcpyHostToDevice`.
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
/// `cudaMemcpyKind::cudaMemcpyDeviceToHost`.
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
/// `cudaHostAllocPortable`: the pinned allocation is usable from any context.
const CUDA_HOST_ALLOC_PORTABLE: c_uint = 0x01;
/// `cudaStreamNonBlocking`: no implicit synchronisation with the null stream.
const CUDA_STREAM_NON_BLOCKING: c_uint = 0x01;
/// `cudaEventBlockingSync | cudaEventDisableTiming`: a waiting thread
/// sleeps instead of spinning, and the event carries no timestamp.
const CUDA_EVENT_FENCE_FLAGS: c_uint = 0x01 | 0x02;
/// `cudaEvent_t`: an opaque handle.
type CudaEventT = *mut c_void;

type CudaSetDevice = unsafe extern "C" fn(c_int) -> CudaErrorT;
type CudaGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> CudaErrorT;
type CudaMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CudaErrorT;
type CudaHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CudaErrorT;
type CudaFreeHost = unsafe extern "C" fn(*mut c_void) -> CudaErrorT;
type CudaMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, CudaStreamT) -> CudaErrorT;
/// `cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)`.
type CudaMemcpy2DAsync = unsafe extern "C" fn(
    *mut c_void,
    usize,
    *const c_void,
    usize,
    usize,
    usize,
    c_int,
    CudaStreamT,
) -> CudaErrorT;
type CudaStreamCreateWithFlags = unsafe extern "C" fn(*mut CudaStreamT, c_uint) -> CudaErrorT;
type CudaStreamSynchronize = unsafe extern "C" fn(CudaStreamT) -> CudaErrorT;
type CudaStreamDestroy = unsafe extern "C" fn(CudaStreamT) -> CudaErrorT;
type CudaGetErrorString = unsafe extern "C" fn(CudaErrorT) -> *const c_char;
type CudaRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> CudaErrorT;
type CudaEventCreateWithFlags = unsafe extern "C" fn(*mut CudaEventT, c_uint) -> CudaErrorT;
type CudaEventRecord = unsafe extern "C" fn(CudaEventT, CudaStreamT) -> CudaErrorT;
type CudaEventSynchronize = unsafe extern "C" fn(CudaEventT) -> CudaErrorT;
type CudaEventDestroy = unsafe extern "C" fn(CudaEventT) -> CudaErrorT;

/// `libcudart`, loaded, with the symbols this crate uses resolved.
pub struct Cudart {
    /// The candidate name that loaded.
    pub loaded_from: String,
    set_device: CudaSetDevice,
    get_device_count: CudaGetDeviceCount,
    mem_get_info: CudaMemGetInfo,
    host_alloc: CudaHostAlloc,
    free_host: CudaFreeHost,
    memcpy_async: CudaMemcpyAsync,
    memcpy_2d_async: CudaMemcpy2DAsync,
    stream_create_with_flags: CudaStreamCreateWithFlags,
    stream_synchronize: CudaStreamSynchronize,
    stream_destroy: CudaStreamDestroy,
    get_error_string: CudaGetErrorString,
    runtime_get_version: CudaRuntimeGetVersion,
    event_create_with_flags: CudaEventCreateWithFlags,
    event_record: CudaEventRecord,
    event_synchronize: CudaEventSynchronize,
    event_destroy: CudaEventDestroy,
    /// Keeps every pointer above valid.
    _lib: Library,
}

impl std::fmt::Debug for Cudart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cudart")
            .field("loaded_from", &self.loaded_from)
            .finish_non_exhaustive()
    }
}

/// A device ordinal as the C `int` cudart takes.
fn device_index(device: u32) -> Result<c_int, CudaError> {
    c_int::try_from(device).map_err(|_| CudaError::Runtime {
        call: "cudaSetDevice",
        code: CUDA_ERROR_INVALID_DEVICE,
    })
}

impl Cudart {
    /// Load the first of `candidates` that opens and resolve every symbol.
    pub fn load(candidates: &[&str]) -> Result<Cudart, CudaError> {
        const LIB: &str = "libcudart";
        let (lib, loaded_from) = open_first(LIB, candidates)?;
        Ok(Cudart {
            loaded_from,
            set_device: symbol::<CudaSetDevice>(&lib, LIB, "cudaSetDevice")?,
            get_device_count: symbol::<CudaGetDeviceCount>(&lib, LIB, "cudaGetDeviceCount")?,
            mem_get_info: symbol::<CudaMemGetInfo>(&lib, LIB, "cudaMemGetInfo")?,
            host_alloc: symbol::<CudaHostAlloc>(&lib, LIB, "cudaHostAlloc")?,
            free_host: symbol::<CudaFreeHost>(&lib, LIB, "cudaFreeHost")?,
            memcpy_async: symbol::<CudaMemcpyAsync>(&lib, LIB, "cudaMemcpyAsync")?,
            memcpy_2d_async: symbol::<CudaMemcpy2DAsync>(&lib, LIB, "cudaMemcpy2DAsync")?,
            stream_create_with_flags: symbol::<CudaStreamCreateWithFlags>(
                &lib,
                LIB,
                "cudaStreamCreateWithFlags",
            )?,
            stream_synchronize: symbol::<CudaStreamSynchronize>(
                &lib,
                LIB,
                "cudaStreamSynchronize",
            )?,
            stream_destroy: symbol::<CudaStreamDestroy>(&lib, LIB, "cudaStreamDestroy")?,
            get_error_string: symbol::<CudaGetErrorString>(&lib, LIB, "cudaGetErrorString")?,
            runtime_get_version: symbol::<CudaRuntimeGetVersion>(
                &lib,
                LIB,
                "cudaRuntimeGetVersion",
            )?,
            event_create_with_flags: symbol::<CudaEventCreateWithFlags>(
                &lib,
                LIB,
                "cudaEventCreateWithFlags",
            )?,
            event_record: symbol::<CudaEventRecord>(&lib, LIB, "cudaEventRecord")?,
            event_synchronize: symbol::<CudaEventSynchronize>(&lib, LIB, "cudaEventSynchronize")?,
            event_destroy: symbol::<CudaEventDestroy>(&lib, LIB, "cudaEventDestroy")?,
            _lib: lib,
        })
    }

    /// Map a `cudaError_t` to a result, logging the runtime's message for
    /// anything but success.
    fn status(&self, call: &'static str, code: CudaErrorT) -> Result<(), CudaError> {
        if code == 0 {
            return Ok(());
        }
        tracing::debug!(call, code, message = %self.error_string(code), "CUDA runtime call failed");
        Err(CudaError::Runtime { call, code })
    }

    /// `cudaGetErrorString`: the runtime's description of `code`.
    pub fn error_string(&self, code: i32) -> String {
        // SAFETY: cudaGetErrorString takes any int and returns either null
        // or a pointer to a static NUL-terminated string owned by the
        // runtime, which lives as long as the library we hold.
        let raw = unsafe { (self.get_error_string)(code) };
        if raw.is_null() {
            return format!("unknown CUDA error {code}");
        }
        // SAFETY: non-null, NUL-terminated, static, as above.
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }

    /// `cudaSetDevice` for the calling thread.
    pub fn set_device(&self, device: u32) -> Result<(), CudaError> {
        let index = device_index(device)?;
        // SAFETY: plain call with a by-value argument; no pointers.
        let code = unsafe { (self.set_device)(index) };
        self.status("cudaSetDevice", code)
    }

    /// `cudaGetDeviceCount`.
    pub fn device_count(&self) -> Result<u32, CudaError> {
        let mut count: c_int = 0;
        // SAFETY: `count` is a live local for the duration of the call.
        let code = unsafe { (self.get_device_count)(&mut count) };
        self.status("cudaGetDeviceCount", code)?;
        Ok(u32::try_from(count).unwrap_or(0))
    }

    /// `cudaRuntimeGetVersion`: `major * 1000 + minor * 10`.
    pub fn runtime_version(&self) -> Result<i32, CudaError> {
        let mut version: c_int = 0;
        // SAFETY: `version` is a live local for the duration of the call.
        let code = unsafe { (self.runtime_get_version)(&mut version) };
        self.status("cudaRuntimeGetVersion", code)?;
        Ok(version)
    }

    /// `cudaMemGetInfo` for the calling thread's current device:
    /// `(free, total)` in bytes.
    pub fn mem_get_info(&self) -> Result<(u64, u64), CudaError> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: both out-pointers are live locals for the duration of the call.
        let code = unsafe { (self.mem_get_info)(&mut free, &mut total) };
        self.status("cudaMemGetInfo", code)?;
        Ok((free as u64, total as u64))
    }

    /// `cudaHostAlloc(nbytes, cudaHostAllocPortable)`, zeroed, freed on drop.
    pub fn host_alloc(lib: &Arc<Cudart>, nbytes: usize) -> Result<HostAllocation, CudaError> {
        if nbytes == 0 {
            return Ok(HostAllocation {
                ptr: NonNull::dangling(),
                len: 0,
                owner: None,
            });
        }
        let mut raw: *mut c_void = ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer for the duration of the call;
        // the flag is a documented cudaHostAlloc flag.
        let code = unsafe { (lib.host_alloc)(&mut raw, nbytes, CUDA_HOST_ALLOC_PORTABLE) };
        lib.status("cudaHostAlloc", code)?;
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or(CudaError::Runtime {
            call: "cudaHostAlloc",
            code: CUDA_ERROR_MEMORY_ALLOCATION,
        })?;
        // SAFETY: cudaHostAlloc succeeded, so `ptr` is `nbytes` writable
        // bytes. Zeroing them makes every later `&[u8]` view of the buffer a
        // view of initialised memory.
        unsafe { ptr::write_bytes(ptr.as_ptr(), 0, nbytes) };
        Ok(HostAllocation {
            ptr,
            len: nbytes,
            owner: Some(Arc::clone(lib)),
        })
    }

    /// `cudaMemcpyAsync(dst, src, HostToDevice, stream)`.
    ///
    /// The copy is asynchronous: `src` must stay alive and unmodified until
    /// `stream` is synchronised, which is the [`crate::DeviceCopier`]
    /// contract its callers uphold.
    pub fn memcpy_h2d(&self, dst: usize, src: &[u8], stream: &Stream) -> Result<(), CudaError> {
        // SAFETY: `dst..dst + src.len()` lies inside a device allocation the
        // caller owns — the DevicePtr contract, range-checked by the caller
        // before this call. `src` is valid for reads of `src.len()` bytes.
        // The runtime only reads `src` until the stream is synchronised, and
        // the caller keeps it alive that long.
        let code = unsafe {
            (self.memcpy_async)(
                dst as *mut c_void,
                src.as_ptr().cast::<c_void>(),
                src.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.raw,
            )
        };
        self.status("cudaMemcpyAsync", code)
    }

    /// `cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height,
    /// HostToDevice, stream)`: `height` rows of `width` bytes, the source
    /// rows `src_pitch` apart from the start of `src`, the destination rows
    /// `dst_pitch` apart from `dst`.
    ///
    /// `src` must hold `(height - 1) * src_pitch + width` bytes and `width`
    /// may not exceed either pitch — checked here, so the runtime is never
    /// handed a shape it would reject or read past. Asynchronous like
    /// [`Cudart::memcpy_h2d`]: `src` stays alive and unmodified until the
    /// stream is synchronised.
    #[allow(clippy::too_many_arguments)]
    pub fn memcpy_2d_h2d(
        &self,
        dst: usize,
        dst_pitch: usize,
        src: &[u8],
        src_pitch: usize,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<(), CudaError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if width > src_pitch || width > dst_pitch {
            return Err(CudaError::BadGeometry {
                width: width as u64,
                src_pitch: src_pitch as u64,
                dst_pitch: dst_pitch as u64,
            });
        }
        let needed = (height - 1)
            .checked_mul(src_pitch)
            .and_then(|n| n.checked_add(width));
        match needed {
            Some(needed) if needed <= src.len() => {}
            _ => {
                return Err(CudaError::Overrun {
                    nbytes: (width as u64).saturating_mul(height as u64),
                    offset: 0,
                    capacity: src.len() as u64,
                });
            }
        }
        // SAFETY: every destination row `dst + i * dst_pitch .. + width`
        // lies inside a device allocation the caller owns — the DevicePtr
        // contract, checked with `Copy2D::check` by the caller before this
        // call. Every source row lies inside `src`, checked just above. The
        // runtime only reads `src` until the stream is synchronised, and
        // the caller keeps it alive that long.
        let code = unsafe {
            (self.memcpy_2d_async)(
                dst as *mut c_void,
                dst_pitch,
                src.as_ptr().cast::<c_void>(),
                src_pitch,
                width,
                height,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.raw,
            )
        };
        self.status("cudaMemcpy2DAsync", code)
    }

    /// `cudaMemcpyAsync(dst, src, DeviceToHost, stream)`.
    ///
    /// Asynchronous: `dst` holds its final bytes only after `stream` is
    /// synchronised; until then the caller neither reads nor frees it.
    pub fn memcpy_d2h(&self, dst: &mut [u8], src: usize, stream: &Stream) -> Result<(), CudaError> {
        // SAFETY: `src..src + dst.len()` lies inside a device allocation the
        // caller owns (the DevicePtr contract, range-checked by the caller).
        // `dst` is valid for writes of `dst.len()` bytes; the runtime writes
        // it until the stream is synchronised and the caller does not read
        // or release it before then.
        let code = unsafe {
            (self.memcpy_async)(
                dst.as_mut_ptr().cast::<c_void>(),
                src as *const c_void,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream.raw,
            )
        };
        self.status("cudaMemcpyAsync", code)
    }
}

/// A `cudaHostAlloc` allocation: page-locked, zeroed, freed on drop.
pub struct HostAllocation {
    ptr: NonNull<u8>,
    len: usize,
    /// `None` only for the zero-length allocation, which owns nothing.
    owner: Option<Arc<Cudart>>,
}

// SAFETY: the allocation is ordinary host memory reached only through the
// slice accessors below, which follow Rust's borrow rules; nothing about it is
// tied to the allocating thread.
unsafe impl Send for HostAllocation {}
// SAFETY: as above — `&HostAllocation` only ever hands out `&[u8]`.
unsafe impl Sync for HostAllocation {}

impl HostAllocation {
    /// The zero-length allocation, which owns nothing and frees nothing.
    pub fn empty() -> HostAllocation {
        HostAllocation {
            ptr: NonNull::dangling(),
            len: 0,
            owner: None,
        }
    }

    /// Bytes in the allocation.
    pub fn nbytes(&self) -> usize {
        self.len
    }

    /// The bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr..ptr + len` is the cudaHostAlloc allocation (or a
        // dangling pointer with len 0, which is a valid empty slice), zeroed
        // at creation and alive until drop; `&self` excludes a live `&mut`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as for `as_slice`; `&mut self` makes this the only view.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        let Some(owner) = &self.owner else {
            return;
        };
        // SAFETY: `ptr` came from cudaHostAlloc on the library `owner` keeps
        // loaded, and this is the only place it is freed.
        let code = unsafe { (owner.free_host)(self.ptr.as_ptr().cast::<c_void>()) };
        if code != 0 {
            tracing::warn!(
                code,
                len = self.len,
                "cudaFreeHost failed; pinned memory leaked"
            );
        }
    }
}

/// A non-blocking CUDA stream, destroyed on drop.
pub struct Stream {
    raw: CudaStreamT,
    lib: Arc<Cudart>,
}

// SAFETY: a cudaStream_t is a handle the CUDA runtime documents as usable
// from any host thread, concurrently; nothing here dereferences it.
unsafe impl Send for Stream {}
// SAFETY: as above.
unsafe impl Sync for Stream {}

impl Stream {
    /// `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` on the calling
    /// thread's current device.
    pub fn create(lib: &Arc<Cudart>) -> Result<Stream, CudaError> {
        let mut raw: CudaStreamT = ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer for the duration of the call;
        // the flag is a documented cudaStreamCreateWithFlags flag.
        let code = unsafe { (lib.stream_create_with_flags)(&mut raw, CUDA_STREAM_NON_BLOCKING) };
        lib.status("cudaStreamCreateWithFlags", code)?;
        Ok(Stream {
            raw,
            lib: Arc::clone(lib),
        })
    }

    /// `cudaStreamSynchronize`: wait for everything enqueued so far.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: `raw` came from cudaStreamCreateWithFlags and is destroyed
        // only in `Drop`, so it is live here.
        let code = unsafe { (self.lib.stream_synchronize)(self.raw) };
        self.lib.status("cudaStreamSynchronize", code)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: `raw` came from cudaStreamCreateWithFlags and this is the
        // only place it is destroyed.
        let code = unsafe { (self.lib.stream_destroy)(self.raw) };
        if code != 0 {
            tracing::warn!(code, "cudaStreamDestroy failed");
        }
    }
}

/// A CUDA event recorded on a stream: a point in the stream's history that
/// can be waited for. Destroyed on drop.
pub struct Event {
    raw: CudaEventT,
    lib: Arc<Cudart>,
}

// SAFETY: a cudaEvent_t is a handle the CUDA runtime documents as usable
// from any host thread, concurrently; nothing here dereferences it.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl Event {
    /// `cudaEventCreateWithFlags` then `cudaEventRecord` on `stream`: the
    /// event completes when everything enqueued on `stream` so far has.
    pub fn record(lib: &Arc<Cudart>, stream: &Stream) -> Result<Event, CudaError> {
        let mut raw: CudaEventT = ptr::null_mut();
        // SAFETY: `raw` is a live out-pointer for the duration of the call;
        // the flags are documented cudaEventCreateWithFlags flags.
        let code = unsafe { (lib.event_create_with_flags)(&mut raw, CUDA_EVENT_FENCE_FLAGS) };
        lib.status("cudaEventCreateWithFlags", code)?;
        let event = Event {
            raw,
            lib: Arc::clone(lib),
        };
        // SAFETY: both handles are live: the event was just created and the
        // stream is destroyed only in its own `Drop`.
        let code = unsafe { (lib.event_record)(event.raw, stream.raw) };
        lib.status("cudaEventRecord", code)?;
        Ok(event)
    }

    /// `cudaEventSynchronize`: block until the recorded point is reached.
    pub fn wait(&self) -> Result<(), CudaError> {
        // SAFETY: `raw` came from cudaEventCreateWithFlags and is destroyed
        // only in `Drop`, so it is live here.
        let code = unsafe { (self.lib.event_synchronize)(self.raw) };
        self.lib.status("cudaEventSynchronize", code)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `raw` came from cudaEventCreateWithFlags and this is the
        // only place it is destroyed.
        let code = unsafe { (self.lib.event_destroy)(self.raw) };
        if code != 0 {
            tracing::warn!(code, "cudaEventDestroy failed");
        }
    }
}

// ---------------------------------------------------------------------------
// cufile: types from /usr/local/cuda/include/cufile.h (cuFile 1.15.1)
// ---------------------------------------------------------------------------

/// `CUFILEOP_BASE_ERR` (cufile.h line 73): `CUfileOpError` values above this
/// are cuFile's own; `IS_CUFILE_ERR(err)` (line 177) is `abs(err) > 5000`.
pub const CUFILEOP_BASE_ERR: i32 = 5000;
/// `CU_FILE_SUCCESS` (line 77).
pub const CU_FILE_SUCCESS: c_int = 0;
/// `CU_FILE_CUDA_DRIVER_ERROR` (line 88): when `err` is this, `cu_err`
/// carries the `CUresult`.
pub const CU_FILE_CUDA_DRIVER_ERROR: c_int = CUFILEOP_BASE_ERR + 11;
/// `CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD` (line 290).
pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

/// `CUfileError_t` (cufile.h lines 165–171): `CUfileOpError err; CUresult
/// cu_err;` — two `int`-sized C enums.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CUfileError {
    /// The cuFile status.
    pub err: c_int,
    /// The CUDA driver status when `err` is `CU_FILE_CUDA_DRIVER_ERROR`.
    pub cu_err: c_int,
}

/// The anonymous `handle` union of `CUfileDescr_t` (cufile.h lines 299–302).
#[repr(C)]
#[derive(Clone, Copy)]
pub union CUfileHandleUnion {
    /// Linux: the file descriptor.
    pub fd: c_int,
    /// Windows: unsupported; here so the union is pointer-sized.
    pub handle: *mut c_void,
}

/// `CUfileDescr_t` (cufile.h lines 297–304): `enum CUfileFileHandleType
/// type; union { int fd; void *handle; } handle; const CUfileFSOps_t *fs_ops;`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUfileDescr {
    /// `CU_FILE_HANDLE_TYPE_OPAQUE_FD` for a Linux fd.
    pub type_: c_int,
    /// The fd.
    pub handle: CUfileHandleUnion,
    /// Null: no user-space file system.
    pub fs_ops: *const c_void,
}

/// `CUfileHandle_t` (cufile.h line 310): `void *`.
pub type CUfileHandleT = *mut c_void;

type CuFileDriverOpen = unsafe extern "C" fn() -> CUfileError;
type CuFileDriverClose = unsafe extern "C" fn() -> CUfileError;
type CuFileHandleRegister =
    unsafe extern "C" fn(*mut CUfileHandleT, *mut CUfileDescr) -> CUfileError;
type CuFileHandleDeregister = unsafe extern "C" fn(CUfileHandleT);
type CuFileBufRegister = unsafe extern "C" fn(*const c_void, usize, c_int) -> CUfileError;
type CuFileBufDeregister = unsafe extern "C" fn(*const c_void) -> CUfileError;
type CuFileRead = unsafe extern "C" fn(
    CUfileHandleT,
    *mut c_void,
    usize,
    libc::off_t,
    libc::off_t,
) -> libc::ssize_t;
type CuFileWrite = unsafe extern "C" fn(
    CUfileHandleT,
    *const c_void,
    usize,
    libc::off_t,
    libc::off_t,
) -> libc::ssize_t;
type CuFileGetVersion = unsafe extern "C" fn(*mut c_int) -> CUfileError;

/// `libcufile`, loaded, with the symbols this crate uses resolved.
pub struct CuFileLib {
    /// The candidate name that loaded.
    pub loaded_from: String,
    driver_open: CuFileDriverOpen,
    driver_close: CuFileDriverClose,
    handle_register: CuFileHandleRegister,
    handle_deregister: CuFileHandleDeregister,
    buf_register: CuFileBufRegister,
    buf_deregister: CuFileBufDeregister,
    read: CuFileRead,
    write: CuFileWrite,
    /// Absent in old releases.
    get_version: Option<CuFileGetVersion>,
    /// Keeps every pointer above valid.
    _lib: Library,
}

impl std::fmt::Debug for CuFileLib {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CuFileLib")
            .field("loaded_from", &self.loaded_from)
            .finish_non_exhaustive()
    }
}

/// Map a `CUfileError_t` to a result.
fn cufile_status(call: &'static str, status: CUfileError) -> Result<(), CudaError> {
    if status.err == CU_FILE_SUCCESS {
        return Ok(());
    }
    if status.err == CU_FILE_CUDA_DRIVER_ERROR {
        tracing::debug!(
            call,
            code = status.err,
            cu_result = status.cu_err,
            "cuFile call failed in the CUDA driver"
        );
    } else {
        tracing::debug!(call, code = status.err, "cuFile call failed");
    }
    Err(CudaError::CuFile {
        call,
        code: status.err,
    })
}

/// Map the `ssize_t` from `cuFileRead`/`cuFileWrite` (cufile.h lines
/// 405–414): `>= 0` is bytes moved, `-1` means `errno` holds a file-system
/// error, any other negative value is `-CUfileOpError`. `code` in the error
/// follows the header's `IS_CUFILE_ERR` convention: above 5000 it is a
/// cuFile status, otherwise an `errno`.
pub fn map_io_return(call: &'static str, ret: isize, errno: i32) -> Result<usize, CudaError> {
    match ret {
        n if n >= 0 => Ok(n.unsigned_abs()),
        -1 => Err(CudaError::CuFile { call, code: errno }),
        n => Err(CudaError::CuFile {
            call,
            code: i32::try_from(n.unsigned_abs()).unwrap_or(i32::MAX),
        }),
    }
}

impl CuFileLib {
    /// Load the first of `candidates` that opens and resolve every symbol.
    pub fn load(candidates: &[&str]) -> Result<CuFileLib, CudaError> {
        const LIB: &str = "libcufile";
        let (lib, loaded_from) = open_first(LIB, candidates)?;
        // cufile.h line 455: `#define cuFileDriverClose cuFileDriverClose_v2`;
        // the unsuffixed symbol is the pre-1.7 ABI, kept for old libraries.
        let driver_close = symbol::<CuFileDriverClose>(&lib, LIB, "cuFileDriverClose_v2")
            .or_else(|_| symbol::<CuFileDriverClose>(&lib, LIB, "cuFileDriverClose"))?;
        Ok(CuFileLib {
            loaded_from,
            driver_open: symbol::<CuFileDriverOpen>(&lib, LIB, "cuFileDriverOpen")?,
            driver_close,
            handle_register: symbol::<CuFileHandleRegister>(&lib, LIB, "cuFileHandleRegister")?,
            handle_deregister: symbol::<CuFileHandleDeregister>(
                &lib,
                LIB,
                "cuFileHandleDeregister",
            )?,
            buf_register: symbol::<CuFileBufRegister>(&lib, LIB, "cuFileBufRegister")?,
            buf_deregister: symbol::<CuFileBufDeregister>(&lib, LIB, "cuFileBufDeregister")?,
            read: symbol::<CuFileRead>(&lib, LIB, "cuFileRead")?,
            write: symbol::<CuFileWrite>(&lib, LIB, "cuFileWrite")?,
            get_version: symbol::<CuFileGetVersion>(&lib, LIB, "cuFileGetVersion").ok(),
            _lib: lib,
        })
    }

    /// `cuFileGetVersion`, when the library has it.
    pub fn version(&self) -> Option<i32> {
        let get_version = self.get_version?;
        let mut version: c_int = 0;
        // SAFETY: `version` is a live local for the duration of the call.
        let status = unsafe { get_version(&mut version) };
        (status.err == CU_FILE_SUCCESS).then_some(version)
    }

    /// `cuFileDriverOpen`.
    pub fn driver_open(&self) -> Result<(), CudaError> {
        // SAFETY: no arguments; the library is loaded.
        let status = unsafe { (self.driver_open)() };
        cufile_status("cuFileDriverOpen", status)
    }

    /// `cuFileDriverClose` (`_v2` where present).
    pub fn driver_close(&self) -> Result<(), CudaError> {
        // SAFETY: no arguments; the library is loaded.
        let status = unsafe { (self.driver_close)() };
        cufile_status("cuFileDriverClose", status)
    }

    /// `cuFileHandleRegister` for an open file; deregistered on drop. The
    /// handle borrows the file so the fd it wraps outlives it.
    pub fn handle_register<'a>(&'a self, file: &'a File) -> Result<FileHandle<'a>, CudaError> {
        let mut descr = CUfileDescr {
            type_: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: CUfileHandleUnion {
                handle: ptr::null_mut(),
            },
            fs_ops: ptr::null(),
        };
        descr.handle.fd = file.as_raw_fd();
        let mut raw: CUfileHandleT = ptr::null_mut();
        // SAFETY: both pointers are to live locals laid out as cufile.h
        // declares them (checked by the size tests below); the descriptor
        // names an fd that `file` keeps open for the handle's lifetime.
        let status = unsafe { (self.handle_register)(&mut raw, &mut descr) };
        cufile_status("cuFileHandleRegister", status)?;
        Ok(FileHandle {
            raw,
            lib: self,
            _file: file,
        })
    }

    /// `cuFileBufRegister(base, len, 0)`; deregistered on drop.
    pub fn buf_register(&self, base: usize, len: usize) -> Result<BufRegistration<'_>, CudaError> {
        // SAFETY: `base..base + len` is a device allocation the caller owns
        // (the DevicePtr contract); cuFile only records the mapping.
        let status = unsafe { (self.buf_register)(base as *const c_void, len, 0) };
        cufile_status("cuFileBufRegister", status)?;
        Ok(BufRegistration { base, lib: self })
    }

    /// One `cuFileRead` call: up to `size` bytes from `file_offset` into the
    /// device buffer at `base + buf_offset`. May return short.
    pub fn read(
        &self,
        handle: &FileHandle<'_>,
        base: usize,
        size: usize,
        file_offset: libc::off_t,
        buf_offset: libc::off_t,
    ) -> Result<usize, CudaError> {
        // SAFETY: `handle` is a live registration; `base + buf_offset ..
        // + size` lies inside a device allocation the caller owns and has
        // range-checked (the DevicePtr contract). cuFileRead is synchronous,
        // so the buffer is not touched after this returns.
        let ret = unsafe {
            (self.read)(
                handle.raw,
                base as *mut c_void,
                size,
                file_offset,
                buf_offset,
            )
        };
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        map_io_return("cuFileRead", ret, errno)
    }

    /// One `cuFileWrite` call: up to `size` bytes from the device buffer at
    /// `base + buf_offset` to `file_offset`. May return short.
    pub fn write(
        &self,
        handle: &FileHandle<'_>,
        base: usize,
        size: usize,
        file_offset: libc::off_t,
        buf_offset: libc::off_t,
    ) -> Result<usize, CudaError> {
        // SAFETY: as for `read`; the device range is only read from.
        let ret = unsafe {
            (self.write)(
                handle.raw,
                base as *const c_void,
                size,
                file_offset,
                buf_offset,
            )
        };
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        map_io_return("cuFileWrite", ret, errno)
    }
}

/// A registered cuFile handle, deregistered on drop.
pub struct FileHandle<'a> {
    raw: CUfileHandleT,
    lib: &'a CuFileLib,
    _file: &'a File,
}

impl Drop for FileHandle<'_> {
    fn drop(&mut self) {
        // SAFETY: `raw` came from cuFileHandleRegister, the file it wraps is
        // still open (borrowed), and this is the only deregistration.
        unsafe { (self.lib.handle_deregister)(self.raw) };
    }
}

/// A registered device buffer, deregistered on drop.
pub struct BufRegistration<'a> {
    base: usize,
    lib: &'a CuFileLib,
}

impl Drop for BufRegistration<'_> {
    fn drop(&mut self) {
        // SAFETY: `base` was registered by cuFileBufRegister and this is the
        // only deregistration.
        let status = unsafe { (self.lib.buf_deregister)(self.base as *const c_void) };
        if status.err != CU_FILE_SUCCESS {
            tracing::warn!(code = status.err, "cuFileBufDeregister failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::*;

    /// cufile.h lines 165–171: two `int` enums, no padding.
    #[test]
    fn cufile_error_layout_matches_header() {
        assert_eq!(size_of::<CUfileError>(), 8);
        assert_eq!(align_of::<CUfileError>(), 4);
    }

    /// cufile.h lines 297–304: a 4-byte enum, 4 bytes of padding, a
    /// pointer-sized union, a pointer — 24 bytes, pointer-aligned.
    #[test]
    fn cufile_descr_layout_matches_header() {
        assert_eq!(size_of::<CUfileHandleUnion>(), size_of::<*mut c_void>());
        assert_eq!(size_of::<CUfileDescr>(), 24);
        assert_eq!(align_of::<CUfileDescr>(), 8);
        assert_eq!(std::mem::offset_of!(CUfileDescr, handle), 8);
        assert_eq!(std::mem::offset_of!(CUfileDescr, fs_ops), 16);
        assert_eq!(size_of::<CUfileHandleT>(), 8);
    }

    #[test]
    fn io_return_convention() {
        assert!(matches!(map_io_return("cuFileRead", 4096, 0), Ok(4096)));
        assert!(matches!(map_io_return("cuFileRead", 0, 0), Ok(0)));
        assert!(matches!(
            map_io_return("cuFileRead", -1, libc::EINVAL),
            Err(CudaError::CuFile { call: "cuFileRead", code }) if code == libc::EINVAL
        ));
        assert!(matches!(
            map_io_return("cuFileRead", -5011, 0),
            Err(CudaError::CuFile {
                call: "cuFileRead",
                code: 5011
            })
        ));
    }

    #[test]
    fn missing_library_is_unavailable_with_every_candidate_named() {
        let err = Cudart::load(&["/nonexistent/libcudart.so.99", "libfst-no-such-lib.so"])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.starts_with("libcudart is not available: "), "{err}");
        assert!(err.contains("/nonexistent/libcudart.so.99: "), "{err}");
        assert!(err.contains("libfst-no-such-lib.so: "), "{err}");
        assert!(matches!(
            CuFileLib::load(&["/nonexistent/libcufile.so.0"]),
            Err(CudaError::Unavailable {
                library: "libcufile",
                ..
            })
        ));
        assert!(matches!(
            CuFileLib::load(&[]),
            Err(CudaError::Unavailable { library: "libcufile", reason }) if reason.contains("no candidate")
        ));
    }

    #[test]
    fn device_index_rejects_ordinals_beyond_c_int() {
        assert!(device_index(0).is_ok());
        assert!(matches!(
            device_index(u32::MAX),
            Err(CudaError::Runtime {
                call: "cudaSetDevice",
                code: CUDA_ERROR_INVALID_DEVICE
            })
        ));
    }
}
