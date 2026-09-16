//! `fst-cuda`: the CUDA runtime and cuFile, loaded at run time.
//!
//! Nothing here is linked against CUDA. `libcudart` and `libcufile` are opened
//! with `dlopen` when first asked for, so the crate builds and its tests run
//! on a machine with neither, and a wheel built anywhere works on a machine
//! with either. A machine without them gets [`CudaError::Unavailable`] with
//! the library name, never a load-time failure.
//!
//! Two capabilities, behind two traits so they can be simulated:
//!
//! * [`DeviceCopier`]: pinned host buffers and asynchronous host-to-device /
//!   device-to-host copies on a stream, contiguous or strided ([`Copy2D`]).
//!   The staging half of every device read and write when GPUDirect Storage
//!   is absent.
//! * [`DirectStorage`]: cuFile — register a file handle and a device buffer,
//!   read a file range straight into device memory. The GDS half.
//!
//! Device memory is never allocated here. The caller (the Python layer, via
//! torch) owns every device buffer and hands in a [`DevicePtr`]; this crate
//! fills it. That keeps accounting with the framework's allocator and means a
//! tensor's lifetime is never tied to an object in this crate.
//!
//! Layout: `ffi` is the one module allowed `unsafe` — `dlopen`, the C
//! signatures, and a safe wrapper per call. [`cudart::CudaRuntime`] and
//! [`cufile::CuFile`] are the real implementations over those wrappers;
//! [`sim::SimCuda`] and [`sim_direct::SimDirect`] are the simulated ones;
//! [`probe()`] reports what the machine has for `explain()`.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use std::path::PathBuf;

/// A device buffer the caller owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePtr {
    /// Raw device address.
    pub address: usize,
    /// Bytes the buffer holds.
    pub nbytes: u64,
    /// Device ordinal.
    pub device: u32,
}

/// What can go wrong talking to CUDA.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    /// The library could not be loaded.
    #[error("{library} is not available: {reason}")]
    Unavailable {
        /// `libcudart` or `libcufile`.
        library: &'static str,
        /// What `dlopen` or the symbol lookup said.
        reason: String,
    },
    /// A runtime call failed.
    #[error("{call} failed with CUDA error {code}")]
    Runtime {
        /// The function.
        call: &'static str,
        /// `cudaError_t` value.
        code: i32,
    },
    /// A cuFile call failed.
    #[error("{call} failed with cuFile error {code}")]
    CuFile {
        /// The function.
        call: &'static str,
        /// `CUfileError_t.err` value.
        code: i32,
    },
    /// A copy was asked to overrun a buffer.
    #[error("copy of {nbytes} bytes at offset {offset} overruns a {capacity}-byte buffer")]
    Overrun {
        /// Bytes asked for.
        nbytes: u64,
        /// Offset into the buffer.
        offset: u64,
        /// The buffer's size.
        capacity: u64,
    },
    /// A 2-D copy's rows are wider than one of its pitches, so rows would
    /// overlap; `cudaMemcpy2DAsync` refuses it and so does this crate,
    /// before the call.
    #[error(
        "2-D copy rows of {width} bytes exceed the source pitch {src_pitch} or destination pitch {dst_pitch}"
    )]
    BadGeometry {
        /// Bytes per row.
        width: u64,
        /// Bytes between consecutive source rows.
        src_pitch: u64,
        /// Bytes between consecutive destination rows.
        dst_pitch: u64,
    },
    /// A file could not be registered with cuFile.
    #[error("cuFile could not register {path}: {reason}")]
    Register {
        /// The file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
    /// A file ended before the requested range did.
    #[error("{path} ended after {got} of {wanted} bytes requested at offset {offset}")]
    ShortRead {
        /// The file.
        path: PathBuf,
        /// Where the read started.
        offset: u64,
        /// Bytes asked for.
        wanted: u64,
        /// Bytes the file had.
        got: u64,
    },
}

/// The geometry of a strided host-to-device copy: `height` rows of `width`
/// bytes, read from `src_offset` at `src_pitch` apart, written at
/// `dst_offset` at `dst_pitch` apart — `cudaMemcpy2DAsync`'s arguments. A
/// strided tensor slice staged through one pinned buffer lands with one of
/// these: the rows are the slice's runs, the source pitch is the run
/// stride in the file, and the destination pitch is the run length, so the
/// rows land back to back. `width` may not exceed either pitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copy2D {
    /// Byte offset of the first row within the source buffer.
    pub src_offset: u64,
    /// Bytes between the starts of consecutive source rows.
    pub src_pitch: u64,
    /// Bytes per row.
    pub width: u64,
    /// Rows.
    pub height: u64,
    /// Byte offset of the first row within the destination buffer.
    pub dst_offset: u64,
    /// Bytes between the starts of consecutive destination rows.
    pub dst_pitch: u64,
}

impl Copy2D {
    /// Bytes the copy moves: `width * height`.
    pub fn nbytes(&self) -> u64 {
        self.width.saturating_mul(self.height)
    }

    /// The last byte a copy of this shape touches in a buffer, exclusive,
    /// counted from `offset` with `pitch` between rows: `offset + (height -
    /// 1) * pitch + width`. `None` when it overflows, or for an empty copy.
    pub fn extent(offset: u64, pitch: u64, width: u64, height: u64) -> Option<u64> {
        if width == 0 || height == 0 {
            return None;
        }
        (height - 1)
            .checked_mul(pitch)?
            .checked_add(width)?
            .checked_add(offset)
    }

    /// Refuse a shape whose rows are wider than a pitch, or whose rows run
    /// past `src_capacity` or `dst_capacity`. Pure; every implementation
    /// runs it before touching a buffer. An empty copy (zero rows or zero
    /// width) always passes.
    pub fn check(&self, src_capacity: u64, dst_capacity: u64) -> Result<(), CudaError> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        if self.width > self.src_pitch || self.width > self.dst_pitch {
            return Err(CudaError::BadGeometry {
                width: self.width,
                src_pitch: self.src_pitch,
                dst_pitch: self.dst_pitch,
            });
        }
        let overrun = |offset: u64, capacity: u64| CudaError::Overrun {
            nbytes: self.nbytes(),
            offset,
            capacity,
        };
        match Copy2D::extent(self.src_offset, self.src_pitch, self.width, self.height) {
            Some(end) if end <= src_capacity => {}
            _ => return Err(overrun(self.src_offset, src_capacity)),
        }
        match Copy2D::extent(self.dst_offset, self.dst_pitch, self.width, self.height) {
            Some(end) if end <= dst_capacity => Ok(()),
            _ => Err(overrun(self.dst_offset, dst_capacity)),
        }
    }
}

/// A page-locked host buffer suitable for asynchronous DMA.
pub trait PinnedBuffer: Send {
    /// The bytes.
    fn as_mut_slice(&mut self) -> &mut [u8];
    /// The bytes.
    fn as_slice(&self) -> &[u8];
    /// Capacity in bytes.
    fn capacity(&self) -> u64;
}

/// A point in the copy stream's history: everything enqueued before
/// [`DeviceCopier::record`] returned it has finished once [`Fence::wait`]
/// returns. Lets a caller wait for its own copies while other threads keep
/// enqueueing, where [`DeviceCopier::synchronize`] waits for everyone's.
pub trait Fence: Send + Sync {
    /// Block until the recorded point is reached.
    fn wait(&self) -> Result<(), CudaError>;
}

/// Pinned host memory and asynchronous copies on one stream.
pub trait DeviceCopier: Send + Sync {
    /// Allocate a pinned host buffer.
    fn alloc_pinned(&self, nbytes: u64) -> Result<Box<dyn PinnedBuffer>, CudaError>;

    /// Enqueue a copy of `src[..nbytes]` to `dst` at `offset`.
    fn copy_to_device(
        &self,
        src: &dyn PinnedBuffer,
        nbytes: u64,
        dst: DevicePtr,
        offset: u64,
    ) -> Result<(), CudaError>;

    /// Enqueue a strided copy of `shape.height` rows of `shape.width` bytes
    /// from `src` into `dst` — `cudaMemcpy2DAsync`. Both ends are checked
    /// with [`Copy2D::check`] before anything is enqueued; an empty shape
    /// enqueues nothing.
    fn copy_to_device_2d(
        &self,
        src: &dyn PinnedBuffer,
        dst: DevicePtr,
        shape: Copy2D,
    ) -> Result<(), CudaError>;

    /// Enqueue a copy of `nbytes` from `src` at `offset` into `dst`.
    fn copy_to_host(
        &self,
        src: DevicePtr,
        offset: u64,
        nbytes: u64,
        dst: &mut dyn PinnedBuffer,
    ) -> Result<(), CudaError>;

    /// Wait for every enqueued copy.
    fn synchronize(&self) -> Result<(), CudaError>;

    /// A fence for the copies enqueued so far.
    fn record(&self) -> Result<Box<dyn Fence>, CudaError>;

    /// Free bytes on `device` per `cudaMemGetInfo`.
    fn free_bytes(&self, device: u32) -> Result<u64, CudaError>;
}

/// GPUDirect Storage reads.
pub trait DirectStorage: Send + Sync {
    /// Read `nbytes` from `path` at `file_offset` straight into `dst` at
    /// `dst_offset`.
    fn read_into_device(
        &self,
        path: &std::path::Path,
        file_offset: u64,
        nbytes: u64,
        dst: DevicePtr,
        dst_offset: u64,
    ) -> Result<(), CudaError>;
}

pub mod cudart;
pub mod cufile;
#[allow(unsafe_code)]
mod ffi;
pub mod probe;
pub mod sim;
pub mod sim_direct;

pub use cudart::CudaRuntime;
pub use cufile::CuFile;
pub use probe::{CudaAvailability, probe};
