//! [`CuFile`]: [`DirectStorage`] over `libcufile`, loaded at run time.
//!
//! The driver is opened when the value is built and closed when it drops;
//! one value per process is the intended use. A read opens the file with
//! `O_DIRECT` where the mount allows (falling back to a plain open on
//! `EINVAL`, which is what NFS without `nvidia_fs` says), registers the
//! handle, calls `cuFileRead` until the range is done — it may return short —
//! and deregisters. When the destination is at least [`BUF_REGISTER_MIN`]
//! bytes the buffer is registered too, if cuFile agrees; otherwise the read
//! goes through cuFile's own bounce buffers.
//!
//! Without the `nvidia_fs` kernel module cuFile runs in *compat mode*
//! (`allow_compat_mode` in `/etc/cufile.json`): the same calls, served by
//! POSIX reads into pinned staging inside the library. The
//! [`CuFile::stats`] counters say which path each read took.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cudart::check_range;
use crate::ffi;
use crate::{CudaError, DevicePtr, DirectStorage};

/// Destinations at least this large get `cuFileBufRegister`ed; smaller ones
/// use cuFile's internal buffers, which cost less than a registration.
pub const BUF_REGISTER_MIN: u64 = 1 << 20;

#[cfg(target_os = "linux")]
const O_DIRECT: i32 = libc::O_DIRECT;
/// No `O_DIRECT` off Linux; the "direct" open is then a plain open.
#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;

/// What the reads so far did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CuFileStats {
    /// Files opened with `O_DIRECT`.
    pub direct_opens: u64,
    /// Files the mount refused `O_DIRECT` for, opened plainly.
    pub plain_opens: u64,
    /// Destination buffers registered with `cuFileBufRegister`.
    pub buffers_registered: u64,
    /// Registrations skipped: destination too small, or cuFile refused.
    pub buffer_registrations_skipped: u64,
}

#[derive(Default)]
struct Counters {
    direct_opens: AtomicU64,
    plain_opens: AtomicU64,
    buffers_registered: AtomicU64,
    buffer_registrations_skipped: AtomicU64,
}

/// Open `path` read-only for cuFile: `O_DIRECT` first, a plain open when the
/// mount answers `EINVAL`. Returns the file and whether it is direct.
pub fn open_for_cufile(path: &Path) -> Result<(File, bool), CudaError> {
    let register = |e: std::io::Error| CudaError::Register {
        path: path.to_path_buf(),
        reason: format!("open: {e}"),
    };
    match OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)
    {
        Ok(file) => Ok((file, O_DIRECT != 0)),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
            tracing::debug!(path = %path.display(), "O_DIRECT refused; opening plainly");
            File::open(path).map(|f| (f, false)).map_err(register)
        }
        Err(e) => Err(register(e)),
    }
}

/// `base + done` as the `off_t` cuFile takes; refused, never wrapped, when it
/// does not fit.
fn offset(base: u64, done: u64, path: &Path) -> Result<libc::off_t, CudaError> {
    base.checked_add(done)
        .and_then(|v| libc::off_t::try_from(v).ok())
        .ok_or_else(|| CudaError::Register {
            path: path.to_path_buf(),
            reason: format!("offset {base} + {done} does not fit off_t"),
        })
}

/// The cuFile driver, open.
#[derive(Debug)]
pub struct CuFile {
    lib: ffi::CuFileLib,
    /// For `cudaSetDevice` before each read: cuFile resolves the destination
    /// pointer against the calling thread's context.
    cudart: Arc<ffi::Cudart>,
    counters: Counters,
}

impl std::fmt::Debug for Counters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Counters")
    }
}

impl CuFile {
    /// Load `libcufile` and `libcudart` from the usual places and open the
    /// driver. Without either library: [`CudaError::Unavailable`] naming it;
    /// a driver that will not open: [`CudaError::CuFile`] from
    /// `cuFileDriverOpen`.
    pub fn load() -> Result<CuFile, CudaError> {
        CuFile::load_from(ffi::CUFILE_CANDIDATES, ffi::CUDART_CANDIDATES)
    }

    /// [`CuFile::load`] with explicit candidate lists for both libraries.
    pub fn load_from(cufile: &[&str], cudart: &[&str]) -> Result<CuFile, CudaError> {
        let lib = ffi::CuFileLib::load(cufile)?;
        let cudart = Arc::new(ffi::Cudart::load(cudart)?);
        lib.driver_open()?;
        Ok(CuFile {
            lib,
            cudart,
            counters: Counters::default(),
        })
    }

    /// Which library name or path loaded.
    pub fn library(&self) -> &str {
        &self.lib.loaded_from
    }

    /// `cuFileGetVersion`, when the library has it.
    pub fn version(&self) -> Option<i32> {
        self.lib.version()
    }

    /// Counters for the reads so far.
    pub fn stats(&self) -> CuFileStats {
        CuFileStats {
            direct_opens: self.counters.direct_opens.load(Ordering::Relaxed),
            plain_opens: self.counters.plain_opens.load(Ordering::Relaxed),
            buffers_registered: self.counters.buffers_registered.load(Ordering::Relaxed),
            buffer_registrations_skipped: self
                .counters
                .buffer_registrations_skipped
                .load(Ordering::Relaxed),
        }
    }

    fn open(&self, path: &Path) -> Result<File, CudaError> {
        let (file, direct) = open_for_cufile(path)?;
        let counter = if direct {
            &self.counters.direct_opens
        } else {
            &self.counters.plain_opens
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(file)
    }

    /// Register `buffer` when the transfer is large enough to pay for it and
    /// cuFile accepts it; otherwise `None` and cuFile stages internally.
    fn maybe_register(&self, buffer: DevicePtr, nbytes: u64) -> Option<ffi::BufRegistration<'_>> {
        let registered = (nbytes >= BUF_REGISTER_MIN)
            .then(|| usize::try_from(buffer.nbytes).ok())
            .flatten()
            .and_then(|len| match self.lib.buf_register(buffer.address, len) {
                Ok(registration) => Some(registration),
                Err(e) => {
                    tracing::debug!(error = %e, "cuFileBufRegister refused; staging internally");
                    None
                }
            });
        let counter = if registered.is_some() {
            &self.counters.buffers_registered
        } else {
            &self.counters.buffer_registrations_skipped
        };
        counter.fetch_add(1, Ordering::Relaxed);
        registered
    }

    /// Write `nbytes` from `src` at `src_offset` to `path` at `file_offset`
    /// through `cuFileWrite`. The file must exist and be writable; the
    /// write-engine's GDS path, when there is one.
    pub fn write_from_device(
        &self,
        path: &Path,
        file_offset: u64,
        nbytes: u64,
        src: DevicePtr,
        src_offset: u64,
    ) -> Result<(), CudaError> {
        check_range(nbytes, src_offset, src.nbytes)?;
        if nbytes == 0 {
            return Ok(());
        }
        let file = OpenOptions::new()
            .write(true)
            .custom_flags(O_DIRECT)
            .open(path)
            .or_else(|e| match e.raw_os_error() {
                Some(code) if code == libc::EINVAL => OpenOptions::new().write(true).open(path),
                _ => Err(e),
            })
            .map_err(|e| CudaError::Register {
                path: path.to_path_buf(),
                reason: format!("open: {e}"),
            })?;
        self.cudart.set_device(src.device)?;
        let handle = self
            .lib
            .handle_register(&file)
            .map_err(|e| CudaError::Register {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })?;
        let _registration = self.maybe_register(src, nbytes);
        let mut done = 0u64;
        while done < nbytes {
            let remaining = usize::try_from(nbytes - done).unwrap_or(usize::MAX);
            let n = self.lib.write(
                &handle,
                src.address,
                remaining,
                offset(file_offset, done, path)?,
                offset(src_offset, done, path)?,
            )?;
            if n == 0 {
                return Err(CudaError::CuFile {
                    call: "cuFileWrite",
                    code: libc::EIO,
                });
            }
            done += n as u64;
        }
        Ok(())
    }
}

impl Drop for CuFile {
    fn drop(&mut self) {
        if let Err(e) = self.lib.driver_close() {
            tracing::warn!(error = %e, "cuFileDriverClose failed");
        }
    }
}

impl DirectStorage for CuFile {
    fn read_into_device(
        &self,
        path: &Path,
        file_offset: u64,
        nbytes: u64,
        dst: DevicePtr,
        dst_offset: u64,
    ) -> Result<(), CudaError> {
        check_range(nbytes, dst_offset, dst.nbytes)?;
        if nbytes == 0 {
            return Ok(());
        }
        let file = self.open(path)?;
        let file_len = file
            .metadata()
            .map_err(|e| CudaError::Register {
                path: path.to_path_buf(),
                reason: format!("fstat: {e}"),
            })?
            .len();
        if check_range(nbytes, file_offset, file_len).is_err() {
            return Err(CudaError::ShortRead {
                path: path.to_path_buf(),
                offset: file_offset,
                wanted: nbytes,
                got: file_len.saturating_sub(file_offset).min(nbytes),
            });
        }
        self.cudart.set_device(dst.device)?;
        let handle = self
            .lib
            .handle_register(&file)
            .map_err(|e| CudaError::Register {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })?;
        let _registration = self.maybe_register(dst, nbytes);
        let mut done = 0u64;
        while done < nbytes {
            let remaining = usize::try_from(nbytes - done).unwrap_or(usize::MAX);
            let n = self.lib.read(
                &handle,
                dst.address,
                remaining,
                offset(file_offset, done, path)?,
                offset(dst_offset, done, path)?,
            )?;
            if n == 0 {
                return Err(CudaError::ShortRead {
                    path: path.to_path_buf(),
                    offset: file_offset,
                    wanted: nbytes,
                    got: done,
                });
            }
            done += n as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fst-cuda-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn missing_library_names_libcufile_first() {
        match CuFile::load_from(
            &["/nonexistent/libcufile.so.0"],
            &["/nonexistent/libcudart.so"],
        ) {
            Err(CudaError::Unavailable { library, reason }) => {
                assert_eq!(library, "libcufile");
                assert!(reason.contains("/nonexistent/libcufile.so.0"), "{reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn open_for_cufile_opens_files_and_names_missing_ones() {
        let path = scratch("present.bin");
        File::create(&path).unwrap().write_all(b"hello").unwrap();
        let (file, _direct) = open_for_cufile(&path).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 5);
        drop(file);
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();

        let missing = scratch("missing.bin");
        match open_for_cufile(&missing) {
            Err(CudaError::Register { path, reason }) => {
                assert_eq!(path, missing);
                assert!(reason.starts_with("open: "), "{reason}");
            }
            other => panic!("expected Register, got {other:?}"),
        }
        std::fs::remove_dir_all(missing.parent().unwrap()).unwrap();
    }

    #[test]
    fn offsets_beyond_off_t_are_refused_before_any_call() {
        let path = Path::new("/x");
        assert_eq!(offset(4096, 8, path).ok(), Some(4104));
        assert!(matches!(
            offset(u64::MAX, 0, path),
            Err(CudaError::Register { .. })
        ));
        assert!(matches!(
            offset(u64::MAX, 1, path),
            Err(CudaError::Register { .. })
        ));
    }
}
