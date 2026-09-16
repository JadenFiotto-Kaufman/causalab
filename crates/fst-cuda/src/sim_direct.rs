//! A simulated [`DirectStorage`]: cuFile reads served from an in-memory file
//! map straight into [`SimCuda`]'s device memory, every call logged, faults
//! on a schedule. What the read engine's cuFile path — and its fallback to
//! pread — is tested against without a GPU or `libcufile`.
//!
//! The faults mirror what the real [`crate::CuFile`] reports: a file on a
//! mount libcufile cannot build a handle for is [`CudaError::Register`] on
//! every read of it; a library that cannot be loaded is
//! [`CudaError::Unavailable`]; anything else cuFile says is
//! [`CudaError::CuFile`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::cudart::check_range;
use crate::sim::SimCuda;
use crate::{CudaError, DevicePtr, DirectStorage};

/// One recorded [`DirectStorage::read_into_device`] call, faulted or not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectCall {
    /// The file.
    pub path: PathBuf,
    /// Where in the file the read started.
    pub file_offset: u64,
    /// Bytes asked for.
    pub nbytes: u64,
    /// Destination address.
    pub address: usize,
    /// Offset into the destination.
    pub dst_offset: u64,
}

/// A scheduled failure. `nth` counts calls from zero, as
/// `fst_core::storage::sim::Fault` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectFault {
    /// Every read of `path` fails with [`CudaError::Register`]: the file sits
    /// on a mount libcufile cannot register.
    Register {
        /// The file.
        path: PathBuf,
        /// What registration said.
        reason: String,
    },
    /// The `nth` call fails with [`CudaError::Unavailable`] for `libcufile`.
    /// One-shot, so a caller that retries instead of giving up is caught.
    Unavailable {
        /// Which call.
        nth: usize,
        /// What `dlopen` said.
        reason: String,
    },
    /// The `nth` call fails with [`CudaError::CuFile`].
    CuFile {
        /// Which call.
        nth: usize,
        /// The function.
        call: &'static str,
        /// The cuFile error code.
        code: i32,
    },
}

impl DirectFault {
    fn fires(&self, path: &Path, index: usize) -> bool {
        match self {
            DirectFault::Register { path: p, .. } => p == path,
            DirectFault::Unavailable { nth, .. } | DirectFault::CuFile { nth, .. } => *nth == index,
        }
    }

    fn error(&self) -> CudaError {
        match self {
            DirectFault::Register { path, reason } => CudaError::Register {
                path: path.clone(),
                reason: reason.clone(),
            },
            DirectFault::Unavailable { reason, .. } => CudaError::Unavailable {
                library: "libcufile",
                reason: reason.clone(),
            },
            DirectFault::CuFile { call, code, .. } => CudaError::CuFile { call, code: *code },
        }
    }
}

#[derive(Default)]
struct State {
    files: BTreeMap<PathBuf, Vec<u8>>,
    faults: Vec<DirectFault>,
    log: Vec<DirectCall>,
}

/// The simulated cuFile. Device memory is the [`SimCuda`] it was built over,
/// so bytes it lands are visible through [`SimCuda::device_bytes`]; nothing
/// it does appears in the [`SimCuda`] log.
pub struct SimDirect {
    cuda: SimCuda,
    state: Mutex<State>,
}

impl SimDirect {
    /// A cuFile over `cuda`'s device memory with no files and no faults.
    pub fn new(cuda: SimCuda) -> Self {
        SimDirect {
            cuda,
            state: Mutex::default(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Store an object under `path`, replacing any previous one.
    pub fn put(&self, path: impl Into<PathBuf>, bytes: impl Into<Vec<u8>>) {
        self.lock().files.insert(path.into(), bytes.into());
    }

    /// Schedule a fault. Faults are checked in arming order; the first that
    /// applies to a call fires.
    pub fn arm(&self, fault: DirectFault) {
        self.lock().faults.push(fault);
    }

    /// Every call so far, in order, including the ones a fault failed.
    pub fn log(&self) -> Vec<DirectCall> {
        self.lock().log.clone()
    }
}

impl DirectStorage for SimDirect {
    fn read_into_device(
        &self,
        path: &Path,
        file_offset: u64,
        nbytes: u64,
        dst: DevicePtr,
        dst_offset: u64,
    ) -> Result<(), CudaError> {
        let mut state = self.lock();
        let index = state.log.len();
        state.log.push(DirectCall {
            path: path.to_path_buf(),
            file_offset,
            nbytes,
            address: dst.address,
            dst_offset,
        });
        if let Some(fault) = state.faults.iter().find(|f| f.fires(path, index)) {
            return Err(fault.error());
        }
        check_range(nbytes, dst_offset, dst.nbytes)?;
        // a file that cannot be opened is a registration failure, as in `CuFile`
        let file = state.files.get(path).ok_or_else(|| CudaError::Register {
            path: path.to_path_buf(),
            reason: "open: no such file".into(),
        })?;
        let len = file.len() as u64;
        if check_range(nbytes, file_offset, len).is_err() {
            return Err(CudaError::ShortRead {
                path: path.to_path_buf(),
                offset: file_offset,
                wanted: nbytes,
                got: len.saturating_sub(file_offset).min(nbytes),
            });
        }
        let bytes = file[file_offset as usize..(file_offset + nbytes) as usize].to_vec();
        drop(state);
        self.cuda.write_device(dst, dst_offset, &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_land_in_device_memory_and_are_logged() {
        let cuda = SimCuda::new(1 << 30);
        let direct = SimDirect::new(cuda.clone());
        direct.put("w", b"0123456789".to_vec());
        let dst = cuda.alloc_device(8, 0);
        direct
            .read_into_device(Path::new("w"), 2, 4, dst, 3)
            .unwrap();
        assert_eq!(cuda.device_bytes(dst), Some(b"\0\0\x002345\0".to_vec()));
        assert_eq!(
            direct.log(),
            vec![DirectCall {
                path: "w".into(),
                file_offset: 2,
                nbytes: 4,
                address: dst.address,
                dst_offset: 3,
            }]
        );
        assert!(cuda.log().is_empty(), "cuFile reads never touch the copier");
        assert!(matches!(
            direct.read_into_device(Path::new("w"), 0, 4, dst, 6),
            Err(CudaError::Overrun {
                nbytes: 4,
                offset: 6,
                capacity: 8
            })
        ));
        assert!(matches!(
            direct.read_into_device(Path::new("w"), 8, 4, dst, 0),
            Err(CudaError::ShortRead {
                offset: 8,
                wanted: 4,
                got: 2,
                ..
            })
        ));
        assert!(matches!(
            direct.read_into_device(Path::new("missing"), 0, 1, dst, 0),
            Err(CudaError::Register { .. })
        ));
        assert_eq!(direct.log().len(), 4);
    }

    #[test]
    fn faults_fire_as_armed() {
        let cuda = SimCuda::new(1 << 30);
        let direct = SimDirect::new(cuda.clone());
        direct.put("a", vec![1; 4]);
        direct.put("b", vec![2; 4]);
        direct.arm(DirectFault::Register {
            path: "b".into(),
            reason: "5027".into(),
        });
        direct.arm(DirectFault::CuFile {
            nth: 2,
            call: "cuFileRead",
            code: -5,
        });
        direct.arm(DirectFault::Unavailable {
            nth: 3,
            reason: "gone".into(),
        });
        let dst = cuda.alloc_device(4, 0);
        let read = |path: &str| direct.read_into_device(Path::new(path), 0, 4, dst, 0);
        assert!(read("a").is_ok());
        assert!(
            matches!(read("b"), Err(CudaError::Register { path, reason }) if path == Path::new("b") && reason == "5027")
        );
        assert!(matches!(
            read("a"),
            Err(CudaError::CuFile {
                call: "cuFileRead",
                code: -5
            })
        ));
        assert!(matches!(
            read("a"),
            Err(CudaError::Unavailable { library: "libcufile", reason }) if reason == "gone"
        ));
        assert!(read("a").is_ok(), "nth faults are one-shot");
        assert!(read("b").is_err(), "a path fault persists");
        assert_eq!(direct.log().len(), 6);
    }
}
