//! Memory-mapped reads. What the reference library does; the page cache does
//! the readahead and a copy out of the mapping is the read.
//!
//! The one `unsafe` in this crate lives here: mapping a file is unsound if
//! another process truncates it while mapped. That is a property of `mmap`
//! itself, not something a wrapper can check, and the reference library
//! accepts the same risk. Writers are not offered; `posix` writes.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::StorageError;

use super::{PartWriter, RangeReader, Storage};

/// The `mmap` backend (read only).
#[derive(Debug, Clone, Copy, Default)]
pub struct MmapStorage;

impl Storage for MmapStorage {
    fn name(&self) -> &'static str {
        "mmap"
    }

    fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError> {
        let file = File::open(path).map_err(|source| StorageError::Io {
            op: "open",
            path: path.to_owned(),
            source,
        })?;
        // SAFETY: the mapping is read-only and private to this process; the
        // documented hazard (a concurrent truncation by another process) is
        // inherent to mmap and shared with every mmap-based reader.
        #[allow(unsafe_code)]
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| StorageError::Io {
            op: "mmap",
            path: path.to_owned(),
            source,
        })?;
        Ok(Box::new(MmapReader {
            map,
            path: path.to_owned(),
        }))
    }

    fn create_writer(&self, _path: &Path) -> Result<Box<dyn PartWriter>, StorageError> {
        Err(StorageError::Unsupported {
            backend: "mmap",
            op: "write",
        })
    }

    fn rename(&self, _from: &Path, _to: &Path, _durable: bool) -> Result<(), StorageError> {
        Err(StorageError::Unsupported {
            backend: "mmap",
            op: "rename",
        })
    }

    fn remove(&self, _path: &Path) -> Result<(), StorageError> {
        Err(StorageError::Unsupported {
            backend: "mmap",
            op: "remove",
        })
    }
}

struct MmapReader {
    map: memmap2::Mmap,
    path: PathBuf,
}

impl RangeReader for MmapReader {
    fn len(&self) -> u64 {
        self.map.len() as u64
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), StorageError> {
        let end = offset + dst.len() as u64;
        if end > self.len() {
            return Err(StorageError::OutOfRange {
                path: self.path.clone(),
                start: offset,
                end,
                len: self.len(),
            });
        }
        dst.copy_from_slice(&self.map[offset as usize..end as usize]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_match_the_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("obj");
        std::fs::write(&path, b"0123456789").unwrap_or_else(|e| panic!("{e}"));
        let reader = MmapStorage
            .open_reader(&path)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut buf = [0u8; 3];
        reader
            .read_at(7, &mut buf)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&buf, b"789");
        assert!(matches!(
            MmapStorage.create_writer(&path),
            Err(StorageError::Unsupported {
                backend: "mmap",
                op: "write"
            })
        ));
        assert!(matches!(
            MmapStorage.rename(&path, &path, false),
            Err(StorageError::Unsupported { op: "rename", .. })
        ));
        assert!(matches!(
            MmapStorage.remove(&path),
            Err(StorageError::Unsupported { op: "remove", .. })
        ));
    }
}
