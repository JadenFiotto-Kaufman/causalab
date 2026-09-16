//! Plain files through `pread` and `write`. The backend every machine has.
//!
//! Reads are positional (`read_exact_at`), so one open file serves many
//! threads without a shared cursor. Writes are sequential `write_all` per
//! part; the file is created fresh and closed by `finish`. Durability is
//! `fsync` of the file *and* of its directory: the first makes the bytes
//! stable, the second makes the name stable, and a crash between the two
//! otherwise leaves a complete file nobody can find.

use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::error::StorageError;

use super::{PartWriter, RangeReader, Storage};

/// The `pread`/`write` backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct PosixStorage;

impl Storage for PosixStorage {
    fn name(&self) -> &'static str {
        "posix"
    }

    fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError> {
        let file = File::open(path).map_err(|source| StorageError::Io {
            op: "open",
            path: path.to_owned(),
            source,
        })?;
        let len = file
            .metadata()
            .map_err(|source| StorageError::Io {
                op: "stat",
                path: path.to_owned(),
                source,
            })?
            .len();
        Ok(Box::new(PosixReader {
            file,
            path: path.to_owned(),
            len,
        }))
    }

    fn create_writer(&self, path: &Path) -> Result<Box<dyn PartWriter>, StorageError> {
        let file = File::create(path).map_err(|source| StorageError::Io {
            op: "create",
            path: path.to_owned(),
            source,
        })?;
        Ok(Box::new(PosixWriter {
            file,
            path: path.to_owned(),
        }))
    }

    fn rename(&self, from: &Path, to: &Path, durable: bool) -> Result<(), StorageError> {
        std::fs::rename(from, to).map_err(|source| StorageError::Io {
            op: "rename",
            path: from.to_owned(),
            source,
        })?;
        if durable {
            sync_directory_of(to)?;
        }
        Ok(())
    }

    fn remove(&self, path: &Path) -> Result<(), StorageError> {
        std::fs::remove_file(path).map_err(|source| StorageError::Io {
            op: "remove",
            path: path.to_owned(),
            source,
        })
    }
}

/// `fsync` the directory holding `path`, so an entry created or renamed in it
/// survives a crash. A bare file name lives in the working directory.
fn sync_directory_of(path: &Path) -> Result<(), StorageError> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_owned(),
        _ => PathBuf::from("."),
    };
    let io = |source| StorageError::Io {
        op: "fsync directory",
        path: dir.clone(),
        source,
    };
    File::open(&dir).and_then(|d| d.sync_all()).map_err(io)
}

struct PosixReader {
    file: File,
    path: PathBuf,
    len: u64,
}

impl RangeReader for PosixReader {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), StorageError> {
        let end = offset + dst.len() as u64;
        if end > self.len {
            return Err(StorageError::OutOfRange {
                path: self.path.clone(),
                start: offset,
                end,
                len: self.len,
            });
        }
        self.file
            .read_exact_at(dst, offset)
            .map_err(|source| match source.kind() {
                std::io::ErrorKind::UnexpectedEof => StorageError::ShortRead {
                    path: self.path.clone(),
                    offset,
                    wanted: dst.len(),
                    got: 0,
                },
                _ => StorageError::Io {
                    op: "pread",
                    path: self.path.clone(),
                    source,
                },
            })
    }
}

struct PosixWriter {
    file: File,
    path: PathBuf,
}

impl PartWriter for PosixWriter {
    fn write_parts(&mut self, parts: &[&[u8]]) -> Result<(), StorageError> {
        for part in parts {
            self.file
                .write_all(part)
                .map_err(|source| StorageError::Io {
                    op: "write",
                    path: self.path.clone(),
                    source,
                })?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>, durable: bool) -> Result<(), StorageError> {
        // `File` is unbuffered; there is nothing to flush before the sync
        if durable {
            self.file.sync_all().map_err(|source| StorageError::Io {
                op: "fsync",
                path: self.path.clone(),
                source,
            })?;
            sync_directory_of(&self.path)?;
        }
        drop(self.file);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_a_real_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("obj");
        let store = PosixStorage;
        let mut writer = store.create_writer(&path).unwrap_or_else(|e| panic!("{e}"));
        writer
            .write_parts(&[b"hello ", b""])
            .unwrap_or_else(|e| panic!("{e}"));
        writer
            .write_parts(&[b"world"])
            .unwrap_or_else(|e| panic!("{e}"));
        writer.finish(true).unwrap_or_else(|e| panic!("{e}"));
        let reader = store.open_reader(&path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(reader.len(), 11);
        let mut buf = [0u8; 5];
        reader
            .read_at(6, &mut buf)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&buf, b"world");
        assert!(matches!(
            reader.read_at(9, &mut buf),
            Err(StorageError::OutOfRange { .. })
        ));
        assert!(matches!(
            store.open_reader(&dir.path().join("missing")),
            Err(StorageError::Io { op: "open", .. })
        ));
    }

    #[test]
    fn rename_replaces_and_remove_unlinks() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let from = dir.path().join("from");
        let to = dir.path().join("to");
        std::fs::write(&from, b"new").unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(&to, b"old").unwrap_or_else(|e| panic!("{e}"));
        let store = PosixStorage;
        store
            .rename(&from, &to, true)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(!from.exists());
        assert_eq!(std::fs::read(&to).unwrap_or_default(), b"new");
        store.remove(&to).unwrap_or_else(|e| panic!("{e}"));
        assert!(!to.exists());
        assert!(matches!(
            store.remove(&to),
            Err(StorageError::Io { op: "remove", .. })
        ));
        assert!(matches!(
            store.rename(&from, &to, false),
            Err(StorageError::Io { op: "rename", .. })
        ));
    }

    #[test]
    fn bare_file_names_sync_the_working_directory() {
        // a relative name has an empty parent; the directory to sync is "."
        sync_directory_of(Path::new("obj")).unwrap_or_else(|e| panic!("{e}"));
    }
}
