//! The simulated backend: objects in memory, every operation recorded, faults
//! on a schedule. What the orchestration is tested against.
//!
//! Determinism is the point. A test builds a [`SimStorage`], seeds objects,
//! optionally arms a [`Fault`] for the *n*-th matching operation, runs the
//! code under test, and asserts on the exact [`Op`] log — which ranges were
//! read, in what pieces, which parts were written — and on the bytes that
//! landed. No timing, no disk, no threads it did not start itself.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::StorageError;

use super::{PartWriter, RangeReader, Storage};

/// One recorded operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// A reader was opened.
    Open(PathBuf),
    /// A range was read: path, offset, length.
    Read(PathBuf, u64, u64),
    /// A writer was created.
    Create(PathBuf),
    /// Parts were written: path, the part lengths.
    Write(PathBuf, Vec<u64>),
    /// A writer was finished: path, whether durability was asked for.
    Finish(PathBuf, bool),
    /// An object was renamed: from, to, whether durability was asked for.
    Rename(PathBuf, PathBuf, bool),
    /// An object was removed.
    Remove(PathBuf),
}

/// A scheduled failure: the `nth` (zero-based) operation matching `kind`
/// fails with `message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fault {
    /// Which operation family fails.
    pub kind: FaultKind,
    /// How many matching operations succeed first.
    pub nth: usize,
    /// The error text.
    pub message: String,
}

/// Operation families a [`Fault`] can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// Opening a reader.
    Open,
    /// A range read.
    Read,
    /// Creating a writer.
    Create,
    /// A part-wise write.
    Write,
    /// Finishing a writer.
    Finish,
    /// A rename.
    Rename,
    /// A removal.
    Remove,
}

#[derive(Default)]
struct State {
    objects: BTreeMap<PathBuf, Vec<u8>>,
    log: Vec<Op>,
    faults: Vec<Fault>,
    seen: BTreeMap<u8, usize>,
}

impl State {
    fn record(&mut self, op: Op) {
        self.log.push(op);
    }

    /// Count this operation against every armed fault of its kind; fire the
    /// one whose turn it is.
    fn check_fault(&mut self, kind: FaultKind) -> Result<(), StorageError> {
        let counter = self.seen.entry(kind as u8).or_insert(0);
        let index = *counter;
        *counter += 1;
        if let Some(position) = self
            .faults
            .iter()
            .position(|f| f.kind == kind && f.nth == index)
        {
            let fault = self.faults.remove(position);
            return Err(StorageError::Simulated(fault.message));
        }
        Ok(())
    }
}

/// In-memory storage with an operation log and fault injection. Cheap to
/// clone: clones share the same objects and log.
#[derive(Clone, Default)]
pub struct SimStorage {
    state: Arc<Mutex<State>>,
}

impl SimStorage {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // a poisoned lock means a test thread panicked; carry on with the
        // data, the panic is already the failure
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Put an object in the store.
    pub fn put(&self, path: impl Into<PathBuf>, bytes: impl Into<Vec<u8>>) {
        self.lock().objects.insert(path.into(), bytes.into());
    }

    /// A copy of an object's bytes, if it exists.
    pub fn get(&self, path: &Path) -> Option<Vec<u8>> {
        self.lock().objects.get(path).cloned()
    }

    /// Every object's path, in order.
    pub fn paths(&self) -> Vec<PathBuf> {
        self.lock().objects.keys().cloned().collect()
    }

    /// Arm a fault.
    pub fn arm(&self, fault: Fault) {
        self.lock().faults.push(fault);
    }

    /// The operations so far, in order.
    pub fn log(&self) -> Vec<Op> {
        self.lock().log.clone()
    }

    /// Forget the log (objects and armed faults stay).
    pub fn clear_log(&self) {
        self.lock().log.clear();
    }

    /// Bytes read so far, summed over every `Read` in the log.
    pub fn bytes_read(&self) -> u64 {
        self.lock()
            .log
            .iter()
            .map(|op| match op {
                Op::Read(_, _, len) => *len,
                _ => 0,
            })
            .sum()
    }
}

impl Storage for SimStorage {
    fn name(&self) -> &'static str {
        "sim"
    }

    fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError> {
        let mut state = self.lock();
        state.record(Op::Open(path.to_owned()));
        state.check_fault(FaultKind::Open)?;
        if !state.objects.contains_key(path) {
            return Err(StorageError::Io {
                op: "open",
                path: path.to_owned(),
                source: std::io::Error::from(std::io::ErrorKind::NotFound),
            });
        }
        drop(state);
        Ok(Box::new(SimReader {
            store: self.clone(),
            path: path.to_owned(),
        }))
    }

    fn create_writer(&self, path: &Path) -> Result<Box<dyn PartWriter>, StorageError> {
        let mut state = self.lock();
        state.record(Op::Create(path.to_owned()));
        state.check_fault(FaultKind::Create)?;
        state.objects.insert(path.to_owned(), Vec::new());
        drop(state);
        Ok(Box::new(SimWriter {
            store: self.clone(),
            path: path.to_owned(),
        }))
    }

    fn rename(&self, from: &Path, to: &Path, durable: bool) -> Result<(), StorageError> {
        let mut state = self.lock();
        state.record(Op::Rename(from.to_owned(), to.to_owned(), durable));
        state.check_fault(FaultKind::Rename)?;
        let bytes = state
            .objects
            .remove(from)
            .ok_or_else(|| not_found("rename", from))?;
        state.objects.insert(to.to_owned(), bytes);
        Ok(())
    }

    fn remove(&self, path: &Path) -> Result<(), StorageError> {
        let mut state = self.lock();
        state.record(Op::Remove(path.to_owned()));
        state.check_fault(FaultKind::Remove)?;
        state
            .objects
            .remove(path)
            .map(drop)
            .ok_or_else(|| not_found("remove", path))
    }
}

fn not_found(op: &'static str, path: &Path) -> StorageError {
    StorageError::Io {
        op,
        path: path.to_owned(),
        source: std::io::Error::from(std::io::ErrorKind::NotFound),
    }
}

struct SimReader {
    store: SimStorage,
    path: PathBuf,
}

impl RangeReader for SimReader {
    fn len(&self) -> u64 {
        self.store
            .lock()
            .objects
            .get(&self.path)
            .map_or(0, |b| b.len() as u64)
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), StorageError> {
        let mut state = self.store.lock();
        state.record(Op::Read(self.path.clone(), offset, dst.len() as u64));
        state.check_fault(FaultKind::Read)?;
        let object = state
            .objects
            .get(&self.path)
            .ok_or_else(|| StorageError::Io {
                op: "read",
                path: self.path.clone(),
                source: std::io::Error::from(std::io::ErrorKind::NotFound),
            })?;
        let end = offset + dst.len() as u64;
        if end > object.len() as u64 {
            return Err(StorageError::OutOfRange {
                path: self.path.clone(),
                start: offset,
                end,
                len: object.len() as u64,
            });
        }
        dst.copy_from_slice(&object[offset as usize..end as usize]);
        Ok(())
    }
}

struct SimWriter {
    store: SimStorage,
    path: PathBuf,
}

impl PartWriter for SimWriter {
    fn write_parts(&mut self, parts: &[&[u8]]) -> Result<(), StorageError> {
        let mut state = self.store.lock();
        state.record(Op::Write(
            self.path.clone(),
            parts.iter().map(|p| p.len() as u64).collect(),
        ));
        state.check_fault(FaultKind::Write)?;
        let object = state.objects.entry(self.path.clone()).or_default();
        for part in parts {
            object.extend_from_slice(part);
        }
        Ok(())
    }

    fn finish(self: Box<Self>, durable: bool) -> Result<(), StorageError> {
        let mut state = self.store.lock();
        state.record(Op::Finish(self.path.clone(), durable));
        state.check_fault(FaultKind::Finish)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_are_exact_and_logged() {
        let store = SimStorage::new();
        store.put("a", b"0123456789".to_vec());
        let reader = store
            .open_reader(Path::new("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        let mut buf = [0u8; 4];
        reader
            .read_at(3, &mut buf)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&buf, b"3456");
        assert!(matches!(
            reader.read_at(8, &mut buf),
            Err(StorageError::OutOfRange {
                start: 8,
                end: 12,
                len: 10,
                ..
            })
        ));
        assert_eq!(
            store.log(),
            vec![
                Op::Open("a".into()),
                Op::Read("a".into(), 3, 4),
                Op::Read("a".into(), 8, 4),
            ]
        );
    }

    #[test]
    fn faults_fire_on_schedule_once() {
        let store = SimStorage::new();
        store.put("a", vec![0; 16]);
        store.arm(Fault {
            kind: FaultKind::Read,
            nth: 1,
            message: "eio".into(),
        });
        let reader = store
            .open_reader(Path::new("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        let mut buf = [0u8; 2];
        assert!(reader.read_at(0, &mut buf).is_ok());
        assert!(
            matches!(reader.read_at(0, &mut buf), Err(StorageError::Simulated(m)) if m == "eio")
        );
        assert!(reader.read_at(0, &mut buf).is_ok());
    }

    #[test]
    fn writes_concatenate_parts() {
        let store = SimStorage::new();
        let mut writer = store
            .create_writer(Path::new("out"))
            .unwrap_or_else(|e| panic!("{e}"));
        writer
            .write_parts(&[b"ab", b"", b"cde"])
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(store.get(Path::new("out")), Some(b"abcde".to_vec()));
        assert_eq!(store.log()[1], Op::Write("out".into(), vec![2, 0, 3]));
        writer.finish(true).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(store.log()[2], Op::Finish("out".into(), true));
    }

    #[test]
    fn rename_moves_and_remove_forgets() {
        let store = SimStorage::new();
        store.put("tmp", b"new".to_vec());
        store.put("final", b"old".to_vec());
        store
            .rename(Path::new("tmp"), Path::new("final"), true)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(store.paths(), vec![PathBuf::from("final")]);
        assert_eq!(store.get(Path::new("final")), Some(b"new".to_vec()));
        assert!(matches!(
            store.rename(Path::new("tmp"), Path::new("final"), false),
            Err(StorageError::Io { op: "rename", .. })
        ));
        store
            .remove(Path::new("final"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(store.paths().is_empty());
        assert!(matches!(
            store.remove(Path::new("final")),
            Err(StorageError::Io { op: "remove", .. })
        ));
        assert_eq!(
            store.log(),
            vec![
                Op::Rename("tmp".into(), "final".into(), true),
                Op::Rename("tmp".into(), "final".into(), false),
                Op::Remove("final".into()),
                Op::Remove("final".into()),
            ]
        );
    }

    #[test]
    fn finish_rename_and_remove_faults_fire() {
        let store = SimStorage::new();
        for (kind, message) in [
            (FaultKind::Finish, "fsync"),
            (FaultKind::Rename, "rename"),
            (FaultKind::Remove, "remove"),
        ] {
            store.arm(Fault {
                kind,
                nth: 0,
                message: message.into(),
            });
        }
        let writer = store
            .create_writer(Path::new("a"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(writer.finish(false), Err(StorageError::Simulated(m)) if m == "fsync"));
        assert!(matches!(
            store.rename(Path::new("a"), Path::new("b"), false),
            Err(StorageError::Simulated(m)) if m == "rename"
        ));
        assert!(matches!(
            store.remove(Path::new("a")),
            Err(StorageError::Simulated(m)) if m == "remove"
        ));
        // the fault fired before the operation, so the object is untouched
        assert_eq!(store.paths(), vec![PathBuf::from("a")]);
    }
}
