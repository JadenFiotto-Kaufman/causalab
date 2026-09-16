//! The I/O boundary.
//!
//! Everything above this module speaks in two verbs: *read this byte range
//! into that buffer* and *write these parts as one object*. The real backends
//! ([`posix`], [`mmap`]) implement them against the operating system; the
//! [`sim`] backend implements them against memory, records every call, and
//! fails on a schedule, so the orchestration in [`crate::plan`] and above can
//! be tested for exact behaviour with no disk and no timing.
//!
//! Device memory never appears here. A GPU destination is reached through a
//! host staging buffer that *is* a `&mut [u8]` from this module's point of
//! view; the copy onward is `fst-cuda`'s business.

use std::path::Path;

use crate::error::StorageError;

pub mod mmap;
pub mod posix;
pub mod sim;

/// A byte-addressable object that can be read in ranges, from many threads.
pub trait RangeReader: Send + Sync {
    /// The object's length in bytes.
    fn len(&self) -> u64;

    /// Whether the object is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill `dst` from `offset`. Fills it completely or errors: a short read
    /// is [`StorageError::ShortRead`], never a silent partial buffer.
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), StorageError>;
}

/// A sink that takes an object as consecutive parts and writes them as one
/// contiguous object, without joining them in memory first.
///
/// `write_parts` may be called any number of times; every call appends its
/// parts after the previous call's. [`PartWriter::finish`] completes the
/// object. A writer dropped without `finish` leaves whatever was written so
/// far under the object's name; the caller (see [`crate::write`]) removes it.
pub trait PartWriter {
    /// Append `parts` back to back, in order.
    fn write_parts(&mut self, parts: &[&[u8]]) -> Result<(), StorageError>;

    /// Complete and close the object. With `durable` the bytes are forced to
    /// stable storage (`fsync` of the file and its directory) before this
    /// returns; without it they are left to the operating system.
    fn finish(self: Box<Self>, durable: bool) -> Result<(), StorageError>;
}

/// How a backend opens objects. One value per backend, no state.
pub trait Storage: Send + Sync {
    /// The backend's name, for plans and errors.
    fn name(&self) -> &'static str;

    /// Open `path` for range reads.
    fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError>;

    /// Create (or truncate) `path` for one part-wise write.
    fn create_writer(&self, path: &Path) -> Result<Box<dyn PartWriter>, StorageError>;

    /// Move the object at `from` to `to`, replacing anything at `to`, so that
    /// a concurrent reader sees either the old object or the new one and
    /// never a partial one. With `durable` the new name is forced to stable
    /// storage (`fsync` of the directory) before this returns.
    fn rename(&self, from: &Path, to: &Path, durable: bool) -> Result<(), StorageError>;

    /// Remove the object at `path`. Removing a missing object is an error.
    fn remove(&self, path: &Path) -> Result<(), StorageError>;
}

/// A read the planner has scheduled: one contiguous byte range of one object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReadRange {
    /// Absolute start within the object.
    pub offset: u64,
    /// Bytes to read.
    pub len: u64,
}

impl ReadRange {
    /// Split into pieces of at most `block` bytes, in order. `block` of zero
    /// means no split.
    pub fn split(&self, block: u64) -> Vec<ReadRange> {
        if block == 0 || self.len <= block {
            return vec![self.clone()];
        }
        let mut pieces = Vec::with_capacity(self.len.div_ceil(block) as usize);
        let mut offset = self.offset;
        let end = self.offset + self.len;
        while offset < end {
            let len = block.min(end - offset);
            pieces.push(ReadRange { offset, len });
            offset += len;
        }
        pieces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_covers_exactly_once() {
        let range = ReadRange {
            offset: 10,
            len: 25,
        };
        let pieces = range.split(8);
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0], ReadRange { offset: 10, len: 8 });
        assert_eq!(pieces[3], ReadRange { offset: 34, len: 1 });
        assert_eq!(pieces.iter().map(|p| p.len).sum::<u64>(), 25);
        assert_eq!(range.split(0), vec![range.clone()]);
        assert_eq!(range.split(100), vec![range]);
    }
}
