//! Draining device-resident parts through a ring of pinned host buffers.
//!
//! A device part is copied to the host in chunks of at most
//! [`StagingRing::bytes`], each into one of [`StagingRing::count`] pinned
//! slots, and each chunk is written as soon as its copy is complete. The
//! copies of the next chunks are enqueued *before* the current chunk is
//! written, so the device-to-host transfer of chunk `k+1..k+count-1` runs
//! while chunk `k` goes to storage.
//!
//! Slot reuse is the invariant that matters: a slot is refilled only after
//! the chunk it held has been written, and a chunk is written only after a
//! `synchronize` that covers its copy. [`fst_cuda::DeviceCopier`] offers one
//! stream-wide `synchronize`, not per-copy fences, so the ring synchronizes
//! whenever it is about to write a chunk whose copy has not yet been covered
//! — once per wrap of the ring in steady state, once per chunk with two slots
//! (classic double buffering), and with one slot there is no overlap at all.
//! Per-buffer events (`cudaEventRecord` / `cudaEventSynchronize`) would let
//! the ring wait for exactly the chunk it needs and are a later refinement of
//! the copier trait; the protocol here needs no change to adopt them.

use std::path::Path;

use fst_cuda::{DeviceCopier, DevicePtr, PinnedBuffer};

use crate::storage::PartWriter;

use super::{Stage, WriteError};

/// Shape of the pinned staging ring used for device-resident parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingRing {
    /// Pinned buffers in the ring; also the most chunk copies in flight.
    pub count: usize,
    /// Bytes per buffer, and so the chunk size. A part smaller than this
    /// gets buffers sized to the largest device part instead.
    pub bytes: u64,
}

impl Default for StagingRing {
    /// Four buffers of the planner's staging size (16 MiB): three copies in
    /// flight behind the write, 64 MiB pinned at most.
    fn default() -> Self {
        StagingRing {
            count: 4,
            bytes: crate::plan::STAGING_BUFFER_BYTES,
        }
    }
}

impl StagingRing {
    pub(super) fn validate(self, path: &Path) -> Result<(), WriteError> {
        if self.count == 0 || self.bytes == 0 {
            return Err(WriteError::EmptyStaging {
                path: path.to_owned(),
                count: self.count,
                bytes: self.bytes,
            });
        }
        Ok(())
    }
}

/// Which part a chunk belongs to, for error messages.
#[derive(Clone, Copy)]
pub(super) struct PartRef<'a> {
    pub path: &'a Path,
    pub index: usize,
    pub name: &'a str,
}

/// The ring, bound to one copier. Slots are allocated as chunks first need
/// them, so a payload whose device parts fit in one chunk pins one buffer.
pub(super) struct Ring<'c> {
    copier: &'c dyn DeviceCopier,
    slots: Vec<Box<dyn PinnedBuffer>>,
    count: usize,
    chunk_bytes: u64,
}

impl<'c> Ring<'c> {
    /// A ring for parts of at most `largest_part` bytes: buffers are never
    /// larger than the biggest part they will hold.
    pub(super) fn new(copier: &'c dyn DeviceCopier, shape: StagingRing, largest_part: u64) -> Self {
        Ring {
            copier,
            slots: Vec::with_capacity(shape.count),
            count: shape.count,
            chunk_bytes: shape.bytes.min(largest_part).max(1),
        }
    }

    fn slot(&mut self, chunk: usize, part: PartRef<'_>) -> Result<usize, WriteError> {
        let slot = chunk % self.count;
        if slot == self.slots.len() {
            let buffer = self
                .copier
                .alloc_pinned(self.chunk_bytes)
                .map_err(|source| cuda_error(part, source))?;
            self.slots.push(buffer);
        }
        Ok(slot)
    }

    /// Copy `nbytes` of `ptr` to the host chunk by chunk and write each chunk
    /// through `writer`. Returns the number of chunks.
    pub(super) fn drain(
        &mut self,
        ptr: DevicePtr,
        nbytes: u64,
        writer: &mut dyn PartWriter,
        part: PartRef<'_>,
    ) -> Result<usize, WriteError> {
        let chunk_bytes = self.chunk_bytes;
        let chunks = nbytes.div_ceil(chunk_bytes) as usize;
        let chunk_len = |i: usize| chunk_bytes.min(nbytes - i as u64 * chunk_bytes);
        // chunks [0, enqueued) have copies issued; [0, synced) are complete
        let mut enqueued = 0usize;
        let mut synced = 0usize;
        for i in 0..chunks {
            // chunk i's copy must be issued (only the one-slot ring gets here
            // without it); its slot is free because chunk i - count is written
            while enqueued <= i {
                self.enqueue(ptr, enqueued, chunk_len(enqueued), part)?;
                enqueued += 1;
            }
            if i >= synced {
                self.copier
                    .synchronize()
                    .map_err(|source| cuda_error(part, source))?;
                synced = enqueued;
            }
            // look ahead: fill every free slot so those copies overlap the
            // write below; slot (e % count) is free once chunk e - count is
            // written, i.e. while e < i + count
            while enqueued < chunks && enqueued < i + self.count {
                self.enqueue(ptr, enqueued, chunk_len(enqueued), part)?;
                enqueued += 1;
            }
            let slot = i % self.count;
            let bytes = &self.slots[slot].as_slice()[..chunk_len(i) as usize];
            writer
                .write_parts(&[bytes])
                .map_err(|source| WriteError::Storage {
                    path: part.path.to_owned(),
                    stage: Stage::Part {
                        index: part.index,
                        name: part.name.to_owned(),
                    },
                    source,
                })?;
        }
        Ok(chunks)
    }

    fn enqueue(
        &mut self,
        ptr: DevicePtr,
        chunk: usize,
        len: u64,
        part: PartRef<'_>,
    ) -> Result<(), WriteError> {
        let slot = self.slot(chunk, part)?;
        let offset = chunk as u64 * self.chunk_bytes;
        self.copier
            .copy_to_host(ptr, offset, len, self.slots[slot].as_mut())
            .map_err(|source| cuda_error(part, source))
    }
}

fn cuda_error(part: PartRef<'_>, source: fst_cuda::CudaError) -> WriteError {
    WriteError::Cuda {
        path: part.path.to_owned(),
        index: part.index,
        name: part.name.to_owned(),
        source,
    }
}
