//! The workers: files in flight, pieces per file, first failure cancels.
//!
//! Two nested levels, each a `std::thread::scope`: `files_in_flight` file
//! workers pull files off a shared queue, open one reader each, and run
//! `readers_per_file` piece workers over that file's pieces. A level with a
//! concurrency of one runs inline on the caller's thread, so a plan with
//! `concurrency() == 1` executes in job order with no threads at all —
//! what tests use for exact operation-log assertions.
//!
//! Each piece takes one of three routes — host pread, cuFile into device
//! memory, or pread into staging and a device copy — and is counted on the
//! route it landed by. A file under a cuFile plan starts on cuFile and
//! switches to staging, for good, on its first `Register` failure; the job
//! switches as a whole on `Unavailable`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use fst_cuda::{Copy2D, CudaError, DeviceCopier, DevicePtr, DirectStorage};

use crate::plan::ReadPlan;
use crate::storage::{RangeReader, ReadRange, Storage};

use super::staging::LazyRing;
use super::{Dest, Fallback, Placement, ReadError, Transfer};

/// Where one piece lands.
pub(super) enum PieceDest<'a> {
    /// The exact sub-slice of the caller's host buffer.
    Host(&'a mut [u8]),
    /// The caller's device buffer at the piece's offset.
    Device {
        /// The buffer.
        ptr: DevicePtr,
        /// Offset of this piece within it.
        offset: u64,
    },
}

/// One read the engine issues: at most `split_bytes` (and, for a device
/// destination, at most one staging buffer) of one file.
pub(super) struct Piece<'a> {
    pub(super) range: ReadRange,
    pub(super) dest: PieceDest<'a>,
    /// The piece's placements, relative to its range and its destination;
    /// empty when the piece lands whole.
    pub(super) placements: Vec<Placement>,
}

impl Piece<'_> {
    /// Bytes this piece lands.
    fn landed_bytes(&self) -> u64 {
        if self.placements.is_empty() {
            self.range.len
        } else {
            self.placements.iter().map(|p| p.len).sum()
        }
    }
}

/// Split a file's transfers into pieces. Host transfers split at
/// `split_bytes`; device transfers additionally never exceed one staging
/// buffer (`device_block`), so any of them can fall back to staging. Zero
/// means unsplit. Empty transfers issue no piece.
///
/// A placed transfer is cut along its placements (see [`split_placed`]);
/// a single placement covering the whole range is the same as none and is
/// normalised away, so a whole run coalesced with nothing is an ordinary
/// piece.
pub(super) fn split_file<'a>(
    transfers: Vec<Transfer<'a>>,
    split_bytes: u64,
    device_block: Option<u64>,
) -> Vec<Piece<'a>> {
    let mut pieces = Vec::new();
    for transfer in transfers {
        if transfer.range.len == 0 {
            continue;
        }
        let mut placements = transfer.placements;
        if let [only] = placements.as_slice()
            && only.src == 0
            && only.len == transfer.range.len
        {
            placements.clear();
        }
        let device_block = match device_block {
            Some(block) if split_bytes == 0 => block,
            Some(block) => block.min(split_bytes),
            None => split_bytes,
        };
        match transfer.dest {
            Dest::Host(slice) if placements.is_empty() => {
                let mut rest = slice;
                for range in transfer.range.split(split_bytes) {
                    let (head, tail) = rest.split_at_mut(range.len as usize);
                    rest = tail;
                    pieces.push(Piece {
                        range,
                        dest: PieceDest::Host(head),
                        placements: Vec::new(),
                    });
                }
            }
            Dest::Host(slice) => {
                let mut rest = slice;
                for (range, placements) in split_placed(&transfer.range, &placements, split_bytes) {
                    let landed: u64 = placements.iter().map(|p| p.len).sum();
                    let (head, tail) = rest.split_at_mut(landed as usize);
                    rest = tail;
                    pieces.push(Piece {
                        range,
                        dest: PieceDest::Host(head),
                        placements,
                    });
                }
            }
            Dest::Device { ptr, offset } if placements.is_empty() => {
                for range in transfer.range.split(device_block) {
                    let delta = range.offset - transfer.range.offset;
                    pieces.push(Piece {
                        range,
                        dest: PieceDest::Device {
                            ptr,
                            offset: offset + delta,
                        },
                        placements: Vec::new(),
                    });
                }
            }
            Dest::Device { ptr, offset } => {
                let mut landed = 0u64;
                for (range, placements) in split_placed(&transfer.range, &placements, device_block)
                {
                    let piece_landed: u64 = placements.iter().map(|p| p.len).sum();
                    pieces.push(Piece {
                        range,
                        dest: PieceDest::Device {
                            ptr,
                            offset: offset + landed,
                        },
                        placements,
                    });
                    landed += piece_landed;
                }
            }
        }
    }
    pieces
}

/// Cut a placed range into pieces of at most `block` bytes (zero: one
/// piece), each with its placements re-based to the piece's own start and
/// landing. Every piece begins at a placement's byte and ends at one: a
/// piece takes whole placements while they fit, stops at the end of the
/// last that does, and the next piece starts at the next placement — so a
/// regular pattern stays regular piece by piece and no piece begins or
/// ends inside a gap. Only a placement longer than `block` on its own is
/// cut, and then it spans pieces. Placements are validated before this runs
/// (sorted, disjoint, packed), so the pieces' landings are consecutive.
pub(super) fn split_placed(
    range: &ReadRange,
    placements: &[Placement],
    block: u64,
) -> Vec<(ReadRange, Vec<Placement>)> {
    let mut out = Vec::new();
    let mut cursor = 0u64;
    let mut index = 0;
    while index < placements.len() {
        let start = cursor.max(placements[index].src);
        let limit = if block == 0 {
            range.len
        } else {
            (start + block).min(range.len)
        };
        let landing = placements[index].dst + start.saturating_sub(placements[index].src);
        let mut piece = Vec::new();
        let mut end = start;
        while index < placements.len() && placements[index].src < limit {
            let p = placements[index];
            let from = p.src.max(start);
            let to = p.src_end().min(limit);
            if to < p.src_end() && !piece.is_empty() {
                // does not fit whole: it opens the next piece instead
                break;
            }
            piece.push(Placement {
                src: from - start,
                dst: p.dst + (from - p.src) - landing,
                len: to - from,
            });
            end = to;
            if p.src_end() <= limit {
                index += 1;
            } else {
                break;
            }
        }
        out.push((
            ReadRange {
                offset: range.offset + start,
                len: end - start,
            },
            piece,
        ));
        cursor = end;
    }
    out
}

/// The 2-D copy that lands `placements` from a staging slot at `dst_offset`
/// on the device, when they are a regular pattern: equal lengths, a
/// constant source stride and a constant destination stride. `None` for an
/// irregular pattern, which lands one placement at a time.
pub(super) fn regular_2d(placements: &[Placement], dst_offset: u64) -> Option<Copy2D> {
    let first = placements.first()?;
    let width = first.len;
    let (src_pitch, dst_pitch) = match placements.get(1) {
        Some(second) => (
            second.src.checked_sub(first.src)?,
            second.dst.checked_sub(first.dst)?,
        ),
        None => (width, width),
    };
    let regular = placements.windows(2).all(|pair| {
        pair[1].len == width
            && pair[1].src.checked_sub(pair[0].src) == Some(src_pitch)
            && pair[1].dst.checked_sub(pair[0].dst) == Some(dst_pitch)
    });
    if !regular {
        return None;
    }
    Some(Copy2D {
        src_offset: first.src,
        src_pitch,
        width,
        height: placements.len() as u64,
        dst_offset: dst_offset + first.dst,
        dst_pitch,
    })
}

/// What the workers counted.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) cufile_bytes: u64,
    pub(super) staged_bytes: u64,
    pub(super) host_bytes: u64,
    pub(super) pieces: usize,
    pub(super) gap_bytes: u64,
    pub(super) placed_pieces: usize,
    pub(super) scatter_copies: usize,
    pub(super) files_opened: usize,
    pub(super) fallbacks: Vec<Fallback>,
}

/// How a piece landed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Host,
    CuFile,
    Staged,
}

/// `'a` is the borrows the workers share; `'r` the copier the ring holds.
struct Shared<'job, 'a, 'r> {
    storage: &'a dyn Storage,
    copier: Option<&'a dyn DeviceCopier>,
    /// cuFile, when the plan chose it and the caller supplied it.
    direct: Option<&'a dyn DirectStorage>,
    ring: &'a LazyRing<'r>,
    readers_per_file: usize,
    files: Mutex<VecDeque<(PathBuf, Vec<Piece<'job>>)>>,
    cancel: AtomicBool,
    /// cuFile is off for the rest of the job: `libcufile` said `Unavailable`.
    direct_disabled: AtomicBool,
    failure: Mutex<Option<ReadError>>,
    fallbacks: Mutex<Vec<Fallback>>,
    cufile_bytes: AtomicU64,
    staged_bytes: AtomicU64,
    host_bytes: AtomicU64,
    pieces: AtomicUsize,
    gap_bytes: AtomicU64,
    placed_pieces: AtomicUsize,
    scatter_copies: AtomicUsize,
    files_opened: AtomicUsize,
}

impl Shared<'_, '_, '_> {
    /// Record the first failure and raise the cancel flag. The flag is set
    /// after the error is stored, so anyone who sees the flag finds the error.
    fn fail(&self, error: ReadError) {
        let mut failure = lock(&self.failure);
        if failure.is_none() {
            *failure = Some(error);
        }
        drop(failure);
        self.cancel.store(true, Ordering::Release);
        if let Some(ring) = self.ring.allocated() {
            ring.wake_all();
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }

    fn record_fallback(&self, path: Option<&Path>, reason: &CudaError) {
        lock(&self.fallbacks).push(Fallback {
            path: path.map(Path::to_owned),
            reason: reason.to_string(),
        });
    }

    /// `libcufile` is gone: the rest of the job stages. Recorded once.
    fn disable_direct(&self, reason: &CudaError) {
        if !self.direct_disabled.swap(true, Ordering::AcqRel) {
            self.record_fallback(None, reason);
        }
    }

    /// Count a piece that landed: its bytes read on `route`, the gap bytes
    /// dropped, and whether it went through placements.
    fn landed(&self, route: Route, piece_bytes: u64, landed_bytes: u64, placed: bool) {
        let counter = match route {
            Route::Host => &self.host_bytes,
            Route::CuFile => &self.cufile_bytes,
            Route::Staged => &self.staged_bytes,
        };
        counter.fetch_add(piece_bytes, Ordering::Relaxed);
        self.gap_bytes
            .fetch_add(piece_bytes - landed_bytes, Ordering::Relaxed);
        self.pieces.fetch_add(1, Ordering::Relaxed);
        if placed {
            self.placed_pieces.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run the job's files as the plan says. Returns the counts, or the first
/// failure once every worker has stopped.
pub(super) fn run<'job, 'a, 'r>(
    plan: &ReadPlan,
    storage: &'a dyn Storage,
    copier: Option<&'a dyn DeviceCopier>,
    direct: Option<&'a dyn DirectStorage>,
    ring: &'a LazyRing<'r>,
    files: Vec<(PathBuf, Vec<Piece<'job>>)>,
) -> Result<Counts, ReadError> {
    let file_workers = plan.files_in_flight.max(1).min(files.len());
    let shared = Shared {
        storage,
        copier,
        direct,
        ring,
        readers_per_file: plan.readers_per_file.max(1),
        files: Mutex::new(files.into()),
        cancel: AtomicBool::new(false),
        direct_disabled: AtomicBool::new(false),
        failure: Mutex::new(None),
        fallbacks: Mutex::new(Vec::new()),
        cufile_bytes: AtomicU64::new(0),
        staged_bytes: AtomicU64::new(0),
        host_bytes: AtomicU64::new(0),
        pieces: AtomicUsize::new(0),
        gap_bytes: AtomicU64::new(0),
        placed_pieces: AtomicUsize::new(0),
        scatter_copies: AtomicUsize::new(0),
        files_opened: AtomicUsize::new(0),
    };
    run_workers(file_workers, || file_worker(&shared));
    let failure = lock(&shared.failure).take();
    match failure {
        Some(error) => Err(error),
        None => Ok(Counts {
            cufile_bytes: shared.cufile_bytes.load(Ordering::Relaxed),
            staged_bytes: shared.staged_bytes.load(Ordering::Relaxed),
            host_bytes: shared.host_bytes.load(Ordering::Relaxed),
            pieces: shared.pieces.load(Ordering::Relaxed),
            gap_bytes: shared.gap_bytes.load(Ordering::Relaxed),
            placed_pieces: shared.placed_pieces.load(Ordering::Relaxed),
            scatter_copies: shared.scatter_copies.load(Ordering::Relaxed),
            files_opened: shared.files_opened.load(Ordering::Relaxed),
            fallbacks: std::mem::take(&mut *lock(&shared.fallbacks)),
        }),
    }
}

/// Run `work` on `n` threads: `n - 1` spawned in a scope plus the calling
/// thread. `n <= 1` runs it inline, once, with no thread.
fn run_workers<F: Fn() + Sync>(n: usize, work: F) {
    if n <= 1 {
        work();
        return;
    }
    std::thread::scope(|scope| {
        for _ in 1..n {
            scope.spawn(&work);
        }
        work();
    });
}

/// One file's context for its piece workers.
struct FileCtx<'job, 'a> {
    path: &'a Path,
    /// The `Storage` reader, opened on first need: before any piece under
    /// pread and mmap, on the first host piece or fallback under cuFile.
    /// `Some(None)` once an open has failed and been reported.
    reader: OnceLock<Option<Box<dyn RangeReader>>>,
    /// Device pieces of this file still go through cuFile.
    direct: AtomicBool,
    queue: Mutex<VecDeque<Piece<'job>>>,
}

impl FileCtx<'_, '_> {
    /// The reader, opening it on the first call. `None` when the open failed
    /// — reported by whoever tried — and the piece should be abandoned.
    fn reader(&self, shared: &Shared<'_, '_, '_>) -> Option<&dyn RangeReader> {
        self.reader
            .get_or_init(|| match shared.storage.open_reader(self.path) {
                Ok(reader) => {
                    shared.files_opened.fetch_add(1, Ordering::Relaxed);
                    Some(reader)
                }
                Err(source) => {
                    shared.fail(ReadError::Open {
                        path: self.path.to_owned(),
                        source,
                    });
                    None
                }
            })
            .as_deref()
    }

    /// cuFile, if this file's device pieces still go through it.
    fn direct<'s>(&self, shared: &Shared<'_, 's, '_>) -> Option<&'s dyn DirectStorage> {
        if self.direct.load(Ordering::Acquire) && !shared.direct_disabled.load(Ordering::Acquire) {
            shared.direct
        } else {
            None
        }
    }

    /// cuFile could not register this file: its remaining device pieces
    /// stage. Recorded once per file.
    fn fall_back(&self, shared: &Shared<'_, '_, '_>, reason: &CudaError) {
        if self.direct.swap(false, Ordering::AcqRel) {
            shared.record_fallback(Some(self.path), reason);
        }
    }
}

fn file_worker(shared: &Shared<'_, '_, '_>) {
    // the cancel check at the top of the loop is the only way out on
    // failure: `fail` records and raises, the loop sees the flag
    loop {
        if shared.cancelled() {
            return;
        }
        let Some((path, pieces)) = lock(&shared.files).pop_front() else {
            return;
        };
        let piece_workers = shared.readers_per_file.min(pieces.len());
        let file = FileCtx {
            path: &path,
            reader: OnceLock::new(),
            direct: AtomicBool::new(shared.direct.is_some()),
            queue: Mutex::new(pieces.into()),
        };
        // without cuFile every piece needs the reader: open it first, so an
        // unopenable file fails before any piece and a file with nothing to
        // read is still opened once
        if shared.direct.is_none() && file.reader(shared).is_none() {
            continue;
        }
        run_workers(piece_workers, || piece_worker(shared, &file));
    }
}

fn piece_worker(shared: &Shared<'_, '_, '_>, file: &FileCtx<'_, '_>) {
    // one scratch buffer per worker for placed host pieces, grown to the
    // largest piece and reused: a coalesced read lands through it
    let mut scratch = Vec::new();
    loop {
        if shared.cancelled() {
            return;
        }
        let Some(piece) = lock(&file.queue).pop_front() else {
            return;
        };
        let nbytes = piece.range.len;
        let landed = piece.landed_bytes();
        let placed = !piece.placements.is_empty();
        if let Some(route) = read_piece(shared, file, piece, &mut scratch) {
            shared.landed(route, nbytes, landed, placed);
        }
    }
}

/// Read one piece to its destination. Returns the route it landed by, or
/// `None` when the piece did not land: its failure has been recorded with
/// [`Shared::fail`], or the job was already failing and it was abandoned
/// unread. Failures are recorded *before* a staging slot is released, so a
/// waiter woken by the release sees the cancel flag and never starts a read
/// after the failure.
fn read_piece(
    shared: &Shared<'_, '_, '_>,
    file: &FileCtx<'_, '_>,
    piece: Piece<'_>,
    scratch: &mut Vec<u8>,
) -> Option<Route> {
    let range = piece.range;
    let placements = piece.placements;
    match piece.dest {
        PieceDest::Host(slice) if placements.is_empty() => {
            let reader = file.reader(shared)?;
            if let Err(source) = reader.read_at(range.offset, slice) {
                shared.fail(ReadError::Read {
                    path: file.path.to_owned(),
                    range,
                    source,
                });
                return None;
            }
            Some(Route::Host)
        }
        PieceDest::Host(slice) => {
            let reader = file.reader(shared)?;
            scratch.resize(range.len as usize, 0);
            if let Err(source) = reader.read_at(range.offset, scratch) {
                shared.fail(ReadError::Read {
                    path: file.path.to_owned(),
                    range,
                    source,
                });
                return None;
            }
            for p in &placements {
                let (src, dst, len) = (p.src as usize, p.dst as usize, p.len as usize);
                slice[dst..dst + len].copy_from_slice(&scratch[src..src + len]);
            }
            Some(Route::Host)
        }
        PieceDest::Device { ptr, offset } => {
            // cuFile lands a range whole: a placed piece stages by design
            if placements.is_empty()
                && let Some(direct) = file.direct(shared)
            {
                match direct.read_into_device(file.path, range.offset, range.len, ptr, offset) {
                    Ok(()) => return Some(Route::CuFile),
                    // the two answers that mean "not this way", not "broken":
                    // the file's mount cannot be registered; the library is gone
                    Err(source @ CudaError::Register { .. }) => file.fall_back(shared, &source),
                    Err(source @ CudaError::Unavailable { .. }) => shared.disable_direct(&source),
                    Err(source) => {
                        shared.fail(ReadError::DirectRead {
                            path: file.path.to_owned(),
                            range,
                            source,
                        });
                        return None;
                    }
                }
            }
            stage_piece(shared, file, range, ptr, offset, &placements).then_some(Route::Staged)
        }
    }
}

/// Enqueue the copies that land a staged piece: one contiguous copy when it
/// has no placements, one 2-D copy when its placements are regular, else
/// one copy per placement (counted in `scatter_copies`).
fn enqueue_copies(
    shared: &Shared<'_, '_, '_>,
    copier: &dyn DeviceCopier,
    slot: &dyn fst_cuda::PinnedBuffer,
    range: &ReadRange,
    ptr: DevicePtr,
    offset: u64,
    placements: &[Placement],
) -> Result<(), CudaError> {
    if placements.is_empty() {
        return copier.copy_to_device(slot, range.len, ptr, offset);
    }
    if let Some(shape) = regular_2d(placements, offset) {
        return copier.copy_to_device_2d(slot, ptr, shape);
    }
    shared
        .scatter_copies
        .fetch_add(placements.len(), Ordering::Relaxed);
    for p in placements {
        copier.copy_to_device_2d(
            slot,
            ptr,
            Copy2D {
                src_offset: p.src,
                src_pitch: p.len,
                width: p.len,
                height: 1,
                dst_offset: offset + p.dst,
                dst_pitch: p.len,
            },
        )?;
    }
    Ok(())
}

/// Read one device piece through pread into a staging slot and enqueue its
/// copies. `true` when the piece landed.
fn stage_piece(
    shared: &Shared<'_, '_, '_>,
    file: &FileCtx<'_, '_>,
    range: ReadRange,
    ptr: DevicePtr,
    offset: u64,
    placements: &[Placement],
) -> bool {
    let Some(reader) = file.reader(shared) else {
        return false;
    };
    // validation guaranteed the copier; the variant is the typed answer if
    // a caller ever reaches here without it
    let Some(copier) = shared.copier else {
        shared.fail(ReadError::CopierRequired);
        return false;
    };
    let ring = match shared.ring.get() {
        Ok(Some(ring)) => ring,
        Ok(None) => return false,
        Err(error) => {
            shared.fail(error);
            return false;
        }
    };
    let mut slot = match ring.acquire(&shared.cancel) {
        Ok(Some(slot)) => slot,
        Ok(None) => return false,
        Err(error) => {
            shared.fail(error);
            return false;
        }
    };
    // `split_file` sizes device pieces to the ring; a typed error rather
    // than an index panic, because a worker that unwinds holding a slot
    // would strand the ring's waiters
    let len = range.len as usize;
    let capacity = slot.capacity();
    let Some(window) = slot.as_mut_slice().get_mut(..len) else {
        shared.fail(ReadError::PieceExceedsStaging {
            path: file.path.to_owned(),
            range,
            capacity,
        });
        ring.release_clean(slot);
        return false;
    };
    if let Err(source) = reader.read_at(range.offset, window) {
        shared.fail(ReadError::Read {
            path: file.path.to_owned(),
            range,
            source,
        });
        ring.release_clean(slot);
        return false;
    }
    let copied = enqueue_copies(
        shared,
        copier,
        slot.as_ref(),
        &range,
        ptr,
        offset,
        placements,
    );
    let landed = match copied {
        Ok(()) => true,
        Err(source) => {
            shared.fail(ReadError::Copy {
                path: file.path.to_owned(),
                range,
                source,
            });
            false
        }
    };
    // dirty either way: a failed enqueue may still have touched the stream.
    // The fence recorded now completes when this slot's copies have read it,
    // which is when the ring may hand it out again; without a fence the slot
    // sits out the job and the final synchronize settles it.
    match copier.record() {
        Ok(fence) => ring.release_dirty(slot, fence),
        Err(source) => {
            ring.release_parked(slot);
            if landed {
                shared.fail(ReadError::Staging { source });
                return false;
            }
        }
    }
    landed
}
