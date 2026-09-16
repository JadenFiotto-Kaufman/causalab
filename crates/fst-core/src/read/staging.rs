//! The pinned staging ring for device destinations.
//!
//! `count` pinned buffers of `bytes` each, allocated once per job. A worker
//! takes a free slot, reads a piece into it, enqueues the copy to the device
//! and hands the slot back *dirty* behind a fence recorded on the copy
//! stream right after its copies. A dirty slot holds bytes an asynchronous
//! copy may still be reading, so it is not free again until its fence has
//! completed.
//!
//! Reuse rule. Dirty slots queue roughly in the order their fences were
//! recorded (record and release are two calls, so two workers can swap
//! places), which on one stream is about the order their copies complete.
//! A slot's own fence is recorded after its own copies, so waiting on the
//! entry that is popped is always sufficient for that slot; the order only
//! decides which slot a worker gets. A worker that needs a slot and finds
//! none free takes the front dirty slot, waits on that slot's fence alone
//! (with the lock released, so other workers keep reading and other waiters
//! take the next slots), and uses it. Nothing
//! drains the stream: a job of many small pieces cycles its slots as fast
//! as the copy engine lands them, instead of stalling every worker on a
//! whole-stream synchronize each time the free list empties. The final
//! synchronize in `execute` still settles whatever is left.
//!
//! A slot whose fence could not be recorded is parked: never reused in this
//! job, settled by that final synchronize.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use fst_cuda::{DeviceCopier, Fence, PinnedBuffer};

use super::ReadError;

/// A ring allocated on first use. A `CuFile` job only pays for pinned
/// memory if some piece actually stages; a pread job forces it before any
/// I/O so an allocation failure is reported before a byte moves.
pub(super) struct LazyRing<'a> {
    copier: Option<&'a dyn DeviceCopier>,
    /// `(count, bytes)`; `None` when the job has no device destination.
    geometry: Option<(usize, u64)>,
    /// `Some(None)` once an allocation has failed: the failure was reported
    /// by the worker that tried, later askers just stand down.
    cell: OnceLock<Option<Ring>>,
}

impl<'a> LazyRing<'a> {
    pub(super) fn new(
        copier: Option<&'a dyn DeviceCopier>,
        geometry: Option<(usize, u64)>,
    ) -> Self {
        LazyRing {
            copier,
            geometry,
            cell: OnceLock::new(),
        }
    }

    /// The ring, allocating it on the first call. `Err` from the call that
    /// tried and failed; `Ok(None)` afterwards, or when there is nothing to
    /// allocate — the caller then abandons its piece, the job is already
    /// failing.
    pub(super) fn get(&self) -> Result<Option<&Ring>, ReadError> {
        let mut error = None;
        let ring = self.cell.get_or_init(|| {
            let (Some(copier), Some((count, bytes))) = (self.copier, self.geometry) else {
                // validation guaranteed both for any job with a device
                // destination; the variants are the typed answer otherwise
                error = Some(if self.copier.is_none() {
                    ReadError::CopierRequired
                } else {
                    ReadError::StagingRequired
                });
                return None;
            };
            match Ring::allocate(copier, count, bytes) {
                Ok(ring) => Some(ring),
                Err(e) => {
                    error = Some(e);
                    None
                }
            }
        });
        match (ring, error) {
            (Some(ring), _) => Ok(Some(ring)),
            (None, Some(error)) => Err(error),
            (None, None) => Ok(None),
        }
    }

    /// The ring if it has been allocated.
    pub(super) fn allocated(&self) -> Option<&Ring> {
        self.cell.get().and_then(Option::as_ref)
    }
}

/// A pinned buffer on loan from the ring.
pub(super) type Slot = Box<dyn PinnedBuffer>;

#[derive(Default)]
struct State {
    /// Slots no enqueued copy reads from.
    free: Vec<Slot>,
    /// Slots an enqueued copy may still read from, oldest first, each behind
    /// the fence recorded after its copies.
    dirty: VecDeque<(Slot, Box<dyn Fence>)>,
    /// Slots with copies enqueued and no fence to wait on: out of the ring
    /// until the job's final synchronize.
    parked: Vec<Slot>,
}

/// The ring.
pub(super) struct Ring {
    state: Mutex<State>,
    changed: Condvar,
}

impl Ring {
    /// Allocate `count` pinned buffers of `bytes` each.
    pub(super) fn allocate(
        copier: &dyn DeviceCopier,
        count: usize,
        bytes: u64,
    ) -> Result<Self, ReadError> {
        let mut free = Vec::with_capacity(count);
        for _ in 0..count {
            free.push(
                copier
                    .alloc_pinned(bytes)
                    .map_err(|source| ReadError::Staging { source })?,
            );
        }
        Ok(Ring {
            state: Mutex::new(State {
                free,
                dirty: VecDeque::new(),
                parked: Vec::new(),
            }),
            changed: Condvar::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // a poisoned lock means another worker panicked, which the scope
        // will report; the ring's bookkeeping is still consistent
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take a slot: a free one, else the oldest dirty one once its fence has
    /// completed. Returns `None` if `cancel` was raised while waiting, so a
    /// cancelled worker never starts a read it would have to abandon.
    pub(super) fn acquire(&self, cancel: &AtomicBool) -> Result<Option<Slot>, ReadError> {
        let mut state = self.lock();
        loop {
            // cancel before free: a failure is recorded before the failing
            // worker releases its slot, so nobody starts a read after it
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            if let Some(slot) = state.free.pop() {
                return Ok(Some(slot));
            }
            if let Some((slot, fence)) = state.dirty.pop_front() {
                drop(state);
                return match fence.wait() {
                    // a waiter on a fence is not on the condvar, so `wake_all`
                    // does not reach it: re-check before handing out the slot
                    Ok(()) if cancel.load(Ordering::Acquire) => {
                        self.release_clean(slot);
                        Ok(None)
                    }
                    Ok(()) => Ok(Some(slot)),
                    Err(source) => {
                        // the copy may still be reading it: park it for the
                        // final synchronize in `execute`
                        self.lock().parked.push(slot);
                        self.changed.notify_all();
                        Err(ReadError::Staging { source })
                    }
                };
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Return a slot no copy was enqueued from.
    pub(super) fn release_clean(&self, slot: Slot) {
        self.lock().free.push(slot);
        self.changed.notify_all();
    }

    /// Return a slot whose copies are enqueued, behind the fence recorded
    /// after them.
    pub(super) fn release_dirty(&self, slot: Slot, fence: Box<dyn Fence>) {
        self.lock().dirty.push_back((slot, fence));
        self.changed.notify_all();
    }

    /// Return a slot whose copies are enqueued with no fence to wait on: it
    /// sits out the rest of the job.
    pub(super) fn release_parked(&self, slot: Slot) {
        self.lock().parked.push(slot);
    }

    /// Wake every waiter, so a cancelled job's waiters see the flag.
    pub(super) fn wake_all(&self) {
        self.changed.notify_all();
    }
}
