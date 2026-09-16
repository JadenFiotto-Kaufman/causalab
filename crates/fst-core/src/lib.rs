//! `fst-core`: the safetensors container and the machinery that moves it.
//!
//! Everything here is free of Python, CUDA and any particular framework.
//! The crate is organised around one question — *given this machine and this
//! request, what is the fastest correct way to move these bytes?* — split into
//! parts that can each be tested on their own:
//!
//! * [`format`]: the container grammar. Header parsing and building, the
//!   dtype table, the tensor layout. Byte-identical to the reference
//!   `safetensors` library on output, strict on input.
//! * [`storage`]: the I/O boundary as traits — a range reader, a part writer —
//!   with the real backends (`posix`, `mmap`) and a [`storage::sim`] backend
//!   that records every operation and injects faults on a schedule, so the
//!   orchestration above it is tested deterministically.
//! * [`env`]: what the machine is. Storage class per path, GPUDirect Storage
//!   availability, CPU count. Probed once, carried as a value.
//! * [`profile`]: the calibration figures — per storage class and mount,
//!   how fast one file goes, how the aggregate grows with files in flight,
//!   whether splitting helps, whether cuFile registers; and the device path.
//!   A JSON document a measuring tool emits; a built-in one carries the
//!   measurements in `docs/fastersafetensors.md`.
//! * [`plan`]: the pure decision. An [`env::Env`], a [`profile::Profile`] and
//!   a request in, a plan out — how many files in flight, how each is split,
//!   which transport, what staging — with the reasoning attached, so
//!   `explain()` can print it.
//! * [`write`]: the write engine. A [`write::Payload`] — built header plus
//!   the caller's host or device buffers — streamed part by part through a
//!   [`storage::PartWriter`], device parts staged through pinned memory,
//!   optionally durable and atomic.
//! * [`select`]: N-d selections over a tensor — the box a tensor-parallel
//!   rank wants — as the byte runs they read, and those runs coalesced into
//!   reads with placements, under a policy the profile sets.
//! * [`read`]: the plan executed. Files in flight, pieces per file, one
//!   reader per file, every destination byte written once; device
//!   destinations reached through `fst-cuda`'s staging traits (the traits
//!   only — `fst-cuda` links nothing, so this crate still builds anywhere).
//!
//! The rules in [`plan`] encode measurements, not guesses; the numbers come
//! from the [`profile`] and each reason names the entry it used. See
//! `docs/fastersafetensors.md` at the repository root.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod env;
pub mod error;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod format;
pub mod plan;
pub mod profile;
pub mod read;
pub mod select;
pub mod storage;
pub mod write;

pub use error::{FormatError, PlanError, ProfileError, SelectError, StorageError};
pub use read::ReadError;
pub use write::WriteError;
