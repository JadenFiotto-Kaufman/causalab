//! The read engine against the simulators, by example and by property, plus
//! one round trip through the real backends. Helpers live here; the cases
//! are grouped by what they pin: bytes landing, refusals, faults, properties,
//! the cuFile transport with its fallback to pread, and placed transfers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fst_cuda::DeviceCopier;
use fst_cuda::sim::{Call, SimCuda};
use proptest::prelude::*;

use crate::plan::{ReadPlan, Staging, Transport};
use crate::storage::mmap::MmapStorage;
use crate::storage::posix::PosixStorage;
use crate::storage::sim::{Fault, FaultKind, Op, SimStorage};
use crate::storage::{ReadRange, Storage};

use super::{Dest, FileJob, ReadError, ReadJob, Transfer, execute};

/// A byte no seeded object ever contains, so an untouched destination is
/// recognisable.
const SENTINEL: u8 = 0xAA;

/// Deterministic bytes that never equal [`SENTINEL`].
fn seeded(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((x >> 33) % 254) as u8;
            if b >= SENTINEL { b + 1 } else { b }
        })
        .collect()
}

fn plan(
    files_in_flight: usize,
    split_bytes: u64,
    readers_per_file: usize,
    staging: Staging,
) -> ReadPlan {
    ReadPlan {
        files_in_flight,
        split_bytes,
        readers_per_file,
        transport: Transport::Pread,
        staging,
        reasons: vec![],
        profile_source: String::new(),
    }
}

fn range(offset: u64, len: u64) -> ReadRange {
    ReadRange { offset, len }
}

fn ok<T>(result: Result<T, ReadError>) -> T {
    result.unwrap_or_else(|e| panic!("{e}"))
}

/// Reads per path from a sim log, as `(offset, len)` sorted.
fn reads_by_path(log: &[Op]) -> BTreeMap<PathBuf, Vec<(u64, u64)>> {
    let mut out: BTreeMap<PathBuf, Vec<(u64, u64)>> = BTreeMap::new();
    for op in log {
        if let Op::Read(path, offset, len) = op {
            out.entry(path.clone()).or_default().push((*offset, *len));
        }
    }
    for reads in out.values_mut() {
        reads.sort_unstable();
    }
    out
}

fn opens(log: &[Op]) -> Vec<PathBuf> {
    log.iter()
        .filter_map(|op| match op {
            Op::Open(p) => Some(p.clone()),
            _ => None,
        })
        .collect()
}

/// At most `count` copies may be outstanding: each enqueue consumes a slot,
/// a fence wait frees the one slot behind it, a synchronize frees them all.
fn assert_ring_bound(calls: &[Call], count: usize) {
    let mut outstanding = 0usize;
    for call in calls {
        match call {
            Call::ToDevice(..) => {
                outstanding += 1;
                assert!(
                    outstanding <= count,
                    "{outstanding} copies outstanding with a ring of {count}: {calls:?}"
                );
            }
            Call::Wait => outstanding = outstanding.saturating_sub(1),
            Call::Synchronize => outstanding = 0,
            _ => {}
        }
    }
}

/// The pieces a range is expected to be read in.
fn expected_pieces(r: &ReadRange, split_bytes: u64, staging_bytes: Option<u64>) -> Vec<(u64, u64)> {
    let block = match staging_bytes {
        Some(b) if split_bytes == 0 => b,
        Some(b) => b.min(split_bytes),
        None => split_bytes,
    };
    r.split(block)
        .into_iter()
        .map(|p| (p.offset, p.len))
        .collect()
}

mod direct;
mod faults;
mod landing;
mod placements;
mod props;
mod refusals;

// ---- real backends -----------------------------------------------------

#[test]
fn real_backends_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("weights");
    let bytes = seeded(1000, 42);
    std::fs::write(&path, &bytes).unwrap();
    let backends: [(&dyn Storage, Transport); 2] = [
        (&PosixStorage, Transport::Pread),
        (&MmapStorage, Transport::Mmap),
    ];
    for (storage, transport) in backends {
        let mut p = plan(1, 64, 3, Staging::None);
        p.transport = transport;
        let mut head = vec![SENTINEL; 300];
        let mut tail = vec![SENTINEL; 150];
        let job = ReadJob {
            files: vec![FileJob {
                path: path.clone(),
                transfers: vec![
                    Transfer {
                        range: range(0, 300),
                        dest: Dest::Host(&mut head),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(850, 150),
                        dest: Dest::Host(&mut tail),
                        placements: Vec::new(),
                    },
                ],
            }],
        };
        let report = ok(execute(&p, storage, None, None, job));
        assert_eq!(head, bytes[..300], "{}", storage.name());
        assert_eq!(tail, bytes[850..], "{}", storage.name());
        assert_eq!(report.bytes_read, 450);
        assert_eq!(report.pieces, 5 + 3);
        assert_eq!(report.files_opened, 1);
        // a range past the end is the backend's typed refusal, named by the engine
        let mut over = vec![0u8; 8];
        let job = ReadJob {
            files: vec![FileJob {
                path: path.clone(),
                transfers: vec![Transfer {
                    range: range(996, 8),
                    dest: Dest::Host(&mut over),
                    placements: Vec::new(),
                }],
            }],
        };
        assert!(matches!(
            execute(&p, storage, None, None, job),
            Err(ReadError::Read { range: r, source: crate::StorageError::OutOfRange { .. }, .. }) if r == range(996, 8)
        ));
    }
}
