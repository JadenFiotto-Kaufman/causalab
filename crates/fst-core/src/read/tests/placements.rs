//! Placed transfers: coalesced reads landing through placements, on the host
//! through scratch and on a device through one 2-D copy per regular piece or
//! one copy per placement otherwise.

use fst_cuda::Copy2D;
use fst_cuda::sim_direct::SimDirect;

use super::*;
use crate::format::Dtype;
use crate::read::{PlacementFault, ReadReport};
use crate::select::{CoalescePolicy, Placement, Read, Selection, coalesce};

/// A tensor of `shape` × `dtype` stored at `start` in object `name`: the
/// object is `start` bytes of filler then the tensor's bytes, seeded.
struct Stored {
    start: u64,
    bytes: Vec<u8>,
}

fn store_tensor(
    store: &SimStorage,
    name: &'static str,
    start: u64,
    nbytes: u64,
    seed: u64,
) -> Stored {
    let bytes = seeded((start + nbytes) as usize, seed);
    store.put(name, bytes.clone());
    Stored { start, bytes }
}

/// The bytes a selection's runs pick out of the stored tensor, in order:
/// the reference every landing is compared with.
fn reference(stored: &Stored, s: &Selection, dtype: Dtype) -> Vec<u8> {
    s.runs(dtype)
        .unwrap_or_else(|e| panic!("{e}"))
        .iter()
        .flat_map(|r| {
            let from = (stored.start + r.offset) as usize;
            stored.bytes[from..from + r.len as usize].to_vec()
        })
        .collect()
}

fn reads_for(s: &Selection, dtype: Dtype, policy: CoalescePolicy) -> Vec<Read> {
    coalesce(&s.runs(dtype).unwrap_or_else(|e| panic!("{e}")), &policy)
}

/// Host transfers for `reads`, each landing in its own slice of `dest`.
fn host_transfers<'a>(reads: &[Read], start: u64, mut dest: &'a mut [u8]) -> Vec<Transfer<'a>> {
    let mut out = Vec::with_capacity(reads.len());
    for read in reads {
        let (head, tail) = dest.split_at_mut(read.wanted() as usize);
        dest = tail;
        out.push(Transfer {
            range: read.in_file(start),
            dest: Dest::Host(head),
            placements: read.placements.clone(),
        });
    }
    out
}

/// Device transfers for `reads`, landing back to back in `ptr` from `base`.
fn device_transfers<'a>(
    reads: &[Read],
    start: u64,
    ptr: fst_cuda::DevicePtr,
    base: u64,
) -> Vec<Transfer<'a>> {
    reads
        .iter()
        .map(|read| Transfer {
            range: read.in_file(start),
            dest: Dest::Device {
                ptr,
                offset: base + read.dst,
            },
            placements: read.placements.clone(),
        })
        .collect()
}

fn policy(max_gap_bytes: u64, max_read_bytes: u64) -> CoalescePolicy {
    CoalescePolicy {
        max_gap_bytes,
        max_read_bytes,
    }
}

fn two_d_calls(calls: &[Call]) -> Vec<Copy2D> {
    calls
        .iter()
        .filter_map(|c| match c {
            Call::ToDevice2D(_, shape) => Some(*shape),
            _ => None,
        })
        .collect()
}

fn assert_no_gap_boundaries(reads: &[(u64, u64)], runs: &[(u64, u64)]) {
    // every piece begins at a run's byte and ends at a run's byte
    for &(offset, len) in reads {
        let end = offset + len;
        assert!(
            runs.iter().any(|&(o, l)| (o..o + l).contains(&offset)),
            "piece {offset}..{end} starts in a gap"
        );
        assert!(
            runs.iter()
                .any(|&(o, l)| (o..=o + l).contains(&end) && end > o),
            "piece {offset}..{end} ends in a gap"
        );
    }
}

// ---- host ---------------------------------------------------------------

#[test]
fn strided_host_reads_land_exactly_through_coalesced_pieces() {
    // [6, 16] bf16, sharded on the inner dim over 4 ranks: 6 runs of 8 bytes
    // with 24-byte gaps, at 32-byte stride
    let store = SimStorage::new();
    let stored = store_tensor(&store, "w", 40, 6 * 16 * 2, 1);
    let s = Selection::shard(&[6, 16], 1, 1, 4).unwrap_or_else(|e| panic!("{e}"));
    let expected = reference(&stored, &s, Dtype::BF16);
    assert_eq!(expected.len(), 48);
    // a gap of 24 merges every run; a cap of 100 cuts the 6 runs into two reads
    let reads = reads_for(&s, Dtype::BF16, policy(24, 100));
    assert_eq!(
        reads
            .iter()
            .map(|r| (r.offset, r.len, r.dst))
            .collect::<Vec<_>>(),
        vec![(8, 72, 0), (104, 72, 24)]
    );
    let mut dest = vec![SENTINEL; 48];
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: host_transfers(&reads, stored.start, &mut dest),
        }],
    };
    // split at 50: two runs fit a piece, the third opens the next, so no
    // piece touches a gap at either end
    let report = ok(execute(
        &plan(1, 50, 1, Staging::None),
        &store,
        None,
        None,
        job,
    ));
    assert_eq!(dest, expected);
    assert_eq!(report.bytes_read - report.gap_bytes, 48);
    assert_eq!(report.host_bytes, report.bytes_read);
    assert_eq!(report.pieces, report.placed_pieces);
    assert_eq!(report.scatter_copies, 0);
    let log = store.log();
    let pieces = &reads_by_path(&log)[Path::new("w")];
    // pieces of at most 50 bytes that never start or end in a gap: each
    // read's first piece takes its first two runs (48..88, 40 bytes) and
    // stops at the run end rather than reading into the gap; the third run
    // is its own piece
    assert_eq!(pieces, &vec![(48, 40), (112, 8), (144, 40), (208, 8)]);
    let runs: Vec<(u64, u64)> = s
        .runs(Dtype::BF16)
        .unwrap_or_else(|e| panic!("{e}"))
        .iter()
        .map(|r| (stored.start + r.offset, r.len))
        .collect();
    assert_no_gap_boundaries(pieces, &runs);
    // the gap bytes read are only those inside a piece: 24 per piece of two runs
    assert_eq!(report.bytes_read, 40 + 8 + 40 + 8);
    assert_eq!(report.gap_bytes, 48);
}

#[test]
fn a_full_selection_is_an_ordinary_transfer() {
    let store = SimStorage::new();
    let stored = store_tensor(&store, "w", 8, 64, 2);
    let s = Selection::full(&[4, 16]).unwrap_or_else(|e| panic!("{e}"));
    let reads = reads_for(&s, Dtype::U8, policy(64, 1 << 20));
    assert_eq!(reads.len(), 1);
    assert!(reads[0].is_contiguous());
    let mut dest = vec![SENTINEL; 64];
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: host_transfers(&reads, stored.start, &mut dest),
        }],
    };
    let report = ok(execute(
        &plan(1, 16, 1, Staging::None),
        &store,
        None,
        None,
        job,
    ));
    assert_eq!(dest, stored.bytes[8..]);
    // the single whole-range placement was normalised away: plain pieces
    assert_eq!(report.pieces, 4);
    assert_eq!(report.placed_pieces, 0);
    assert_eq!(report.gap_bytes, 0);
}

// ---- device -------------------------------------------------------------

#[test]
fn regular_device_placements_land_with_one_2d_copy_per_piece() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    // [8, 8] f32 sharded on the inner dim over 2: 8 runs of 16 bytes at
    // 32-byte stride, gap 16
    let stored = store_tensor(&store, "w", 16, 8 * 8 * 4, 3);
    let s = Selection::shard(&[8, 8], 1, 1, 2).unwrap_or_else(|e| panic!("{e}"));
    let expected = reference(&stored, &s, Dtype::F32);
    // one read of everything (gap 16, cap generous)
    let reads = reads_for(&s, Dtype::F32, policy(16, 1 << 20));
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0].placements.len(), 8);
    let ptr = cuda.alloc_device(128 + 8, 0);
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: device_transfers(&reads, stored.start, ptr, 8),
        }],
    };
    // staging of 100 bytes cuts the 240-byte read into pieces of at most
    // 100, each trimmed to run boundaries
    let staging = Staging::Pinned {
        count: 2,
        bytes: 100,
    };
    let report = ok(execute(
        &plan(1, 0, 1, staging),
        &store,
        Some(&cuda),
        None,
        job,
    ));
    let landed = cuda.device_bytes(ptr).unwrap_or_default();
    assert_eq!(&landed[8..], &expected[..]);
    assert_eq!(&landed[..8], &[0u8; 8]);
    assert_eq!(report.scatter_copies, 0);
    assert_eq!(report.staged_bytes, report.bytes_read);
    assert_eq!(report.bytes_read - report.gap_bytes, 128);
    let calls = cuda.log();
    let shapes = two_d_calls(&calls);
    // one 2-D copy per piece, rows of 16 at source pitch 32 landing
    // contiguously (destination pitch 16); the heights sum to the runs
    assert_eq!(shapes.len(), report.placed_pieces);
    assert_eq!(report.placed_pieces, report.pieces);
    assert_eq!(shapes.iter().map(|c| c.height).sum::<u64>(), 8);
    for shape in &shapes {
        assert_eq!(
            (shape.width, shape.src_pitch, shape.dst_pitch),
            (16, 32, 16)
        );
        assert_eq!(shape.src_offset, 0, "every piece starts at a run");
    }
    assert!(!calls.iter().any(|c| matches!(c, Call::ToDevice(..))));
    assert_eq!(calls.last(), Some(&Call::Synchronize));
    // pieces never exceed the staging buffer and never end in a gap
    let pieces = &reads_by_path(&store.log())[Path::new("w")];
    assert!(pieces.iter().all(|&(_, l)| l <= 100));
    let runs: Vec<(u64, u64)> = s
        .runs(Dtype::F32)
        .unwrap_or_else(|e| panic!("{e}"))
        .iter()
        .map(|r| (stored.start + r.offset, r.len))
        .collect();
    assert_no_gap_boundaries(pieces, &runs);
}

#[test]
fn irregular_device_placements_fall_back_to_one_copy_each_and_are_counted() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let bytes = seeded(64, 4);
    store.put("w", bytes.clone());
    // three runs of unequal length: no single 2-D shape fits
    let placements = vec![
        Placement {
            src: 0,
            dst: 0,
            len: 4,
        },
        Placement {
            src: 10,
            dst: 4,
            len: 6,
        },
        Placement {
            src: 30,
            dst: 10,
            len: 2,
        },
    ];
    let ptr = cuda.alloc_device(12, 0);
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: vec![Transfer {
                range: range(8, 32),
                dest: Dest::Device { ptr, offset: 0 },
                placements,
            }],
        }],
    };
    let staging = Staging::Pinned {
        count: 1,
        bytes: 64,
    };
    let report = ok(execute(
        &plan(1, 0, 1, staging),
        &store,
        Some(&cuda),
        None,
        job,
    ));
    let mut expected = bytes[8..12].to_vec();
    expected.extend_from_slice(&bytes[18..24]);
    expected.extend_from_slice(&bytes[38..40]);
    assert_eq!(cuda.device_bytes(ptr), Some(expected));
    assert_eq!(report.pieces, 1);
    assert_eq!(report.placed_pieces, 1);
    assert_eq!(report.scatter_copies, 3);
    assert_eq!(report.gap_bytes, 32 - 12);
    let shapes = two_d_calls(&cuda.log());
    assert_eq!(shapes.len(), 3);
    assert!(shapes.iter().all(|c| c.height == 1));
    assert_eq!(
        shapes
            .iter()
            .map(|c| (c.src_offset, c.dst_offset, c.width))
            .collect::<Vec<_>>(),
        vec![(0, 0, 4), (10, 4, 6), (30, 10, 2)]
    );
}

#[test]
fn placed_device_pieces_stage_under_a_cufile_plan_by_design() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let bytes = seeded(64, 5);
    store.put("w", bytes.clone());
    direct.put("w", bytes.clone());
    let s = Selection::shard(&[4, 16], 1, 0, 2).unwrap_or_else(|e| panic!("{e}"));
    let reads = reads_for(&s, Dtype::U8, policy(8, 1 << 20));
    let ptr = cuda.alloc_device(32 + 8, 0);
    let mut transfers = device_transfers(&reads, 0, ptr, 0);
    // and one whole-range transfer that cuFile can take
    transfers.push(Transfer {
        range: range(56, 8),
        dest: Dest::Device { ptr, offset: 32 },
        placements: Vec::new(),
    });
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers,
        }],
    };
    let mut p = plan(1, 0, 1, Staging::None);
    p.transport = Transport::CuFile;
    let report = ok(execute(&p, &store, Some(&cuda), Some(&direct), job));
    let mut expected: Vec<u8> = (0..4)
        .flat_map(|row| bytes[row * 16..row * 16 + 8].to_vec())
        .collect();
    expected.extend_from_slice(&bytes[56..]);
    assert_eq!(cuda.device_bytes(ptr), Some(expected));
    // the placed piece staged, the plain one went through cuFile, and
    // neither is a fallback
    assert_eq!(report.staged_bytes, 56);
    assert_eq!(report.cufile_bytes, 8);
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    assert_eq!(direct.log().len(), 1);
    assert_eq!(two_d_calls(&cuda.log()).len(), 1);
}

// ---- faults and refusals -----------------------------------------------

#[test]
fn a_fault_on_a_coalesced_read_names_the_piece_range() {
    let store = SimStorage::new();
    let stored = store_tensor(&store, "w", 0, 64, 6);
    let s = Selection::shard(&[4, 16], 1, 1, 2).unwrap_or_else(|e| panic!("{e}"));
    let reads = reads_for(&s, Dtype::U8, policy(8, 1 << 20));
    assert_eq!(reads.len(), 1);
    store.arm(Fault {
        kind: FaultKind::Read,
        nth: 0,
        message: "eio".into(),
    });
    let mut dest = vec![SENTINEL; 32];
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: host_transfers(&reads, stored.start, &mut dest),
        }],
    };
    let err = execute(&plan(1, 0, 1, Staging::None), &store, None, None, job).err();
    // the range named is the coalesced read, 8..64, not one of its runs
    assert!(
        matches!(&err, Some(ReadError::Read { path, range: r, source: crate::StorageError::Simulated(m) })
            if path == Path::new("w") && *r == range(8, 56) && m == "eio"),
        "{err:?}"
    );
    assert_eq!(dest, vec![SENTINEL; 32], "nothing landed");
}

#[test]
fn invalid_placements_are_refused_before_io() {
    let store = SimStorage::new();
    store.put("w", seeded(64, 7));
    let cuda = SimCuda::new(1 << 30);
    let ptr = cuda.alloc_device(8, 0);
    let p = |src, dst, len| Placement { src, dst, len };
    let attempt = |placements: Vec<Placement>, dest_len: u64| {
        let mut dest = vec![0u8; dest_len as usize];
        let job = ReadJob {
            files: vec![FileJob {
                path: "w".into(),
                transfers: vec![Transfer {
                    range: range(0, 16),
                    dest: Dest::Host(&mut dest),
                    placements,
                }],
            }],
        };
        execute(&plan(1, 0, 1, Staging::None), &store, None, None, job).err()
    };
    assert!(matches!(
        attempt(vec![p(0, 0, 4), p(8, 4, 12)], 16),
        Some(ReadError::InvalidPlacement {
            index: 1,
            fault: PlacementFault::OutOfRange {
                src: 8,
                len: 12,
                range_len: 16
            },
            ..
        })
    ));
    assert!(matches!(
        attempt(vec![p(0, 0, 4), p(2, 4, 4)], 8),
        Some(ReadError::InvalidPlacement {
            index: 1,
            fault: PlacementFault::Unordered {
                src: 2,
                previous_end: 4
            },
            ..
        })
    ));
    assert!(matches!(
        attempt(vec![p(0, 0, 0)], 0),
        Some(ReadError::InvalidPlacement {
            index: 0,
            fault: PlacementFault::Unordered { .. },
            ..
        })
    ));
    assert!(matches!(
        attempt(vec![p(0, 0, 4), p(8, 6, 4)], 8),
        Some(ReadError::InvalidPlacement {
            index: 1,
            fault: PlacementFault::NotPacked {
                dst: 6,
                expected: 4
            },
            ..
        })
    ));
    assert!(matches!(
        attempt(vec![p(0, 0, 4), p(8, 4, 4)], 9),
        Some(ReadError::PlacedDestinationMismatch {
            wanted: 8,
            available: 9,
            ..
        })
    ));
    // a device buffer too small for the placements from its offset
    let job = ReadJob {
        files: vec![FileJob {
            path: "w".into(),
            transfers: vec![Transfer {
                range: range(0, 16),
                dest: Dest::Device { ptr, offset: 2 },
                placements: vec![p(0, 0, 4), p(8, 4, 4)],
            }],
        }],
    };
    let staging = Staging::Pinned {
        count: 1,
        bytes: 16,
    };
    assert!(matches!(
        execute(&plan(1, 0, 1, staging), &store, Some(&cuda), None, job).err(),
        Some(ReadError::PlacedDestinationMismatch {
            wanted: 8,
            available: 6,
            ..
        })
    ));
    assert!(store.log().is_empty());
    assert!(cuda.log().is_empty());
}

// ---- properties --------------------------------------------------------

#[derive(Debug, Clone)]
struct Case {
    shape: Vec<u64>,
    ranges: Vec<std::ops::Range<u64>>,
    dtype: Dtype,
    max_gap: u64,
    max_read: u64,
    split_bytes: u64,
    device: Option<(usize, u64)>,
}

fn arb_case() -> impl Strategy<Value = Case> {
    prop::collection::vec(1u64..6, 1..4).prop_flat_map(|shape| {
        let ranges: Vec<_> = shape
            .iter()
            .map(|&extent| {
                (0..extent).prop_flat_map(move |start| (Just(start), start + 1..=extent))
            })
            .collect();
        (
            Just(shape),
            ranges,
            prop::sample::select(vec![Dtype::U8, Dtype::BF16, Dtype::F32, Dtype::F64]),
            0u64..64,
            1u64..200,
            prop_oneof![Just(0u64), 1u64..40],
            prop::option::of((1usize..3, 1u64..40)),
        )
            .prop_map(
                |(shape, ranges, dtype, max_gap, max_read, split_bytes, device)| Case {
                    shape,
                    ranges: ranges.into_iter().map(|(s, e)| s..e).collect(),
                    dtype,
                    max_gap,
                    max_read,
                    split_bytes,
                    device,
                },
            )
    })
}

fn check_report(report: &ReadReport, wanted: u64, reads: &[Read]) {
    assert_eq!(report.bytes_read - report.gap_bytes, wanted);
    // pieces read at most the coalesced reads: cutting at run ends drops
    // the gaps that fall between pieces
    let read_bytes: u64 = reads.iter().map(|r| r.len).sum();
    assert!(
        report.bytes_read <= read_bytes,
        "{} > {read_bytes}",
        report.bytes_read
    );
    assert!(report.placed_pieces <= report.pieces);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn random_selections_land_exactly_through_coalesced_pieces(case in arb_case()) {
        let store = SimStorage::new();
        let cuda = SimCuda::new(1 << 30);
        let s = Selection::new(&case.shape, case.ranges.clone()).unwrap_or_else(|e| panic!("{e}"));
        let nbytes = s.nbytes(case.dtype).unwrap_or_else(|e| panic!("{e}"));
        let total = case.shape.iter().product::<u64>() * u64::from(case.dtype.bits()) / 8;
        let start = 5;
        let stored = store_tensor(&store, "t", start, total, 77);
        let expected = reference(&stored, &s, case.dtype);
        let reads = reads_for(&s, case.dtype, policy(case.max_gap, case.max_read));
        let wanted: u64 = reads.iter().map(Read::wanted).sum();
        prop_assert_eq!(wanted, nbytes);
        match case.device {
            None => {
                let mut dest = vec![SENTINEL; nbytes as usize];
                let job = ReadJob { files: vec![FileJob { path: "t".into(), transfers: host_transfers(&reads, start, &mut dest) }] };
                let report = execute(&plan(1, case.split_bytes, 2, Staging::None), &store, None, None, job).unwrap_or_else(|e| panic!("{e}"));
                prop_assert_eq!(dest, expected);
                check_report(&report, wanted, &reads);
                prop_assert!(cuda.log().is_empty());
            }
            Some((count, bytes)) => {
                let ptr = cuda.alloc_device(nbytes + 3, 0);
                let job = ReadJob { files: vec![FileJob { path: "t".into(), transfers: device_transfers(&reads, start, ptr, 3) }] };
                let staging = Staging::Pinned { count, bytes };
                let report = execute(&plan(1, case.split_bytes, 2, staging), &store, Some(&cuda), None, job).unwrap_or_else(|e| panic!("{e}"));
                let landed = cuda.device_bytes(ptr).unwrap_or_default();
                prop_assert_eq!(&landed[3..], &expected[..]);
                prop_assert_eq!(&landed[..3], &[0u8; 3]);
                check_report(&report, wanted, &reads);
                let calls = cuda.log();
                prop_assert_eq!(calls.last(), Some(&Call::Synchronize));
                assert_ring_bound(&calls, count);
                // a piece never exceeds the staging buffer
                let pieces = reads_by_path(&store.log()).remove(Path::new("t")).unwrap_or_default();
                prop_assert!(pieces.iter().all(|&(_, l)| l <= bytes), "{pieces:?}");
                // one 2-D copy per regular placed piece, one per placement
                // for the irregular ones, each landing its rows contiguously
                let shapes = two_d_calls(&calls);
                prop_assert!(shapes.len() >= report.placed_pieces);
                prop_assert!(shapes.len() <= report.placed_pieces + report.scatter_copies);
                prop_assert!(shapes.iter().all(|c| c.dst_pitch == c.width && c.src_pitch >= c.width));
                // a box is regular within a piece unless it is partial on two
                // or more dimensions
                let partial = s.ranges().iter().zip(s.shape()).filter(|(r, e)| r.start != 0 || r.end != **e).count();
                if partial <= 1 {
                    prop_assert_eq!(report.scatter_copies, 0, "{:?}", shapes);
                }
            }
        }
        // no piece starts or ends inside a gap
        let pieces = reads_by_path(&store.log()).remove(Path::new("t")).unwrap_or_default();
        let runs: Vec<(u64, u64)> = s.runs(case.dtype).unwrap_or_else(|e| panic!("{e}")).iter().map(|r| (start + r.offset, r.len)).collect();
        assert_no_gap_boundaries(&pieces, &runs);
        if case.split_bytes > 0 && case.device.is_none() {
            prop_assert!(pieces.iter().all(|&(_, l)| l <= case.split_bytes), "{pieces:?}");
        }
    }
}
