//! Every destination byte lands exactly once, host and device, and the
//! operation log follows the plan.

use super::*;

// ---- host destinations -------------------------------------------------

#[test]
fn host_bytes_land_once_in_job_order_when_single_threaded() {
    let store = SimStorage::new();
    let a = seeded(32, 1);
    let b = seeded(20, 2);
    store.put("a", a.clone());
    store.put("b", b.clone());
    let mut a0 = vec![SENTINEL; 10];
    let mut a1 = vec![SENTINEL; 7];
    let mut b0 = vec![SENTINEL; 20];
    let job = ReadJob {
        files: vec![
            FileJob {
                path: "a".into(),
                transfers: vec![
                    Transfer {
                        range: range(2, 10),
                        dest: Dest::Host(&mut a0),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(20, 7),
                        dest: Dest::Host(&mut a1),
                        placements: Vec::new(),
                    },
                ],
            },
            FileJob {
                path: "b".into(),
                transfers: vec![Transfer {
                    range: range(0, 20),
                    dest: Dest::Host(&mut b0),
                    placements: Vec::new(),
                }],
            },
        ],
    };
    let report = ok(execute(
        &plan(1, 4, 1, Staging::None),
        &store,
        None,
        None,
        job,
    ));
    assert_eq!(a0, a[2..12]);
    assert_eq!(a1, a[20..27]);
    assert_eq!(b0, b);
    assert_eq!(report.bytes_read, 37);
    assert_eq!(report.pieces, 3 + 2 + 5);
    assert_eq!(report.files_opened, 2);
    // concurrency 1: exact order, files then transfers then pieces
    assert_eq!(
        store.log(),
        vec![
            Op::Open("a".into()),
            Op::Read("a".into(), 2, 4),
            Op::Read("a".into(), 6, 4),
            Op::Read("a".into(), 10, 2),
            Op::Read("a".into(), 20, 4),
            Op::Read("a".into(), 24, 3),
            Op::Open("b".into()),
            Op::Read("b".into(), 0, 4),
            Op::Read("b".into(), 4, 4),
            Op::Read("b".into(), 8, 4),
            Op::Read("b".into(), 12, 4),
            Op::Read("b".into(), 16, 4),
        ]
    );
}

#[test]
fn host_bytes_land_once_concurrently() {
    let store = SimStorage::new();
    let objects: Vec<Vec<u8>> = (0..9).map(|i| seeded(100 + i * 7, i as u64)).collect();
    for (i, bytes) in objects.iter().enumerate() {
        store.put(format!("f{i}"), bytes.clone());
    }
    let mut dests: Vec<Vec<u8>> = objects.iter().map(|o| vec![SENTINEL; o.len()]).collect();
    let files = dests
        .iter_mut()
        .enumerate()
        .map(|(i, dest)| {
            let len = dest.len() as u64;
            // two transfers per file, issued out of file order on purpose
            let (tail, head) = dest.split_at_mut(30);
            FileJob {
                path: format!("f{i}").into(),
                transfers: vec![
                    Transfer {
                        range: range(30, len - 30),
                        dest: Dest::Host(head),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(0, 30),
                        dest: Dest::Host(tail),
                        placements: Vec::new(),
                    },
                ],
            }
        })
        .collect();
    let p = plan(3, 16, 4, Staging::None);
    let report = ok(execute(&p, &store, None, None, ReadJob { files }));
    assert_eq!(dests, objects);
    assert_eq!(report.files_opened, 9);
    assert_eq!(
        report.bytes_read,
        objects.iter().map(|o| o.len() as u64).sum::<u64>()
    );
    let log = store.log();
    let mut opened = opens(&log);
    opened.sort();
    assert_eq!(
        opened,
        (0..9)
            .map(|i| PathBuf::from(format!("f{i}")))
            .collect::<Vec<_>>()
    );
    for (path, reads) in reads_by_path(&log) {
        let len = objects[path.to_string_lossy()[1..].parse::<usize>().unwrap()].len() as u64;
        assert!(reads.iter().all(|&(_, l)| l <= 16));
        let mut expected = expected_pieces(&range(30, len - 30), 16, None);
        expected.extend(expected_pieces(&range(0, 30), 16, None));
        expected.sort_unstable();
        assert_eq!(reads, expected, "{path:?}");
    }
}

#[test]
fn zero_length_transfers_issue_no_io() {
    let store = SimStorage::new();
    store.put("a", seeded(8, 3));
    let mut empty: Vec<u8> = vec![];
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(4, 0),
                dest: Dest::Host(&mut empty),
                placements: Vec::new(),
            }],
        }],
    };
    let report = ok(execute(
        &plan(1, 0, 1, Staging::None),
        &store,
        None,
        None,
        job,
    ));
    assert_eq!(report.pieces, 0);
    assert_eq!(report.bytes_read, 0);
    assert_eq!(report.files_opened, 1);
    assert_eq!(store.log(), vec![Op::Open("a".into())]);
}

// ---- device destinations -----------------------------------------------

#[test]
fn device_bytes_land_through_staging_and_are_synchronized() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let a = seeded(50, 4);
    let b = seeded(23, 5);
    store.put("a", a.clone());
    store.put("b", b.clone());
    // one device buffer per file; transfers packed back to back
    let da = cuda.alloc_device(40, 0);
    let db = cuda.alloc_device(23, 0);
    let job = ReadJob {
        files: vec![
            FileJob {
                path: "a".into(),
                transfers: vec![
                    Transfer {
                        range: range(10, 30),
                        dest: Dest::Device {
                            ptr: da,
                            offset: 10,
                        },
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(0, 10),
                        dest: Dest::Device { ptr: da, offset: 0 },
                        placements: Vec::new(),
                    },
                ],
            },
            FileJob {
                path: "b".into(),
                transfers: vec![Transfer {
                    range: range(0, 23),
                    dest: Dest::Device { ptr: db, offset: 0 },
                    placements: Vec::new(),
                }],
            },
        ],
    };
    let staging = Staging::Pinned { count: 2, bytes: 8 };
    let report = ok(execute(
        &plan(2, 0, 2, staging),
        &store,
        Some(&cuda),
        None,
        job,
    ));
    assert_eq!(cuda.device_bytes(da), Some(a[..40].to_vec()));
    assert_eq!(cuda.device_bytes(db), Some(b));
    assert_eq!(report.bytes_read, 63);
    assert_eq!(report.files_opened, 2);
    let calls = cuda.log();
    assert_eq!(
        calls
            .iter()
            .filter(|c| matches!(c, Call::AllocPinned(8)))
            .count(),
        2
    );
    assert_eq!(calls.last(), Some(&Call::Synchronize));
    assert_eq!(
        calls
            .iter()
            .filter_map(|c| match c {
                Call::ToDevice(n, ..) => Some(*n),
                _ => None,
            })
            .sum::<u64>(),
        63
    );
    assert_ring_bound(&calls, 2);
    // unsplit plan, but a piece never exceeds a staging buffer
    let log = store.log();
    for reads in reads_by_path(&log).values() {
        assert!(reads.iter().all(|&(_, l)| l <= 8), "{reads:?}");
    }
    let mut expected = expected_pieces(&range(10, 30), 0, Some(8));
    expected.extend(expected_pieces(&range(0, 10), 0, Some(8)));
    expected.sort_unstable();
    assert_eq!(reads_by_path(&log)[Path::new("a")], expected);
}

#[test]
fn staging_ring_smaller_than_concurrency_still_completes() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let objects: Vec<Vec<u8>> = (0..6).map(|i| seeded(64, 10 + i)).collect();
    let ptrs: Vec<_> = objects
        .iter()
        .map(|o| cuda.alloc_device(o.len() as u64, 0))
        .collect();
    for (i, o) in objects.iter().enumerate() {
        store.put(format!("f{i}"), o.clone());
    }
    let files = ptrs
        .iter()
        .enumerate()
        .map(|(i, ptr)| FileJob {
            path: format!("f{i}").into(),
            transfers: vec![Transfer {
                range: range(0, 64),
                dest: Dest::Device {
                    ptr: *ptr,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        })
        .collect();
    let staging = Staging::Pinned { count: 1, bytes: 4 };
    ok(execute(
        &plan(3, 8, 2, staging),
        &store,
        Some(&cuda),
        None,
        ReadJob { files },
    ));
    for (o, ptr) in objects.iter().zip(&ptrs) {
        assert_eq!(cuda.device_bytes(*ptr).as_deref(), Some(o.as_slice()));
    }
    let calls = cuda.log();
    assert_ring_bound(&calls, 1);
    assert_eq!(calls.last(), Some(&Call::Synchronize));
    // split 8 but staging 4: pieces are 4
    assert!(
        reads_by_path(&store.log())
            .values()
            .flatten()
            .all(|&(_, l)| l == 4)
    );
}

#[test]
fn slots_recycle_on_their_own_fences_without_draining_the_stream() {
    // 96 pieces through a ring of one slot: every reuse waits on the fence
    // recorded after that slot's copy, and the only synchronize is the final
    // one in `execute`. A stream drain per reuse is what made small pieces
    // slow.
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let objects: Vec<Vec<u8>> = (0..6).map(|i| seeded(64, 20 + i)).collect();
    let ptrs: Vec<_> = objects
        .iter()
        .map(|o| cuda.alloc_device(o.len() as u64, 0))
        .collect();
    for (i, o) in objects.iter().enumerate() {
        store.put(format!("f{i}"), o.clone());
    }
    let files = ptrs
        .iter()
        .enumerate()
        .map(|(i, ptr)| FileJob {
            path: format!("f{i}").into(),
            transfers: vec![Transfer {
                range: range(0, 64),
                dest: Dest::Device {
                    ptr: *ptr,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        })
        .collect();
    let staging = Staging::Pinned { count: 1, bytes: 4 };
    let report = ok(execute(
        &plan(3, 0, 4, staging),
        &store,
        Some(&cuda),
        None,
        ReadJob { files },
    ));
    for (o, ptr) in objects.iter().zip(&ptrs) {
        assert_eq!(cuda.device_bytes(*ptr).as_deref(), Some(o.as_slice()));
    }
    assert_eq!(report.pieces, 96);
    let calls = cuda.log();
    let count = |wanted: fn(&Call) -> bool| calls.iter().filter(|c| wanted(c)).count();
    assert_eq!(count(|c| matches!(c, Call::Synchronize)), 1);
    assert_eq!(calls.last(), Some(&Call::Synchronize));
    assert_eq!(count(|c| matches!(c, Call::Record)), 96);
    // every piece but the first found the slot dirty
    assert_eq!(count(|c| matches!(c, Call::Wait)), 95);
    assert_ring_bound(&calls, 1);
}

#[test]
fn device_copy_failure_is_typed_and_synchronized() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    store.put("a", seeded(16, 6));
    // an address the sim never handed out: the copy fails at the runtime
    let bogus = fst_cuda::DevicePtr {
        address: 7,
        nbytes: 16,
        device: 0,
    };
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 16),
                dest: Dest::Device {
                    ptr: bogus,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        }],
    };
    let staging = Staging::Pinned {
        count: 2,
        bytes: 16,
    };
    let err = execute(&plan(1, 0, 1, staging), &store, Some(&cuda), None, job).err();
    assert!(
        matches!(
            &err,
            Some(ReadError::Copy { path, range: r, source: fst_cuda::CudaError::Runtime { .. } })
                if path == Path::new("a") && *r == range(0, 16)
        ),
        "{err:?}"
    );
    assert_eq!(cuda.log().last(), Some(&Call::Synchronize));
}
