//! Faults: what is guaranteed after the first failure, and that the plan's
//! concurrency bounds hold.

use super::*;

// ---- faults ------------------------------------------------------------

#[test]
fn open_fault_names_the_path() {
    let store = SimStorage::new();
    store.put("a", seeded(8, 9));
    let mut dest = vec![SENTINEL; 8];
    let job = ReadJob {
        files: vec![FileJob {
            path: "missing".into(),
            transfers: vec![Transfer {
                range: range(0, 8),
                dest: Dest::Host(&mut dest),
                placements: Vec::new(),
            }],
        }],
    };
    let err = execute(&plan(1, 0, 1, Staging::None), &store, None, None, job).err();
    assert!(
        matches!(&err, Some(ReadError::Open { path, source: crate::StorageError::Io { op: "open", .. } })
            if path == Path::new("missing")),
        "{err:?}"
    );
    assert_eq!(dest, vec![SENTINEL; 8]);
}

#[test]
fn read_fault_single_threaded_stops_exactly_there() {
    let store = SimStorage::new();
    let a = seeded(12, 10);
    store.put("a", a.clone());
    store.put("b", seeded(4, 11));
    store.arm(Fault {
        kind: FaultKind::Read,
        nth: 1,
        message: "eio".into(),
    });
    let mut d0 = vec![SENTINEL; 4];
    let mut d1 = vec![SENTINEL; 4];
    let mut d2 = vec![SENTINEL; 4];
    let mut db = vec![SENTINEL; 4];
    let job = ReadJob {
        files: vec![
            FileJob {
                path: "a".into(),
                transfers: vec![
                    Transfer {
                        range: range(0, 4),
                        dest: Dest::Host(&mut d0),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(4, 4),
                        dest: Dest::Host(&mut d1),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(8, 4),
                        dest: Dest::Host(&mut d2),
                        placements: Vec::new(),
                    },
                ],
            },
            FileJob {
                path: "b".into(),
                transfers: vec![Transfer {
                    range: range(0, 4),
                    dest: Dest::Host(&mut db),
                    placements: Vec::new(),
                }],
            },
        ],
    };
    let err = execute(&plan(1, 0, 1, Staging::None), &store, None, None, job).err();
    assert!(
        matches!(&err, Some(ReadError::Read { path, range: r, source: crate::StorageError::Simulated(m) })
            if path == Path::new("a") && *r == range(4, 4) && m == "eio"),
        "{err:?}"
    );
    // the failing read is the last operation: nothing after it starts
    assert_eq!(
        store.log(),
        vec![
            Op::Open("a".into()),
            Op::Read("a".into(), 0, 4),
            Op::Read("a".into(), 4, 4)
        ]
    );
    assert_eq!(d0, a[..4]);
    assert_eq!(
        d1,
        vec![SENTINEL; 4],
        "the failing piece's destination is untouched by the sim"
    );
    assert_eq!(d2, vec![SENTINEL; 4]);
    assert_eq!(db, vec![SENTINEL; 4]);
}

/// Concurrently, what is guaranteed: the error names the faulting piece; no
/// piece starts after the failure is recorded, so with far more pieces than
/// workers some are never read; every piece that never started has an
/// untouched destination; every piece that completed has the right bytes.
#[test]
fn read_fault_concurrent_leaves_unstarted_destinations_untouched() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let n_files = 16usize;
    let per_file = 64u64;
    let objects: Vec<Vec<u8>> = (0..n_files)
        .map(|i| seeded(per_file as usize, 20 + i as u64))
        .collect();
    for (i, o) in objects.iter().enumerate() {
        store.put(format!("f{i}"), o.clone());
    }
    store.arm(Fault {
        kind: FaultKind::Read,
        nth: 5,
        message: "eio".into(),
    });
    let mut host: Vec<Vec<u8>> = (0..n_files / 2)
        .map(|_| vec![SENTINEL; per_file as usize])
        .collect();
    let ptrs: Vec<_> = (0..n_files / 2)
        .map(|_| cuda.alloc_device(per_file, 0))
        .collect();
    // device buffers start as zero in the sim; fill them with the sentinel
    for ptr in &ptrs {
        let mut pinned = cuda.alloc_pinned(per_file).unwrap();
        pinned.as_mut_slice().fill(SENTINEL);
        cuda.copy_to_device(pinned.as_ref(), per_file, *ptr, 0)
            .unwrap();
    }
    let mut files = Vec::new();
    for (i, dest) in host.iter_mut().enumerate() {
        files.push(FileJob {
            path: format!("f{i}").into(),
            transfers: vec![Transfer {
                range: range(0, per_file),
                dest: Dest::Host(dest),
                placements: Vec::new(),
            }],
        });
    }
    for (i, ptr) in ptrs.iter().enumerate() {
        files.push(FileJob {
            path: format!("f{}", i + n_files / 2).into(),
            transfers: vec![Transfer {
                range: range(0, per_file),
                dest: Dest::Device {
                    ptr: *ptr,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        });
    }
    let staging = Staging::Pinned { count: 4, bytes: 8 };
    let err = execute(
        &plan(4, 8, 2, staging),
        &store,
        Some(&cuda),
        None,
        ReadJob { files },
    )
    .err();
    let Some(ReadError::Read {
        path: failed_path,
        range: failed_range,
        ..
    }) = &err
    else {
        panic!("{err:?}");
    };
    let log = store.log();
    let started = reads_by_path(&log);
    assert!(started[failed_path].contains(&(failed_range.offset, failed_range.len)));
    let total_pieces = n_files as u64 * per_file / 8;
    let n_started: usize = started.values().map(Vec::len).sum();
    assert!(
        n_started < total_pieces as usize,
        "{n_started} of {total_pieces} pieces started"
    );
    let device_bytes: Vec<Vec<u8>> = ptrs
        .iter()
        .map(|p| cuda.device_bytes(*p).unwrap())
        .collect();
    for i in 0..n_files {
        let path = PathBuf::from(format!("f{i}"));
        let landed = if i < n_files / 2 {
            &host[i]
        } else {
            &device_bytes[i - n_files / 2]
        };
        let reads = started.get(&path).cloned().unwrap_or_default();
        for piece in 0..per_file / 8 {
            let (lo, hi) = ((piece * 8) as usize, (piece * 8 + 8) as usize);
            let this = (piece * 8, 8);
            if !reads.contains(&this) {
                assert_eq!(
                    &landed[lo..hi],
                    &[SENTINEL; 8],
                    "{path:?} piece {piece} never started"
                );
            } else if !(path == *failed_path && this == (failed_range.offset, failed_range.len)) {
                // started and not the faulting one: it either completed with
                // the right bytes or was cut off untouched by a cancel that
                // beat its copy; never partially or wrongly written
                let got = &landed[lo..hi];
                assert!(
                    got == &objects[i][lo..hi] || got == [SENTINEL; 8],
                    "{path:?} piece {piece}: {got:?}"
                );
            }
        }
    }
    assert_eq!(cuda.log().last(), Some(&Call::Synchronize));
    assert_ring_bound(&cuda.log()[ptrs.len()..], 4);
}

/// With one staging slot every device read holds the slot, and a failure is
/// recorded before the slot is released, so the read that fails is the last
/// read there is — deterministically, whatever the thread schedule.
#[test]
fn read_fault_with_one_staging_slot_stops_exactly_there() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let a = seeded(24, 12);
    store.put("a", a.clone());
    store.arm(Fault {
        kind: FaultKind::Read,
        nth: 2,
        message: "eio".into(),
    });
    let ptr = cuda.alloc_device(24, 0);
    let mut pinned = cuda.alloc_pinned(24).unwrap();
    pinned.as_mut_slice().fill(SENTINEL);
    cuda.copy_to_device(pinned.as_ref(), 24, ptr, 0).unwrap();
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 24),
                dest: Dest::Device { ptr, offset: 0 },
                placements: Vec::new(),
            }],
        }],
    };
    let staging = Staging::Pinned { count: 1, bytes: 4 };
    let err = execute(&plan(1, 4, 3, staging), &store, Some(&cuda), None, job).err();
    let Some(ReadError::Read {
        path,
        range: failed,
        source: crate::StorageError::Simulated(m),
    }) = &err
    else {
        panic!("{err:?}");
    };
    assert_eq!(path, Path::new("a"));
    assert_eq!(m, "eio");
    let reads = &reads_by_path(&store.log())[Path::new("a")];
    assert_eq!(
        reads.len(),
        3,
        "two reads succeed, the third fails, none after: {reads:?}"
    );
    assert!(reads.contains(&(failed.offset, failed.len)));
    let landed = cuda.device_bytes(ptr).unwrap();
    for piece in 0..6u64 {
        let (lo, hi) = ((piece * 4) as usize, (piece * 4 + 4) as usize);
        let expected: &[u8] = if reads.contains(&(piece * 4, 4)) && piece * 4 != failed.offset {
            &a[lo..hi]
        } else {
            &[SENTINEL; 4]
        };
        assert_eq!(&landed[lo..hi], expected, "piece {piece}");
    }
    assert_eq!(cuda.log().last(), Some(&Call::Synchronize));
}

/// A storage wrapper that watches concurrency: readers alive and reads in
/// flight per path, with a small delay per read so overlap is observable.
mod throttle {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::StorageError;
    use crate::storage::sim::SimStorage;
    use crate::storage::{PartWriter, RangeReader, Storage};

    #[derive(Default)]
    pub struct Watch {
        pub readers_alive: AtomicUsize,
        pub max_readers_alive: AtomicUsize,
        pub in_flight: Mutex<BTreeMap<PathBuf, usize>>,
        pub max_in_flight: Mutex<BTreeMap<PathBuf, usize>>,
    }

    fn bump_max(max: &AtomicUsize, now: usize) {
        max.fetch_max(now, Ordering::SeqCst);
    }

    pub struct Throttled {
        pub inner: SimStorage,
        pub watch: Arc<Watch>,
    }

    impl Storage for Throttled {
        fn name(&self) -> &'static str {
            "throttled"
        }
        fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError> {
            let inner = self.inner.open_reader(path)?;
            let now = self.watch.readers_alive.fetch_add(1, Ordering::SeqCst) + 1;
            bump_max(&self.watch.max_readers_alive, now);
            Ok(Box::new(Reader {
                inner,
                path: path.to_owned(),
                watch: self.watch.clone(),
            }))
        }
        fn create_writer(&self, path: &Path) -> Result<Box<dyn PartWriter>, StorageError> {
            self.inner.create_writer(path)
        }
        fn rename(&self, from: &Path, to: &Path, durable: bool) -> Result<(), StorageError> {
            self.inner.rename(from, to, durable)
        }
        fn remove(&self, path: &Path) -> Result<(), StorageError> {
            self.inner.remove(path)
        }
    }

    struct Reader {
        inner: Box<dyn RangeReader>,
        path: PathBuf,
        watch: Arc<Watch>,
    }

    impl RangeReader for Reader {
        fn len(&self) -> u64 {
            self.inner.len()
        }
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), StorageError> {
            {
                let mut in_flight = self.watch.in_flight.lock().unwrap();
                let now = in_flight.entry(self.path.clone()).or_default();
                *now += 1;
                let now = *now;
                let mut max = self.watch.max_in_flight.lock().unwrap();
                let m = max.entry(self.path.clone()).or_default();
                *m = (*m).max(now);
            }
            std::thread::sleep(Duration::from_millis(1));
            let result = self.inner.read_at(offset, dst);
            *self
                .watch
                .in_flight
                .lock()
                .unwrap()
                .get_mut(&self.path)
                .unwrap() -= 1;
            result
        }
    }

    impl Drop for Reader {
        fn drop(&mut self) {
            self.watch.readers_alive.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// At most `files_in_flight` readers alive and `readers_per_file` reads in
/// flight per file. The bounds are exact guarantees; that they are reached
/// (so a scheduler that ignored them would be caught) relies on the 1 ms
/// delay per read, which makes overlap practically certain.
#[test]
fn concurrency_never_exceeds_the_plan() {
    let store = SimStorage::new();
    let n_files = 12usize;
    let objects: Vec<Vec<u8>> = (0..n_files).map(|i| seeded(64, 200 + i as u64)).collect();
    for (i, o) in objects.iter().enumerate() {
        store.put(format!("f{i}"), o.clone());
    }
    let watch = std::sync::Arc::new(throttle::Watch::default());
    let throttled = throttle::Throttled {
        inner: store.clone(),
        watch: watch.clone(),
    };
    let mut dests: Vec<Vec<u8>> = objects.iter().map(|o| vec![SENTINEL; o.len()]).collect();
    let files = dests
        .iter_mut()
        .enumerate()
        .map(|(i, dest)| FileJob {
            path: format!("f{i}").into(),
            transfers: vec![Transfer {
                range: range(0, 64),
                dest: Dest::Host(dest),
                placements: Vec::new(),
            }],
        })
        .collect();
    let p = plan(3, 4, 2, Staging::None);
    ok(execute(&p, &throttled, None, None, ReadJob { files }));
    assert_eq!(dests, objects);
    let max_alive = watch
        .max_readers_alive
        .load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        max_alive <= 3,
        "{max_alive} readers alive with files_in_flight 3"
    );
    assert_eq!(max_alive, 3, "the plan's files in flight were not used");
    let max_in_flight = watch.max_in_flight.lock().unwrap().clone();
    for (path, m) in &max_in_flight {
        assert!(
            *m <= 2,
            "{path:?}: {m} reads in flight with readers_per_file 2"
        );
    }
    assert!(
        max_in_flight.values().any(|&m| m == 2),
        "readers per file were not used: {max_in_flight:?}"
    );
    assert_eq!(
        watch
            .readers_alive
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "readers closed on return"
    );
}
