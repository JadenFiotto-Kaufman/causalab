//! The cuFile transport: device pieces through `DirectStorage` with no
//! staging, host pieces through the `Storage`, and the run-time fallback to
//! pread when registration fails per file, when `libcufile` is gone, or when
//! no `DirectStorage` was passed at all.

use fst_cuda::CudaError;
use fst_cuda::sim_direct::{DirectFault, SimDirect};

use super::*;
use crate::read::Fallback;

fn cufile_plan(
    files_in_flight: usize,
    split_bytes: u64,
    readers_per_file: usize,
    staging: Staging,
) -> ReadPlan {
    let mut p = plan(files_in_flight, split_bytes, readers_per_file, staging);
    p.transport = Transport::CuFile;
    p
}

/// The same object in both simulators: cuFile and the pread fallback read
/// the same file.
fn seed(store: &SimStorage, direct: &SimDirect, name: &str, bytes: &[u8]) {
    store.put(name, bytes.to_vec());
    direct.put(name, bytes.to_vec());
}

/// A device buffer full of the sentinel, so untouched bytes are visible.
fn sentinel_device(cuda: &SimCuda, nbytes: u64) -> fst_cuda::DevicePtr {
    let ptr = cuda.alloc_device(nbytes, 0);
    cuda.write_device(ptr, 0, &vec![SENTINEL; nbytes as usize])
        .unwrap();
    ptr
}

fn direct_paths(direct: &SimDirect) -> Vec<PathBuf> {
    direct.log().into_iter().map(|c| c.path).collect()
}

fn pinned_allocs(calls: &[Call]) -> Vec<u64> {
    calls
        .iter()
        .filter_map(|c| match c {
            Call::AllocPinned(n) => Some(*n),
            _ => None,
        })
        .collect()
}

// ---- (a) the transport itself -------------------------------------------

#[test]
fn cufile_plan_lands_bytes_through_direct_storage_without_staging() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let a = seeded(50, 30);
    let b = seeded(23, 31);
    seed(&store, &direct, "a", &a);
    seed(&store, &direct, "b", &b);
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
    let p = cufile_plan(2, 0, 2, Staging::None);
    let report = ok(execute(&p, &store, Some(&cuda), Some(&direct), job));
    assert_eq!(cuda.device_bytes(da), Some(a[..40].to_vec()));
    assert_eq!(cuda.device_bytes(db), Some(b));
    assert_eq!(report.bytes_read, 63);
    assert_eq!(report.cufile_bytes, 63);
    assert_eq!(report.staged_bytes, 0);
    assert_eq!(report.host_bytes, 0);
    assert_eq!(report.pieces, 3);
    assert_eq!(report.files_opened, 0, "cuFile opens its own handles");
    assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
    // no pinned memory, no copies, no synchronize, no pread
    assert!(cuda.log().is_empty(), "{:?}", cuda.log());
    assert!(store.log().is_empty(), "{:?}", store.log());
    let mut paths = direct_paths(&direct);
    paths.sort();
    assert_eq!(paths, vec![PathBuf::from("a"), "a".into(), "b".into()]);
}

// ---- (b) registration fails for one file -------------------------------

#[test]
fn register_failure_falls_back_that_file_only() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let a = seeded(16, 40);
    let b = seeded(12, 41);
    let c = seeded(8, 42);
    seed(&store, &direct, "a", &a);
    seed(&store, &direct, "b", &b);
    seed(&store, &direct, "c", &c);
    direct.arm(DirectFault::Register {
        path: "b".into(),
        reason: "cuFileHandleRegister failed with cuFile error 5027".into(),
    });
    let ptrs: Vec<_> = [16, 12, 8]
        .iter()
        .map(|&n| cuda.alloc_device(n, 0))
        .collect();
    let files = ["a", "b", "c"]
        .iter()
        .zip(&ptrs)
        .map(|(name, ptr)| FileJob {
            path: (*name).into(),
            transfers: vec![Transfer {
                range: range(0, ptr.nbytes),
                dest: Dest::Device {
                    ptr: *ptr,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        })
        .collect();
    // concurrency 1 for exact logs; Staging::None so the fallback ring is
    // the planner's pread geometry
    let p = cufile_plan(1, 8, 1, Staging::None);
    let report = ok(execute(
        &p,
        &store,
        Some(&cuda),
        Some(&direct),
        ReadJob { files },
    ));
    assert_eq!(cuda.device_bytes(ptrs[0]), Some(a.clone()));
    assert_eq!(cuda.device_bytes(ptrs[1]), Some(b.clone()));
    assert_eq!(cuda.device_bytes(ptrs[2]), Some(c.clone()));
    assert_eq!(report.bytes_read, 36);
    assert_eq!(report.cufile_bytes, 24, "a and c went through cuFile");
    assert_eq!(report.staged_bytes, 12, "b was staged");
    assert_eq!(report.files_opened, 1, "only b needed a pread reader");
    assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
    let fallback = &report.fallbacks[0];
    assert_eq!(fallback.path.as_deref(), Some(Path::new("b")));
    assert!(fallback.reason.contains("5027"), "{}", fallback.reason);
    assert!(fallback.reason.contains('b'), "{}", fallback.reason);
    // cuFile was asked once for b, then never again for that file
    assert_eq!(
        direct_paths(&direct),
        vec![PathBuf::from("a"), "a".into(), "b".into(), "c".into()]
    );
    // pread saw only b, in the pieces the plan split it into
    assert_eq!(
        store.log(),
        vec![
            Op::Open("b".into()),
            Op::Read("b".into(), 0, 8),
            Op::Read("b".into(), 8, 4),
        ]
    );
    // the fallback ring: the planner's rule, (1 * 1 * 2).clamp(2, 64) 16 MiB
    // buffers, allocated only once b fell back, then synchronized
    let calls = cuda.log();
    assert_eq!(
        pinned_allocs(&calls),
        vec![crate::plan::STAGING_BUFFER_BYTES; 2]
    );
    assert_eq!(calls.last(), Some(&Call::Synchronize));
    assert_ring_bound(&calls, 2);
}

/// Registration can also fail after a file's earlier pieces went through
/// cuFile (the real driver registers on every read). The file switches from
/// that piece on; every byte still lands once.
#[test]
fn register_failure_mid_file_switches_the_rest_of_the_file() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let a = seeded(24, 43);
    seed(&store, &direct, "a", &a);
    // the sim's `Register` fault is per path and persistent (a mount that
    // never registers); this wrapper arms it after the first call, so the
    // first piece goes through and the second is refused
    struct Second(SimDirect);
    impl fst_cuda::DirectStorage for Second {
        fn read_into_device(
            &self,
            path: &Path,
            file_offset: u64,
            nbytes: u64,
            dst: fst_cuda::DevicePtr,
            dst_offset: u64,
        ) -> Result<(), CudaError> {
            if self.0.log().len() == 1 {
                self.0.arm(DirectFault::Register {
                    path: path.to_owned(),
                    reason: "second registration refused".into(),
                });
            }
            self.0
                .read_into_device(path, file_offset, nbytes, dst, dst_offset)
        }
    }
    let second = Second(direct);
    let ptr = cuda.alloc_device(24, 0);
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
    let p = cufile_plan(1, 8, 1, Staging::Pinned { count: 1, bytes: 8 });
    let report = ok(execute(&p, &store, Some(&cuda), Some(&second), job));
    assert_eq!(cuda.device_bytes(ptr), Some(a));
    assert_eq!(report.cufile_bytes, 8);
    assert_eq!(report.staged_bytes, 16);
    assert_eq!(report.fallbacks.len(), 1);
    assert_eq!(report.fallbacks[0].path.as_deref(), Some(Path::new("a")));
    assert_eq!(
        second.0.log().len(),
        2,
        "first piece landed, second refused, third never asked"
    );
    assert_eq!(
        store.log(),
        vec![
            Op::Open("a".into()),
            Op::Read("a".into(), 8, 8),
            Op::Read("a".into(), 16, 8),
        ]
    );
}

// ---- (c) libcufile unavailable -----------------------------------------

/// `Unavailable` is about the library, not a file, so it is one fallback for
/// the job: cuFile is not asked again for any file. The sim's fault is
/// one-shot, so a retry would have succeeded — `cufile_bytes == 0` proves
/// there was none.
#[test]
fn unavailable_switches_the_whole_job_to_pread_once() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let objects: Vec<Vec<u8>> = (0..3).map(|i| seeded(16, 50 + i)).collect();
    for (i, o) in objects.iter().enumerate() {
        seed(&store, &direct, &format!("f{i}"), o);
    }
    direct.arm(DirectFault::Unavailable {
        nth: 0,
        reason: "libcufile.so.0: cannot open shared object file".into(),
    });
    let ptrs: Vec<_> = objects.iter().map(|_| cuda.alloc_device(16, 0)).collect();
    let files = ptrs
        .iter()
        .enumerate()
        .map(|(i, ptr)| FileJob {
            path: format!("f{i}").into(),
            transfers: vec![Transfer {
                range: range(0, 16),
                dest: Dest::Device {
                    ptr: *ptr,
                    offset: 0,
                },
                placements: Vec::new(),
            }],
        })
        .collect();
    let p = cufile_plan(
        1,
        0,
        1,
        Staging::Pinned {
            count: 1,
            bytes: 64,
        },
    );
    let report = ok(execute(
        &p,
        &store,
        Some(&cuda),
        Some(&direct),
        ReadJob { files },
    ));
    for (o, ptr) in objects.iter().zip(&ptrs) {
        assert_eq!(cuda.device_bytes(*ptr).as_deref(), Some(o.as_slice()));
    }
    assert_eq!(report.cufile_bytes, 0);
    assert_eq!(report.staged_bytes, 48);
    assert_eq!(report.files_opened, 3);
    assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
    assert_eq!(report.fallbacks[0].path, None);
    assert!(
        report.fallbacks[0].reason.contains("libcufile"),
        "{}",
        report.fallbacks[0].reason
    );
    assert_eq!(direct.log().len(), 1, "one call, then never again");
    let mut opened = opens(&store.log());
    opened.sort();
    assert_eq!(opened, vec![PathBuf::from("f0"), "f1".into(), "f2".into()]);
    assert_eq!(pinned_allocs(&cuda.log()), vec![64]);
}

// ---- (d) any other cuFile error fails the job ---------------------------

#[test]
fn other_cufile_error_fails_the_job_naming_path_and_range() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let a = seeded(16, 60);
    let b = seeded(16, 61);
    seed(&store, &direct, "a", &a);
    seed(&store, &direct, "b", &b);
    // a's two pieces land; b's first piece is the third call
    direct.arm(DirectFault::CuFile {
        nth: 2,
        call: "cuFileRead",
        code: -5,
    });
    let da = sentinel_device(&cuda, 16);
    let db = sentinel_device(&cuda, 16);
    let files = [("a", da), ("b", db)]
        .into_iter()
        .map(|(name, ptr)| FileJob {
            path: name.into(),
            transfers: vec![Transfer {
                range: range(0, 16),
                dest: Dest::Device { ptr, offset: 0 },
                placements: Vec::new(),
            }],
        })
        .collect();
    let p = cufile_plan(1, 8, 1, Staging::None);
    let err = execute(&p, &store, Some(&cuda), Some(&direct), ReadJob { files }).err();
    assert!(
        matches!(
            &err,
            Some(ReadError::DirectRead { path, range: r, source: CudaError::CuFile { call: "cuFileRead", code: -5 } })
                if path == Path::new("b") && *r == range(0, 8)
        ),
        "{err:?}"
    );
    assert_eq!(
        cuda.device_bytes(da),
        Some(a),
        "completed pieces are intact"
    );
    assert_eq!(
        cuda.device_bytes(db),
        Some(vec![SENTINEL; 16]),
        "the failing piece and everything after it never landed"
    );
    assert_eq!(direct.log().len(), 3, "nothing starts after the failure");
    assert!(store.log().is_empty(), "no fallback for a cuFile I/O error");
    assert!(cuda.log().is_empty(), "no ring was ever needed");
}

// ---- (e) host destinations under a cuFile plan --------------------------

#[test]
fn cufile_plan_reads_host_destinations_through_storage() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    let a = seeded(20, 70);
    let b = seeded(16, 71);
    let c = seeded(8, 72);
    seed(&store, &direct, "a", &a);
    seed(&store, &direct, "b", &b);
    seed(&store, &direct, "c", &c);
    let mut ha = vec![SENTINEL; 20];
    let mut hc = vec![SENTINEL; 4];
    let db = cuda.alloc_device(16, 0);
    let dc = cuda.alloc_device(4, 0);
    let job = ReadJob {
        files: vec![
            FileJob {
                path: "a".into(),
                transfers: vec![Transfer {
                    range: range(0, 20),
                    dest: Dest::Host(&mut ha),
                    placements: Vec::new(),
                }],
            },
            FileJob {
                path: "b".into(),
                transfers: vec![Transfer {
                    range: range(0, 16),
                    dest: Dest::Device { ptr: db, offset: 0 },
                    placements: Vec::new(),
                }],
            },
            // one file, both kinds of destination
            FileJob {
                path: "c".into(),
                transfers: vec![
                    Transfer {
                        range: range(0, 4),
                        dest: Dest::Host(&mut hc),
                        placements: Vec::new(),
                    },
                    Transfer {
                        range: range(4, 4),
                        dest: Dest::Device { ptr: dc, offset: 0 },
                        placements: Vec::new(),
                    },
                ],
            },
        ],
    };
    let p = cufile_plan(1, 0, 1, Staging::None);
    let report = ok(execute(&p, &store, Some(&cuda), Some(&direct), job));
    assert_eq!(ha, a);
    assert_eq!(hc, c[..4]);
    assert_eq!(cuda.device_bytes(db), Some(b));
    assert_eq!(cuda.device_bytes(dc), Some(c[4..].to_vec()));
    assert_eq!(report.host_bytes, 24, "host pieces are pread, and say so");
    assert_eq!(report.cufile_bytes, 20);
    assert_eq!(report.staged_bytes, 0);
    assert_eq!(report.files_opened, 2, "a and c needed readers; b did not");
    assert!(report.fallbacks.is_empty(), "host reads are not a fallback");
    assert_eq!(
        store.log(),
        vec![
            Op::Open("a".into()),
            Op::Read("a".into(), 0, 20),
            Op::Open("c".into()),
            Op::Read("c".into(), 0, 4),
        ]
    );
    assert_eq!(direct_paths(&direct), vec![PathBuf::from("b"), "c".into()]);
    assert!(cuda.log().is_empty());
}

// ---- (f) no DirectStorage at all ---------------------------------------

#[test]
fn cufile_plan_without_direct_storage_runs_on_pread() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let a = seeded(16, 80);
    let b = seeded(8, 81);
    store.put("a", a.clone());
    store.put("b", b.clone());
    let da = cuda.alloc_device(16, 0);
    let db = cuda.alloc_device(8, 0);
    let files = [("a", da), ("b", db)]
        .into_iter()
        .map(|(name, ptr)| FileJob {
            path: name.into(),
            transfers: vec![Transfer {
                range: range(0, ptr.nbytes),
                dest: Dest::Device { ptr, offset: 0 },
                placements: Vec::new(),
            }],
        })
        .collect();
    let p = cufile_plan(
        1,
        0,
        1,
        Staging::Pinned {
            count: 2,
            bytes: 64,
        },
    );
    let report = ok(execute(&p, &store, Some(&cuda), None, ReadJob { files }));
    assert_eq!(cuda.device_bytes(da), Some(a));
    assert_eq!(cuda.device_bytes(db), Some(b));
    assert_eq!(report.cufile_bytes, 0);
    assert_eq!(report.staged_bytes, 24);
    assert_eq!(
        report.fallbacks,
        vec![Fallback {
            path: None,
            reason: "no DirectStorage provided".into(),
        }]
    );
    // the whole job stages, so the ring is allocated up front as for pread
    let calls = cuda.log();
    assert_eq!(pinned_allocs(&calls), vec![64, 64]);
    assert_eq!(calls.last(), Some(&Call::Synchronize));

    // a host-only job under a cuFile plan has nothing to fall back from
    let mut host = vec![SENTINEL; 16];
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 16),
                dest: Dest::Host(&mut host),
                placements: Vec::new(),
            }],
        }],
    };
    let report = ok(execute(&p, &store, None, None, job));
    assert!(report.fallbacks.is_empty());
    assert_eq!(report.host_bytes, 16);
}

#[test]
fn cufile_plan_with_device_destinations_still_needs_a_copier() {
    let store = SimStorage::new();
    let cuda = SimCuda::new(1 << 30);
    let direct = SimDirect::new(cuda.clone());
    seed(&store, &direct, "a", &seeded(8, 90));
    let ptr = cuda.alloc_device(8, 0);
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 8),
                dest: Dest::Device { ptr, offset: 0 },
                placements: Vec::new(),
            }],
        }],
    };
    let p = cufile_plan(1, 0, 1, Staging::None);
    // the fallback must always be possible: no copier, no job
    assert!(matches!(
        execute(&p, &store, None, Some(&direct), job),
        Err(ReadError::CopierRequired)
    ));
    assert!(direct.log().is_empty());
    assert!(store.log().is_empty());
}

// ---- (g) property: random register faults ------------------------------

#[derive(Debug, Clone)]
struct DirectFileSpec {
    len: u64,
    ranges: Vec<ReadRange>,
    /// cuFile refuses to register this file.
    refused: bool,
}

fn direct_file_spec() -> impl Strategy<Value = DirectFileSpec> {
    (1u64..64, any::<bool>()).prop_flat_map(|(len, refused)| {
        prop::collection::vec((0..len, 1u64..=64), 1..4).prop_map(move |raw| DirectFileSpec {
            len,
            ranges: raw
                .into_iter()
                .map(|(offset, want)| ReadRange {
                    offset,
                    len: want.min(len - offset),
                })
                .collect(),
            refused,
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]
    #[test]
    fn random_register_faults_are_byte_exact_and_accounted(
        specs in prop::collection::vec(direct_file_spec(), 1..5),
        files_in_flight in 1usize..4,
        split_bytes in prop_oneof![Just(0u64), 1u64..16],
        readers_per_file in 1usize..4,
        (count, bytes) in (1usize..4, 1u64..8),
    ) {
        let store = SimStorage::new();
        let cuda = SimCuda::new(1 << 30);
        let direct = SimDirect::new(cuda.clone());
        let objects: Vec<Vec<u8>> = specs.iter().enumerate().map(|(i, s)| seeded(s.len as usize, 300 + i as u64)).collect();
        for (i, (spec, o)) in specs.iter().zip(&objects).enumerate() {
            seed(&store, &direct, &format!("f{i}"), o);
            if spec.refused {
                direct.arm(DirectFault::Register { path: format!("f{i}").into(), reason: "5027".into() });
            }
        }
        let ptrs: Vec<_> = specs.iter().map(|s| cuda.alloc_device(s.ranges.iter().map(|r| r.len).sum(), 0)).collect();
        let mut files = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let mut packed = 0u64;
            let transfers = spec.ranges.iter().map(|r| {
                let d = Dest::Device { ptr: ptrs[i], offset: packed };
                packed += r.len;
                Transfer { range: r.clone(), dest: d, placements: Vec::new() }
            }).collect();
            files.push(FileJob { path: format!("f{i}").into(), transfers });
        }
        let p = cufile_plan(files_in_flight, split_bytes, readers_per_file, Staging::Pinned { count, bytes });
        let report = execute(&p, &store, Some(&cuda), Some(&direct), ReadJob { files }).unwrap_or_else(|e| panic!("{e}"));

        // byte-exact, whichever route each file took
        for (i, spec) in specs.iter().enumerate() {
            let expected: Vec<u8> = spec.ranges.iter().flat_map(|r| objects[i][r.offset as usize..(r.offset + r.len) as usize].to_vec()).collect();
            prop_assert_eq!(cuda.device_bytes(ptrs[i]), Some(expected), "file {}", i);
        }
        let per_file: Vec<u64> = specs.iter().map(|s| s.ranges.iter().map(|r| r.len).sum()).collect();
        let total: u64 = per_file.iter().sum();
        let refused: u64 = specs.iter().zip(&per_file).filter(|(s, _)| s.refused).map(|(_, n)| n).sum();
        prop_assert_eq!(report.bytes_read, total);
        prop_assert_eq!(report.cufile_bytes + report.staged_bytes, total);
        prop_assert_eq!(report.host_bytes, 0);
        prop_assert_eq!(report.staged_bytes, refused, "exactly the refused files staged");

        // one fallback per refused file, naming it; pread opened exactly those
        let mut refused_paths: Vec<PathBuf> = specs.iter().enumerate().filter(|(_, s)| s.refused).map(|(i, _)| format!("f{i}").into()).collect();
        refused_paths.sort();
        let mut fell_back: Vec<PathBuf> = report.fallbacks.iter().filter_map(|f| f.path.clone()).collect();
        fell_back.sort();
        prop_assert_eq!(&fell_back, &refused_paths);
        prop_assert_eq!(report.fallbacks.len(), refused_paths.len(), "{:?}", report.fallbacks);
        prop_assert_eq!(report.files_opened, refused_paths.len());
        let mut opened = opens(&store.log());
        opened.sort();
        prop_assert_eq!(&opened, &refused_paths);
        // cuFile was asked about every file, and landed exactly the unrefused ones
        let asked: std::collections::BTreeSet<PathBuf> = direct_paths(&direct).into_iter().collect();
        prop_assert_eq!(asked.len(), specs.len());

        // the ring exists only if something staged, and then it is the plan's
        let calls = cuda.log();
        if refused_paths.is_empty() {
            prop_assert!(calls.is_empty(), "{:?}", calls);
        } else {
            prop_assert_eq!(pinned_allocs(&calls), vec![bytes; count]);
            prop_assert_eq!(calls.last(), Some(&Call::Synchronize));
            assert_ring_bound(&calls, count);
        }
    }
}
