use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fst_cuda::sim::{Call, SimCuda};
use fst_cuda::{CudaError, DeviceCopier, DevicePtr, PinnedBuffer};
use proptest::prelude::*;

use super::*;
use crate::format::{Dtype, Layout, numel, parse_object};
use crate::storage::RangeReader;
use crate::storage::posix::PosixStorage;
use crate::storage::sim::{Fault, FaultKind, Op, SimStorage};

use crate::fixtures::{MIXED, SINGLE_U8};

/// A fixture split into the pieces a caller would hand us: specs in *name*
/// order (deliberately not data order) and one host slice per spec.
struct Split<'a> {
    layout: Layout,
    specs: Vec<TensorSpec>,
    parts: Vec<Part<'a>>,
}

fn split(fixture: &[u8]) -> Split<'_> {
    let layout = parse_object(fixture, None).unwrap_or_else(|e| panic!("{e}"));
    let mut specs = Vec::new();
    let mut parts = Vec::new();
    for (name, info) in &layout.tensors {
        specs.push(TensorSpec {
            name: name.clone(),
            dtype: info.dtype,
            shape: info.shape.clone(),
            nbytes: info.nbytes(),
        });
        let [start, end] = layout.absolute_range(name).unwrap();
        parts.push(Part::Host(&fixture[start as usize..end as usize]));
    }
    Split {
        layout,
        specs,
        parts,
    }
}

fn ok<T>(r: Result<T, WriteError>) -> T {
    r.unwrap_or_else(|e| panic!("{e}"))
}

fn options(atomic: bool) -> WriteOptions {
    WriteOptions {
        atomic,
        ..WriteOptions::default()
    }
}

// ---------------------------------------------------------------- fixtures

#[test]
fn fixtures_round_trip_byte_identical_through_sim_posix_and_vec() {
    let dir = tempfile::tempdir().unwrap();
    for (name, fixture) in [("mixed", MIXED), ("single_u8", SINGLE_U8)] {
        let s = split(fixture);
        let payload = ok(Payload::new(
            &s.specs,
            &s.parts,
            s.layout.metadata.as_deref(),
        ));
        assert_eq!(payload.header().layout, s.layout);
        assert_eq!(payload.nbytes() as usize, fixture.len());

        let store = SimStorage::new();
        let report = ok(write_object(
            &store,
            Path::new(name),
            &payload,
            &options(false),
            None,
        ));
        assert_eq!(
            store.get(Path::new(name)).as_deref(),
            Some(fixture),
            "{name} via sim"
        );
        assert_eq!(report.bytes as usize, fixture.len());
        assert_eq!(report.parts, s.specs.len());
        assert_eq!(report.staging_chunks, 0);

        for atomic in [false, true] {
            let path = dir.path().join(format!("{name}-{atomic}.safetensors"));
            ok(write_object(
                &PosixStorage,
                &path,
                &payload,
                &options(atomic),
                None,
            ));
            assert_eq!(std::fs::read(&path).unwrap(), fixture, "{name} via posix");
        }

        let host: Vec<&[u8]> = s
            .parts
            .iter()
            .map(|p| match p {
                Part::Host(b) => *b,
                Part::Device { .. } => unreachable!(),
            })
            .collect();
        let bytes = ok(serialize_to_vec(
            &s.specs,
            &host,
            s.layout.metadata.as_deref(),
        ));
        assert_eq!(bytes, fixture, "{name} via serialize_to_vec");
    }
}

// --------------------------------------------------------------------- sim

#[test]
fn host_parts_stream_in_header_order_as_one_write_each() {
    let s = split(MIXED);
    let payload = ok(Payload::new(
        &s.specs,
        &s.parts,
        s.layout.metadata.as_deref(),
    ));
    let store = SimStorage::new();
    ok(write_object(
        &store,
        Path::new("m"),
        &payload,
        &options(false),
        None,
    ));
    let header_len = payload.header().bytes.len() as u64;
    // data order is a/bias (I64, 24 bytes), b/weight (F32, 24), a/half (BF16, 4),
    // empty (F16, 0 -- skipped), c/flag (BOOL, 2)
    let names: Vec<&str> = payload.names().collect();
    assert_eq!(names, ["a/bias", "b/weight", "a/half", "empty", "c/flag"]);
    assert_eq!(
        store.log(),
        vec![
            Op::Create("m".into()),
            Op::Write("m".into(), vec![header_len]),
            Op::Write("m".into(), vec![24]),
            Op::Write("m".into(), vec![24]),
            Op::Write("m".into(), vec![4]),
            Op::Write("m".into(), vec![2]),
            Op::Finish("m".into(), false),
        ]
    );
}

#[test]
fn atomic_writes_go_to_a_sibling_temp_name_then_rename() {
    let s = split(SINGLE_U8);
    let payload = ok(Payload::new(&s.specs, &s.parts, None));
    let store = SimStorage::new();
    let path = PathBuf::from("dir/obj.safetensors");
    let mut opts = options(true);
    opts.durable = true;
    ok(write_object(&store, &path, &payload, &opts, None));
    let log = store.log();
    let Op::Create(temp) = &log[0] else {
        panic!("{log:?}")
    };
    assert_eq!(temp.parent(), path.parent(), "temp is a sibling");
    assert_ne!(temp, &path);
    assert_eq!(log[log.len() - 2], Op::Finish(temp.clone(), true));
    assert_eq!(
        log[log.len() - 1],
        Op::Rename(temp.clone(), path.clone(), true)
    );
    assert_eq!(
        store.paths(),
        vec![path.clone()],
        "only the final name remains"
    );
    assert_eq!(store.get(&path).as_deref(), Some(SINGLE_U8));
}

#[test]
fn write_fault_names_path_and_part() {
    let s = split(MIXED);
    let payload = ok(Payload::new(
        &s.specs,
        &s.parts,
        s.layout.metadata.as_deref(),
    ));
    // writes: 0 header, 1 a/bias, 2 b/weight
    let fault = || Fault {
        kind: FaultKind::Write,
        nth: 2,
        message: "enospc".into(),
    };
    let b_weight = s.specs.iter().position(|t| t.name == "b/weight").unwrap();

    // atomic: nothing is left under any name
    let store = SimStorage::new();
    store.arm(fault());
    let err = write_object(&store, Path::new("m"), &payload, &options(true), None)
        .err()
        .unwrap();
    match &err {
        WriteError::Storage {
            path,
            stage: Stage::Part { index, name },
            source: StorageError::Simulated(m),
        } => {
            assert_eq!(path, Path::new("m"));
            assert_eq!(*index, b_weight);
            assert_eq!(name, "b/weight");
            assert_eq!(m, "enospc");
        }
        other => panic!("{other}"),
    }
    assert!(store.paths().is_empty(), "{:?}", store.paths());
    assert!(matches!(store.log().last(), Some(Op::Remove(_))));

    // not atomic: the partial object stays under the final name, exactly the
    // bytes that were written before the fault (header and the first part)
    let store = SimStorage::new();
    store.arm(fault());
    let err = write_object(&store, Path::new("m"), &payload, &options(false), None)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        WriteError::Storage {
            stage: Stage::Part { .. },
            ..
        }
    ));
    let written = payload.header().bytes.len() + 24;
    assert_eq!(
        store.get(Path::new("m")).as_deref(),
        Some(&MIXED[..written])
    );
    assert!(!store.log().iter().any(|op| matches!(op, Op::Remove(_))));
}

#[test]
fn header_finish_and_rename_faults_are_staged_and_leave_nothing() {
    let s = split(SINGLE_U8);
    let payload = ok(Payload::new(&s.specs, &s.parts, None));
    for (kind, nth, expected) in [
        (FaultKind::Write, 0, Stage::Header),
        (FaultKind::Finish, 0, Stage::Finish),
        (FaultKind::Rename, 0, Stage::Rename),
    ] {
        let store = SimStorage::new();
        store.arm(Fault {
            kind,
            nth,
            message: "boom".into(),
        });
        let err = write_object(&store, Path::new("o"), &payload, &options(true), None)
            .err()
            .unwrap();
        match err {
            WriteError::Storage { path, stage, .. } => {
                assert_eq!(path, Path::new("o"));
                assert_eq!(stage, expected);
            }
            other => panic!("{other}"),
        }
        assert!(store.paths().is_empty(), "{expected}: {:?}", store.paths());
    }
}

#[test]
fn create_fault_removes_nothing_and_cleanup_failure_does_not_mask() {
    let s = split(SINGLE_U8);
    let payload = ok(Payload::new(&s.specs, &s.parts, None));

    let store = SimStorage::new();
    store.arm(Fault {
        kind: FaultKind::Create,
        nth: 0,
        message: "eacces".into(),
    });
    let err = write_object(&store, Path::new("o"), &payload, &options(true), None)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        WriteError::Storage {
            stage: Stage::Create,
            ..
        }
    ));
    assert_eq!(
        store.log().len(),
        1,
        "nothing to remove after a failed create"
    );

    let store = SimStorage::new();
    store.arm(Fault {
        kind: FaultKind::Write,
        nth: 1,
        message: "eio".into(),
    });
    store.arm(Fault {
        kind: FaultKind::Remove,
        nth: 0,
        message: "cleanup failed too".into(),
    });
    let err = write_object(&store, Path::new("o"), &payload, &options(true), None)
        .err()
        .unwrap();
    assert!(matches!(
        err,
        WriteError::Storage {
            stage: Stage::Part { index: 0, .. },
            source: StorageError::Simulated(ref m),
            ..
        } if m == "eio"
    ));
}

#[test]
fn payload_and_option_validation() {
    let spec = |name: &str, nbytes| TensorSpec {
        name: name.into(),
        dtype: Dtype::U8,
        shape: vec![nbytes],
        nbytes,
    };
    let data = [0u8; 4];
    assert!(matches!(
        Payload::new(&[spec("a", 4)], &[], None),
        Err(WriteError::PartCount {
            expected: 1,
            count: 0
        })
    ));
    assert!(matches!(
        Payload::new(&[spec("a", 3)], &[Part::Host(&data)], None),
        Err(WriteError::PartLength {
            index: 0,
            expected: 3,
            actual: 4,
            ..
        })
    ));
    assert!(matches!(
        Payload::new(&[spec("a", 5)], &[Part::Host(&data[..1])], None),
        Err(WriteError::PartLength { .. })
    ));
    assert!(matches!(
        Payload::new(&[spec("__metadata__", 4)], &[Part::Host(&data)], None),
        Err(WriteError::Format(FormatError::ReservedName))
    ));
    let cuda = SimCuda::new(1 << 20);
    let small = cuda.alloc_device(2, 0);
    assert!(matches!(
        Payload::new(
            &[spec("a", 4)],
            &[Part::Device {
                ptr: small,
                nbytes: 4
            }],
            None
        ),
        Err(WriteError::DeviceOverrun {
            index: 0,
            nbytes: 4,
            capacity: 2,
            ..
        })
    ));

    let ptr = cuda.alloc_device(4, 0);
    let device = ok(Payload::new(
        &[spec("a", 4)],
        &[Part::Device { ptr, nbytes: 4 }],
        None,
    ));
    assert!(device.has_device_parts());
    let store = SimStorage::new();
    assert!(matches!(
        write_object(&store, Path::new("o"), &device, &options(false), None),
        Err(WriteError::NoCopier { .. })
    ));
    let mut empty = options(false);
    empty.staging = StagingRing { count: 0, bytes: 8 };
    assert!(matches!(
        write_object(&store, Path::new("o"), &device, &empty, Some(&cuda)),
        Err(WriteError::EmptyStaging {
            count: 0,
            bytes: 8,
            ..
        })
    ));
    assert!(store.log().is_empty(), "refused before touching storage");

    let host = ok(Payload::new(&[spec("a", 4)], &[Part::Host(&data)], None));
    assert!(matches!(
        write_object(&store, Path::new("/"), &host, &options(true), None),
        Err(WriteError::NoFileName { .. })
    ));
}

// ------------------------------------------------------------------ device

fn device_payload<'a>(cuda: &SimCuda, nbytes: u64) -> (Payload<'a>, Vec<u8>) {
    let ptr = cuda.alloc_device(nbytes, 0);
    let bytes: Vec<u8> = (0..nbytes).map(|i| (i * 7 + 3) as u8).collect();
    let mut pinned = cuda.alloc_pinned(nbytes.max(1)).unwrap();
    pinned.as_mut_slice()[..nbytes as usize].copy_from_slice(&bytes);
    cuda.copy_to_device(pinned.as_ref(), nbytes, ptr, 0)
        .unwrap();
    let spec = TensorSpec {
        name: "act".into(),
        dtype: Dtype::U8,
        shape: vec![nbytes],
        nbytes,
    };
    let payload = ok(Payload::new(&[spec], &[Part::Device { ptr, nbytes }], None));
    (payload, bytes)
}

fn staging(count: usize, bytes: u64) -> WriteOptions {
    WriteOptions {
        staging: StagingRing { count, bytes },
        ..WriteOptions::default()
    }
}

#[test]
fn device_parts_equal_simulated_device_memory_beside_host_parts() {
    let cuda = SimCuda::new(1 << 30);
    let a = cuda.alloc_device(10, 0);
    let mut pinned = cuda.alloc_pinned(10).unwrap();
    pinned.as_mut_slice().copy_from_slice(b"0123456789");
    cuda.copy_to_device(pinned.as_ref(), 10, a, 0).unwrap();
    let host = [9u8; 6];
    let specs = [
        TensorSpec {
            name: "host".into(),
            dtype: Dtype::U8,
            shape: vec![6],
            nbytes: 6,
        },
        TensorSpec {
            name: "device".into(),
            dtype: Dtype::U8,
            shape: vec![10],
            nbytes: 10,
        },
    ];
    let parts = [Part::Host(&host), Part::Device { ptr: a, nbytes: 10 }];
    let payload = ok(Payload::new(&specs, &parts, None));
    let store = SimStorage::new();
    let report = ok(write_object(
        &store,
        Path::new("o"),
        &payload,
        &staging(2, 4),
        Some(&cuda),
    ));
    let mut expected = payload.header().bytes.clone();
    expected.extend_from_slice(b"0123456789"); // "device" < "host" by name
    expected.extend_from_slice(&host);
    assert_eq!(store.get(Path::new("o")), Some(expected));
    assert_eq!(report.staging_chunks, 3);
    // 10 bytes in 4-byte chunks: 4, 4, 2 -- each chunk is its own write
    let writes: Vec<Vec<u64>> = store
        .log()
        .into_iter()
        .filter_map(|op| match op {
            Op::Write(_, lens) => Some(lens),
            _ => None,
        })
        .collect();
    assert_eq!(writes[1..], [vec![4], vec![4], vec![2], vec![6]]);
    // the ring allocates lazily: two slots, sized to the chunk
    let allocs = cuda
        .log()
        .iter()
        .filter(|c| matches!(c, Call::AllocPinned(4)))
        .count();
    assert_eq!(allocs, 2);
}

/// One event stream over both simulators, so the order of copies, syncs and
/// writes can be checked against each other.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    ToHost { offset: u64 },
    Sync,
    Write(u64),
}

type Trace = Arc<Mutex<Vec<Event>>>;

struct TracedCuda(SimCuda, Trace);

impl DeviceCopier for TracedCuda {
    fn alloc_pinned(&self, nbytes: u64) -> Result<Box<dyn PinnedBuffer>, CudaError> {
        self.0.alloc_pinned(nbytes)
    }
    fn copy_to_device(
        &self,
        src: &dyn PinnedBuffer,
        nbytes: u64,
        dst: DevicePtr,
        offset: u64,
    ) -> Result<(), CudaError> {
        self.0.copy_to_device(src, nbytes, dst, offset)
    }
    fn copy_to_device_2d(
        &self,
        src: &dyn PinnedBuffer,
        dst: DevicePtr,
        shape: fst_cuda::Copy2D,
    ) -> Result<(), CudaError> {
        self.0.copy_to_device_2d(src, dst, shape)
    }
    fn copy_to_host(
        &self,
        src: DevicePtr,
        offset: u64,
        nbytes: u64,
        dst: &mut dyn PinnedBuffer,
    ) -> Result<(), CudaError> {
        self.1.lock().unwrap().push(Event::ToHost { offset });
        self.0.copy_to_host(src, offset, nbytes, dst)
    }
    fn synchronize(&self) -> Result<(), CudaError> {
        self.1.lock().unwrap().push(Event::Sync);
        self.0.synchronize()
    }
    fn record(&self) -> Result<Box<dyn fst_cuda::Fence>, CudaError> {
        // the traced copier stands in for a stream; a fence over it is the
        // inner simulator's, which orders after every copy enqueued so far
        self.0.record()
    }
    fn free_bytes(&self, device: u32) -> Result<u64, CudaError> {
        self.0.free_bytes(device)
    }
}

struct TracedStorage(SimStorage, Trace);

impl Storage for TracedStorage {
    fn name(&self) -> &'static str {
        "traced"
    }
    fn open_reader(&self, path: &Path) -> Result<Box<dyn RangeReader>, StorageError> {
        self.0.open_reader(path)
    }
    fn create_writer(&self, path: &Path) -> Result<Box<dyn PartWriter>, StorageError> {
        Ok(Box::new(TracedWriter(
            self.0.create_writer(path)?,
            self.1.clone(),
        )))
    }
    fn rename(&self, from: &Path, to: &Path, durable: bool) -> Result<(), StorageError> {
        self.0.rename(from, to, durable)
    }
    fn remove(&self, path: &Path) -> Result<(), StorageError> {
        self.0.remove(path)
    }
}

struct TracedWriter(Box<dyn PartWriter>, Trace);

impl PartWriter for TracedWriter {
    fn write_parts(&mut self, parts: &[&[u8]]) -> Result<(), StorageError> {
        let total = parts.iter().map(|p| p.len() as u64).sum();
        self.1.lock().unwrap().push(Event::Write(total));
        self.0.write_parts(parts)
    }
    fn finish(self: Box<Self>, durable: bool) -> Result<(), StorageError> {
        self.0.finish(durable)
    }
}

/// The staging protocol, from the combined event stream of one device part:
/// every chunk's copy is enqueued, then synchronized, then written; a slot is
/// refilled only after the chunk it held was written; never more than `count`
/// copies are outstanding.
fn check_staging_protocol(events: &[Event], chunks: usize, count: usize, chunk_bytes: u64) {
    let pos = |pred: &dyn Fn(&Event) -> bool| -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter(|(_, e)| pred(e))
            .map(|(i, _)| i)
            .collect()
    };
    let to_host = pos(&|e| matches!(e, Event::ToHost { .. }));
    let syncs = pos(&|e| matches!(e, Event::Sync));
    let writes = pos(&|e| matches!(e, Event::Write(_)));
    assert_eq!(to_host.len(), chunks, "one copy per chunk");
    assert_eq!(
        writes.len(),
        chunks + 1,
        "the header and one write per chunk"
    );
    let chunk_writes = &writes[1..];
    for (k, &copy) in to_host.iter().enumerate() {
        assert_eq!(
            events[copy],
            Event::ToHost {
                offset: k as u64 * chunk_bytes
            },
            "chunks are copied in order"
        );
        let write = chunk_writes[k];
        assert!(
            copy < write,
            "chunk {k} written before its copy was enqueued"
        );
        assert!(
            syncs.iter().any(|&s| copy < s && s < write),
            "chunk {k} written without a synchronize after its copy"
        );
        if k >= count {
            assert!(
                chunk_writes[k - count] < copy,
                "slot reused for chunk {k} before chunk {} was written",
                k - count
            );
        }
    }
    // outstanding copies never exceed the ring
    let mut outstanding = 0usize;
    for event in events {
        match event {
            Event::ToHost { .. } => {
                outstanding += 1;
                assert!(
                    outstanding <= count,
                    "{outstanding} copies in flight with {count} slots"
                );
            }
            Event::Write(_) if outstanding > 0 => outstanding -= 1,
            _ => {}
        }
    }
}

#[test]
fn staging_ring_overlaps_the_next_copy_with_the_current_write() {
    let trace: Trace = Arc::default();
    let cuda = TracedCuda(SimCuda::new(1 << 30), trace.clone());
    let store = TracedStorage(SimStorage::new(), trace.clone());
    let (payload, bytes) = device_payload(&cuda.0, 50);
    let report = ok(write_object(
        &store,
        Path::new("o"),
        &payload,
        &staging(3, 8),
        Some(&cuda),
    ));
    assert_eq!(report.staging_chunks, 7);
    let mut expected = payload.header().bytes.clone();
    expected.extend_from_slice(&bytes);
    assert_eq!(store.0.get(Path::new("o")), Some(expected));
    let events = trace.lock().unwrap().clone();
    check_staging_protocol(&events, 7, 3, 8);
    // with three slots the copy of chunk k+1 (and k+2) is in flight while
    // chunk k is written: the write of chunk 0 is preceded by the copies of
    // chunks 1 and 2, not just its own
    let first_write = events
        .iter()
        .position(|e| matches!(e, Event::Write(8)))
        .unwrap();
    let copies_before = events[..first_write]
        .iter()
        .filter(|e| matches!(e, Event::ToHost { .. }))
        .count();
    assert_eq!(copies_before, 3);
    // a single slot cannot overlap: copy, sync, write, repeat
    let trace: Trace = Arc::default();
    let cuda = TracedCuda(SimCuda::new(1 << 30), trace.clone());
    let store = TracedStorage(SimStorage::new(), trace.clone());
    let (payload, _) = device_payload(&cuda.0, 20);
    ok(write_object(
        &store,
        Path::new("o"),
        &payload,
        &staging(1, 8),
        Some(&cuda),
    ));
    let events = trace.lock().unwrap().clone();
    check_staging_protocol(&events, 3, 1, 8);
    assert_eq!(
        events[1..],
        [
            Event::ToHost { offset: 0 },
            Event::Sync,
            Event::Write(8),
            Event::ToHost { offset: 8 },
            Event::Sync,
            Event::Write(8),
            Event::ToHost { offset: 16 },
            Event::Sync,
            Event::Write(4),
        ]
    );
}

proptest! {
    #[test]
    fn device_chunking_never_splits_wrong(
        nbytes in 0u64..300,
        chunk_bytes in 1u64..64,
        count in 1usize..5,
    ) {
        let trace: Trace = Arc::default();
        let cuda = TracedCuda(SimCuda::new(1 << 30), trace.clone());
        let store = TracedStorage(SimStorage::new(), trace.clone());
        let (payload, bytes) = device_payload(&cuda.0, nbytes);
        let report = ok(write_object(&store, Path::new("o"), &payload, &staging(count, chunk_bytes), Some(&cuda)));
        let mut expected = payload.header().bytes.clone();
        expected.extend_from_slice(&bytes);
        prop_assert_eq!(store.0.get(Path::new("o")), Some(expected));
        let chunks = nbytes.div_ceil(chunk_bytes) as usize;
        prop_assert_eq!(report.staging_chunks, chunks);
        prop_assert_eq!(report.bytes, payload.nbytes());
        let events = trace.lock().unwrap().clone();
        check_staging_protocol(&events, chunks, count, chunk_bytes);
        let allocs = cuda.0.log().iter().filter(|c| matches!(c, Call::AllocPinned(_))).count();
        // one allocation seeded the device in `device_payload`; the ring's are the rest
        prop_assert_eq!(allocs - 1, count.min(chunks), "slots are allocated as chunks need them");
    }
}

// ---------------------------------------------------------------- proptest

fn arb_spec() -> impl Strategy<Value = TensorSpec> {
    (
        "[a-z][a-z0-9_./]{0,12}",
        proptest::sample::select(Dtype::ALL.to_vec()),
        proptest::collection::vec(0u64..5, 0..4),
    )
        .prop_filter_map("whole bytes", |(name, dtype, shape)| {
            let nbytes = dtype.nbytes(numel(&shape)).ok()?;
            Some(TensorSpec {
                name,
                dtype,
                shape,
                nbytes,
            })
        })
}

fn arb_specs() -> impl Strategy<Value = Vec<TensorSpec>> {
    proptest::collection::vec(arb_spec(), 0..12).prop_map(|mut v| {
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v.dedup_by(|a, b| a.name == b.name);
        v
    })
}

fn arb_metadata() -> impl Strategy<Value = Option<Vec<(String, String)>>> {
    proptest::option::of(
        proptest::collection::btree_map("[a-z]{1,6}", "[ -~]{0,10}", 0..3)
            .prop_map(|m| m.into_iter().collect::<Vec<_>>()),
    )
}

fn buffers_for(specs: &[TensorSpec], seed: u8) -> Vec<Vec<u8>> {
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (0..s.nbytes)
                .map(|j| (j as u8).wrapping_mul(31).wrapping_add(i as u8 ^ seed))
                .collect()
        })
        .collect()
}

proptest! {
    #[test]
    fn serialize_to_vec_parses_back_with_parts_in_header_order(
        specs in arb_specs(),
        metadata in arb_metadata(),
        seed in any::<u8>(),
    ) {
        let buffers = buffers_for(&specs, seed);
        let parts: Vec<&[u8]> = buffers.iter().map(Vec::as_slice).collect();
        let bytes = ok(serialize_to_vec(&specs, &parts, metadata.as_deref()));
        let built = format::build_header(&specs, metadata.as_deref()).unwrap();
        let layout = parse_object(&bytes, None).unwrap_or_else(|e| panic!("{e}"));
        prop_assert_eq!(&layout, &built.layout);
        prop_assert_eq!(bytes.len() as u64, layout.object_len());
        let mut data = Vec::new();
        for &i in &built.order {
            data.extend_from_slice(&buffers[i]);
        }
        prop_assert_eq!(&bytes[layout.data_start() as usize..], &data[..]);
        // and the streaming path produces the same bytes
        let host: Vec<Part> = parts.iter().map(|p| Part::Host(p)).collect();
        let payload = ok(Payload::new(&specs, &host, metadata.as_deref()));
        let store = SimStorage::new();
        ok(write_object(&store, Path::new("o"), &payload, &options(true), None));
        prop_assert_eq!(store.get(Path::new("o")), Some(bytes));
    }
}

// ------------------------------------------------------------------- posix

#[test]
fn posix_durable_and_atomic_round_trips_leave_only_the_final_file() {
    let s = split(MIXED);
    let payload = ok(Payload::new(
        &s.specs,
        &s.parts,
        s.layout.metadata.as_deref(),
    ));
    for (durable, atomic) in [(false, false), (true, false), (false, true), (true, true)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        // an existing object under the final name is replaced whole
        std::fs::write(&path, b"stale").unwrap();
        let opts = WriteOptions {
            durable,
            atomic,
            ..WriteOptions::default()
        };
        let report = ok(write_object(&PosixStorage, &path, &payload, &opts, None));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            MIXED,
            "durable={durable} atomic={atomic}"
        );
        assert_eq!(report.path, path);
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["m.safetensors"], "no temp name remains");
    }
}

#[test]
fn posix_atomic_failure_leaves_the_old_object_and_no_temp() {
    let s = split(MIXED);
    let payload = ok(Payload::new(
        &s.specs,
        &s.parts,
        s.layout.metadata.as_deref(),
    ));
    let dir = tempfile::tempdir().unwrap();
    // the final name is a directory: the rename must fail
    let path = dir.path().join("m.safetensors");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("child"), b"x").unwrap();
    let err = write_object(&PosixStorage, &path, &payload, &options(true), None)
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            WriteError::Storage {
                stage: Stage::Rename,
                ..
            }
        ),
        "{err}"
    );
    assert!(path.is_dir(), "the old object is untouched");
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["m.safetensors"], "the temp file was removed");
}
