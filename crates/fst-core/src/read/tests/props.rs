//! Random layouts of files, ranges and plans: byte-exact and tiling
//! invariants.

use super::*;

// ---- properties --------------------------------------------------------

#[derive(Debug, Clone)]
struct FileSpec {
    len: u64,
    ranges: Vec<ReadRange>,
}

fn file_spec() -> impl Strategy<Value = FileSpec> {
    (1u64..64).prop_flat_map(|len| {
        prop::collection::vec((0..len, 1u64..=64), 1..4).prop_map(move |raw| FileSpec {
            len,
            ranges: raw
                .into_iter()
                .map(|(offset, want)| ReadRange {
                    offset,
                    len: want.min(len - offset),
                })
                .collect(),
        })
    })
}

#[derive(Debug, Clone)]
struct PlanSpec {
    files_in_flight: usize,
    split_bytes: u64,
    readers_per_file: usize,
    device: Option<(usize, u64)>,
}

fn plan_spec() -> impl Strategy<Value = PlanSpec> {
    (
        1usize..8,
        prop_oneof![Just(0u64), 1u64..16],
        1usize..4,
        prop::option::of((1usize..4, 1u64..8)),
    )
        .prop_map(
            |(files_in_flight, split_bytes, readers_per_file, device)| PlanSpec {
                files_in_flight,
                split_bytes,
                readers_per_file,
                device,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]
    #[test]
    fn random_jobs_land_exactly_once_and_tile(
        specs in prop::collection::vec(file_spec(), 1..5),
        p in plan_spec(),
    ) {
        let store = SimStorage::new();
        let cuda = SimCuda::new(1 << 30);
        let objects: Vec<Vec<u8>> = specs.iter().enumerate().map(|(i, s)| seeded(s.len as usize, 100 + i as u64)).collect();
        for (i, o) in objects.iter().enumerate() {
            store.put(format!("f{i}"), o.clone());
        }
        let staging = match p.device {
            Some((count, bytes)) => Staging::Pinned { count, bytes },
            None => Staging::None,
        };
        let rp = plan(p.files_in_flight, p.split_bytes, p.readers_per_file, staging);
        // host: one buffer per transfer; device: one buffer per file, transfers packed
        let mut host: Vec<Vec<u8>> = specs.iter().flat_map(|s| s.ranges.iter().map(|r| vec![SENTINEL; r.len as usize])).collect();
        let ptrs: Vec<_> = specs.iter().map(|s| cuda.alloc_device(s.ranges.iter().map(|r| r.len).sum(), 0)).collect();
        let mut host_iter = host.iter_mut();
        let mut files = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let mut packed = 0u64;
            let transfers = spec.ranges.iter().map(|r| {
                let dest = if p.device.is_some() {
                    let d = Dest::Device { ptr: ptrs[i], offset: packed };
                    packed += r.len;
                    d
                } else {
                    Dest::Host(host_iter.next().unwrap())
                };
                Transfer { range: r.clone(), dest, placements: Vec::new() }
            }).collect();
            files.push(FileJob { path: format!("f{i}").into(), transfers });
        }
        let report = execute(&rp, &store, Some(&cuda), None, ReadJob { files }).unwrap_or_else(|e| panic!("{e}"));

        // every byte landed exactly where it should
        let wanted: u64 = specs.iter().flat_map(|s| s.ranges.iter().map(|r| r.len)).sum();
        prop_assert_eq!(report.bytes_read, wanted);
        prop_assert_eq!(report.files_opened, specs.len());
        if p.device.is_some() {
            for (i, spec) in specs.iter().enumerate() {
                let expected: Vec<u8> = spec.ranges.iter().flat_map(|r| objects[i][r.offset as usize..(r.offset + r.len) as usize].to_vec()).collect();
                prop_assert_eq!(cuda.device_bytes(ptrs[i]), Some(expected));
            }
        } else {
            let mut got = host.iter();
            for (i, spec) in specs.iter().enumerate() {
                for r in &spec.ranges {
                    prop_assert_eq!(got.next().unwrap().as_slice(), &objects[i][r.offset as usize..(r.offset + r.len) as usize]);
                }
            }
        }

        // the operation log respects the plan
        let log = store.log();
        let mut opened = opens(&log);
        opened.sort();
        prop_assert_eq!(opened, (0..specs.len()).map(|i| PathBuf::from(format!("f{i}"))).collect::<Vec<_>>());
        let reads = reads_by_path(&log);
        let staging_bytes = p.device.map(|(_, b)| b);
        for (i, spec) in specs.iter().enumerate() {
            let mut expected: Vec<(u64, u64)> = spec.ranges.iter().flat_map(|r| expected_pieces(r, p.split_bytes, staging_bytes)).collect();
            expected.sort_unstable();
            let actual = reads.get(Path::new(&format!("f{i}"))).cloned().unwrap_or_default();
            prop_assert_eq!(&actual, &expected, "file {} pieces", i);
            if p.split_bytes > 0 {
                prop_assert!(actual.iter().all(|&(_, l)| l <= p.split_bytes));
            }
            prop_assert_eq!(report.pieces, reads.values().map(Vec::len).sum::<usize>());
        }
        if let Some((count, _)) = p.device {
            let calls = cuda.log();
            prop_assert_eq!(calls.iter().filter(|c| matches!(c, Call::AllocPinned(_))).count(), count);
            prop_assert_eq!(calls.last(), Some(&Call::Synchronize));
            assert_ring_bound(&calls, count);
        } else {
            prop_assert!(cuda.log().is_empty());
        }
    }
}
