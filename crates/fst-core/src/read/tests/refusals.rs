//! Jobs the engine refuses before any I/O, with typed errors.

use super::*;

// ---- refusals before any I/O -------------------------------------------

#[test]
fn destination_mismatch_is_refused_before_io() {
    let store = SimStorage::new();
    store.put("a", seeded(16, 7));
    let mut short = vec![0u8; 3];
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 4),
                dest: Dest::Host(&mut short),
                placements: Vec::new(),
            }],
        }],
    };
    let err = execute(&plan(1, 0, 1, Staging::None), &store, None, None, job).err();
    assert!(
        matches!(&err, Some(ReadError::DestinationMismatch { path, range: r, available: 3 })
            if path == Path::new("a") && *r == range(0, 4)),
        "{err:?}"
    );
    let cuda = SimCuda::new(1 << 30);
    let ptr = cuda.alloc_device(8, 0);
    let job = ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 4),
                dest: Dest::Device { ptr, offset: 6 },
                placements: Vec::new(),
            }],
        }],
    };
    let staging = Staging::Pinned { count: 1, bytes: 4 };
    let err = execute(&plan(1, 0, 1, staging), &store, Some(&cuda), None, job).err();
    assert!(
        matches!(
            err,
            Some(ReadError::DestinationMismatch { available: 2, .. })
        ),
        "{err:?}"
    );
    assert!(store.log().is_empty());
    assert!(cuda.log().is_empty());
}

#[test]
fn device_destinations_need_a_copier_and_staging() {
    let store = SimStorage::new();
    store.put("a", seeded(8, 8));
    let cuda = SimCuda::new(1 << 30);
    let ptr = cuda.alloc_device(8, 0);
    let job = || ReadJob {
        files: vec![FileJob {
            path: "a".into(),
            transfers: vec![Transfer {
                range: range(0, 8),
                dest: Dest::Device { ptr, offset: 0 },
                placements: Vec::new(),
            }],
        }],
    };
    let staged = plan(1, 0, 1, Staging::Pinned { count: 2, bytes: 8 });
    assert!(matches!(
        execute(&staged, &store, None, None, job()),
        Err(ReadError::CopierRequired)
    ));
    let unstaged = plan(1, 0, 1, Staging::None);
    assert!(matches!(
        execute(&unstaged, &store, Some(&cuda), None, job()),
        Err(ReadError::StagingRequired)
    ));
    let empty = plan(1, 0, 1, Staging::Pinned { count: 0, bytes: 8 });
    assert!(matches!(
        execute(&empty, &store, Some(&cuda), None, job()),
        Err(ReadError::EmptyStaging { count: 0, bytes: 8 })
    ));
    assert!(store.log().is_empty());
    assert!(cuda.log().is_empty());
}
