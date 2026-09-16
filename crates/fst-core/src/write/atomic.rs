//! Temp-then-rename: the object is written under a sibling name and renamed
//! into place once complete, so a reader of the final name sees the previous
//! object or the new one and never a prefix of the new one.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::Storage;

use super::{Stage, WriteError};

/// Distinguishes temp names created by this process; the pid distinguishes
/// processes sharing a directory.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// `dir/.name.<pid>.<n>.tmp` beside `dir/name`. A sibling, so the rename
/// stays within one file system and is atomic.
pub(super) fn temp_name(path: &Path) -> Result<PathBuf, WriteError> {
    let name = path.file_name().ok_or_else(|| WriteError::NoFileName {
        path: path.to_owned(),
    })?;
    let mut temp = OsString::from(".");
    temp.push(name);
    temp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(path.with_file_name(temp))
}

/// Move the finished temp object into place.
pub(super) fn commit(
    storage: &dyn Storage,
    temp: &Path,
    path: &Path,
    durable: bool,
) -> Result<(), WriteError> {
    storage
        .rename(temp, path, durable)
        .map_err(|source| WriteError::Storage {
            path: path.to_owned(),
            stage: Stage::Rename,
            source,
        })
}

/// Best-effort removal of a temp object after a failure. The failure being
/// reported is the write's; a second failure here is logged, never returned.
pub(super) fn discard(storage: &dyn Storage, temp: &Path) {
    if let Err(err) = storage.remove(temp) {
        tracing::warn!(temp = %temp.display(), error = %err, "could not remove temp object");
    }
}
