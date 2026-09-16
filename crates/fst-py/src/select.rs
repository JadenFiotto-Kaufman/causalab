//! Selections resolved to reads: the Python layer describes a box over a
//! tensor; `fst_core::select` says which bytes to read and how they land.
//!
//! Python parses an index into per-dimension ranges (and keeps the view that
//! turns the box into the result). Everything from there — the runs of the
//! box, coalescing them under the profile's policy for the file's storage,
//! the placements — is the core's, exposed here as [`select_reads`], so the
//! rows Python hands `read_job` are `fst_core::select::Read`s shifted by the
//! tensor's start and nothing more. [`shard_ranges`] is
//! `Selection::shard`: the ranges of one of `world` equal parts.

use std::ops::Range;
use std::path::PathBuf;

use fst_core::env::Env;
use fst_core::format::Dtype;
use fst_core::plan::coalesce_policy_for;
use fst_core::profile::Profile;
use fst_core::select::{self, CoalesceSummary, Selection};
use fst_core::{FormatError, SelectError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::IntoPyErr;

/// `(path, tensor name, shape, ranges, dtype)`: one box over one tensor of
/// one file, the dtype by its header name.
type PyItem = (PathBuf, String, Vec<u64>, Vec<(u64, u64)>, String);
/// `(offset within the tensor, nbytes, dst within the selection's
/// destination, placements)`: a `select::Read`; `placements` is `None`
/// when the read is one run landing whole.
type PyRead = (u64, u64, u64, Option<Vec<(u64, u64, u64)>>);

/// What resolving a selection can refuse.
#[derive(Debug, thiserror::Error)]
enum ResolveError {
    #[error(transparent)]
    Select(#[from] SelectError),
    #[error(transparent)]
    Format(#[from] FormatError),
}

impl IntoPyErr for ResolveError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        match self {
            ResolveError::Select(err) => err.into_py_err(py),
            ResolveError::Format(err) => err.into_py_err(py),
        }
    }
}

fn ranges(pairs: &[(u64, u64)]) -> Vec<Range<u64>> {
    pairs.iter().map(|&(start, end)| start..end).collect()
}

fn py_read(read: &select::Read) -> PyRead {
    let placements = (!read.is_contiguous()).then(|| {
        read.placements
            .iter()
            .map(|p| (p.src, p.dst, p.len))
            .collect()
    });
    (read.offset, read.len, read.dst, placements)
}

/// The reads for every item under the profile's policy for its file, and
/// the summary of what coalescing did over all of them.
fn resolve(
    env: &Env,
    profile: &Profile,
    items: &[PyItem],
) -> Result<(Vec<Vec<PyRead>>, CoalesceSummary), ResolveError> {
    let mut summary = CoalesceSummary::default();
    let mut out = Vec::with_capacity(items.len());
    for (path, name, shape, pairs, dtype_name) in items {
        let dtype = Dtype::parse(dtype_name).ok_or_else(|| FormatError::UnknownDtype {
            name: name.clone(),
            dtype: dtype_name.clone(),
        })?;
        let selection = Selection::new(shape, ranges(pairs))?;
        let runs = selection.runs(dtype)?;
        let policy = coalesce_policy_for(&profile.lookup(env, path).entry);
        let reads = select::coalesce(&runs, &policy);
        summary.add(&reads);
        out.push(reads.iter().map(py_read).collect());
    }
    Ok((out, summary))
}

/// Resolve boxes to reads. `items` is `[(path, name, shape, ranges, dtype)]`
/// — `ranges` one `(start, end)` per dimension, `dtype` the header's name.
/// Returns `(reads, summary)`: `reads[i]` is `[(offset, nbytes, dst,
/// placements)]` for `items[i]` — offsets within the tensor, `dst` within
/// the selection's contiguous destination, `placements` `None` for a read
/// that is one run landing whole — coalesced under the profile's policy for
/// the file's storage class (the same rule `plan_read` explains), and
/// `summary` is `{runs, reads, wanted_bytes, read_bytes, amplification}`
/// over all items, to hand back to `plan_read`. A box outside its shape, a
/// misaligned sub-byte run, or an unknown dtype is a typed error before
/// anything is read. The machine is probed once per call.
#[pyfunction]
pub fn select_reads<'py>(
    py: Python<'py>,
    items: Vec<PyItem>,
) -> PyResult<(Vec<Vec<PyRead>>, Bound<'py, PyDict>)> {
    let profile = crate::profile::current(py)?;
    let (reads, summary) = py
        .detach(|| {
            let env = Env::probe();
            resolve(&env, &profile, &items)
        })
        .map_err(|e| e.into_py_err(py))?;
    let dict = PyDict::new(py);
    dict.set_item("runs", summary.runs)?;
    dict.set_item("reads", summary.reads)?;
    dict.set_item("wanted_bytes", summary.wanted_bytes)?;
    dict.set_item("read_bytes", summary.read_bytes)?;
    dict.set_item("amplification", summary.amplification())?;
    Ok((reads, dict))
}

/// The `(start, end)` per dimension of shard `rank` of `world` equal shards
/// of a tensor of `shape` along `dim` — what `torch.chunk(t, world,
/// dim)[rank]` selects. `SelectError` when `dim` is not a dimension, `rank`
/// is not below `world`, or the dimension does not divide.
#[pyfunction]
pub fn shard_ranges(
    py: Python<'_>,
    shape: Vec<u64>,
    dim: usize,
    rank: u64,
    world: u64,
) -> PyResult<Vec<(u64, u64)>> {
    let selection = Selection::shard(&shape, dim, rank, world).map_err(|e| e.into_py_err(py))?;
    Ok(selection
        .ranges()
        .iter()
        .map(|r| (r.start, r.end))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_boxes_are_one_plain_read_and_inner_cuts_coalesce() {
        let env = Env::probe();
        let profile = Profile::default_measured();
        let path = PathBuf::from("/tmp/x.safetensors");
        let items = vec![
            (
                path.clone(),
                "a".to_owned(),
                vec![8, 6],
                vec![(0, 8), (0, 6)],
                "F32".to_owned(),
            ),
            (
                path.clone(),
                "b".to_owned(),
                vec![8, 6],
                vec![(0, 8), (1, 3)],
                "F32".to_owned(),
            ),
            (
                path.clone(),
                "c".to_owned(),
                vec![8, 6],
                vec![(2, 5), (0, 6)],
                "BF16".to_owned(),
            ),
        ];
        let (reads, summary) = resolve(&env, &profile, &items).unwrap();
        assert_eq!(reads[0], vec![(0, 192, 0, None)]);
        // eight 8-byte runs with 16-byte gaps: far under any policy's gap
        let (offset, len, dst, placements) = &reads[1][0];
        assert_eq!((offset, len, dst), (&4, &(7 * 24 + 8), &0));
        assert_eq!(placements.as_ref().map(Vec::len), Some(8));
        assert_eq!(reads[2], vec![(2 * 12, 3 * 12, 0, None)]);
        assert_eq!(summary.runs, 10);
        assert_eq!(summary.reads, 3);
        assert_eq!(summary.wanted_bytes, 192 + 64 + 36);
        // an unknown dtype and a box outside its shape are refused
        let bad = vec![(
            path.clone(),
            "a".to_owned(),
            vec![2],
            vec![(0, 2)],
            "Q7".to_owned(),
        )];
        assert!(matches!(
            resolve(&env, &profile, &bad),
            Err(ResolveError::Format(_))
        ));
        let outside = vec![(path, "a".to_owned(), vec![2], vec![(0, 3)], "U8".to_owned())];
        assert!(matches!(
            resolve(&env, &profile, &outside),
            Err(ResolveError::Select(SelectError::OutOfBounds { .. }))
        ));
    }
}
