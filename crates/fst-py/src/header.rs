//! The header grammar, parsed and built.

use std::path::{Path, PathBuf};

use fst_core::format::{self, Dtype, Layout, TensorSpec};
use fst_core::storage::Storage;
use fst_core::storage::posix::PosixStorage;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::errors::IntoPyErr;

/// `(name, dtype, shape, absolute start, absolute end)` per tensor.
type TensorRow = (String, String, Vec<u64>, u64, u64);
/// `(header_len, tensors, metadata)`.
type ParsedHeader = (u64, Vec<TensorRow>, Option<Vec<(String, String)>>);
/// `(name, dtype, shape, nbytes)` as the Python layer describes a tensor.
pub type SpecRow = (String, String, Vec<u64>, u64);

/// The rows as `TensorSpec`s; an unknown dtype name is
/// [`FormatError::UnknownDtype`](fst_core::FormatError::UnknownDtype).
pub fn tensor_specs(rows: &[SpecRow]) -> Result<Vec<TensorSpec>, fst_core::FormatError> {
    rows.iter()
        .map(|(name, dtype, shape, nbytes)| {
            let parsed =
                Dtype::parse(dtype).ok_or_else(|| fst_core::FormatError::UnknownDtype {
                    name: name.clone(),
                    dtype: dtype.clone(),
                })?;
            Ok(TensorSpec {
                name: name.clone(),
                dtype: parsed,
                shape: shape.clone(),
                nbytes: *nbytes,
            })
        })
        .collect()
}

fn rows(layout: &Layout) -> ParsedHeader {
    let base = layout.data_start();
    let tensors = layout
        .tensors
        .iter()
        .map(|(name, info)| {
            (
                name.clone(),
                info.dtype.name().to_owned(),
                info.shape.clone(),
                base + info.data_offsets[0],
                base + info.data_offsets[1],
            )
        })
        .collect();
    (layout.header_len, tensors, layout.metadata.clone())
}

/// Parse a safetensors header from the start of `bytes`; returns
/// `(header_len, [(name, dtype, shape, start, end)], metadata)` with tensor
/// ranges absolute within the object and `metadata` `None` when the header
/// carries no `__metadata__` key. The object is checked to be complete.
#[pyfunction]
pub fn parse_header(py: Python<'_>, bytes: &[u8]) -> PyResult<ParsedHeader> {
    let layout = format::parse_object(bytes, None).map_err(|e| e.into_py_err(py))?;
    Ok(rows(&layout))
}

/// Read `bytes[..min(8 + header_len, file_len)]`, letting `parse_object`
/// name a truncation rather than the storage layer.
fn read_header_prefix(path: &Path) -> Result<(Vec<u8>, u64), fst_core::StorageError> {
    let reader = PosixStorage.open_reader(path)?;
    let len = reader.len();
    let prefix_len = (format::LENGTH_PREFIX_BYTES as u64).min(len);
    let mut prefix = vec![0u8; prefix_len as usize];
    reader.read_at(0, &mut prefix)?;
    let Ok(header_len) = format::parse_length_prefix(&prefix) else {
        return Ok((prefix, len));
    };
    let want = (format::LENGTH_PREFIX_BYTES as u64 + header_len).min(len);
    let mut bytes = vec![0u8; want as usize];
    bytes[..prefix.len()].copy_from_slice(&prefix);
    reader.read_at(prefix_len, &mut bytes[prefix.len()..])?;
    Ok((bytes, len))
}

/// [`parse_header`] over a file: reads only the prefix and the header, and
/// validates the data section against the file's size. I/O runs with the
/// GIL released.
#[pyfunction]
pub fn parse_file_header(py: Python<'_>, path: PathBuf) -> PyResult<ParsedHeader> {
    let (bytes, file_len) = py
        .detach(|| read_header_prefix(&path))
        .map_err(|e| e.into_py_err(py))?;
    let layout = format::parse_object(&bytes, Some(file_len)).map_err(|e| e.into_py_err(py))?;
    Ok(rows(&layout))
}

/// Build the header for `specs` — `(name, dtype, shape, nbytes)` each — and
/// `metadata` (`None`: no `__metadata__` key; `[]`: an empty one; order is
/// kept). Returns `(header_bytes, order)` where `order` indexes `specs` in the
/// order their buffers follow the header.
#[pyfunction]
#[pyo3(signature = (specs, metadata=None))]
pub fn build_header<'py>(
    py: Python<'py>,
    specs: Vec<SpecRow>,
    metadata: Option<Vec<(String, String)>>,
) -> PyResult<(Bound<'py, PyBytes>, Vec<usize>)> {
    let specs = tensor_specs(&specs).map_err(|e| e.into_py_err(py))?;
    let built = format::build_header(&specs, metadata.as_deref()).map_err(|e| e.into_py_err(py))?;
    Ok((PyBytes::new(py, &built.bytes), built.order))
}
