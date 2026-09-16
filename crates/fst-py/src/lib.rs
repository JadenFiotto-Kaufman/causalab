//! `causalab.io.fastersafetensors._core`: the Python face of `fst-core`.
//!
//! Thin by design. Every function here is a direct wrapper over one `fst-core`
//! operation: the header grammar ([`header`]), the planner and the probe
//! ([`planning`]), selections resolved to reads ([`select`]), and byte
//! movement into caller-owned buffers ([`io`]). All
//! I/O runs with the GIL released (`Python::detach`); errors cross the
//! boundary as the exception classes of `causalab.io.fastersafetensors.errors`
//! ([`errors`]).
//!
//! **Ownership.** Rust never allocates a tensor. The Python layer allocates
//! every destination and source with torch and passes addresses (`read_job`,
//! `write_object`); the engines fill or drain them and hold nothing
//! afterwards. The CUDA runtime, when a job touches a device, is loaded once
//! per device and kept ([`io`]).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

mod errors;
mod header;
mod io;
mod planning;
mod profile;
mod select;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(header::parse_header, m)?)?;
    m.add_function(wrap_pyfunction!(header::parse_file_header, m)?)?;
    m.add_function(wrap_pyfunction!(header::build_header, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_job, m)?)?;
    m.add_function(wrap_pyfunction!(io::write_object, m)?)?;
    m.add_function(wrap_pyfunction!(planning::probe_env, m)?)?;
    m.add_function(wrap_pyfunction!(planning::storage_classes, m)?)?;
    m.add_function(wrap_pyfunction!(planning::plan_read, m)?)?;
    m.add_function(wrap_pyfunction!(planning::probe_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(select::select_reads, m)?)?;
    m.add_function(wrap_pyfunction!(select::shard_ranges, m)?)?;
    Ok(())
}
