//! The probe and the planner, as dictionaries Python can print.

use std::path::PathBuf;

use fst_core::env::Env;
use fst_core::plan::{self, Destination, FileSpec, ReadRequest, Staging, Transport};
use fst_core::select::CoalesceSummary;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::IntoPyErr;

/// Observe this machine. Keys: `mounts` (`[(point, fs_type)]`, longest
/// mount points first), `nvidia_fs_loaded`, `cufile_library` (path or
/// `None`), `cpus`.
#[pyfunction]
pub fn probe_env(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let env = py.detach(Env::probe);
    env_dict(py, &env)
}

fn env_dict<'py>(py: Python<'py>, env: &Env) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let mounts: Vec<(String, String)> = env
        .mounts
        .iter()
        .map(|m| (m.point.display().to_string(), m.fs_type.clone()))
        .collect();
    dict.set_item("mounts", mounts)?;
    dict.set_item("nvidia_fs_loaded", env.gds.module_loaded)?;
    dict.set_item(
        "cufile_library",
        env.gds
            .cufile_library
            .as_ref()
            .map(|p| p.display().to_string()),
    )?;
    dict.set_item("cpus", env.cpus)?;
    Ok(dict)
}

/// What CUDA this machine has, for `explain()`: `cudart` and `cufile` are
/// `(available, detail)` — the version and which library loaded, or why
/// not — and `devices` is what `cudaGetDeviceCount` reports. Loads the
/// libraries; opens no driver, allocates nothing.
#[pyfunction]
pub fn probe_cuda(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let cuda = py.detach(fst_cuda::probe);
    let dict = PyDict::new(py);
    let flat = |r: &Result<String, String>| match r {
        Ok(text) => (true, text.clone()),
        Err(text) => (false, text.clone()),
    };
    dict.set_item("cudart", flat(&cuda.cudart))?;
    dict.set_item("cufile", flat(&cuda.cufile))?;
    dict.set_item("devices", cuda.devices)?;
    Ok(dict)
}

/// The storage class of each path, by the planner's names (`LocalBlock`,
/// `Nfs`, `Fuse`, `Ram`, `OtherNetwork("lustre")`, `Other("")`).
#[pyfunction]
pub fn storage_classes(py: Python<'_>, paths: Vec<PathBuf>) -> Vec<String> {
    let env = py.detach(Env::probe);
    paths
        .iter()
        .map(|p| format!("{:?}", env.storage_class(p)))
        .collect()
}

/// `"cpu"` / `None` is the host; `"cuda"` or `"cuda:N"` a device, with
/// `free_bytes` as caller-reported aggregate headroom (unknown: no fit check).
fn destination(device: Option<&str>, free_bytes: Option<u64>) -> PyResult<Destination> {
    let Some(device) = device else {
        return Ok(Destination::Host);
    };
    if device == "cpu" {
        return Ok(Destination::Host);
    }
    let Some(rest) = device.strip_prefix("cuda") else {
        return Err(PyValueError::new_err(format!(
            "device must be 'cpu', 'cuda' or 'cuda:N', got {device:?}"
        )));
    };
    let ordinal = match rest.strip_prefix(':') {
        None if rest.is_empty() => 0,
        Some(n) => n
            .parse::<u32>()
            .map_err(|_| PyValueError::new_err(format!("bad device ordinal in {device:?}")))?,
        None => {
            return Err(PyValueError::new_err(format!(
                "device must be 'cpu', 'cuda' or 'cuda:N', got {device:?}"
            )));
        }
    };
    Ok(Destination::Device {
        device: ordinal,
        free_bytes: free_bytes.unwrap_or(u64::MAX),
    })
}

/// Plan a read of `paths` (with `wanted_bytes[i]` bytes wanted from each)
/// into `device` on this machine. Keys: `files_in_flight`, `split_bytes`
/// (0: unsplit), `readers_per_file`, `transport` (`"pread"`, `"mmap"`,
/// `"cufile"`), `staging` (`None` or `(count, bytes)` of pinned buffers),
/// `reasons` (one line per decision). `coalesced` is `(runs, reads,
/// wanted_bytes, read_bytes)` as `select_reads` summarised the request's
/// selections; when it merged anything the reasons say so and by how much.
/// `allocation_bytes` overrides storage bytes for the device fit check,
/// accounting for destination tensors and any simultaneously live scratch.
#[pyfunction]
#[pyo3(signature = (paths, wanted_bytes, device=None, free_bytes=None, coalesced=None, allocation_bytes=None))]
pub fn plan_read<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    wanted_bytes: Vec<u64>,
    device: Option<&str>,
    free_bytes: Option<u64>,
    coalesced: Option<(usize, usize, u64, u64)>,
    allocation_bytes: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    if paths.len() != wanted_bytes.len() {
        return Err(PyValueError::new_err(format!(
            "{} paths but {} wanted_bytes",
            paths.len(),
            wanted_bytes.len()
        )));
    }
    let destination = destination(device, free_bytes)?;
    let request = ReadRequest {
        files: paths
            .into_iter()
            .zip(wanted_bytes)
            .map(|(path, wanted_bytes)| FileSpec { path, wanted_bytes })
            .collect(),
        destination,
        allocation_bytes,
        coalesced: coalesced.map(|(runs, reads, wanted_bytes, read_bytes)| CoalesceSummary {
            runs,
            reads,
            wanted_bytes,
            read_bytes,
        }),
    };
    let env = py.detach(Env::probe);
    let profile = crate::profile::current(py)?;
    let plan = plan::plan_read(&env, &profile, &request).map_err(|e| e.into_py_err(py))?;
    let dict = PyDict::new(py);
    dict.set_item("files_in_flight", plan.files_in_flight)?;
    dict.set_item("split_bytes", plan.split_bytes)?;
    dict.set_item("readers_per_file", plan.readers_per_file)?;
    dict.set_item(
        "transport",
        match plan.transport {
            Transport::Pread => "pread",
            Transport::Mmap => "mmap",
            Transport::CuFile => "cufile",
        },
    )?;
    dict.set_item(
        "staging",
        match plan.staging {
            Staging::None => None,
            Staging::Pinned { count, bytes } => Some((count, bytes)),
        },
    )?;
    dict.set_item("reasons", plan.reasons)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_strings() {
        assert_eq!(destination(None, None).unwrap(), Destination::Host);
        assert_eq!(destination(Some("cpu"), None).unwrap(), Destination::Host);
        assert_eq!(
            destination(Some("cuda"), Some(7)).unwrap(),
            Destination::Device {
                device: 0,
                free_bytes: 7
            }
        );
        assert_eq!(
            destination(Some("cuda:3"), None).unwrap(),
            Destination::Device {
                device: 3,
                free_bytes: u64::MAX
            }
        );
        pyo3::Python::initialize();
        pyo3::Python::attach(|_py| {
            assert!(destination(Some("mps"), None).is_err());
            assert!(destination(Some("cuda:x"), None).is_err());
        });
    }
}
