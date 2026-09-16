//! What this machine has: the two libraries and the device count, as
//! strings `explain()` can print. Loads the libraries and asks their
//! versions; opens no driver, allocates nothing.

use std::fmt;

use crate::ffi;

/// The CUDA picture, for printing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaAvailability {
    /// `Ok`: runtime version and which library name loaded. `Err`: why not.
    pub cudart: Result<String, String>,
    /// `Ok`: cuFile version (when the library reports one) and which library
    /// name loaded. `Err`: why not.
    pub cufile: Result<String, String>,
    /// Devices `cudaGetDeviceCount` reports; 0 when the runtime is absent or
    /// has no driver to talk to.
    pub devices: u32,
}

impl fmt::Display for CudaAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn line(f: &mut fmt::Formatter<'_>, name: &str, r: &Result<String, String>) -> fmt::Result {
            match r {
                Ok(v) => writeln!(f, "{name}: {v}"),
                Err(e) => writeln!(f, "{name}: absent ({e})"),
            }
        }
        line(f, "cudart", &self.cudart)?;
        line(f, "cufile", &self.cufile)?;
        write!(f, "devices: {}", self.devices)
    }
}

/// Probe the usual library locations.
pub fn probe() -> CudaAvailability {
    probe_from(ffi::CUDART_CANDIDATES, ffi::CUFILE_CANDIDATES)
}

/// [`probe`] with explicit candidate lists.
pub fn probe_from(cudart: &[&str], cufile: &[&str]) -> CudaAvailability {
    let (cudart, devices) = match ffi::Cudart::load(cudart) {
        Ok(lib) => {
            let version = match lib.runtime_version() {
                Ok(v) => format!("{}.{}", v / 1000, (v % 1000) / 10),
                Err(e) => format!("version unknown: {e}"),
            };
            let devices = lib.device_count().unwrap_or(0);
            (Ok(format!("{version} via {}", lib.loaded_from)), devices)
        }
        Err(e) => (Err(e.to_string()), 0),
    };
    let cufile = ffi::CuFileLib::load(cufile)
        .map(|lib| match lib.version() {
            Some(v) => format!("{} via {}", format_cufile_version(v), lib.loaded_from),
            None => lib.loaded_from.clone(),
        })
        .map_err(|e| e.to_string());
    CudaAvailability {
        cudart,
        cufile,
        devices,
    }
}

/// `cuFileGetVersion` packs `major * 1000 + minor * 10 + patch`
/// (1.15.1 reports 1151).
fn format_cufile_version(v: i32) -> String {
    format!("{}.{}.{}", v / 1000, (v % 1000) / 10, v % 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_libraries_are_reported_not_fatal() {
        let a = probe_from(
            &["/nonexistent/libcudart.so"],
            &["/nonexistent/libcufile.so"],
        );
        assert_eq!(a.devices, 0);
        assert!(
            a.cudart.as_ref().is_err_and(|e| e.contains("libcudart")),
            "{a:?}"
        );
        assert!(
            a.cufile.as_ref().is_err_and(|e| e.contains("libcufile")),
            "{a:?}"
        );
        let text = a.to_string();
        assert!(text.starts_with("cudart: absent ("), "{text}");
        assert!(text.ends_with("devices: 0"), "{text}");
    }

    #[test]
    fn versions_format() {
        assert_eq!(format_cufile_version(1151), "1.15.1");
    }
}
