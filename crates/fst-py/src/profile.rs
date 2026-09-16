//! Runtime profile selection shared by planning and selection coalescing.
//!
//! Hardware names alone do not identify a storage calibration. Selection is
//! explicit, via a JSON path in FASTERSAFETENSORS_PROFILE. This is an
//! experimental override, not evidence that the calibration fits this host.

use std::ffi::OsStr;
use std::path::Path;

use fst_core::profile::Profile;
use pyo3::prelude::*;

use crate::errors::IntoPyErr;

fn selected(value: Option<&OsStr>) -> Result<Profile, fst_core::ProfileError> {
    let base = Profile::default_measured();
    let Some(value) = value else {
        let mut profile = base;
        profile.source = format!(
            "{} (FASTERSAFETENSORS_PROFILE unset; no hardware auto-detection)",
            profile.source
        );
        return Ok(profile);
    };
    let over = Profile::from_path(Path::new(value))?;
    let mut profile = Profile::merge(&base, &over);
    profile.source = format!(
        "FASTERSAFETENSORS_PROFILE={value:?} (explicit override; applicability not verified): {}",
        profile.source
    );
    Ok(profile)
}

pub fn current(py: Python<'_>) -> PyResult<Profile> {
    py.detach(|| selected(std::env::var_os("FASTERSAFETENSORS_PROFILE").as_deref()))
        .map_err(|e| e.into_py_err(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_explicit_and_missing_profile_is_an_error() {
        assert!(selected(None).unwrap().source.contains("unset"));
        assert!(selected(Some(OsStr::new("/nonexistent/fst-profile.json"))).is_err());
    }
}
