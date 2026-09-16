//! Mapping the Rust error enums onto `causalab.io.fastersafetensors.errors`.
//!
//! The exception classes are defined in Python (`errors.py`), where they can
//! inherit from both the package base and the matching built-in
//! (`FormatError` is a `ValueError`, `StorageError` an `OSError`). Rust looks
//! them up by name at raise time; the module is already imported whenever
//! `_core` is, so the lookup is a dictionary hit.
//!
//! Domain enums (`FormatError`, `StorageError`, `PlanError`, `SelectError`,
//! `CudaError`) map
//! to the class of the same name. The engines' enums (`ReadError`,
//! `WriteError`) are composites: a variant that wraps a domain error maps to
//! that domain's class, so a missing file is a `StorageError` whichever
//! engine met it; a variant describing a job the engine cannot run (a
//! destination too small, a plan without staging) maps to `ReadError` /
//! `WriteError`. Every `match` is exhaustive so a new variant is a compile
//! error here, not a bare `RuntimeError` at run time.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

const ERRORS_MODULE: &str = "causalab.io.fastersafetensors.errors";

fn class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyType>> {
    py.import(ERRORS_MODULE)?
        .getattr(name)?
        .cast_into::<PyType>()
        .map_err(PyErr::from)
}

/// Raise `causalab.io.fastersafetensors.errors.<name>(message)`; if the class cannot be
/// found the import error itself is raised, never a bare string.
pub fn raise(py: Python<'_>, name: &str, message: String) -> PyErr {
    match class(py, name) {
        Ok(cls) => PyErr::from_type(cls, message),
        Err(err) => err,
    }
}

/// Convert a domain error into its Python exception.
pub trait IntoPyErr {
    /// The exception, with the error's `Display` text as its message.
    fn into_py_err(self, py: Python<'_>) -> PyErr;
}

impl IntoPyErr for fst_core::FormatError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        raise(py, "FormatError", self.to_string())
    }
}

impl IntoPyErr for fst_core::StorageError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        raise(py, "StorageError", self.to_string())
    }
}

impl IntoPyErr for fst_core::PlanError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        raise(py, "PlanError", self.to_string())
    }
}

impl IntoPyErr for fst_core::ProfileError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        raise(py, "ProfileError", self.to_string())
    }
}

impl IntoPyErr for fst_core::SelectError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        raise(py, "SelectError", self.to_string())
    }
}

impl IntoPyErr for fst_cuda::CudaError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        use fst_cuda::CudaError::*;
        // one domain, one class; listed so a new variant is decided here
        let class = match &self {
            Unavailable { .. }
            | Runtime { .. }
            | CuFile { .. }
            | Overrun { .. }
            | BadGeometry { .. }
            | Register { .. }
            | ShortRead { .. } => "CudaError",
        };
        raise(py, class, self.to_string())
    }
}

impl IntoPyErr for fst_core::ReadError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        use fst_core::ReadError::*;
        let message = self.to_string();
        let class = match self {
            Open { .. } | Read { .. } => "StorageError",
            Staging { .. } | Copy { .. } | DirectRead { .. } => "CudaError",
            CopierRequired
            | StagingRequired
            | EmptyStaging { .. }
            | DestinationMismatch { .. }
            | PlacedDestinationMismatch { .. }
            | InvalidPlacement { .. }
            | PieceExceedsStaging { .. } => "ReadError",
        };
        raise(py, class, message)
    }
}

impl IntoPyErr for fst_core::WriteError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        use fst_core::WriteError::*;
        let message = self.to_string();
        let class = match self {
            Format(_) => "FormatError",
            Storage { .. } => "StorageError",
            Cuda { .. } => "CudaError",
            PartCount { .. }
            | PartLength { .. }
            | DeviceOverrun { .. }
            | NoCopier { .. }
            | EmptyStaging { .. }
            | NoFileName { .. } => "WriteError",
        };
        raise(py, class, message)
    }
}

/// What can go wrong between the Python call and the engine.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// The read engine refused or failed.
    #[error(transparent)]
    Read(#[from] fst_core::ReadError),
    /// The write engine refused or failed.
    #[error(transparent)]
    Write(#[from] fst_core::WriteError),
    /// The CUDA runtime could not be loaded for the device asked for.
    #[error(transparent)]
    Cuda(#[from] fst_cuda::CudaError),
    /// The caller's arguments do not describe a valid job.
    #[error("{0}")]
    Invalid(String),
}

impl IntoPyErr for IoError {
    fn into_py_err(self, py: Python<'_>) -> PyErr {
        match self {
            IoError::Read(err) => err.into_py_err(py),
            IoError::Write(err) => err.into_py_err(py),
            IoError::Cuda(err) => err.into_py_err(py),
            IoError::Invalid(message) => PyValueError::new_err(message),
        }
    }
}
