"""Typed exceptions, one family per domain, mirroring the Rust error enums.

The domain enums (``FormatError``, ``StorageError``, ``PlanError``,
``SelectError``, ``CudaError``) map to the class of the same name. The engines' enums are
composites: a read or write that failed *in* storage or CUDA raises that
domain's class with the engine's message (so a missing file is a
``StorageError`` whichever path met it); a job the engine cannot run at all
raises ``ReadError`` / ``WriteError``.
"""

from __future__ import annotations


class FasterSafetensorsError(Exception):
    """Base of everything this package raises on purpose."""


class FormatError(FasterSafetensorsError, ValueError):
    """The safetensors container grammar was violated."""


class StorageError(FasterSafetensorsError, OSError):
    """Moving bytes failed."""


class SelectError(FasterSafetensorsError, ValueError):
    """A selection does not fit the tensor it names: a shard that does not
    divide the dimension, a box outside the shape, a sub-byte run off a byte
    boundary (the Rust ``SelectError``), or — raised in Python — an index the
    tensor cannot take, a selection for a tensor the request does not load.
    Always before any byte moves."""


class PlanError(FasterSafetensorsError):
    """The request cannot be planned for this machine, or the plan names a
    transport this build cannot execute."""


class ProfileError(FasterSafetensorsError, ValueError):
    """The explicitly selected calibration could not be read or validated."""


class CudaError(FasterSafetensorsError):
    """The CUDA runtime or cuFile refused, or is absent."""


class ReadError(FasterSafetensorsError):
    """The read job and its plan disagree: a destination too small for its
    range, device destinations without staging or a copier, an empty staging
    ring. Names a bug in the caller of ``_core.read_job``, not a bad file."""


class WriteError(FasterSafetensorsError):
    """The write payload is inconsistent: a part whose length disagrees with
    its spec, a device part without a copier, a path with no file name to
    derive a temporary name from."""
