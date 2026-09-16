"""A parsed header as Python sees it: where every tensor is and what it is."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import prod

import torch

import causalab.io.fastersafetensors._core as _core
from ._dtypes import torch_dtype

PathLike = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One tensor of a header, with its byte range absolute within the object."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.dtype)

    @property
    def itemsize(self) -> int:
        return self.torch_dtype.itemsize

    def empty(self, device: torch.device) -> torch.Tensor:
        """An uninitialised destination tensor of this entry's dtype and shape."""
        return torch.empty(self.shape, dtype=self.torch_dtype, device=device)


@dataclass(frozen=True, slots=True)
class Layout:
    """A parsed header: tensors by name (sorted), metadata as written."""

    header_len: int
    tensors: dict[str, TensorEntry]
    metadata: dict[str, str] | None

    @classmethod
    def _from_rows(
        cls,
        rows: tuple[
            int,
            list[tuple[str, str, list[int], int, int]],
            list[tuple[str, str]] | None,
        ],
    ) -> Layout:
        header_len, tensors, metadata = rows
        entries = {
            name: TensorEntry(name, dtype, tuple(shape), start, end)
            for name, dtype, shape, start, end in tensors
        }
        return cls(header_len, entries, None if metadata is None else dict(metadata))

    @classmethod
    def from_bytes(cls, data: bytes) -> Layout:
        """Parse a whole in-memory object; raises ``FormatError`` on truncation."""
        return cls._from_rows(_core.parse_header(data))

    @classmethod
    def from_file(cls, path: PathLike) -> Layout:
        """Parse a file's header, reading only the header; the data section is
        validated against the file size."""
        return cls._from_rows(_core.parse_file_header(os.fspath(path)))

    @property
    def data_start(self) -> int:
        return 8 + self.header_len

    def in_data_order(self) -> list[TensorEntry]:
        """Tensors in the order their bytes appear in the data section."""
        return sorted(self.tensors.values(), key=lambda t: (t.start, t.end, t.name))

    def wanted_bytes(self, names: list[str] | None = None) -> int:
        """Bytes a read of ``names`` (all when ``None``) touches."""
        entries = (
            self.tensors.values() if names is None else (self.tensors[n] for n in names)
        )
        return sum(t.nbytes for t in entries)
