"""An in-process NDIF that runs the *deserialized* program.

nnsight's ``remote="local"`` serializes a block, deserializes it with the
caller's own modules hidden, and then **executes the original tracer against
the original frame** — so it checks pickling and imports, and nothing past
them: the block still reads the client's objects, runs on the client's model
and leaves its results in the client's variables. None of what separates a
client from a server is exercised.

:class:`FaithfulServer` stands in for the network instead. The client is
what production uses — a weight-free bundle under ``remote=True`` — and the
"server" is a separately loaded model. nnsight's
:class:`~nnsight.intervention.backends.remote.RemoteBackend` keeps its own
serialization and its own push into the caller's frame; only ``request`` (the
websocket round trip) is replaced, by what an NDIF model actor does with the
payload:

* deserialize against **the server model's** persistent objects, the client's
  non-installed modules hidden (``causalab`` is installed where a real block
  runs, so it stays importable);
* run the **restored** tracer on its **restored** frame, bracketed in a trace
  scope, and collect the block variables marked by ``nnsight.save`` by
  identity — NDIF's ``execute_traced_block``;
* send the saves home through ``torch.save`` / ``torch.load`` onto the CPU —
  the result blob.

So a block that reads a client object, mutates one, saves into a slot no
block variable names, or flips the client's config instead of the server's
fails here as it fails on NDIF.
"""

from __future__ import annotations

import dataclasses
import io
import linecache
from typing import Any

import pytest
import torch

__all__ = ["FaithfulServer", "Job"]


@dataclasses.dataclass(frozen=True)
class Job:
    """One request the server ran: the payload's size as pickled and as
    NDIF's zstd level compresses it, and the names that came back."""

    raw_bytes: int
    zstd_bytes: int
    returned: tuple[str, ...]


class FaithfulServer:
    """Route every ``remote=True`` run of this process to ``model``.

    ``model`` is the server's nnsight model (a bundle's ``.model``), loaded
    apart from whatever the client holds. ``jobs`` records each request.
    """

    def __init__(self, model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from nnsight.intervention.backends import local, remote

        self.model = model
        self.jobs: list[Job] = []
        monkeypatch.setattr(
            local, "_SERVER_MODULES", {*local._SERVER_MODULES, "causalab"}
        )
        server = self

        def request(backend: Any, _request: Any, tracer: Any) -> dict[str, Any]:
            return server.run(backend, tracer)

        monkeypatch.setattr(remote.RemoteBackend, "request", request)

    def run(self, backend: Any, tracer: Any) -> dict[str, Any]:
        """What a model actor does with one request."""
        import zstandard
        from nnsight.intervention.backends.local import LocalSimulationBackend
        from nnsight.schema.request import RequestModel
        from nnsight.tracing.tracer import _saves, dec, inc

        blob = backend._serialize(tracer)
        raw = RequestModel.serialize(tracer, compress=False)
        hider = LocalSimulationBackend(self.model)
        # deserialize registers the block's source under the client's
        # filename; a real server is another process, so put the entry back
        filename = tracer.info.frame.f_code.co_filename
        line = linecache.cache.get(filename)
        hidden = hider._hide_local_modules()
        try:
            restored = RequestModel.deserialize(
                blob,
                self.model._remoteable_persistent_objects(),
                compress=backend.compress,
            )
        finally:
            hider._restore(hidden)
            if line is not None:
                linecache.cache[filename] = line
            else:
                linecache.cache.pop(filename, None)

        inc()
        try:
            restored.execute(restored.info.code)
            saves = _saves()
            saved = {
                name: value
                for name, value in restored.info.frame.f_locals.items()
                if id(value) in saves
            }
        finally:
            dec()

        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        result = torch.load(buffer, map_location="cpu", weights_only=False)
        self.jobs.append(
            Job(
                raw_bytes=len(raw),
                zstd_bytes=len(zstandard.ZstdCompressor(level=6).compress(raw)),
                returned=tuple(sorted(result)),
            )
        )
        return result
