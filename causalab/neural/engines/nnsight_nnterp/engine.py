"""The nnsight + nnterp engine's entry point.

The same shape as the other engines': capability and component declarations
for routing, a loader, an executor factory and the shared execution
orchestration. It is **not registered** in the closed engine registry
(``registry.ENGINES``): its component set is computed from the family taps
it can land rather than read from a capability row, and routing to it is a
caller's explicit choice (``run_protocol(..., [NnterpEngine()])``).
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

from causalab.neural.engines.nnsight_nnterp.executor import NnterpExecutor
from causalab.neural.engines.nnsight_nnterp.loading import NnterpBundle, load_model
from causalab.neural.engines.nnsight_nnterp.sources import components_addressed
from causalab.neural.shared.execution import execute_request
from causalab.neural.shared.services import (
    check_caller_bundle,
    load_table,
    load_tensors,
    resolve_roles,
)
from causalab.protocol.canonical import canonical_model
from causalab.protocol.engine import Engine, ExecutionRequest, RunResult
from causalab.protocol.errors import ProtocolError
from causalab.protocol.registry import CAPABILITIES, FAMILIES
from causalab.protocol.schema import Document

__all__ = ["NnterpEngine", "module_boundary_components", "served_components"]


def module_boundary_components() -> frozenset[str]:
    """Every component some registered family taps at a module boundary —
    what this engine can land. A ``from_row`` tap is one too: the resolver
    reads its child off the row's per-family address (the pre-RoPE
    projections, the value states, the gate), and the executor lands it as
    any other envoy side."""
    return frozenset(
        component
        for adapter in FAMILIES.values()
        for component, tap in adapter.taps.items()
        if tap.kind in ("in", "out")
    )


def served_components() -> frozenset[str]:
    """What this engine lands: every module boundary, plus every interior the
    address table reaches through ``.source``
    (:func:`~causalab.neural.engines.nnsight_nnterp.sources.components_addressed`)."""
    return module_boundary_components() | components_addressed()


def _write_verbs(components: frozenset[str]) -> frozenset[str]:
    """The coarse §8 verbs the rows charge for writes at ``components``."""
    return frozenset(
        row.write_capability
        for component, row in CAPABILITIES.items()
        if component in components and row.write_capability is not None
    )


class NnterpEngine(Engine):
    """The nnsight + nnterp engine. ``components`` is :func:`served_components`,
    the union over every registered tree, so a document naming an interior the
    loaded tree has no address for routes here and is refused by name at run
    time (:meth:`NnterpExecutor._address`)."""

    name = "nnsight_nnterp"
    components = served_components()
    writable_components = served_components()
    capabilities = frozenset(
        {"paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    ) | _write_verbs(served_components())
    is_local = True

    def __init__(
        self, *, device: str = "cpu", bundle: NnterpBundle | None = None
    ) -> None:
        self.device = device
        self.bundle = bundle

    @property
    def model_source(self) -> str:
        return "caller" if self.bundle is not None else "loaded"

    def execute(self, request: ExecutionRequest) -> RunResult:
        return execute_request(
            request,
            engine_name=self.name,
            executor_factory=lambda doc, req, coords, _interning: self._executor(
                doc, req, coords=coords
            ),
            train_runner=None,
        )

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        coords: Mapping[str, Any] | None = None,
    ) -> NnterpExecutor:
        realization = canonical_model(doc.raw["model"])
        if request.decoding is not None:
            raise ProtocolError(
                "P4",
                "this request carries a decoding block, which the nnsight_nnterp "
                "engine does not serve — its continuation is the greedy decode "
                "alone, and it publishes no continuations file; the reference "
                "engine serves the request",
            )
        if realization.get("quantization") is not None:
            raise ProtocolError(
                "P4",
                "this document declares weight quantization, which the "
                "nnsight_nnterp engine has not verified through its loader — "
                "the reference engine serves it",
            )
        if self.bundle is not None:
            check_caller_bundle(self.bundle, realization, device=self.device)
            bundle = self.bundle
        else:
            bundle = load_model(
                str(doc.model.key),
                str(doc.model.revision),
                dtype=str(realization["dtype"]),
                device=self.device,
                **(
                    {"attn_implementation": realization["attn_implementation"]}
                    if "attn_implementation" in realization
                    else {}
                ),
            )
        role_rows, role_fields = resolve_roles(doc, request)
        return NnterpExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, request),
            load_table=functools.partial(load_table, request),
            coords=coords,
        )
