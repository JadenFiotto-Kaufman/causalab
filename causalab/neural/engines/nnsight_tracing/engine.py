"""The nnsight engine's entry point.

Same shape as the reference engine's: capability and component declarations
for routing, a loader, an executor factory, and the shared execution
orchestration. What it does **not** declare says as much as what it does:

* ``grad`` — training through traces is real design work;
  ``train`` documents route to the reference engine.
* ``quantized_weights`` — unverified through nnsight's loader; refused until
  someone needs it and proves it.

The components only this engine serves — the fused-forward interiors of the
experts and the DeltaNet kernel, and the decode-side DeltaNet state — route
here by name: the
reference engine simply does not declare them.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import NnsightBundle, load_model
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
from causalab.protocol.registry import components_served_by, write_capabilities
from causalab.protocol.schema import Document

__all__ = ["NnsightEngine"]


class NnsightEngine(Engine):
    name = "nnsight"
    capabilities = frozenset(
        {
            "paired_forward",
            "full_logits",
            "pytorch_fn_local",
            # continuation reads through one model.generate trace, decode
            # steps walked with tracer.iter; writes stay in the prefill,
            # as everywhere
            "generate",
        }
        # the write verbs the capability rows charge for the components this
        # engine serves — the pattern's write lands on the softmax's output
        # *inside* the eager function ('attn_weights_2' in the attention address
        # table), where the value multiply consumes it; a write to the mixer's
        # returned attn_weights would reach nothing
    ) | write_capabilities("nnsight")
    # Which components this engine serves is a row in the capability registry
    # (`registry.CAPABILITIES`, the `reads` cell): the whole vocabulary but
    # the `delta_*` set, like the reference engine — module
    # boundaries land on envoys, the attention interior through the `.source`
    # address table, and 'attention_result' (derived by re-invoking the
    # o-projection) works because an envoy outside a trace calls its
    # underlying module. The per-expert MoE interior and the DeltaNet
    # interior are the vocabulary only this engine serves; routing lands
    # them here by name. Read-only / swap-only components and stream
    # constraints are *protocol policy* (the rows' `writes` and `stream`
    # cells), not capability gaps — the same argument the reference engine's
    # declaration makes, and why `writable_components` is the same set.
    #
    # The `delta_*` vocabulary is the *reference engine's* DeltaNet
    # interior: the kernel boundary is reached by swapping the modeling
    # file's module globals for the dynamic extent of one mixer forward, and
    # the per-step interior by stepping the recurrent kernel inside the
    # swapped globals — pytorch_hooks mechanisms with no nnsight equivalent
    # (this engine's DeltaNet interior is the `deltanet_*` set). The
    # `delta_*` module-boundary taps (qkv, gate, premix) are ordinary envoy
    # reads and would very likely work here unchanged — but nothing exercises
    # them on this engine, and declaring support this engine has never been
    # tested for is the claim worth not making.
    components = components_served_by("nnsight")
    writable_components = components
    is_local = True

    def __init__(
        self, *, device: str = "cpu", bundle: NnsightBundle | None = None
    ) -> None:
        # placement is execution (the engine's call, §8); precision is not —
        # dtype comes from each point's own `model` section. `bundle` is a
        # caller-owned model run instead of loading one (spec §9): checked
        # against each document's realization before any forward, never
        # loaded, moved, freed or re-moded by this engine
        self.device = device
        self.bundle = bundle

    @property
    def model_source(self) -> str:
        """``"caller"`` when this engine runs a bundle handed to it, else
        ``"loaded"`` — execution provenance for the run receipt (§8)."""
        return "caller" if self.bundle is not None else "loaded"

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        return execute_request(
            request,
            engine_name=self.name,
            # the trace executor does not consult the shared ForwardCache, so
            # it takes the campaign's interning handle and drops it; §3's
            # cross-point sharing is unclaimed here and RunResult.forwards
            # stays 0 rather than reporting a count nothing measured
            executor_factory=lambda doc, req, coords, _interning: self._executor(
                doc, req, coords=coords
            ),
            train_runner=None,
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        coords: Mapping[str, Any] | None = None,
    ) -> TracePointExecutor:
        realization = canonical_model(doc.raw["model"])
        if realization.get("quantization") is not None:
            raise ProtocolError(
                "P4",
                "this document declares weight quantization, which the "
                "nnsight engine has not verified through its loader — its "
                "'quantized_weights' capability is absent, so routing should "
                "not have sent it here; the reference engine serves it",
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
        return TracePointExecutor(
            doc,
            bundle,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, request),
            load_table=functools.partial(load_table, request),
            coords=coords,
        )
