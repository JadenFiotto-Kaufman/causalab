"""The nnterp engine's entry point.

The same shape as the reference engine's: capability and component
declarations for routing, a loader, an executor factory and the shared
execution orchestration. Which components it serves is the capability
registry's ``reads`` cell (``registry.CAPABILITIES``): the whole vocabulary
but the three per-token DeltaNet faces the chunked prefill kernel never
materializes (``delta_kv_mem``, ``delta_state_update``, ``delta_state``),
which route to the reference engine by name. Read-only and swap-only
components and stream constraints are protocol policy (the rows' ``writes``
and ``stream`` cells), not capability gaps, so ``writable_components`` is the
same set. It does not declare ``quantized_weights`` — unverified through
this loader.

It declares ``grad``, wherever its forwards run: a ``train`` document is
fitted by ``train.run_training`` on the shared loop — featurizer slots, fp32
losses, evals on epoch boundaries, so none of ``train_free_params``,
``train_loss_precision`` and ``train_eval_updates``. A ``remote`` engine runs
the whole fit as one NDIF job (``fit.run_fit``): the stages, the optimizer
and the autograd graph live where the model is, and the fitted state comes
home.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping, Sequence

from causalab.neural.engines.nnterp_engine.executor import NnterpExecutor
from causalab.neural.engines.nnterp_engine.loading import NnterpBundle, load_model
from causalab.neural.engines.nnterp_engine.train import run_training
from causalab.neural.shared.execution import TrainOutcome, execute_request
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

__all__ = ["NnterpEngine"]


class NnterpEngine(Engine):
    """The nnterp engine. ``components`` is the registry's row set, the union
    over every registered tree, so a document naming an interior the loaded
    tree has no address for routes here and is refused by name at run time
    (:meth:`NnterpExecutor._address`)."""

    name = "nnterp"
    components = components_served_by("nnterp")
    writable_components = components
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    ) | write_capabilities("nnterp")
    is_local = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        bundle: NnterpBundle | None = None,
        remote: bool | str | None = None,
    ) -> None:
        self.device = device
        self.bundle = bundle
        #: Where the forwards run: ``False`` here, ``True`` (or a host URL) on
        #: NDIF against a weight-free bundle, ``"local"`` through nnsight's
        #: in-process dry run of the remote path against a loaded one.
        #: ``None`` inherits the bundle's own — here for a bundle this engine
        #: loads, on NDIF for a weight-free bundle the caller hands in.
        self.remote = remote

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
            train_runner=self._train,
        )

    def _train(
        self,
        docs: Sequence[Document],
        executors: Sequence[NnterpExecutor],
        request: ExecutionRequest,
    ) -> list[TrainOutcome]:
        return run_training(docs, executors, request)

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
                "this request carries a decoding block, which the nnterp "
                "engine does not serve — its continuation is the greedy decode "
                "alone, and it publishes no continuations file; the reference "
                "engine serves the request",
            )
        if realization.get("quantization") is not None:
            raise ProtocolError(
                "P4",
                "this document declares weight quantization, which the "
                "nnterp engine has not verified through its loader — its "
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
                # "local" dry-runs the remote path in this process, so it
                # needs the weights here
                remote=bool(self.remote) and self.remote != "local",
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
            remote=self.remote,
        )
