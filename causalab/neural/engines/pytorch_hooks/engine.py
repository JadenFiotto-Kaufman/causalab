"""The reference engine: compiled interventions on native pytorch hooks.

Implements the spec §8 services on the two supported architecture families.
All **seven** capabilities in :data:`~causalab.protocol.engine.CAPABILITIES`
— this engine is the only one declaring the full set:

* ``grad`` (the train loop, train.py) — the reason a ``train`` document
  routes here;
* ``paired_forward`` (cross-input operand flow via the lazy group executor);
* ``full_logits`` (lm_head is an ordinary tap wherever the document needs
  the whole projection — a read of every position, a write at the head, a
  decode; a read at named positions is served by projecting the gathered
  ``ln_final`` rows through the same module instead, ``shared/head.py``);
* ``generate`` (the greedy decode in executor.py, which interventions reach
  only through the prefill);
* ``pytorch_fn_local`` (this engine is local);
* ``quantized_weights`` (a document may declare ``model.quantization``);
* ``writable_attention_probs`` — a write to the attention pattern reaches the
  output through the eager attention function rather than a forward hook,
  because the hook fires after the pattern has already been consumed (see
  ``attention_interface.py``).

The engine-neutral half of ``execute`` — metric lowering and the output
tables — lives in :mod:`causalab.neural.shared.execution`; this module keeps
what is genuinely this engine's: its loader, its executor, its train loop.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.neural.engines.pytorch_hooks.executor import Interning, PointExecutor
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.execution import execute_request
from causalab.neural.shared.services import (
    check_caller_bundle,
    load_table,
    load_tensors,
    resolve_roles,
)
from causalab.protocol.canonical import canonical_model
from causalab.protocol.examples import example_labels
from causalab.protocol.engine import (
    CONTINUATIONS_FILE,
    Engine,
    ExecutionRequest,
    RunResult,
)
from causalab.protocol.tables import write_table
from causalab.protocol.registry import components_served_by, write_capabilities
from causalab.protocol.schema import Document

__all__ = ["PytorchHooksEngine"]


class PytorchHooksEngine(Engine):
    name = "pytorch_hooks"
    # The engine-level verbs, plus the write verbs the capability rows charge
    # for the components this engine serves (`writable_attention_probs`: the
    # pattern's write goes through the eager attention function, not a hook).
    capabilities = frozenset(
        {
            "grad",
            "paired_forward",
            "full_logits",
            "generate",
            "pytorch_fn_local",
            "quantized_weights",
        }
    ) | write_capabilities("pytorch_hooks")
    # Which components this engine serves is a row in the capability registry
    # (`registry.CAPABILITIES`, the `reads` cell), not a literal here: the
    # module-boundary and attention-interface vocabulary, writes included, and
    # the routed-expert interior reached by wrapping the grouped
    # experts dispatch. What the rows leave to the nnsight engine:
    # `expert_permutation` (the serving kernel's own bookkeeping, a `.source`
    # line with no dispatch-slot face) and the Gated DeltaNet interior —
    # tensors inside a fused forward where no hook can reach. Read-only /
    # swap-only components and stream constraints are *protocol policy* (the
    # rows' `writes` and `stream` cells, applied by the shared executor and
    # `validate`), not capability gaps: declaring router_logits unwritable
    # here would turn "a write here reaches nothing, write router_scores
    # instead" into "try another engine", the wrong answer for every engine —
    # which is why `writable_components` is the same set.
    components = components_served_by("pytorch_hooks")
    writable_components = components
    is_local = True

    def __init__(
        self,
        *,
        device: str = "cpu",
        cuda_graphs: bool = False,
        batch_rows: int | None = None,
        fit_rows: int | None = None,
        bundle: ModelBundle | None = None,
    ) -> None:
        """``device`` places the model; ``batch_rows`` bounds how many rows one
        no-grad forward covers; ``fit_rows`` bounds how many rows one **grad**
        forward of a fit covers; ``bundle`` is a caller-owned model to run
        instead of loading one.

        The document's ``model.attn_implementation`` selects the attention
        backend. Attention-interior forwards temporarily use eager and restore
        the selection on every exit path.

        Placement and geometry are execution (the engine's call, §8);
        precision is not — dtype and quantization come from each point's own
        ``model`` section. With ``batch_rows`` set, a forward group over more
        rows runs as several forwards over row windows whose captures are
        concatenated in row order (executor.py); the numbers equal the
        single-forward run up to dtype rounding, and nothing about it enters
        a digest or a stamp. It bounds every no-grad forward — document runs
        and ``train.eval`` passes — but not a training minibatch:
        ``train.batch.pairs`` is the document's own batching knob for grad
        forwards, so the execution bound applies to the no-grad passes and to
        ``train.eval``.

        ``fit_rows`` is the grad-forward counterpart: the points of a swept
        campaign that declare ``train`` are fitted together as a cohort, one
        forward per optimizer step over the concatenation of every member's
        minibatch, and ``fit_rows`` bounds how many rows that forward covers
        — the members are packed into forwards under the bound, a member's
        own minibatch (its ``train.batch.pairs`` rows) is never split, and
        ``None`` measures the bound on the cohort's first step from the
        device's free memory (``budget.py``; unbounded off CUDA), reporting
        it as ``execution.fit_rows_resolved``. Like ``batch_rows`` it
        is execution, never identity: nothing about it enters a canonical
        form, a digest or a stamp, and the run receipt (``execution.fit_rows``)
        is its one recorder. Both bounds are this engine's defaults; a
        request's own ``execution`` block overrides either for that request
        (:meth:`effective_batch_rows`, :meth:`effective_fit_rows`).

        With ``bundle`` set (built by :meth:`ModelBundle.from_model`, spec §9),
        the executor runs that model and :func:`load_model` is never called:
        the engine never loads, moves, frees or changes the training mode of
        a caller-owned model,
        and every hook it installs is removed on every exit path. Before any
        forward the document's canonical ``model`` realization is checked
        against the bundle's ``key`` / ``revision`` / ``dtype`` /
        ``quantization`` and ``device`` against this argument
        (:func:`~causalab.neural.shared.services.check_caller_bundle`); a
        disagreement refuses. The run receipt says which way the model came
        in (:attr:`model_source`), and nothing else does.

        Raises:
            ValueError: ``batch_rows`` or ``fit_rows`` is not a positive row
                count. This is the one runtime check; the CLI's argparse type
                refuses the same values before they reach here, and the
                executor and the train loop trust what the engine hands them.
        """
        self.device = device
        self.cuda_graphs = cuda_graphs
        self.batch_rows = _row_bound("batch_rows", batch_rows)
        self.fit_rows = _row_bound("fit_rows", fit_rows)
        self.bundle = bundle

    @property
    def model_source(self) -> str:
        """``"caller"`` when this engine runs a bundle handed to it, else
        ``"loaded"`` — execution provenance for the run receipt (§8), never
        part of a canonical form, a digest or a stamp."""
        return "caller" if self.bundle is not None else "loaded"

    def effective_batch_rows(self, request: ExecutionRequest) -> int | None:
        """The no-grad row bound this ``request`` runs under: its own
        ``execution.batch_rows`` when the key is present — ``None`` there
        meaning unbounded for this request — else this engine's.

        Raises:
            ValueError: the override is not a positive row count.
        """
        return _row_bound(
            "batch_rows", request.execution.get("batch_rows", self.batch_rows)
        )

    def effective_fit_rows(self, request: ExecutionRequest) -> int | None:
        """The grad-forward row bound this ``request``'s fits run under: its
        own ``execution.fit_rows`` when the key is present — ``None`` there
        meaning every cohort member in one forward — else this engine's.

        Raises:
            ValueError: the override is not a positive row count.
        """
        return _row_bound("fit_rows", request.execution.get("fit_rows", self.fit_rows))

    # ------------------------------------------------------------------ #

    def execute(self, request: ExecutionRequest) -> RunResult:
        from causalab.neural.engines.pytorch_hooks.train import run_cohort_training

        # one executor per point, kept so a decoding request's continuations
        # can be published after the campaign ran (workflow spec §2.7)
        executors: list[PointExecutor] = []

        def factory(
            doc: Document,
            req: ExecutionRequest,
            coords: Mapping[str, Any],
            interning: Interning | None,
        ) -> PointExecutor:
            executor = self._executor(doc, req, coords=coords, interning=interning)
            executors.append(executor)
            return executor

        graph_cache = FitGraphCache() if self.cuda_graphs else None
        try:
            result = execute_request(
                request,
                engine_name=self.name,
                executor_factory=factory,
                train_runner=functools.partial(
                    run_cohort_training,
                    fit_rows=self.effective_fit_rows(request),
                    graph_cache=graph_cache,
                ),
                # this engine's executor consults the shared ForwardCache, so it
                # claims §3's cross-point interning and can report what it paid
                intern_forwards=True,
            )
        finally:
            if graph_cache is not None:
                graph_cache.close()
            for executor in executors:
                if isinstance(executor, GraphExecutor):
                    executor.close()
        if request.decoding is None:
            return result
        # a request that declared its decoding gets the continuation as an
        # output — request-keyed, not a `save` kind, so no document digest
        # moves (workflow spec §2.7)
        path = _write_continuations(request, executors)
        return dataclasses.replace(
            result, files={**result.files, CONTINUATIONS_FILE: path}
        )

    # ------------------------------------------------------------------ #

    def _executor(
        self,
        doc: Document,
        request: ExecutionRequest,
        *,
        grad_enabled: bool = False,
        coords: Mapping[str, Any] | None = None,
        interning: Interning | None = None,
    ) -> PointExecutor:
        realization = canonical_model(doc.raw["model"])
        if self.bundle is not None:
            # a caller-owned model: checked against the document, never
            # loaded, never inserted into load_model's cache
            check_caller_bundle(self.bundle, realization, device=self.device)
            bundle = self.bundle
        else:
            bundle = load_model(
                str(doc.model.key),
                str(doc.model.revision),
                dtype=str(realization["dtype"]),
                device=self.device,
                quantization=realization.get("quantization"),
                **(
                    {"attn_implementation": realization["attn_implementation"]}
                    if "attn_implementation" in realization
                    else {}
                ),
            )
        role_rows, role_fields = resolve_roles(doc, request)
        executor = make_executor(
            doc,
            bundle,
            cuda_graphs=self.cuda_graphs,
            decoding=request.decoding,
            role_rows=role_rows,
            role_fields=role_fields,
            load_tensors=functools.partial(load_tensors, request),
            load_table=functools.partial(load_table, request),
            grad_enabled=grad_enabled,
            coords=coords,
            interning=interning,
            batch_rows=self.effective_batch_rows(request),
        )
        # the request's decode spec rides on the executor (None = the argmax)
        executor.decoding = request.decoding
        return executor


def _write_continuations(
    request: ExecutionRequest, executors: Sequence[PointExecutor]
) -> Path:
    """``continuations.json`` under the request's output directory: one row
    per generated row of every decoding group of every point — ``point``,
    ``point_digest``, ``model``, ``input``, ``example``, ``steps`` (the
    budget), ``width`` (tokens before the first EOS), ``truncated``
    (``width == steps``), the real ``token_ids``, the decoded ``text`` and
    each token's ``[start, end)`` char span in it — the
    :class:`~causalab.neural.shared.encoding.Continuation` as a table."""
    rows: list[dict[str, Any]] = []
    for index, executor in enumerate(executors):
        digest = request.digests[index] if index < len(request.digests) else None
        for (model, input_role), continuation in sorted(
            executor.continuations().items()
        ):
            steps = continuation.steps
            input_ids = executor.input_token_ids(input_role)
            eos_ids = list(executor.eos_token_ids())
            labels = example_labels(executor.role_rows[input_role])
            for row, width in enumerate(continuation.widths):
                text = continuation.texts[row] if row < len(continuation.texts) else ""
                offsets = (
                    continuation.offsets[row] if row < len(continuation.offsets) else ()
                )
                rows.append(
                    {
                        "point": index,
                        "point_digest": digest,
                        "model": model,
                        "input": input_role,
                        "example_id": labels[row],
                        "split": executor.role_rows[input_role][row].get("split"),
                        "input_ids": input_ids[row],
                        "steps": steps,
                        "width": int(width),
                        "truncated": int(width) >= steps,
                        "token_ids": continuation.real_ids(row),
                        "greedy_token_id": (
                            int(continuation.token_ids[row, 0])
                            if (request.decoding or {}).get("mode", "deterministic")
                            == "deterministic"
                            else None
                        ),
                        "emitted_ids": [
                            int(t)
                            for t in continuation.token_ids[
                                row, : int(width) + (int(width) < steps)
                            ]
                        ],
                        "terminal_eos_id": (
                            int(continuation.token_ids[row, width])
                            if int(width) < steps
                            else None
                        ),
                        "padding_ids": [
                            int(t)
                            for t in continuation.token_ids[
                                row, int(width) + (int(width) < steps) :
                            ]
                        ],
                        "stop_reason": "eos" if int(width) < steps else "length",
                        "eos_token_ids": eos_ids,
                        "decoding": dict(request.decoding or {"mode": "deterministic"}),
                        "text": text,
                        "offsets": [list(span) for span in offsets[: int(width)]],
                    }
                )
    target = request.output_dir / CONTINUATIONS_FILE
    write_table(target, rows)
    return target


def _row_bound(name: str, value: Any) -> int | None:
    """``value`` as a row bound: a positive ``int`` (never a ``bool``) or
    ``None`` for unbounded; anything else is a :class:`ValueError` naming
    ``name``, whether it came from the constructor or a request's
    ``execution`` block."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive row count, got {value!r}")
    return value
