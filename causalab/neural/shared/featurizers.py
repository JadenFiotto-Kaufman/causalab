"""Featurizer kinds (spec §2.5): ``featurize(x) → (f, err)``,
``inverse(f, err) → x̂``, per kind, plus composition.

The error-term contract is the load-bearing rule: ``err`` and unselected
dims always come from the **pre-write** value at the address, so a zero
write ablates only the feature contribution and a ``dims`` write is a
subspace swap. One deliberate interpretation (surfaced in the PR): the
spec's kind table writes ``(Qᵀx, 0)`` for ``subspace``, but the contract
paragraph requires the orthogonal complement to survive a swap — so here
``err = x − QQᵀx`` (identically 0 when ``k = d``), matching the oracle's
lossy-split behavior and DAS semantics.

``gate`` is soft during training (``σ(θ/T) ⊙ x``, temperature annealed by
the train loop) and **hard** in eval (``θ > 0``) — the parity oracle's
mask mode pins the hard-eval split. A gate loaded from a fitted ``theta``
(:meth:`Gate.from_theta`) is that eval-mode object and nothing else, so a
DBM fit has a held-out *apply* pass in the same shape DAS does. A gate
declared with ``group: head`` holds one ``theta`` per head of a head-major
component and expands it over the head's coordinates at featurize time
(``repeat_interleave``), so the mask it fits selects heads. A gate declared
with ``group: expert_neuron`` on ``expert_activation`` or
``expert_neuron_output`` holds the whole expert table,
``theta[expert, neuron]``. A token's routed slots look their rows up through
the routing table handed in beside the activation
(``featurize(x, routing=expert_idx)``), so the same neuron of the same expert
has one parameter whichever slot it fills for a token. Both maps come from the
model and the site (:func:`causalab.protocol.registry.gate_group_map`), never
from the document.

Everything here is per-position math on ``(..., d)`` tensors; widths come
from the resolved site, never from the document.

**Device.** Stages are built on CPU and moved to the run's device by
:func:`build_stack`. Building on CPU is deliberate: a ``subspace`` init draws
from a CPU generator, so a seeded init stays bit-identical across devices.
Dtype is *not* forced — every stage casts at the boundary, so featurizers
stay fp32 against a bf16 backbone.

**Seed.** ``subspace`` is the only kind with a random init, and every draw it
makes — the starting rotation, and the completion of the ``matrix_exp`` /
``stiefel`` base, which torch would take from the global RNG — is its own
*local* generator's, so its rotation cannot depend on build order, on whether
a train loop ran, or on whoever else shares the process; building one leaves
the global RNG exactly as it was. :func:`build_stack` takes the ``seed``; the executor resolves it from ``train.seed`` (0 when the document
declares no fit). ``gate`` inits to zeros, or loads a fitted ``theta``; the rest load from
files. A ``hard_concrete`` gate is the one stage that draws *during* a fit —
its training mask is a sample — and it draws only when the train loop hands it
a generator (:meth:`Gate.resample`, once per optimizer step, from the fit's
own CPU generator), never from the global RNG: so the sample is one draw shared
by every read and write the gate sits on in that step, a member of a cohort
samples independently of what fits beside it, and a fit reproduces across
devices.

**Init.** A ``subspace`` may instead *start* from a saved basis (§2.5
``init`` — the first ``k`` columns of a PCA fit at the same site). The seed
then completes that basis to a full frame rather than drawing the start, and
the stage records where it started (:meth:`Subspace.identity_fields`) so the
saved bundle names the basis. Without ``init`` nothing here changes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import torch
import torch.nn.utils.parametrize

from causalab.protocol.bundles import entry_selection
from causalab.protocol.errors import ProtocolError, ValidationError
from causalab.protocol.registry import ModelInfo, gate_group_map, gate_param_shape
from causalab.protocol.schema import (
    GATE_DEAD_RULES,
    GATE_PARAMETRIZATIONS,
    HARD_CONCRETE_STRETCH,
    HARD_CONCRETE_TEMPERATURE,
    FEATURIZER_SLOTS,
    FeaturizerSpec,
    hard_concrete_theta,
    hard_concrete_threshold,
)
from causalab.protocol.shapes import FeatureShape

__all__ = [
    "FeaturizerStack",
    "Stage",
    "StageRecipe",
    "build_recipe",
    "build_stack",
    "featurizer_cache",
]

#: How far ``PᵀP`` may sit from the identity before a saved basis is refused
#: as a ``subspace`` start (:func:`_init_basis`). Loose enough for the fp32
#: bases the pipeline writes — ``fit_pca`` (~1e-6) and a fitted ``subspace``
#: (~1e-6 in the regime fits live in; :class:`Cayley`, *Conditioning*, for
#: the degenerate one that can exceed this) — tight enough that a basis which
#: is merely *close* to a frame — a bf16-rounded one (≈ 3e-3), a hand-scaled
#: one — is caught. The writer side records every fitted rotation's deviation
#: and its verdict against this bar as ``orthonormality_deviation`` /
#: ``within_tolerance`` in ``fit_diagnostics.json``
#: (:func:`~causalab.neural.shared.training.diagnostics.fit_diagnostics`),
#: so a start refused here can be traced to the fit that produced it.
ORTHONORMAL_TOLERANCE = 1e-4


def orthonormality_deviation(q: torch.Tensor) -> float:
    """``max|QᵀQ − I|`` in fp32 — the one quantity :data:`ORTHONORMAL_TOLERANCE`
    is the bar for, so it lives beside it rather than being respelled at each
    of its call sites: :func:`_init_basis` at load, :meth:`Cayley.right_inverse`
    on assignment, and the train loop's ``fit_diagnostics`` at save."""
    with torch.no_grad():
        columns = q.detach().to(torch.float32)
        gram = columns.mT @ columns
        eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        return float((gram - eye).abs().max())


@dataclasses.dataclass
class _Scope:
    """The open :func:`featurizer_cache` scopes: how many are nested, and
    their shared store — ``None`` outside any scope. Keyed by the stage module
    itself, not its ``id``, so an entry keeps its stage alive for the scope
    and a stage built inside the scope can never inherit a freed one's
    address."""

    depth: int = 0
    entries: dict[tuple[Any, ...], Any] | None = None


_SCOPE = _Scope()


@contextlib.contextmanager
def featurizer_cache(*, isolated: bool = False) -> Iterator[None]:
    """Evaluate each stage's derived quantity — a ``subspace``'s rotation
    ``Q``, a ``gate``'s mask — **once** for the block where that changes no
    bit of any result, forward or backward.

    A trained stage is a function of its free parameter, recomputed on every
    access: an orthogonal parametrization on every ``.weight`` read, a gate's
    ``σ(θ/T)`` on every ``featurize``. One optimizer step accesses a member's
    featurizer several times over — its read, and featurize *and* inverse on
    each write, on every hooked layer — and every evaluation is a run of tiny
    kernels (the Cayley map: a k×k inverse and a handful of k×k products) on
    a step loop that is host-bound. Profiled on the standard A3B workflow the
    ``cayley`` map issued more launches per DAS step than the whole MoE
    forward. The train loop opens one scope around a step's eager grad
    forwards and loss, and one around an eval round (an engine's ``train.py``,
    ``shared/training/executors.py``); a CUDA
    graph capture opens one per captured pass (``cuda_graphs.Replay``).

    **What is shared, and why it is exact.** Under ``no_grad`` — the eval
    passes, a CUDA-graph executor's frozen-input pass — every stage's value
    is computed once and reused: the same computation, its result read
    several times. Under grad the value is the same but the *gradient* is
    not, if one tensor serves several consumers: autograd then sums the
    consumers' cotangents at the shared tensor and runs the map's backward
    once — ``Jᵀ(Σcᵢ)`` in place of ``Σ Jᵀcᵢ`` — and a gate's cast to the
    activation's dtype would even accumulate its consumers' bf16 cotangents
    in bf16. Roundoff, but a bf16 model turns an ulp into a flipped rounding
    and thirty Adam steps into a different fit. So under grad a trained
    quantity shares its **forward alone** (:class:`_Shared`): every access
    returns its own autograd node over the one cached value, and that node's
    backward replays the quantity's backward for *its* cotangent through the
    cached graph and hands the parameter the same contributions, in the same
    order, that a fresh graph per access would have — so the parameter's
    gradient accumulates in the exact order it did before. That replay is
    exact when the contributions can be told apart: the ``cayley`` map is
    evaluated on two aliases of its parameter, one per place it enters; a
    gate's mask shares when its graph reaches ``theta`` by exactly one edge
    and no other trainable leaf (a ``leak`` adds a second edge, a pool other
    gates' ``theta`` — those recompute per grad access, and share as a value
    under ``no_grad`` like everything else). The forward launches are what
    the profile counted; the backward's are unchanged.

    A mode-dependent quantity's key carries the stage's mode — a training
    mask and an eval mask never share an entry; the rotation, a function of
    the parameter alone, is keyed without it. Scopes nest; the store is
    dropped when the outermost exits, so nothing from one step reaches the
    next — a scope is
    valid between two mutations of its stages' parameters, and the loop
    opens each one after the optimizer step, the projection, the anneal and
    the mode switch. An ``isolated`` scope does not nest: it runs on a fresh
    store and restores the enclosing one afterwards, so a captured pass
    evaluates every stage inside the capture — a value shared in from the
    warmup pass, or from eager code around the capture, would be baked into
    the graph and replayed stale. :func:`torch.nn.utils.parametrize.cached`
    is the naive version of this for parametrized weights: one tensor for
    every consumer, the summed-cotangent gradient above.
    """
    if isolated:
        saved = (_SCOPE.depth, _SCOPE.entries)
        _SCOPE.depth, _SCOPE.entries = 1, {}
        try:
            yield
        finally:
            _SCOPE.depth, _SCOPE.entries = saved
        return
    if _SCOPE.entries is None:
        _SCOPE.entries = {}
    _SCOPE.depth += 1
    try:
        yield
    finally:
        _SCOPE.depth -= 1
        if _SCOPE.depth == 0:
            _SCOPE.entries = None


def _once(
    stage: "Stage",
    tag: Any,
    compute: Callable[[], torch.Tensor],
    *,
    trainable: bool,
) -> torch.Tensor:
    """``compute()`` once per open :func:`featurizer_cache` scope for this
    stage, mode and ``tag`` — for the accesses that want no graph. An access
    under grad of a ``trainable`` quantity (one that depends on a parameter
    with ``requires_grad``) computes its own, every time: sharing it would
    change the gradient (:func:`featurizer_cache`)."""
    cache = _SCOPE.entries
    if cache is None or (trainable and torch.is_grad_enabled()):
        return compute()
    key = (stage, tag, stage.training)
    value = cache.get(key)
    if value is None:
        value = compute()
        cache[key] = value
    return value


def _leaf_edges(output: torch.Tensor, leaf: torch.Tensor) -> tuple[int, bool]:
    """How ``output``'s graph reaches ``leaf``: the number of edges into its
    accumulator (one per op that consumed the parameter directly), and
    whether any *other* trainable leaf is reached — the two facts that decide
    whether one node per access can replay the graph's backward exactly
    (:class:`_Shared`)."""
    edges = 0
    others = False
    # the wrappers `next_functions` hands out are fresh objects; the visited
    # set keeps them alive so an id is never recycled onto an unvisited node
    seen: dict[int, Any] = {}
    pending = [output.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node
        for next_node, _ in node.next_functions:
            if next_node is None:
                continue
            variable = getattr(next_node, "variable", None)
            if variable is None:
                pending.append(next_node)
            elif variable is leaf:
                edges += 1
            else:
                others = True
    return edges, others


class _Shared:
    """One evaluation of a trained stage's quantity for the scope, serving
    every access exactly (:func:`featurizer_cache`).

    ``graph`` is the quantity with its autograd graph; ``aliases`` the leaves
    a replay differentiates with respect to, one per contribution the
    parameter receives from one evaluation, in the order the standalone
    graph's backward delivers them. A no-grad access gets the detached value.
    A grad access gets a fresh :class:`_Replay` node over that value whose
    backward pushes the access's own cotangent through this graph
    (``retain_graph``) and returns those contributions to the parameter one
    after the other — so the parameter's accumulated gradient is bit for bit
    the one a graph per access produces. No aliases means the quantity does
    not depend on the parameter (a frozen mask): the value serves everyone.

    The graph is built under ``enable_grad`` whatever the mode of the access
    that builds it, so a no-grad access reads a value produced in grad mode
    and an eval round keeps one small graph per stage until the scope
    closes. Neither changes a number: none of these ops (``inv_ex``, the
    k×k products, a sigmoid, ``repeat_interleave``) picks its kernel by grad
    mode — unlike a scalar-vs-tensor divide, which this file does mind."""

    def __init__(
        self,
        original: torch.Tensor,
        graph: torch.Tensor,
        aliases: Sequence[torch.Tensor],
    ) -> None:
        self.original = original
        self.graph = graph
        self.aliases = tuple(aliases)
        self.value = graph.detach()

    @classmethod
    def rotation(cls, cayley: "Cayley", original: torch.Tensor) -> "_Shared":
        """The Cayley map evaluated on two aliases of ``X`` — where it enters
        the frame coordinates ``Q₀ᵀX`` and where it enters the complement
        ``X − Q₀(Q₀ᵀX)`` (:meth:`Cayley.map`); the standalone backward
        delivers the complement's contribution first, then the frame's."""
        frame = original.detach().requires_grad_(True)
        complement = original.detach().requires_grad_(True)
        with torch.enable_grad():
            graph = cayley.map(frame, complement)
        return cls(original, graph, (complement, frame))

    @classmethod
    def single(
        cls, original: torch.Tensor, compute: Callable[[], torch.Tensor]
    ) -> "tuple[_Shared | None, torch.Tensor]":
        """A quantity whose graph reaches ``original`` by one edge and no
        other trainable leaf — then one replay per access is the standalone
        backward exactly — or ``None`` when it does not, and each access must
        compute its own. The evaluation made to decide is returned beside:
        when nothing can be shared it is exactly the first access's own
        graph, so the probe costs that access nothing."""
        with torch.enable_grad():
            graph = compute()
        if not graph.requires_grad:
            return cls(original, graph, ()), graph
        edges, others = _leaf_edges(graph, original)
        if edges != 1 or others:
            return None, graph
        return cls(original, graph, (original,)), graph

    def access(self) -> torch.Tensor:
        if self.aliases and torch.is_grad_enabled() and self.original.requires_grad:
            return cast(
                torch.Tensor,
                _Replay.apply(self, *([self.original] * len(self.aliases))),
            )
        return self.value

    def replay(self, cotangent: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            self.graph, self.aliases, cotangent, retain_graph=True
        )


#: A scope entry saying a quantity cannot be shared under grad — each access
#: computes its own (:meth:`_Shared.single`)
_UNSHARED = object()


class _Replay(torch.autograd.Function):
    """One access's autograd node over a :class:`_Shared`: the value for
    free, the backward replayed for this access alone. The parameter is
    passed once per contribution so the node hands them to the parameter's
    accumulator one after the other, as the quantity's own graph does.

    First order only: the replay is ``autograd.grad`` without
    ``create_graph``, so a backward run with ``create_graph=True`` (a
    Hessian-vector product, a meta-gradient) would not see the map in the
    second-order graph. No caller differentiates twice; one that does must
    evaluate outside the scope."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, shared: _Shared, *originals: torch.Tensor
    ) -> torch.Tensor:
        ctx.shared = shared
        # a distinct tensor object per access over the one storage, not
        # `shared.value` itself: `apply` attaches this node to the tensor it
        # returns, and a second access returning the same object would
        # overwrite the first's node, losing that access's cotangent
        return shared.value.detach()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, cotangent: torch.Tensor
    ) -> tuple[Any, ...]:
        return (None, *ctx.shared.replay(cotangent))


class Stage(torch.nn.Module):
    """One featurizer stage. Subclasses implement ``featurize`` /
    ``inverse``; parameters registered here are what ``train.params``
    optimizes."""

    kind: str = "identity"

    #: Whether ``featurize`` takes the routing table beside the activation —
    #: true of the expert-keyed gate alone (:class:`Gate`, ``expert_neuron``),
    #: whose parameters a token's slots find through ``expert_idx``.
    needs_routing: bool = False

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f

    def slot_params(self) -> dict[str, torch.Tensor]:
        """The auto-declared slots (§2.5), for saving and identity checks."""
        return {}

    def project(self) -> None:
        """Put the parameters back on their feasible set after an optimizer
        step — a no-op for every stage whose parameters are unconstrained or
        constrained by construction (a ``subspace``'s orthogonal
        parametrization). The train loop calls it after every step; a
        ``clamp`` gate clips its mask into ``[0, 1]`` here (§2.5)."""

    def identity_fields(self) -> dict[str, Any]:
        """ArtifactIdentity fields (§8) a bundle this stage is saved into
        carries *beyond* what the document implies — empty unless the stage
        was built from something the document only points at."""
        return {}


class Identity(Stage):
    kind = "identity"


class Cayley(torch.nn.Module):
    """The Cayley map onto the Stiefel manifold, computed at the rank of the
    tangent vector rather than at ``d×d``.

    Given the base frame ``Q₀`` (``(d, k)``, orthonormal) and a free
    ``X ∈ ℝ^{d×k}``, the skew-symmetric ``A = X Q₀ᵀ − Q₀ Xᵀ`` and

        Q(X) = cayley(A) Q₀,   cayley(A) = (I − A/2)⁻¹ (I + A/2).

    ``A Q₀`` ranges over the whole tangent space at ``Q₀``, so this reaches
    every frame the d×d map reaches. Torch's ``orthogonal(...,
    orthogonal_map="cayley")`` computes exactly this map — but by embedding
    ``X`` into a d×d skew matrix, solving a d×d system, and multiplying by a
    d×d ``base`` (completed once, at build time, from the *global* RNG), on
    every ``.weight`` access. At ``d = 4096, k = 8`` that is a 4096³ solve
    to move eight vectors, no cheaper than the matrix exponential.

    **The low-rank form.** Split ``X`` against the frame: ``B = Q₀ᵀX`` and
    ``X⊥ = X − Q₀B``. Only the skew half ``Ω = B − Bᵀ`` of ``B`` reaches
    ``A`` (its symmetric half cancels — those ``k(k+1)/2`` directions of
    ``X`` are redundant, and a loss has zero gradient along them), so

        A = Q₀ Ω Q₀ᵀ + X⊥ Q₀ᵀ − Q₀ X⊥ᵀ = P C Pᵀ,
        P = [X⊥, Q₀] (d×2k),   C = [[0, I], [−I, Ω]].

    The Woodbury identity gives ``(I − A/2)⁻¹ A Q₀ = −2 P (PᵀP − 2C⁻¹)⁻¹
    PᵀQ₀``; with ``X⊥ ⊥ Q₀`` the Gram is block-diagonal and ``PᵀQ₀ = [0; I]``,
    so ``Q(X) = Q₀ − 2 P v`` with ``v = [v₁; v₂]`` solving

        [[X⊥ᵀX⊥ − 2Ω, 2I], [−2I, I]] v = [0; I].

    Eliminating the second block row (``v₂ = I + 2v₁``) leaves one ``k×k``
    system on the Schur complement ``S``:

        S v₁ = −2I,   S = X⊥ᵀX⊥ − 2Ω + 4I.

    ``sym(S) = X⊥ᵀX⊥ + 4I`` is positive definite, so ``S`` is never singular
    — that is the whole invertibility argument. Cost ``O(d k²)`` plus a
    ``k×k`` inverse; nothing larger than ``(d, k)`` is built or saved for
    backward, and the map is a function of ``Q₀`` alone — no RNG.

    **Conditioning.** Scaling ``X⊥``'s columns to unit norm, ``X⊥ = X̃ D``
    with ``D = diag(‖x⊥ⱼ‖)`` clamped at 1 and detached (the value does not
    depend on ``D``, so neither should its gradient), gives

        S = X̃ᵀX̃ − 2D⁻¹ΩD⁻¹ + 4D⁻²,   Q(X) = Q₀ − 2 (X̃ v₁ + Q₀ v₂),
        v₂ = I + 2D⁻¹v₁.

    With generic (near-orthogonal) columns ``X̃ᵀX̃ ≈ I`` and ``κ(S) = O(1)``
    for any ``‖X‖`` — measured at ``d = 4096, k = 32`` the fp32
    orthonormality error stays near 2e-6 up to column norms of 900, where the
    dense fp32 solve has drifted to 3e-6. The limit is **rank-deficient**
    ``X⊥``: two (near-)parallel columns of length ``s`` put an eigenvalue of
    ``≈ 4/s²`` in ``S``, so ``κ(S) ~ s²/2`` — quadratic in ``‖X‖`` where the
    dense ``I − A/2`` degrades only linearly. Measured (fp32, low-rank /
    dense): ``s = 10`` → 3e-6 / 2e-6, ``s = 100`` → 1e-4 / 2e-6, ``s = 900``
    → 1e-2 / 1e-5. At ``k = 1`` a column of length ``s`` is a rotation of the
    base vector by ``2·atan(s/2)`` — 178° at ``s = 100``, the chart near
    saturation. At ``k ≥ 2`` the degenerate case is several columns rotating
    toward one direction, which no such one-plane picture bounds, so rather
    than argue it cannot happen the train loop records each fitted rotation's
    deviation in ``fit_diagnostics.json`` (``orthonormality_deviation``, and
    ``within_tolerance`` against :data:`ORTHONORMAL_TOLERANCE`); a rotation
    past that bar is one a later document cannot name as an ``init``. A pure
    in-frame ``X`` (``X⊥ = 0``) is exact:
    ``S = 4I − 2Ω``.

    ``right_inverse`` is what ``stage.weight = Q`` calls: it rebases the map
    at ``Q`` and returns the zero ``X`` — the identity in ``X`` is the base.

    **Launches.** The map is a run of ~25 tiny kernels, issued on every
    ``.weight`` access unless a :func:`featurizer_cache` scope is open, so
    its spelling minds the count where that costs no bit: the ``k×k``
    identity is a buffer rather than a fresh ``eye`` per call, and the
    inverse is ``inv_ex`` with its host error check off — ``torch.linalg.inv``
    *is* ``inv_ex`` followed by a device-to-host copy of the info tensor, a
    synchronization per call, and the Schur system is nonsingular by the
    argument above. The products with the diagonal ``D⁻¹`` stay GEMMs on
    purpose: spelled as the row and column scalings they are, the *forward*
    is bit-identical but the gradient is not — the autograd engine then sums
    ``X̃``'s three contributions in another order — and the count is the
    same either way. ``tests/neural/shared/test_featurizer_cache.py`` holds
    this spelling bit-identical, forward and backward, to the one it
    replaced."""

    base: torch.Tensor
    eye: torch.Tensor

    def __init__(self, base: torch.Tensor) -> None:
        super().__init__()
        # row-major on purpose: `torch.linalg.qr` hands back a column-major Q
        # and `clone` would keep that layout, so the materialized weight would
        # be a transposed-layout matrix while its saved copy is row-major —
        # and a matmul takes a different kernel path for each, landing one ulp
        # apart. A fit and its reloaded artifact must featurize bit-identically.
        self.register_buffer("base", base.detach().contiguous().clone())
        # the k×k identity `v₂ = I + 2D⁻¹v₁` adds: a constant, so a buffer
        # that follows the module's device and dtype — and not saved state,
        # so a bundle's keys and the train loop's snapshot are unchanged
        self.register_buffer(
            "eye",
            torch.eye(base.shape[-1], dtype=base.dtype, device=base.device),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x, x)

    def map(self, x_frame: torch.Tensor, x_perp: torch.Tensor) -> torch.Tensor:
        """The map with ``X`` named once per place it enters: ``x_frame``
        where the frame coordinates ``Q₀ᵀX`` are taken, ``x_perp`` where the
        complement ``X − Q₀(Q₀ᵀX)`` is. :meth:`forward` passes the one ``X``
        to both; :meth:`_Shared.rotation` passes two aliases of it so the
        parameter's two gradient contributions come back apart."""
        base = self.base
        in_frame = base.mT @ x_frame
        omega = in_frame - in_frame.mT
        perp = x_perp - base @ in_frame
        scale = torch.linalg.vector_norm(perp.detach(), dim=-2).clamp(min=1.0)
        perp = perp / scale
        inv_scale = torch.diag_embed(1.0 / scale)
        schur = (
            perp.mT @ perp
            - 2.0 * inv_scale @ omega @ inv_scale
            + 4.0 * inv_scale @ inv_scale
        )
        # `inv` rather than `solve`: the matrix is k×k, so the cost is the
        # same, and solve's backward (`linalg_lu_solve`) has no MPS kernel in
        # torch 2.9 while inv's is plain matmuls — the map trains on every
        # device the engine accepts. `inv_ex` without the error check: the
        # check is a host synchronization per call, and the system is
        # nonsingular (class docstring).
        inverse = torch.linalg.inv_ex(schur, check_errors=False).inverse
        v1 = inverse @ (-2.0 * inv_scale)
        v2 = self.eye + (2.0 * inv_scale @ v1)
        return base - 2.0 * (perp @ v1 + base @ v2)

    def right_inverse(self, q: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            deviation = orthonormality_deviation(q)
            if deviation > ORTHONORMAL_TOLERANCE:
                raise ValueError(
                    "a subspace weight must be orthonormal (QᵀQ = I; max |QᵀQ − I| "
                    f"= {deviation:.3g}, tolerance {ORTHONORMAL_TOLERANCE:g}); "
                    "refusing to re-orthonormalize silently"
                )
            self.base.copy_(q)
            return torch.zeros_like(q, memory_format=torch.contiguous_format)


class Subspace(Stage):
    """An orthonormal ``(d, k)`` map ``Q``; features are the coordinates in
    its column space, ``err`` the complement (module docstring).

    ``parametrization`` picks how the optimizer's free tensor becomes a
    Stiefel point. ``cayley`` is :class:`Cayley`, ``O(d k²)`` per access.
    ``matrix_exp`` and ``stiefel`` (householder products) are torch's
    ``orthogonal`` maps: ``stiefel`` is also ``O(d k²)``, but ``matrix_exp``
    exponentiates a d×d matrix on every access and is impractical at model
    width — it stays in the vocabulary because the spec (§2.5) and every
    saved rotation's ArtifactIdentity name it.

    ``seed`` picks the initial rotation and is kept on the instance so a
    cached stage can be checked against the seed a later use site asks for
    (:func:`build_stack`).

    ``init`` is an optional orthonormal ``(d, k)`` matrix ``P`` the fit starts
    from (§2.5 ``init`` — the first ``k`` columns of a saved basis). Every
    parametrization here is a *trivialization*: the weight is a map of a free
    parameter that starts at the identity, applied to a fixed base frame, so
    before any step the weight *is* the start (``qr(randn(d, k))`` from the
    seeded generator, or ``P`` verbatim) and the read is ``Pᵀx`` bit-for-bit,
    while the trainable surface is unchanged. For ``cayley`` the base is the
    ``(d, k)`` start itself (:class:`Cayley`). For ``matrix_exp`` and
    ``stiefel`` torch's ``orthogonal`` needs a d×d base: absent ``init`` torch
    completes it from the global RNG, exactly as before; with ``init`` the
    base is ``[P | N]`` orthonormalized by QR, ``N`` drawn from the same
    seeded generator, with its first ``k`` columns set to ``P``, so the
    start is a function of the document alone. ``init_identity`` is what a
    bundle this stage is saved into records about that start
    (:meth:`identity_fields`)."""

    kind = "subspace"

    def __init__(
        self,
        width: int,
        k: int,
        parametrization: str,
        *,
        seed: int = 0,
        init: torch.Tensor | None = None,
        init_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.width = width
        self.map_name = parametrization
        self.init_identity: dict[str, Any] = dict(init_identity or {})
        generator = torch.Generator().manual_seed(seed)
        if init is None:
            start = torch.linalg.qr(torch.randn(width, k, generator=generator))[0]
        else:
            if tuple(init.shape) != (width, k):
                raise ValueError(
                    f"init must be a ({width}, {k}) matrix, got {tuple(init.shape)}"
                )
            start = init.detach().to(torch.float32).clone()
        self.weight = torch.nn.Parameter(start)
        if parametrization == "cayley":
            # the base *is* the start — no d×d completion to seed or replace
            torch.nn.utils.parametrize.register_parametrization(
                self, "weight", Cayley(start)
            )
            return
        orthogonal_map = {
            "matrix_exp": "matrix_exp",
            # a direct Stiefel point via householder products — the map torch
            # provides for rectangular orthogonal parametrizations
            "stiefel": "householder",
        }[parametrization]
        # torch completes the d×d base from the **global** RNG. Fork it away
        # and replace the completion with this generator's, so a rotation is a
        # function of its document alone — not of build order, nor of whoever
        # else shares the process (a served model's is another tenant's).
        with torch.random.fork_rng(devices=[]):  # on CPU, as every stage is
            torch.nn.utils.parametrizations.orthogonal(
                self, "weight", orthogonal_map=orthogonal_map
            )
        complement = torch.randn(width, width - k, generator=generator)
        full = torch.linalg.qr(torch.cat([start, complement], dim=1))[0]
        self.parametrizations.weight[0].base = torch.cat([start, full[:, k:]], dim=1)

    def identity_fields(self) -> dict[str, Any]:
        return dict(self.init_identity)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the constructor's arguments, the free parameter and the
        rest of the ``state_dict``: torch refuses to pickle a parametrized
        module whole (its class is made at registration), and these are what
        a fit moves, so a copy rebuilt from them featurizes bit-identically.
        The free parameter travels as the object it is, not as a copy — an
        optimizer pickled beside the stage holds the same object, and must
        still hold the stage's parameter on the other side. The mode rides
        along; whether the parameter takes a gradient (a §2.11 phase turns
        it off) is the parameter's own."""
        parametrization = self.parametrizations.weight  # type: ignore[union-attr]
        return (
            _rebuild_subspace,
            (
                self.width,
                self.k,
                self.map_name,
                self.seed,
                dict(self.init_identity),
                parametrization.original,
                {
                    key: value.detach()
                    for key, value in self.state_dict().items()
                    if key != _SUBSPACE_ORIGINAL
                },
                self.training,
            ),
        )

    def _q(self) -> torch.Tensor:
        """The rotation — the parametrization evaluated once per open
        :func:`featurizer_cache` scope where that is exact: a trained
        ``cayley`` map shares its forward and replays its backward per access
        (:meth:`_Shared.rotation`); every other map shares under ``no_grad``
        only and is recomputed on a grad access. The entry is keyed without
        the stage's mode on purpose: ``Q`` is a function of ``base`` and the
        parameter alone, the same in ``train()`` and ``eval()``."""
        entries = _SCOPE.entries
        if entries is None:
            return self.weight
        parametrization = self.parametrizations.weight  # type: ignore[union-attr]
        original: torch.Tensor = parametrization.original
        # one map composes `weight`; a second would make `[0]` a wrong
        # rotation inside a scope alone, so the assumption is loud
        assert len(parametrization) == 1, "the rotation is one map; see `_Shared`"
        cayley = parametrization[0]  # type: ignore[index]
        if isinstance(cayley, Cayley) and original.requires_grad:
            key = (self, "rotation")
            shared = entries.get(key)
            if shared is None:
                shared = _Shared.rotation(cayley, original)
                entries[key] = shared
            return shared.access()
        return _once(
            self, "weight", lambda: self.weight, trainable=original.requires_grad
        )

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._q()
        f = x.to(q.dtype) @ q
        return f, x - f @ q.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        q = self._q()
        x = f @ q.T
        return x if err is None else x + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"weight": self._q()}


#: the ``state_dict`` key of the tensor an optimizer steps under any of a
#: subspace's parametrizations
_SUBSPACE_ORIGINAL = "parametrizations.weight.original"


def _rebuild_subspace(
    width: int,
    k: int,
    map_name: str,
    seed: int,
    init_identity: Mapping[str, Any],
    original: torch.nn.Parameter,
    state: Mapping[str, torch.Tensor],
    training: bool,
) -> Subspace:
    """:meth:`Subspace.__reduce__`'s inverse. The construction's own draws
    are thrown away by the state that follows, and they are the constructor's
    own generator's, so unpickling a stage moves no stream a fit reads."""
    stage = Subspace(width, k, map_name, seed=seed, init_identity=init_identity)
    stage.to(original.device)
    stage.load_state_dict(dict(state), strict=False)
    stage.parametrizations.weight.original = original  # type: ignore[union-attr]
    stage.train(training)
    return stage


class LoadedLinear(Stage):
    """A fixed ``(d, k)`` map loaded from an artifact (``pca``, or an
    applied ``subspace`` fit): same math as :class:`Subspace`, no
    parametrization, never trainable."""

    weight: torch.Tensor

    def __init__(self, kind: str, weight: torch.Tensor) -> None:
        super().__init__()
        self.kind = kind
        self.register_buffer("weight", weight)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.weight
        f = x.to(weight.dtype) @ weight
        return f, x - f @ weight.T

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = f @ self.weight.T
        return x if err is None else x + err


class Standardize(Stage):
    kind = "standardize"

    mu: torch.Tensor
    sigma: torch.Tensor

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (x - self.mu) / self.sigma, None

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f * self.sigma + self.mu


class Sae(Stage):
    """A loaded sparse autoencoder: ``(enc(x), x − dec(enc(x)))``."""

    kind = "sae"

    enc: torch.Tensor
    dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def __init__(
        self,
        enc: torch.Tensor,
        dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("enc", enc)
        self.register_buffer("dec", dec)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.enc + self.b_enc)

    def _decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.dec + self.b_dec

    def featurize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = self._encode(x)
        return f, x - self._decode(f)

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        x = self._decode(f)
        return x if err is None else x + err


class Gate(Stage):
    """The DBM gate: soft ``σ(θ/T) ⊙ x`` in training, hard ``θ > 0`` in
    eval. ``temperature`` is the anneal target ``<name>.theta.temperature``.

    ``parametrization`` is how ``theta`` maps to the mask (§2.5,
    ``GATE_PARAMETRIZATIONS``). ``sigmoid`` is the description above. Under
    ``clamp`` the parameter *is* the mask: soft ``m = θ``, projected into
    ``[0, 1]`` after every optimizer step (:meth:`project`), hard ``θ > ½``
    — i.e. ``round`` — and no temperature: the anneal target is refused at
    load and again here. The L1 term and the decisiveness diagnostic read the
    mask through :meth:`soft_mask`, the hard count through :meth:`hard_mask`,
    so neither spells a parametrization out.

    Under ``hard_concrete`` (Louizos, Welling & Kingma 2018) the training
    mask is a *sample* of the stretched, clipped concrete distribution and the
    eval mask its deterministic mean above ½ (:func:`hard_concrete_threshold`).
    The draw is not made here: the train loop calls :meth:`resample` once per
    optimizer step with the fit's generator, and every ``featurize`` of that
    step — the read of the counterfactual *and* the write into the base — reads
    the one cached draw, so ``m·v_cf + (1 − m)·v_base`` is an interchange and
    not two masks (:meth:`sampled_mask`). ``temperature`` is β and ``stretch``
    the ``(γ, ζ)`` of the relaxation; ``init.fill p`` starts the deterministic
    mask at exactly ``p`` by inverting the stretch.

    ``group`` is the unit kind the document authored (§2.5) and ``groups``
    the derived map that goes with it; ``None`` for both is the per-coordinate
    gate. Under ``head``, ``groups`` is ``(heads, head_dim)``: ``theta`` has
    one entry per head and every coordinate of a head receives that entry's
    value, soft or hard. Under ``expert_neuron``, ``groups`` is
    ``(num_experts, d_expert)``: ``theta`` is the whole expert table,
    ``theta[expert, neuron]``, and the site is token-major with ``top_k``
    slots of ``d_expert`` — so ``featurize`` needs the routing table beside
    the activation, and slot *k* of a token receives the row of the expert
    ``expert_idx[..., k]`` names. An expert a token did not activate has no
    slot, so its parameters do not touch that token. Either way the L1 term
    (the mean of the soft mask over ``theta``) and the hard-mask count are
    over units, not coordinates.

    ``dead`` (§2.5 ``GATE_DEAD_RULES``) is what a *training* gate does about a
    unit whose hard mask has closed. Under ``{"freeze_after": n}`` a unit
    hard-off for ``n`` consecutive optimizer steps is frozen: its ``theta`` is
    photographed and restored after every later step (:meth:`project`), so
    the optimizer's momentum cannot reopen it and a pruned unit stays pruned
    — the original DCM behaviour, which makes the sweep a nested sequence by
    construction. Under ``{"leak": ε}`` the training mask's *derivative* is
    floored: the forward value is the map's own ``m``, unchanged, and the
    backward pass sees ``∂m/∂θ + ε`` (:meth:`_mask`, the leaky-ReLU idiom
    ``m + ε·(θ − θ.detach())``), so a unit whose map has saturated at the
    zero pole — ``σ'(θ/T) ≈ 0``, or a concrete sample clipped to 0 — still
    receives ``ε·∂L/∂m``, which is nonzero whenever the loss would rather have
    the unit's counterfactual, and can come back. A *value* floor would not
    do this: ``ε + (1 − ε)·m`` leaves ``∂/∂θ`` scaled by the same vanishing
    ``σ'``, and it routes ``ε`` of the counterfactual through a unit the eval
    mask drops — the co-adaptation a hard mask exists to prevent. The
    eval-mode hard mask is untouched. Both rules are bookkept per
    unit here because :meth:`project` is the one hook the loop calls after
    every step for every stage; ``fit_diagnostics`` reads
    :meth:`dead_diagnostics` — how many units froze, how many were ever
    hard-off and are kept at the end (``reawakened_units``)."""

    kind = "gate"

    def __init__(
        self,
        width: int,
        *,
        group: str | None = None,
        groups: tuple[int, int] | None = None,
        parametrization: str = "sigmoid",
        init: float | torch.Tensor | None = None,
        init_identity: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        stretch: tuple[float, float] | None = None,
        k_schedule: Mapping[str, Any] | None = None,
        stop_grad_shift: bool = False,
        pool: str | None = None,
        dead: Mapping[str, Any] | None = None,
        axis: str | None = None,
        forward: str | None = None,
    ) -> None:
        super().__init__()
        if forward not in (None, "hard"):
            raise ValueError(f"a gate's forward is 'hard' or absent, got {forward!r}")
        #: §2.5 the mapping form of ``parametrization``: ``"hard"`` when the
        #: training forward uses the map's mask thresholded at ½ with the
        #: map's gradient behind it (straight-through); ``None`` is the map's
        #: own forward. Eval is the map's hard split either way.
        #: stored as ``forward_mask``: ``forward`` is ``nn.Module``'s callable
        #: slot, and a string there would make every gate uncallable as a
        #: module (hooks, compile, DDP). The document key and the stamp key
        #: stay ``forward``
        self.forward_mask: str | None = forward
        if axis not in (None, "position"):
            raise ValueError(f"a gate's axis is 'position' or absent, got {axis!r}")
        if axis == "position" and group is not None:
            raise ValueError("a position gate takes no group: it is one θ per position")
        if axis == "position" and pool is not None:
            # the parser refuses this too; a second line of defense, as `group`
            raise ValueError("a position gate takes no pool: no budget over positions")
        #: §2.5 ``axis``: ``"position"`` when θ runs over the addressed token
        #: positions — ``width`` is then the window's length and the mask
        #: broadcasts along the position axis of a ``(rows, positions,
        #: width)`` value — else ``None``, the feature gate
        self.axis: str | None = axis
        if parametrization not in GATE_PARAMETRIZATIONS:
            raise ValueError(
                f"unknown gate parametrization {parametrization!r}; one of "
                f"{list(GATE_PARAMETRIZATIONS)}"
            )
        #: ``hard_concrete`` only: the stretch ``(γ, ζ)`` the concrete sample
        #: is mapped onto before clipping; ``None`` under the other maps
        self.stretch: tuple[float, float] | None = None
        beta: float | None = None
        if parametrization == "hard_concrete":
            beta = (
                HARD_CONCRETE_TEMPERATURE if temperature is None else float(temperature)
            )
            lo, hi = HARD_CONCRETE_STRETCH if stretch is None else stretch
            if beta <= 0.0:
                raise ValueError(f"a hard-concrete temperature is positive, got {beta}")
            if not float(lo) < 0.0 < 1.0 < float(hi):
                raise ValueError(
                    f"stretch [γ, ζ] must satisfy γ < 0 < 1 < ζ, got {stretch}"
                )
            self.stretch = (float(lo), float(hi))
        elif temperature is not None or stretch is not None:
            raise ValueError(
                "temperature and stretch are hard_concrete's constants; a "
                f"{parametrization!r} gate samples nothing"
            )
        #: ``budget`` only (§2.5 ``k_schedule``): how each step draws its
        #: budget, and the cut the eval-mode split is read at (``eval``, or
        #: ``k`` when fixed). A budget gate being fitted needs one; a loaded
        #: one is read out through :attr:`top_k` instead and carries none.
        self.k_schedule: dict[str, Any] | None = None
        self.stop_grad_shift: bool = bool(stop_grad_shift)
        #: ``budget`` only (§2.5 ``pool``): the name of the budget pool this
        #: gate shares one ``k``, one shift and one ranking with, and — once
        #: :func:`link_budget_pools` has run over the point's stages — the
        #: :class:`BudgetPool` itself. ``None`` for a gate that budgets alone.
        self.pool_name: str | None = pool
        self.pool: BudgetPool | None = None
        #: a loaded pooled gate's stamped ``pool_units`` (§8), compared with the
        #: pool the document assembles at the link — ``None`` when unstamped
        self.stamped_pool_units: int | None = None
        if parametrization == "budget":
            if k_schedule is not None:
                self.k_schedule = _checked_k_schedule(k_schedule)
        elif k_schedule is not None or stop_grad_shift:
            raise ValueError(
                "k_schedule and stop_grad_shift belong to the budget "
                f"parametrization; a {parametrization!r} gate draws no budget"
            )
        # `pool` on a non-budget gate is legal only as a pooled READOUT of a
        # loaded gate (`from_theta` with `top_k`); a fit under another map has
        # no budget to share, which `from_theta`'s caller and the parser hold
        if (group is None) != (groups is None):
            raise ValueError("a gate's group kind and group map come together")
        if (
            group in ("head", "site")
            and groups is not None
            and groups[0] * groups[1] != width
        ):
            raise ValueError(f"group map {groups} does not tile a {width}-wide gate")
        if group == "expert_neuron" and groups is not None and width % groups[1]:
            raise ValueError(
                f"expert map {groups}: a {width}-wide gate is not a whole number "
                f"of {groups[1]}-wide expert slots"
            )
        self.width = width
        self.group = group
        self.groups = groups
        self.parametrization = parametrization
        #: the ``init.fill`` the gate started at, when it did (§2.5) —
        #: recorded by ``fit_diagnostics`` so the record says where it began
        self.init_fill: float | None = None
        #: what a bundle this gate is saved into records about a saved start
        #: (``_gate_start``): the ``init_*`` provenance keys, as a subspace's
        self.init_identity: dict[str, Any] = dict(init_identity or {})
        #: the ``init.from_scores`` a start was read from, when it was (§2.5)
        #: — the resolved table path, the mode and the units it put on the
        #: kept pole — recorded by ``fit_diagnostics`` beside ``init_fill``
        self.init_scores: dict[str, Any] | None = None
        shape = gate_param_shape(group, groups, width)
        if init is None:
            # the midpoint mask under every map: σ(0) = ½, or ½ itself
            start = torch.full(shape, 0.5 if parametrization == "clamp" else 0.0)
        elif isinstance(init, torch.Tensor):
            # a saved theta of this very gate, verbatim (§2.5 init.file_path)
            if init.numel() != math.prod(shape):
                raise ValueError(
                    f"a gate over {list(shape)} needs {math.prod(shape)} parameters "
                    f"to start from, got {init.numel()}"
                )
            start = init.detach().to(torch.float32).reshape(shape).clone()
        else:
            # a mask value, mapped into theta by the parametrization
            fill = float(init)
            if not 0.0 <= fill <= 1.0:
                raise ValueError(f"init fill is a mask value in [0, 1], got {fill}")
            if parametrization in ("sigmoid", "hard_concrete", "budget"):
                if self.stretch is None and not 0.0 < fill < 1.0:
                    # only the sigmoid start is a bare logit; the hard-concrete
                    # start inverts the stretch and is finite at both poles
                    raise ValueError(
                        f"a {parametrization} gate starts at θ = logit(fill), which "
                        f"is undefined at fill={fill}; start strictly inside (0, 1), "
                        "or use parametrization 'clamp' or 'hard_concrete' for a "
                        "start at a pole"
                    )
                if self.stretch is not None:
                    # the deterministic mask is clip(σ(θ)·(ζ−γ)+γ), so the θ
                    # whose mask is exactly `fill` inverts the stretch — the
                    # same map that puts the eval split at mask ½
                    theta_0 = hard_concrete_theta(fill, self.stretch)
                else:
                    theta_0 = math.log(fill / (1.0 - fill))
                start = torch.full(shape, theta_0)
            else:
                start = torch.full(shape, fill)
            self.init_fill = fill
        self.theta = torch.nn.Parameter(start)
        #: §2.5 ``dead``: ``freeze_after`` steps, or ``None``
        self.freeze_after: int | None = None
        #: §2.5 ``dead``: the gradient leak ``ε`` added to ``∂m/∂θ``, or ``None``
        self.leak: float | None = None
        if dead is not None:
            rules = dict(dead)
            unknown = sorted(set(rules) - set(GATE_DEAD_RULES))
            if unknown or len(rules) != 1:
                raise ValueError(
                    f"a gate's dead-unit rule is exactly one of {list(GATE_DEAD_RULES)}, "
                    f"got {sorted(rules)}"
                )
            if "freeze_after" in rules:
                n = int(rules["freeze_after"])
                if n < 1:
                    raise ValueError(
                        f"freeze_after counts steps, a positive integer; got {n}"
                    )
                self.freeze_after = n
            else:
                eps = float(rules["leak"])
                if not 0.0 < eps < 1.0:
                    raise ValueError(
                        f"leak is a gradient slope strictly inside (0, 1); got {eps}"
                    )
                self.leak = eps
        # per-unit bookkeeping for `dead` and for `reawakened_units`, kept as
        # buffers so a `.to(device)` moves them with theta; none is a slot
        self.register_buffer("_off_streak", torch.zeros(shape, dtype=torch.long))
        self.register_buffer("_frozen", torch.zeros(shape, dtype=torch.bool))
        self.register_buffer("_frozen_theta", torch.zeros(shape))
        self.register_buffer("_ever_off", torch.zeros(shape, dtype=torch.bool))
        #: the sigmoid gate's anneal target ``T``; under ``hard_concrete`` the
        #: concrete temperature β (annealable the same way)
        self.temperature: float = 1.0 if beta is None else beta
        self.hard_eval: bool = True
        self.capture_temperature: torch.Tensor | None = None
        #: §2.11 ``phases[i].freeze_masks``: the hard mask a phase pinned at its
        #: start, used by every forward while set — training mode included —
        #: so a featurizer trained behind this gate learns under the split it
        #: will be scored through, not under a mask still moving. ``None``
        #: outside such a phase; never saved (a bundle holds ``theta``).
        self.frozen_mask: torch.Tensor | None = None
        #: ``hard_concrete`` only: the uniform draw of the current optimizer
        #: step (:meth:`resample`), ``None`` until the loop makes one
        self._draw: torch.Tensor | None = None
        #: ``budget`` only: the budget ``k`` of the current optimizer step
        #: (:meth:`resample`) and the shift ``c_k`` of the last solve
        #: (:meth:`budget_shift`), the mask's second coordinate beside ``θ``
        self._k: int | None = None
        self._shift: torch.Tensor | None = None
        #: a *loaded* gate's top-k readout (§2.5 ``top_k``): when set, the
        #: eval-mode split is the ``top_k`` largest units of ``theta`` rather
        #: than the map's threshold (:meth:`hard_mask`). Only
        #: :meth:`from_theta` sets it — a gate being fitted has none
        self.top_k: int | None = None

    @property
    def needs_routing(self) -> bool:  # type: ignore[override]
        return self.group == "expert_neuron"

    def _require_linked(self) -> None:
        """A gate that authors a pool is only ever read *through* it. The link
        (:func:`link_budget_pools`) runs after every stack build, so a gate
        reaching a mask unlinked is a broken construction, not a lone gate —
        left alone it would draw its own budget, cut its own ranking and stamp
        no pool, wrong numbers with no trace."""
        if self.pool_name is not None and self.pool is None:
            raise RuntimeError(
                f"gate authors pool {self.pool_name!r} but was never linked — "
                "link_budget_pools runs over the point's stages before any mask "
                "is computed (§2.5 pool)"
            )

    @property
    def samples_per_step(self) -> bool:
        """Whether the training mask depends on a per-step draw the loop makes
        through :meth:`resample` — the hard-concrete noise, or the budget's
        ``k`` — so a read and a write the gate sits on share one mask."""
        return self.parametrization in ("hard_concrete", "budget")

    def _stretched(self, s: torch.Tensor) -> torch.Tensor:
        assert self.stretch is not None
        lo, hi = self.stretch
        return (s * (hi - lo) + lo).clamp(0.0, 1.0)

    def soft_mask(self) -> torch.Tensor:
        """The relaxed mask over ``theta``'s units, the quantity the L1 term
        and the decisiveness diagnostic are about: ``σ(θ/T)`` under
        ``sigmoid``, ``θ`` itself under ``clamp``, and under ``hard_concrete``
        the **deterministic** stretched-and-clipped ``σ(θ)`` — the mask the
        eval-mode split is taken from, with the concrete noise at its mean.
        The mask a training forward *uses* under ``hard_concrete`` is
        :meth:`sampled_mask`."""
        if self.parametrization == "clamp":
            return self.theta
        if self.parametrization == "hard_concrete":
            return self._stretched(torch.sigmoid(self.theta))
        if self.parametrization == "budget":
            # the mask the last forward used, or — before any step, and on a
            # loaded gate — the mask at the eval cut: `Σ m = k` either way
            # (over the pool, when pooled: the shift is the pool's)
            self._require_linked()
            last = self._shift if self.pool is None else self.pool.last_shift
            shift = last if last is not None else self.budget_shift(self.eval_k())
            return torch.sigmoid(self.theta.view(-1) + shift).view(self.theta.shape)
        if self.capture_temperature is not None:
            # CUDA scalar division lowers to reciprocal-multiply. Tensor
            # division rounds differently; preserve the eager CUDA arithmetic.
            return torch.sigmoid(self.theta * self.capture_temperature.reciprocal())
        return torch.sigmoid(self.theta / self.temperature)

    def resample(self, generator: torch.Generator) -> None:
        """Draw this step's concrete noise, ``u ~ U(0, 1)`` per unit, from
        ``generator`` — the fit's own CPU generator, never the global RNG — and
        keep it for every :meth:`featurize` of the step. The train loop calls
        this once per optimizer step: the read of the counterfactual and the
        write into the base then share one mask, so the training intervention
        is the interchange ``m·v_cf + (1 − m)·v_base`` and not two independent
        draws that would double-write or erase a unit. Drawn on CPU and moved
        to ``theta``'s device, so a seeded fit is bit-identical across devices
        (the ``subspace`` init's rule). ``u`` is kept off the endpoints as
        NeuroSurgeon does (``1e-4``), where the logit is undefined."""
        if self.parametrization == "budget":
            self._require_linked()
            if self.pool is not None:
                raise RuntimeError(
                    f"gate in pool {self.pool.name!r} draws no budget of its "
                    "own — the train loop resamples the pool once per step "
                    "(BudgetPool.resample), so every member shares the draw"
                )
            self._k = self._draw_budget(generator)
            return
        if self.parametrization != "hard_concrete":
            raise ValueError("only a hard_concrete or budget gate samples its mask")
        u = torch.rand(self.theta.shape, generator=generator, dtype=torch.float32)
        self._draw = u.clamp(1e-4, 1.0 - 1e-4).to(self.theta.device)

    # ------------------------------------------------------------------ #
    # the budget map (§2.5 `parametrization: budget`)
    # ------------------------------------------------------------------ #

    def _draw_budget(self, generator: torch.Generator) -> int:
        """This step's budget from the schedule (§2.5 ``k_schedule``), as the
        number of units that **take the counterfactual** — the count every
        internal reader (:meth:`budget_mask`, :meth:`hard_mask`) works in.
        The draw itself is :func:`_draw_from_schedule`; under ``of: "kept"``
        the schedule counts the units left clean, so the draw is complemented
        against the unit count (:func:`_as_patched`)."""
        schedule = self._budget_schedule()
        return _as_patched(
            schedule, _draw_from_schedule(schedule, generator), self.theta.numel()
        )

    def _budget_schedule(self) -> dict[str, Any]:
        if self.parametrization != "budget":
            raise ValueError("only a budget gate has a k_schedule")
        if self.k_schedule is None:
            raise ValueError(
                "a budget gate being fitted needs a k_schedule; a loaded one is "
                "read out at top_k"
            )
        return self.k_schedule

    def eval_k(self) -> int:
        """The count the eval-mode split of a ``budget`` gate is cut at: the
        document's ``top_k`` on a loaded gate, else the schedule's ``eval``
        (``k`` when fixed). A budget gate has no threshold — ``θ`` is a
        ranking — so one of the two must name the cut."""
        self._require_linked()
        if self.pool is not None:
            return self.pool.eval_k()
        if self.top_k is not None:
            return self.top_k
        schedule = self._budget_schedule()
        raw = int(schedule["eval"]) if "eval" in schedule else int(schedule["k"])
        return _as_patched(schedule, raw, self.theta.numel())

    def budget_shift(self, k: int) -> torch.Tensor:
        """The scalar ``c_k`` with ``Σ σ(θ + c_k) = k``, by bisection in
        float64 on a bracket that contains it: ``Σσ`` is strictly
        increasing in ``c`` from 0 to the unit count, so for ``0 < k < units``
        the root is unique and 60 halvings of ``[−max θ − 40, −min θ + 40]``
        pin it well below float32 resolution; at the poles ``k = 0`` /
        ``k = units`` there is no finite root and the bracket's end is returned
        (a mask within ``1e−17`` of the pole). Not differentiated through: the
        gradient the shift carries is added in :meth:`budget_mask` from the
        implicit-function rule."""
        if self.pool is not None:
            return self.pool.shift(k)
        # the bisection runs on CPU in float64: a few hundred units, and MPS
        # has no float64 at all
        theta = self.theta.detach().cpu().to(torch.float64).view(-1)
        return torch.tensor(
            _solve_shift(theta, k), dtype=self.theta.dtype, device=self.theta.device
        )

    def budget_mask(self, k: int) -> torch.Tensor:
        """The budget mask at ``k``: ``σ(θ + c_k)`` over ``theta``'s units.
        The shift is solved without gradient (:meth:`budget_shift`); unless
        ``stop_grad_shift``, its implicit gradient is attached — from
        ``F(θ, c) = Σ σ(θ + c) − k = 0``, ``∂c/∂θ_i = −σ'_i / Σ_j σ'_j`` — as
        ``c = c₀ + (w·θ).detach() − w·θ`` with ``w_i = σ'_i / Σ σ'_j`` held
        constant, which is exact to first order and keeps ``Σ m = k`` under any
        step. With ``stop_grad_shift`` the shift is the constant ``c₀``
        (ablation by ``−c_k``). The solve is kept as :attr:`_shift` so
        :meth:`soft_mask` reports the mask the last forward used. In a pool the
        whole computation is the pool's (:meth:`BudgetPool.mask_for`): one
        shift over every member's θ, the weights normalised over the pool."""
        if self.pool is not None:
            return self.pool.mask_for(self, k)
        shift = self.budget_shift(k)
        self._shift = shift
        theta = self.theta.view(-1)
        if not self.stop_grad_shift:
            with torch.no_grad():
                s = torch.sigmoid(theta + shift)
                weights = s * (1.0 - s)
                weights = weights / weights.sum().clamp_min(1e-30)
            correction = (weights * theta).sum()
            shift = shift + correction.detach() - correction
        return torch.sigmoid(theta + shift).view(self.theta.shape)

    def sampled_mask(self) -> torch.Tensor:
        """The hard-concrete mask of the current step (Louizos et al. 2018,
        eq. 10–11): with the step's draw ``u`` (:meth:`resample`),
        ``s = σ((log u − log(1 − u) + θ) / β)``, stretched to ``(γ, ζ)`` and
        clipped to ``[0, 1]`` — a reparametrized sample, so the gradient reaches
        ``θ`` through the sigmoid. Refuses when no draw was made: a training
        forward without the loop's ``resample`` is a broken step, not a mask."""
        if self.parametrization != "hard_concrete":
            raise ValueError("only a hard_concrete gate samples its mask")
        if self._draw is None:
            raise RuntimeError(
                "a hard_concrete gate in training mode has no draw for this step — "
                "the train loop calls Gate.resample(generator) once per optimizer "
                "step before the forward"
            )
        u = self._draw
        s = torch.sigmoid(
            (torch.log(u) - torch.log1p(-u) + self.theta) / self.temperature
        )
        return self._stretched(s)

    def expected_l0(self) -> torch.Tensor:
        """The expected kept fraction per unit of a ``hard_concrete`` gate — the
        ``l0`` penalty's quantity (§2.11): Louizos et al.'s closed form
        ``P(mask ≠ 0) = σ(θ − β · log(−γ/ζ))`` (eq. 12). ``hard_concrete`` only:
        under a deterministic map the relaxed mask is itself the kept
        probability and its mean is the ``l1`` term, so ``l0`` there would be a
        second spelling of ``l1`` — refused at validation (rule 4) and here."""
        if self.parametrization != "hard_concrete":
            raise ValueError(
                "only a hard_concrete gate has an expected L0; the relaxed mask of a "
                f"{self.parametrization!r} gate is deterministic and its mean is 'l1'"
            )
        assert self.stretch is not None
        lo, hi = self.stretch
        return torch.sigmoid(self.theta - self.temperature * math.log(-lo / hi))

    def hard_threshold(self) -> float:
        """The value of ``theta`` above which a unit is kept in eval mode:
        ``0`` under ``sigmoid``; ``½`` under ``clamp``; under ``hard_concrete``
        the θ whose stretched-and-clipped ``σ(θ)`` crosses ½ —
        ``logit((½ − γ) / (ζ − γ))``, exactly ``0`` at the default (symmetric)
        stretch. The arithmetic lives in
        :func:`~causalab.protocol.schema.hard_concrete_threshold`, shared with
        ``analysis.random_mask`` so a control's count is the fit's own."""
        if self.parametrization == "budget":
            raise ValueError(
                "a budget gate has no threshold — theta is a ranking, read out at "
                "a count (k_schedule.eval in a fit, top_k on a loaded gate)"
            )
        if self.parametrization == "clamp":
            return 0.5
        if self.parametrization == "hard_concrete":
            assert self.stretch is not None
            return hard_concrete_threshold(self.stretch)
        return 0.0

    def hard_mask(self) -> torch.Tensor:
        """The eval-mode split over ``theta``'s units, as a 0/1 tensor of
        ``theta``'s dtype: ``θ > 0`` under ``sigmoid`` (and ``hard_concrete``
        at its default stretch), ``θ > ½`` under ``clamp`` — the one number a
        localization claim is about (:meth:`hard_threshold``). Under a
        :attr:`top_k` readout the split is the ``top_k`` first units of
        :meth:`ranking` instead: the same ``theta``, cut at a count rather
        than at the map's threshold."""
        self._require_linked()
        if self.pool is not None:
            # the cut is through the POOLED ranking: this member keeps the
            # units whose pooled rank is below the pool's cut — a budget pool's
            # cut, or the one `top_k` a pooled readout of loaded gates names
            return self.pool.hard_for(self)
        cut = self.eval_k() if self.parametrization == "budget" else self.top_k
        if cut is not None:
            mask = torch.zeros_like(self.theta)
            mask.view(-1)[self.ranking()[:cut]] = 1.0
            return mask
        return (self.theta > self.hard_threshold()).to(self.theta.dtype)

    def ranking(self) -> torch.Tensor:
        """``theta``'s units (flat indices) from the most to the least kept —
        the object a top-k readout cuts and a ``rank`` save records (§2.12).
        Ordered by ``theta`` itself: every map's relaxed mask is monotone in
        ``θ`` (``σ(θ/T)``, ``θ``, the stretched-and-clipped ``σ(θ)``), so the
        order is the soft mask's, and ranking the parameter rather than the
        mask keeps units the clip has saturated to exactly 0 or 1 apart.
        Ties break toward the lower index, so the cut is a function of
        ``theta`` alone."""
        return _stable_ranking(self.theta.detach().view(-1))

    def rank(self) -> torch.Tensor:
        """Each unit's position in :meth:`ranking` (``0`` = kept first), in
        ``theta``'s layout."""
        order = self.ranking()
        rank = torch.empty_like(order)
        rank[order] = torch.arange(order.numel(), device=order.device)
        return rank.view(self.theta.shape)

    def project(self) -> None:  # type: ignore[override]
        """After every optimizer step (the loop's one post-step hook): a
        ``clamp`` gate back onto ``[0, 1]``; then the dead-unit bookkeeping
        (§2.5 ``dead``). The hard split is read *after* the projection so a
        clamp gate's streak counts what its mask does. A frozen unit's
        ``theta`` is restored from the photograph taken when it froze — the
        step that just happened is undone for that unit, and so is every
        later one — which is what makes the freeze hold under Adam, whose
        momentum would keep moving a unit whose gradient was merely zeroed."""
        with torch.no_grad():
            if self.parametrization == "clamp":
                self.theta.clamp_(0.0, 1.0)
            # "off" is the eval-mode split's complement, read through
            # `hard_mask` rather than the threshold: a budget gate has no
            # threshold (its split is the cut at `k_schedule.eval`), and under
            # the threshold maps the two readings are the same tensor
            off = self.hard_mask() == 0
            self._ever_off |= off
            if self.freeze_after is None:
                return
            self._off_streak = torch.where(
                off, self._off_streak + 1, torch.zeros_like(self._off_streak)
            )
            newly = (self._off_streak >= self.freeze_after) & ~self._frozen
            # branch-free on purpose: with nothing newly frozen the photograph
            # and the mask are unchanged, with nothing frozen theta is copied
            # onto itself — the same values a conditional would write, without
            # two device→host reads per gate per update
            self._frozen_theta = torch.where(
                newly, self.theta.detach(), self._frozen_theta
            )
            self._frozen |= newly
            self.theta.copy_(torch.where(self._frozen, self._frozen_theta, self.theta))

    def dead_diagnostics(self) -> dict[str, Any]:
        """What the dead-unit bookkeeping can say at the end of a fit, for
        ``fit_diagnostics.json``: the rule authored (as authored), how many
        units are frozen, and ``reawakened_units`` — units that were hard-off
        after some step and are kept by the final hard mask. The last is
        bookkept under every rule and none: it is the observable a ``leak``
        exists to move, and a frozen gate reports exactly ``0.0``."""
        with torch.no_grad():
            kept = self.hard_mask().bool()
            out: dict[str, Any] = {
                "frozen_units": float(self._frozen.sum()),
                "reawakened_units": float((self._ever_off & kept).sum()),
            }
        if self.freeze_after is not None:
            out["dead"] = {"freeze_after": self.freeze_after}
        elif self.leak is not None:
            out["dead"] = {"leak": self.leak}
        return out

    def _mask(self, routing: torch.Tensor | None = None) -> torch.Tensor:
        """The mask over ``x``'s coordinates: the table mask, looked up per
        routed slot when the gate is expert-keyed."""
        return self._route(self._table_mask(), routing)

    def _table_mask(self) -> torch.Tensor:
        """The mask over the gate's own units, expanded over a ``head`` /
        ``site`` group's coordinates — everything about the mask that does
        not depend on the activation it is applied to, which is what a
        :func:`featurizer_cache` scope shares across the step's accesses."""
        self._require_linked()
        if self.frozen_mask is not None:
            # a phase's photograph: 0/1 already, no gradient — the split below
            # has nothing to do to it, so it is not applied
            mask = self.frozen_mask.to(self.theta.dtype)
        else:
            if self.training and self.parametrization == "hard_concrete":
                mask = self.sampled_mask()
            elif self.training and self.parametrization == "budget":
                k = self._k if self.pool is None else self.pool.k
                if k is None:
                    raise RuntimeError(
                        "a budget gate in training mode has no budget for this step "
                        "— the train loop calls Gate.resample(generator) (or, in a "
                        "pool, BudgetPool.resample) once per optimizer step before "
                        "the forward"
                    )
                mask = self.budget_mask(k)
            elif self.training or not self.hard_eval:
                mask = self.soft_mask()
            else:
                mask = self.hard_mask()
            if self.training and self.forward_mask == "hard":
                # §2.5 forward/backward split: the value is the training mask
                # thresholded at ½ (the map's own split: θ > 0 under sigmoid,
                # θ > ½ under clamp, the sample's or the shifted mask's ½),
                # the gradient the map's — `hard + (soft − soft.detach())`
                mask = (mask > 0.5).to(mask.dtype) + (mask - mask.detach())
        if self.training and self.leak is not None and self.theta.requires_grad:
            # §2.5 `dead.leak`: a gradient leak, not a value floor — the forward
            # mask is the map's own, the backward sees ∂m/∂θ + ε, so a unit
            # saturated at the zero pole still gets ε·∂L/∂m and can climb back;
            # the eval split (and the value the loss sees) is untouched
            mask = mask + self.leak * (self.theta - self.theta.detach())
        if self.group in ("head", "site"):
            # one entry per group, each `groups[1]` coordinates wide; under
            # `site` that is a single entry over the whole width
            assert self.groups is not None
            mask = mask.repeat_interleave(self.groups[1])
        return mask

    def _route(self, mask: torch.Tensor, routing: torch.Tensor | None) -> torch.Tensor:
        """An expert-keyed gate's table rows at the slots ``routing`` names;
        every other gate's mask is already over ``x``'s coordinates."""
        if self.group == "expert_neuron":
            assert self.groups is not None
            if routing is None:
                raise ProtocolError(
                    "P2",
                    "an expert-keyed gate (group 'expert_neuron') needs the "
                    "routing table beside the activation — its parameters are "
                    "looked up per slot through expert_idx, and this path handed "
                    "it none",
                )
            top_k = self.width // self.groups[1]
            if routing.shape[-1] != top_k:
                raise ProtocolError(
                    "P2",
                    f"an expert-keyed gate over {top_k} routed slots was handed a "
                    f"routing table with {routing.shape[-1]} — the two come from "
                    "one tap and cannot disagree",
                )
            # (..., top_k) expert ids -> (..., top_k, d_expert) rows of the
            # table -> the token-major (..., top_k · d_expert) the site has
            mask = mask[routing.long()].reshape(*routing.shape[:-1], self.width)
        return mask

    def featurize(  # type: ignore[override]
        self, x: torch.Tensor, routing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # the table mask once per scope, in `theta`'s dtype; the routing
        # lookup and the cast to `x`'s dtype stay per call — the lookup is
        # the one part that depends on the activation, and casting *after* it
        # keeps the lookup's backward accumulating in `theta`'s dtype as it
        # always has: `mask[routing]`'s backward is a scatter-add in the
        # indexed tensor's dtype, and a bf16 sum of the slots' cotangents is
        # not the fp32 one
        mask = self._route(self._shared_table(), routing).to(x.dtype)
        if self.axis == "position":
            # one scalar per addressed position, over every coordinate: the
            # value is `(…, positions, width)` and θ runs over `positions`
            if x.dim() < 3 or x.shape[-2] != self.width:
                raise ProtocolError(
                    "P2",
                    f"a position gate over {self.width} positions was handed a value "
                    f"of shape {tuple(x.shape)} — it applies to a (rows, positions, "
                    "width) window whose positions axis is its own; a ragged window "
                    "arrives flattened and cannot carry one, and a window of another "
                    "length is another gate's (§2.5 axis)",
                )
            mask = mask.unsqueeze(-1)
        return mask * x, (1.0 - mask) * x

    def _shared_table(self) -> torch.Tensor:
        """The table mask, in ``theta``'s dtype, evaluated once per open
        :func:`featurizer_cache` scope where that is exact: shared as a value
        under ``no_grad``; under grad through one replay node per access when
        the mask's graph reaches ``theta`` by a single edge and nothing else
        trainable (:meth:`_Shared.single`) — a ``leak`` or a pool makes each
        grad access compute its own instead, the value still shared under
        ``no_grad``. Like the subspace's rotation the evaluation is made
        whatever the grad mode of the access that makes it, so a no-grad
        access followed by a grad one costs one evaluation, not two."""
        entries = _SCOPE.entries
        compute = self._table_mask
        if entries is None:
            return compute()
        # the graph entry; the no-grad value `_once` keeps sits under its own tag
        key = (self, "mask graph", self.training)
        shared = entries.get(key)
        if shared is None:
            shared, probe = _Shared.single(self.theta, compute)
            if shared is None:
                entries[key] = _UNSHARED
                if torch.is_grad_enabled():
                    return probe  # this access's own graph; the rest compute theirs
                # the probe is this no-grad access's value: kept for the next
                return _once(self, "mask value", lambda: probe.detach(), trainable=True)
            entries[key] = shared
        if shared is _UNSHARED:
            # what one replay cannot serve is still one exact value under
            # no_grad; a grad access computes its own
            return _once(self, "mask value", compute, trainable=True)
        return shared.access()

    def inverse(self, f: torch.Tensor, err: torch.Tensor | None) -> torch.Tensor:
        return f if err is None else f + err

    def slot_params(self) -> dict[str, torch.Tensor]:
        return {"theta": self.theta}

    def identity_fields(self) -> dict[str, Any]:
        self._require_linked()
        fields = dict(self.init_identity)
        if self.stretch is not None:
            # the hard split depends on the stretch (`hard_threshold`), so a
            # reader of the bundle — `analysis.random_mask` — needs it
            fields["stretch"] = json.dumps(list(self.stretch))
        if self.pool is not None:
            # a pooled member's θ is a ranking only relative to its co-members
            # (§2.5 `pool`): the name is compared by the loader, the unit
            # count is provenance a reader can size the pool's cut by
            fields["pool"] = self.pool.name
            fields["pool_units"] = str(self.pool.units)
        if self.axis is not None:
            # a position gate's θ is one entry per token position — a bundle
            # of it is not a mask over coordinates (§2.5 axis)
            fields["axis"] = self.axis
        if self.forward_mask is not None:
            # provenance, not identity: a straight-through fit's θ reads out
            # through the same hard split as the map's own, so an apply under
            # either spelling is the same mask (§2.5)
            fields["forward"] = self.forward_mask
        return fields

    @classmethod
    def from_theta(
        cls,
        theta: torch.Tensor,
        *,
        group: str | None = None,
        groups: tuple[int, int] | None = None,
        width: int | None = None,
        parametrization: str = "sigmoid",
        temperature: float | None = None,
        stretch: tuple[float, float] | None = None,
        top_k: int | None = None,
        pool: str | None = None,
        axis: str | None = None,
        forward: str | None = None,
    ) -> "Gate":
        """A gate reconstituted from a fitted ``theta`` (§2.5 ``file_path``).

        The number a DBM fit *reports* is scored through its eval-mode hard
        split, so an apply document only reproduces the fit if the reloaded
        stage is the same object a trained gate is after ``stage.eval()``:
        same ``theta``, same ``θ > 0`` mask. ``theta`` is therefore copied
        verbatim — no re-init, no thresholding here — and left untrainable,
        because applying a mask is not resuming a fit (a ``file_path``
        featurizer may not appear in ``train.params``).

        ``groups`` is the map the gate is applied through and ``theta`` must
        hold exactly the parameters that map implies (``gate_param_shape``);
        the caller has already checked it against the map the bundle was
        stamped with. ``width`` is the site width, which only an expert-keyed
        gate cannot recover from its table (the routed slot count is the
        site's, not the table's); for the other gates it is derived when
        omitted.

        ``top_k`` (§2.5) replaces the map's threshold with a count: the hard
        mask becomes the ``top_k`` largest units of ``theta``
        (:meth:`ranking`), ``0`` keeps nothing. A count above the unit count
        names units the gate does not have and is refused; the caller
        reports it with the document's words.
        """
        if width is None:
            if group == "expert_neuron":
                raise ValueError("an expert-keyed gate needs the site width")
            width = int(theta.numel()) if groups is None else groups[0] * groups[1]
        shape = gate_param_shape(group, groups, width)
        count = int(theta.numel())
        if count != math.prod(shape):
            raise ValueError(
                f"a gate over {list(shape)} needs {math.prod(shape)} parameters, "
                f"got {count}"
            )
        if top_k is not None and top_k < 0:
            raise ValueError(
                f"top_k={top_k} — a top-k readout keeps a non-negative count"
            )
        if top_k is not None and pool is None and top_k > count:
            # in a pool the cut is the POOLED count, bounded by the pool's
            # units at the link (`link_budget_pools`), not by this member's
            raise ValueError(
                f"top_k={top_k} on a gate of {count} units — a top-k readout keeps "
                f"between 0 and {count} of them"
            )
        if parametrization == "budget" and top_k is None:
            raise ValueError(
                "a budget gate's theta is a ranking with no threshold — a loaded "
                "one is read out at 'top_k' (§2.5)"
            )
        if pool is not None and top_k is None:
            raise ValueError(
                "a loaded gate in a pool is read out at one pooled 'top_k' — a "
                "pool of loaded gates without a cut has nothing to share (§2.5)"
            )
        gate = cls(
            width,
            group=group,
            groups=groups,
            parametrization=parametrization,
            temperature=temperature,
            stretch=stretch,
            pool=pool,
            axis=axis,
            forward=forward,
        )
        gate.theta = torch.nn.Parameter(
            theta.detach().clone().reshape(shape), requires_grad=False
        )
        gate.top_k = top_k
        return gate


# --------------------------------------------------------------------------- #
# the budget pool (§2.5 `pool`)
# --------------------------------------------------------------------------- #


def _schedule_of(schedule: Mapping[str, Any]) -> str:
    """What a schedule's numbers count (§2.5 ``k_schedule.of``): ``patched``
    — units that take the counterfactual, the gate's own count — unless the
    schedule says ``kept``."""
    return str(schedule.get("of", "patched"))


def _as_patched(schedule: Mapping[str, Any], value: int, units: int) -> int:
    """A schedule number as a **patched** count: itself, or its complement
    against ``units`` when the schedule counts kept units. Every internal reader
    of a budget works in patched units, so a ``kept`` schedule is complemented
    exactly once, here."""
    return units - int(value) if _schedule_of(schedule) == "kept" else int(value)


def _draw_from_schedule(schedule: Mapping[str, Any], generator: torch.Generator) -> int:
    """One draw from a ``k_schedule`` (§2.5), in the schedule's own units:
    ``k`` itself when fixed; an integer uniform on ``[low, high]``; or
    ``round(exp(U(log low, log high)))`` clipped to the bounds — a
    log-uniform curriculum, which spends as many steps between 1 and 2 units as
    between 24 and 48, so the ranking is learned at every scale. One scalar
    from the fit's own generator, so the sequence of budgets is a function of
    ``train.seed`` — and the same sequence whether the gate budgets alone or
    for a pool, since a pool draws exactly one scalar per step too."""
    kind = schedule["kind"]
    if kind == "fixed":
        return int(schedule["k"])
    low, high = int(schedule["low"]), int(schedule["high"])
    u = float(torch.rand((), generator=generator, dtype=torch.float64))
    if kind == "uniform":
        return min(high, low + int(u * (high - low + 1)))
    k = int(round(math.exp(math.log(low) + u * (math.log(high) - math.log(low)))))
    return max(low, min(high, k))


def _stable_ranking(theta: torch.Tensor) -> torch.Tensor:
    """Flat unit indices from the largest θ down, ties toward the lower index,
    **sorted on CPU**: ``stable=True`` is load-bearing (a pooled ranking over
    separately fitted bundles has real ties), and a tie-break that varied by
    backend would make one document cut differently on different machines.
    The order comes back on θ's device."""
    order = torch.argsort(theta.detach().cpu(), descending=True, stable=True)
    return order.to(theta.device)


def _solve_shift(theta: torch.Tensor, k: int) -> float:
    """The scalar ``c_k`` with ``Σ σ(θ + c_k) = k`` over a flat float64
    ``theta``, by bisection on a bracket that contains it:
    ``Σσ`` is strictly increasing in ``c`` from 0 to the unit count, so for
    ``0 < k < units`` the root is unique and 60 halvings of
    ``[−max θ − 40, −min θ + 40]`` pin it well below float32 resolution; at the
    poles ``k = 0`` / ``k = units`` there is no finite root and the bracket's
    end is returned (a mask within ``1e−17`` of the pole)."""
    units = theta.numel()
    if not 0 <= k <= units:
        raise ValueError(f"a budget of {k} on a gate of {units} units")
    lo = -float(theta.max()) - 40.0
    hi = -float(theta.min()) + 40.0
    if k == 0:
        return lo
    if k == units:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(torch.sigmoid(theta + mid).sum()) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class BudgetPool:
    """One ranking over several gates (§2.5 ``pool``), in two roles. On a
    **fit** it is the one budget a set of ``budget`` gates share: one
    ``k`` per optimizer step, one shift ``c_k`` solved over the **concatenation**
    of every member's θ, one ranking cut at one count. It owns no parameters
    and is never saved — a bundle stays one gate's θ, stamped with the pool's
    name and unit count — and it exists so that a mask over units that live at
    different sites (every head, every MLP block and the embedding of a model)
    is *one* budgeted fit over ``N`` units rather than ``2L + 1`` fits over
    their own budgets: the curriculum's ``k ~ LogUniform{1, N − 1}`` is over
    ``N``.

    On a **readout** — every member loaded through ``file_path`` with one
    ``top_k`` — it is the pooled cut alone: the members' θ concatenated, ranked
    once, and each member keeps the units whose pooled rank falls below
    ``top_k``. That role takes gates of any map, since a ranking method's
    curve (MIB's node-level CPR) cuts the *joint* ranking of every unit of a
    model, whether the units were fitted as one budget or as one L1-penalised
    sigmoid mask; nothing is drawn and no shift is solved.

    Members are the point's gates authoring this ``pool`` name, in featurizer
    name order, with one ``k_schedule`` (a fit, budget gates only) or one
    ``top_k`` (loaded, any map), linked by :func:`link_budget_pools` once the
    executor has built them all. Every quantity below is in **patched** units — the count a member's
    :meth:`Gate.hard_mask` sums to — with a ``kept`` schedule complemented once
    against the pool's units, exactly as a lone gate complements against its
    own.

    The implicit gradient of the shared shift is the pool's: ``w_i = σ'_i / Σ_j
    σ'_j`` normalised over *every* member's unit, so a member's mask carries a
    gradient into its co-members' θ through the correction term — which is the
    true derivative of ``c_k`` under ``Σ_pool σ(θ + c_k) = k``."""

    def __init__(self, name: str, members: Sequence["Gate"]) -> None:
        if not members:
            raise ValueError(f"pool {name!r} has no members")
        self.name = name
        self.members: tuple[Gate, ...] = tuple(members)
        #: this step's budget, patched units (:meth:`resample`), or ``None``
        self.k: int | None = None
        #: the shift of the last solve, so :meth:`Gate.soft_mask` on a member
        #: reports the mask the last forward used
        self.last_shift: torch.Tensor | None = None
        #: memo of the quantities every member asks for in one forward — the
        #: pooled θ, the shift at a budget, the pooled ranks — keyed by the
        #: members' ``theta._version`` counters (an optimizer step bumps them),
        #: so M members share one solve and one argsort, exactly: θ cannot
        #: move inside a forward
        self._memo: dict[tuple[Any, ...], Any] = {}

    @property
    def units(self) -> int:
        return sum(member.theta.numel() for member in self.members)

    def _version(self) -> tuple[int, ...]:
        return tuple(int(member.theta._version) for member in self.members)

    def _memoized(self, key: tuple[Any, ...], compute: "Callable[[], Any]") -> Any:
        version = self._version()
        if self._memo and next(iter(self._memo))[0] != version:
            self._memo.clear()  # θ moved: everything below is stale
        full = (version, *key)
        if full not in self._memo:
            self._memo[full] = compute()
        return self._memo[full]

    def theta(self) -> torch.Tensor:
        """The pooled ranking's parameter: every member's θ, flat, in member
        order — the object one shift is solved over and one cut is made in.
        Not memoized: it carries the members' autograd graph, and the caller
        that needs the detached copy memoizes that."""
        return torch.cat([member.theta.view(-1) for member in self.members])

    def _theta_detached(self) -> torch.Tensor:
        return self._memoized(
            ("theta",),
            lambda: torch.cat([m.theta.detach().view(-1) for m in self.members]),
        )

    def _schedule(self) -> dict[str, Any]:
        schedule = self.members[0].k_schedule
        if schedule is None:
            raise ValueError(
                f"pool {self.name!r} is loaded (no k_schedule) and is read "
                "out at its members' top_k"
            )
        return schedule

    def resample(self, generator: torch.Generator) -> None:
        """The pool's one draw for the step (:func:`_draw_from_schedule`),
        complemented against the pool's units under a ``kept`` schedule."""
        schedule = self._schedule()
        self.k = _as_patched(
            schedule, _draw_from_schedule(schedule, generator), self.units
        )

    def eval_k(self) -> int:
        """The pooled count the eval-mode split is cut at: the members' one
        ``top_k`` when loaded, else the schedule's ``eval`` (``k`` when fixed),
        complemented under ``kept``."""
        tops = {member.top_k for member in self.members}
        if tops != {None}:
            if len(tops) != 1:
                raise ValueError(
                    f"pool {self.name!r}: its members are read out at "
                    f"different top_k values {sorted(t for t in tops if t is not None)} "
                    "— one pool, one cut through one ranking"
                )
            return int(next(iter(tops)))
        schedule = self._schedule()
        raw = int(schedule["eval"]) if "eval" in schedule else int(schedule["k"])
        return _as_patched(schedule, raw, self.units)

    def shift(self, k: int) -> torch.Tensor:
        """``c_k`` over the pooled θ (:func:`_solve_shift`), as a scalar tensor
        on the members' dtype and device."""
        anchor = self.members[0].theta

        def solve() -> torch.Tensor:
            # on CPU in float64: device-independent to the bit (a seeded fit is
            # the same fit on CPU, CUDA and MPS) and free of the 60 device syncs
            # the on-device loop paid; MPS has no float64 at all
            theta = self._theta_detached().cpu().to(torch.float64)
            return torch.tensor(
                _solve_shift(theta, k), dtype=anchor.dtype, device=anchor.device
            )

        return self._memoized(("shift", int(k)), solve)

    def mask_for(self, member: "Gate", k: int) -> torch.Tensor:
        """``member``'s training mask at the pooled budget ``k``: ``σ(θ_m + c_k)``
        with the shift solved over the pool and its implicit gradient attached
        over the pool (``Gate.budget_mask``'s rule with the sum over every
        member's units), unless the pool's ``stop_grad_shift``."""
        shift = self.shift(k)
        self.last_shift = shift
        if not member.stop_grad_shift:
            pooled = self.theta()
            with torch.no_grad():
                s = torch.sigmoid(pooled + shift)
                weights = s * (1.0 - s)
                weights = weights / weights.sum().clamp_min(1e-30)
            correction = (weights * pooled).sum()
            shift = shift + correction.detach() - correction
        theta = member.theta.view(-1)
        return torch.sigmoid(theta + shift).view(member.theta.shape)

    def ranking(self) -> torch.Tensor:
        """Pooled unit indices from the most to the least kept (``Gate.ranking``
        over the concatenation, ties toward the lower pooled index)."""
        return self._memoized(
            ("ranking",), lambda: _stable_ranking(self._theta_detached())
        )

    def pooled_rank(self) -> torch.Tensor:
        """Each pooled unit's position in :meth:`ranking` (``0`` = kept first)."""

        def invert() -> torch.Tensor:
            order = self.ranking()
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.numel(), device=order.device)
            return rank

        return self._memoized(("rank",), invert)

    def offset(self, member: "Gate") -> int:
        """Where ``member``'s units start in the pooled layout."""
        start = 0
        for candidate in self.members:
            if candidate is member:
                return start
            start += candidate.theta.numel()
        raise ValueError(f"gate is not a member of pool {self.name!r}")

    def member_rank(self, member: "Gate") -> torch.Tensor:
        """``member``'s units' pooled ranks, in θ's layout."""
        start = self.offset(member)
        count = member.theta.numel()
        return self.pooled_rank()[start : start + count].view(member.theta.shape)

    def hard_for(self, member: "Gate") -> torch.Tensor:
        """``member``'s eval-mode split: the units whose pooled rank is below the
        pool's cut (:meth:`eval_k`), as 0/1 in θ's dtype."""
        return (self.member_rank(member) < self.eval_k()).to(member.theta.dtype)


def link_budget_pools(
    featurizers: Mapping[str, FeaturizerSpec],
    stages: Mapping[str, Stage],
    build: Callable[[str], Stage],
) -> None:
    """Attach one :class:`BudgetPool` to every gate of each ``pool`` the
    document authors (§2.5), building the members not yet built through
    ``build`` — the executor's own ``stage(name)`` — so a pool is complete
    before any member's mask is computed. Idempotent: a pool whose members all
    carry the same pool object is left alone, which is what makes the recursion
    through ``build`` (a sibling's build calls back here) terminate.

    Refuses (P2) a pool that is not a pool: a fitted member under another map
    (only the budget map has a budget to share; a *loaded* member of any map
    joins a pooled readout), members that disagree on ``k_schedule``,
    ``stop_grad_shift`` or ``top_k``, a mix of fitted and loaded members, or a
    schedule number above the pool's units."""
    by_pool: dict[str, list[str]] = {}
    for name in sorted(featurizers):
        spec = featurizers[name]
        if spec.kind == "gate" and isinstance(spec.pool, str):
            by_pool.setdefault(spec.pool, []).append(name)
    linking = _LINKING.setdefault(id(stages), set())
    for pool_name, names in by_pool.items():
        if pool_name in linking:
            continue  # a sibling's build re-entered us: the outer frame finishes
        linking.add(pool_name)
        try:
            _link_pool(pool_name, names, stages, build)
        finally:
            linking.discard(pool_name)
            if not linking:
                _LINKING.pop(id(stages), None)


#: Pools being linked right now, per stage map (``id(stages)``): a member's
#: build calls back into :func:`link_budget_pools`, which would otherwise
#: recurse once per member — depth ``M`` and ``O(M·F)`` re-scans on a 53-gate
#: pool. With the guard the nested call returns at once and the outer frame,
#: which is building the members in order anyway, links the pool: depth 2.
_LINKING: dict[int, set[str]] = {}


def _link_pool(
    pool_name: str,
    names: Sequence[str],
    stages: Mapping[str, Stage],
    build: Callable[[str], Stage],
) -> None:
    members: list[Gate] = []
    for name in names:
        stage = stages[name] if name in stages else build(name)
        if not isinstance(stage, Gate) or (
            stage.parametrization != "budget" and stage.top_k is None
        ):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r} is in pool {pool_name!r} but is neither a "
                "budget gate nor a loaded gate read out at top_k — a pool "
                "shares one budget (which only the budget map draws) or one "
                "pooled cut (§2.5)",
            )
        members.append(stage)
    pools = {id(member.pool) for member in members if member.pool is not None}
    if len(pools) == 1 and all(member.pool is not None for member in members):
        return  # linked already
    loaded = {member.k_schedule is None for member in members}
    if len(loaded) != 1:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r} mixes fitted and loaded gates — a pool is one "
            "ranking, fitted together or read out together (§2.5)",
        )
    schedules = {json.dumps(member.k_schedule, sort_keys=True) for member in members}
    stops = {member.stop_grad_shift for member in members}
    tops = {member.top_k for member in members}
    maps = {member.parametrization for member in members}
    if len(schedules) != 1 or len(stops) != 1 or len(tops) != 1 or len(maps) != 1:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r}: its members {list(names)} disagree on "
            "parametrization, k_schedule, stop_grad_shift or top_k — one pool "
            "is one ranking on one scale, draws one budget and is cut at one "
            "count (§2.5)",
        )
    pool = BudgetPool(pool_name, members)
    for name, member in zip(names, members):
        stamped = member.stamped_pool_units
        if stamped is not None and stamped != pool.units:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: its bundle was fitted in a pool of "
                f"{stamped} units but pool {pool_name!r} here has {pool.units} "
                "— a pooled theta ranks against exactly its co-members, and "
                "these are not them (§2.5, rule 15)",
            )
    top_k = members[0].top_k
    if top_k is not None and top_k > pool.units:
        raise ProtocolError(
            "P2",
            f"pool {pool_name!r}: top_k={top_k} but the pool has {pool.units} "
            "units — a pooled cut keeps at most every unit (§2.5)",
        )
    schedule = members[0].k_schedule
    if schedule is not None:
        for key in ("k", "low", "high", "eval"):
            if key in schedule and int(schedule[key]) > pool.units:
                raise ProtocolError(
                    "P2",
                    f"pool {pool_name!r}: k_schedule.{key}={schedule[key]} but "
                    f"the pool has {pool.units} units — a budget keeps at most "
                    "every unit (§2.5)",
                )
    for member in members:
        member.pool = pool


@dataclasses.dataclass
class FeaturizerStack:
    """A left-to-right composition of stages with a per-stage ``err`` list
    (§2.5). ``names`` aligns with ``stages`` for train-param addressing."""

    names: tuple[str, ...]
    stages: tuple[Stage, ...]

    def featurize(
        self, x: torch.Tensor, *, routing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """``routing`` is the routing table gathered at the same rows and
        positions as ``x`` (``(..., top_k)`` expert ids), which only a stage
        with ``needs_routing`` consumes; every other stage ignores it."""
        errs: list[torch.Tensor | None] = []
        for stage in self.stages:
            if stage.needs_routing:
                x, err = stage.featurize(x, routing=routing)  # type: ignore[call-arg]
            else:
                x, err = stage.featurize(x)
            errs.append(err)
        return x, errs

    def inverse(
        self, f: torch.Tensor, errs: Sequence[torch.Tensor | None]
    ) -> torch.Tensor:
        for stage, err in zip(reversed(self.stages), reversed(list(errs))):
            f = stage.inverse(f, err)
        return f

    @property
    def is_identity(self) -> bool:
        return all(isinstance(s, Identity) for s in self.stages)

    @property
    def needs_routing(self) -> bool:
        """Whether any stage joins its parameters to the activation through
        the routing table (an expert-keyed gate)."""
        return any(stage.needs_routing for stage in self.stages)


def stage_output_width(spec: FeaturizerSpec, input_width: int) -> int | None:
    """The feature width a stage emits, given its input width — the chain
    rule of §2.5's composition: subspace/pca project to ``k``, the
    width-preserving kinds pass ``input_width`` through, and a loaded SAE's
    dictionary size is unknowable from the spec alone (``None``)."""
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    if kind in ("subspace", "pca"):
        return spec.k if isinstance(spec.k, int) else None
    if kind == "sae":
        return None  # the dictionary size lives in the bundle, not the spec
    return input_width


def _stage_width(stage: Stage) -> int | None:
    """The input width a built stage was sized for (for cache-reuse checks)."""
    params = stage.slot_params()
    if isinstance(stage, (Subspace, LoadedLinear)):
        weight = stage.weight if isinstance(stage, LoadedLinear) else params["weight"]
        return int(weight.shape[0])
    if isinstance(stage, Gate):
        return stage.width
    if isinstance(stage, Standardize):
        return int(stage.mu.shape[0])
    return None


@dataclasses.dataclass(frozen=True)
class StageRecipe:
    """Where a document uses one declared featurizer, as far as building it
    goes: the width it is sized to (the site's, folded through the stages
    before it in the chain), the site's shape and component a grouped gate
    derives its map from, and the addressed window's length a position gate
    is sized by. Plain data — with the featurizer specs, the seed and the
    point's coordinates it is everything :func:`build_recipe` needs, so a
    stage can be built where no executor is."""

    name: str
    width: int
    site_shape: FeatureShape | None = None
    site_component: str | None = None
    position_width: int | None = None


def build_recipe(
    recipe: StageRecipe,
    specs: Mapping[str, FeaturizerSpec],
    *,
    stage_cache: dict[str, Stage],
    load_tensors: Any = None,
    load_table: Any = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    coords: Mapping[str, Any] | None = None,
    model_info: ModelInfo | None = None,
) -> Stage:
    """The one stage ``recipe`` names, built into ``stage_cache`` (or found
    there) by :func:`build_stack` — the single construction path, whether an
    executor asks (``ExecutorBase.stage``) or a fit built from its spec
    (``training.state.build_stages``)."""
    build_stack(
        recipe.name,
        dict(specs),
        width=recipe.width,
        load_tensors=load_tensors,
        load_table=load_table,
        stage_cache=stage_cache,
        device=device,
        seed=seed,
        coords=coords,
        site_shape=recipe.site_shape,
        site_component=recipe.site_component,
        model_info=model_info,
        position_width=recipe.position_width,
    )
    return stage_cache[recipe.name]


def build_stack(
    ref: Any,
    specs: dict[str, FeaturizerSpec],
    *,
    width: int,
    load_tensors: Any,
    stage_cache: dict[str, Stage],
    device: str | torch.device = "cpu",
    seed: int = 0,
    coords: Mapping[str, Any] | None = None,
    site_shape: FeatureShape | None = None,
    site_component: str | None = None,
    model_info: ModelInfo | None = None,
    load_table: Any = None,
    position_width: int | None = None,
) -> FeaturizerStack:
    """Build (or reuse from ``stage_cache``) the stack a read/write
    references. ``width`` is the SITE width; each later stage in a
    composition is sized to the *previous stage's output* (the §2.5 chain —
    a gate after a k=3 rotation is a 3-wide gate). ``load_tensors`` supplies
    loaded bundles; caching by name keeps one stage instance per declared
    featurizer, so training one featurizer updates every use site — a name
    reused at a different chain width is a contradiction and refuses.

    ``site_shape`` and ``site_component`` describe the site the chain starts
    at, and ``model_info`` the model; a grouped gate derives its map from them
    (:func:`~causalab.protocol.registry.gate_group_map` — the head layout for
    ``group: head``, the expert table for ``group: expert_neuron``) and
    refuses without them, or after a stage that changed the coordinate basis
    (§5.23) — it groups the component's own coordinates, and after a rotation
    "head" names nothing; a ``standardize`` before it is per coordinate and
    legal.

    ``device`` is the run's device; stages are built on CPU and moved there
    (module docstring). The ``"cpu"`` default leaves CPU-only callers alone.

    ``seed`` is the document's featurizer-init seed (``train.seed``, 0 with no
    fit — ``executor.document_seed``). Explicit rather than read from the global
    RNG because this also runs on apply/inference paths, where a global-RNG init
    would make a rotation depend on construction order. The cache is keyed by
    name, so a cached stage built from a different seed refuses, as with width.

    A ``subspace`` spec may name its **own** ``seed`` (§2.5), which wins. That is
    what makes an *untrained* subspace a random rank-k basis a document can
    sweep — the matched-k random-subspace control, which otherwise needs a
    ``train`` block it has nothing to train.

    ``coords`` are the executing point's sweep coordinates: they select the
    matching entry of a swept bundle when the spec authored no ``entry``
    (§2.5)."""
    if ref is None:
        return FeaturizerStack(names=(), stages=(Identity(),))
    chain = (ref,) if isinstance(ref, str) else tuple(ref)
    stages: list[Stage] = []
    running: int | None = width
    for index, name in enumerate(chain):
        spec = specs[name]
        # a `subspace` may author its own seed (§2.5); absent, the document's
        # seed stands, so nothing about an existing document changes
        stage_seed = spec.seed if isinstance(spec.seed, int) else seed
        groups = _gate_groups(
            name,
            spec,
            width=running,
            before=tuple(
                (
                    member,
                    specs[member].kind
                    if isinstance(specs[member].kind, str)
                    else "identity",
                )
                for member in chain[:index]
            ),
            site_shape=site_shape,
            site_component=site_component,
            model_info=model_info,
        )
        # §2.5 `axis`: a position gate is sized by the addressed window's
        # length, which the executor derives from the entry's `span`, not by
        # the feature width flowing through the chain (which it passes on)
        positional = spec.kind == "gate" and spec.axis == "position"
        stage_width = position_width if positional else running
        if positional and position_width is None:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r} is a position gate but the entry using it "
                "addresses no fixed window — its `pos` must be a `span` [a, b) "
                "of two or more positions (§2.5 axis)",
            )
        if name in stage_cache:
            stage = stage_cache[name]
            built_for = _stage_width(stage)
            if (
                built_for is not None
                and stage_width is not None
                and built_for != stage_width
            ):
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is used at width {stage_width} here but was "
                    f"built for width {built_for} — one featurizer, one width"
                    + (
                        " (a position gate's width is its window's length)"
                        if positional
                        else ""
                    ),
                )
            built_groups = getattr(stage, "groups", None)
            if isinstance(stage, Gate) and built_groups != groups:
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is grouped {groups} here but the cached "
                    f"stage was built over {built_groups} — one featurizer, one "
                    "group map",
                )
            built_seed = getattr(stage, "seed", None)
            if built_seed is not None and built_seed != stage_seed:
                raise ProtocolError(
                    "P2",
                    f"featurizer {name!r} is used at init seed {stage_seed} here "
                    f"but the "
                    f"cached stage was initialised from seed {built_seed} — one "
                    "featurizer, one seed; a stage cache belongs to one point, so "
                    "two points differing in train.seed must not share one",
                )
        else:
            if stage_width is None:
                raise ProtocolError(
                    "P2",
                    f"cannot size featurizer {name!r}: the preceding stage's "
                    "output width is not derivable from its spec",
                )
            stage = _build_stage(
                name,
                spec,
                width=stage_width,
                load_tensors=load_tensors,
                seed=stage_seed,
                coords=coords,
                groups=groups,
                load_table=load_table,
            )
            stage.to(device)  # parameters and registered buffers alike
            # inference documents get eval semantics (a gate's hard split);
            # the train loop flips modes around its steps explicitly
            stage.eval()
            stage_cache[name] = stage
        stages.append(stage)
        if isinstance(stage, Sae):
            running = int(stage.enc.shape[1])
        elif running is not None:
            running = stage_output_width(spec, running)
    return FeaturizerStack(names=chain, stages=tuple(stages))


def _gate_groups(
    name: str,
    spec: FeaturizerSpec,
    *,
    width: int | None,
    before: Sequence[tuple[str, str]],
    site_shape: FeatureShape | None,
    site_component: str | None,
    model_info: ModelInfo | None,
) -> tuple[int, int] | None:
    """The group map a grouped gate is built over — ``(heads, head_dim)`` or
    ``(num_experts, d_expert)`` — or ``None`` for every other stage (module
    docstring, ``group``). ``before`` is ``(name, kind)`` of every stage ahead
    of this one in its chain: rule 23 already refused a grouped gate that is
    not first at load, so a refusal here is the executor holding the same line
    rather than a document problem."""
    if spec.kind != "gate" or not isinstance(spec.group, str):
        return None
    if before:
        member, kind = before[0]
        raise ProtocolError(
            "P2",
            f"featurizer {name!r} is grouped by {spec.group} but follows {member!r} "
            f"({kind!r}) in its chain — a grouped gate acts on the component's own "
            "coordinates, so it must be the first stage of its chain",
        )
    if site_shape is None or site_component is None or width is None:
        raise ProtocolError(
            "P2",
            f"featurizer {name!r} is grouped by {spec.group} but the site it is "
            "built for declares no shape to derive the groups from",
        )
    try:
        return gate_group_map(
            spec.group, site_shape, width, component=site_component, info=model_info
        )
    except ValidationError as err:
        raise ProtocolError("P2", f"featurizer {name!r}: {err.message}") from err


def _check_entry_identity(
    record: Mapping[str, Any],
    spec: FeaturizerSpec,
    what: str,
    *,
    groups: tuple[int, int] | None = None,
) -> None:
    """Refuse an entry whose stamped fit contradicts the spec that selected
    it (§2.5).

    The load-time check (``loader.check_loaded_featurizers``) covers a
    bundle whose entry is knowable there; when the selection is the
    executing point's — implicit matching against a swept producer — this is
    where the claim is finally tested, so "apply the k=8 fit" cannot quietly
    apply the k=32 one. Only the per-entry fields are compared: everything
    file-level was already checked at load.

    A hard-concrete gate's ``stretch`` is compared the same way in both
    directions, since the hard split's threshold is derived from it.

    A gate's grouping is compared in both directions: a bundle fitted per
    head or per expert neuron is not applicable through a per-coordinate gate,
    and the stamped ``group_map`` must be the one this site derives
    (``groups``) — the same head count and head width, or the same expert
    table, not merely the same parameter count.
    """
    for field, value in (
        ("k", spec.k),
        ("parametrization", spec.parametrization),
    ):
        if value is None or not isinstance(value, (int, str)):
            continue
        stamped = record.get(field)
        if stamped is not None and str(stamped) != str(value):
            raise ProtocolError(
                "P2",
                f"{what}: the document says {field}={value!r} but the selected "
                f"entry was fitted with {field}={stamped!r}",
            )
    if spec.kind != "gate":
        return
    # a position gate's θ is one entry per token position; a feature gate's
    # one per coordinate — compared both ways, an unstamped bundle being a
    # feature gate (§2.5 axis)
    declared_axis = spec.axis if isinstance(spec.axis, str) else None
    stamped_axis = record.get("axis")
    if (stamped_axis or None) != declared_axis:
        raise ProtocolError(
            "P2",
            f"{what}: the document's gate runs over "
            f"{declared_axis or 'coordinates'} but the selected entry was fitted "
            f"over {stamped_axis or 'coordinates'} — a mask over positions is not "
            "a mask over coordinates",
        )
    # a gate's map from theta to mask decides its hard split (θ > 0 against
    # θ > ½), so a bundle fitted under one map is not a mask under the other.
    # Unstamped means fitted before the field existed, i.e. sigmoid — both
    # directions are compared, like `group` below
    declared = (
        spec.parametrization if isinstance(spec.parametrization, str) else "sigmoid"
    )
    stamped_param = record.get("parametrization") or "sigmoid"
    if str(stamped_param) != declared:
        raise ProtocolError(
            "P2",
            f"{what}: the document's gate is parametrized {declared!r} but the "
            f"selected entry was fitted {stamped_param!r} — the two hard masks "
            "differ (θ > 0 against θ > ½), so a mask under one map is not a "
            "mask under the other",
        )
    if declared == "hard_concrete":
        # the stretch decides WHERE the hard split falls, θ > logit((½−γ)/(ζ−γ)),
        # so it is compared like the map, in both directions: an authored
        # stretch against the stamped one, and a non-default stamp against a
        # spec authoring none (which implies the default). As JSON, not
        # str(tuple): the stamp is '[-0.1, 1.1]'. This is the check that
        # reaches a swept producer whose entry the executing point selects —
        # the load-time check bails on that case before comparing anything
        stamped_stretch = record.get("stretch")
        if stamped_stretch is not None:
            fitted = [
                float(v)
                for v in (
                    json.loads(stamped_stretch)
                    if isinstance(stamped_stretch, str)
                    else stamped_stretch
                )
            ]
            wanted = (
                [float(v) for v in spec.stretch]
                if spec.stretch is not None
                else list(HARD_CONCRETE_STRETCH)
            )
            if fitted != wanted:
                authored = (
                    f"stretch {wanted}"
                    if spec.stretch is not None
                    else f"no stretch (the default {wanted})"
                )
                raise ProtocolError(
                    "P2",
                    f"{what}: the document's gate declares {authored} but the "
                    f"selected entry was fitted at stretch {fitted} — the two hard "
                    "masks split θ at different thresholds, so declare the same "
                    "stretch (§2.5)",
                )
    group = spec.group if isinstance(spec.group, str) else None
    stamped_group = record.get("group")
    if stamped_group is not None and str(stamped_group) != str(group):
        raise ProtocolError(
            "P2",
            f"{what}: the document declares group={group!r} on the gate but the "
            f"selected entry was fitted with group={stamped_group!r} — a mask "
            "over one kind of unit is not a mask over another",
        )
    stamped_map = record.get("group_map")
    if group is not None and groups is not None and stamped_map is not None:
        want = list(groups)
        got = json.loads(stamped_map) if isinstance(stamped_map, str) else stamped_map
        if list(got) != want:
            raise ProtocolError(
                "P2",
                f"{what}: the site here has {_describe_map(group, want)} but the "
                f"fitted gate was grouped over {_describe_map(group, got)} — same "
                "group kind, different units",
            )


def _describe_map(group: str, group_map: Sequence[int]) -> str:
    """A group map in words: ``8 heads of 32 coordinates``, ``128 experts
    of 32 neurons`` or ``one site of 768 coordinates``."""
    if group == "expert_neuron":
        return f"{group_map[0]} experts of {group_map[1]} neurons"
    if group == "site":
        return f"one site of {group_map[1]} coordinates"
    return f"{group_map[0]} heads of {group_map[1]} coordinates"


def _build_stage(
    name: str,
    spec: FeaturizerSpec,
    *,
    width: int,
    load_tensors: Any,
    seed: int = 0,
    coords: Mapping[str, Any] | None = None,
    groups: tuple[int, int] | None = None,
    load_table: Any = None,
) -> Stage:
    kind = spec.kind if isinstance(spec.kind, str) else "identity"
    group = spec.group if isinstance(spec.group, str) else None
    if isinstance(spec.file_path, str):
        slots = FEATURIZER_SLOTS.get(kind, ())
        if not slots:
            raise ProtocolError(
                "P2", f"featurizer kind {kind!r} cannot be loaded from a file"
            )
        want, implicit = entry_selection(spec.entry, coords, name)
        what = f"featurizer {name!r} ({spec.file_path})"
        point = load_tensors(spec.file_path).point(
            slots[0], want, what=what, implicit=implicit
        )
        # the entry's record carries what a swept producer stamped per entry;
        # the header identity carries what it stamped file-wide (a
        # single-point fit's `parametrization`, `group`) — the check reads
        # both, entry over file, as `entry_identity` does
        _check_entry_identity(
            {**point.identity, **point.record}, spec, what, groups=groups
        )
        slot = point.tensor
        if kind in ("subspace", "pca"):
            return LoadedLinear(kind, slot("weight"))
        if kind == "standardize":
            return Standardize(slot("mu"), slot("sigma"))
        if kind == "sae":
            return Sae(slot("enc"), slot("dec"), slot("b_enc"), slot("b_dec"))
        if kind == "gate":
            theta = slot("theta")
            expected = math.prod(gate_param_shape(group, groups, width))
            positional = spec.axis == "position"
            if theta.numel() != expected:
                # §2.5 `axis`: a position gate's units are positions and its
                # layout is the window, so the message counts what θ counts
                unit = (
                    "positions"
                    if positional
                    else {
                        None: "wide",
                        "head": "heads",
                        "expert_neuron": "expert neurons",
                        "site": "site-wide unit(s)",
                    }[group]
                )
                where = "window" if positional else "site"
                raise ProtocolError(
                    "P2",
                    f"{what}: the fitted gate is {theta.numel()} {unit} but the "
                    f"{where} here is {expected} — a mask is a set of units of "
                    "one activation, so it only applies at the layout it was "
                    "fitted at",
                )
            top_k = _concrete_top_k(spec)
            if _gate_parametrization(spec) == "budget" and top_k is None:
                raise ProtocolError(
                    "P2",
                    f"{what}: the fitted gate is a budget gate — its theta is a "
                    "ranking with no threshold, so the document names the cut: "
                    "author 'top_k' (§2.5)",
                )
            if (
                top_k is not None
                and top_k > expected
                and not isinstance(spec.pool, str)
            ):
                unit = (
                    "positions"
                    if positional
                    else {
                        None: "coordinates",
                        "head": "heads",
                        "expert_neuron": "expert neurons",
                        "site": "site-wide unit(s)",
                    }[group]
                )
                raise ProtocolError(
                    "P2",
                    f"{what}: top_k={top_k} but the gate has {expected} {unit} — a "
                    "top-k readout keeps at most every unit (§2.5)",
                )
            gate = Gate.from_theta(
                theta,
                group=group,
                groups=groups,
                width=width,
                parametrization=_gate_parametrization(spec),
                temperature=_concrete_float(spec.temperature),
                stretch=spec.stretch,
                top_k=top_k,
                pool=spec.pool if isinstance(spec.pool, str) else None,
                axis=spec.axis if isinstance(spec.axis, str) else None,
                forward=spec.forward,
            )
            stamped_units = {**point.identity, **point.record}.get("pool_units")
            if stamped_units is not None:
                # compared with the pool the document assembles, at the link
                gate.stamped_pool_units = int(stamped_units)
            return gate
        raise ProtocolError(
            "P2", f"featurizer kind {kind!r} cannot be loaded from a file"
        )
    if kind == "identity":
        return Identity()
    if kind == "subspace":
        k = spec.k if isinstance(spec.k, int) else None
        parametrization = (
            spec.parametrization if isinstance(spec.parametrization, str) else "cayley"
        )
        if k is None:
            raise ProtocolError("P2", f"subspace featurizer {name!r} needs k")
        if spec.init is None:
            return Subspace(width, k, parametrization, seed=seed)
        basis, identity = _init_basis(
            name, spec, load_tensors, width=width, k=k, coords=coords
        )
        return Subspace(
            width, k, parametrization, seed=seed, init=basis, init_identity=identity
        )
    if kind == "gate":
        # θ starts at the midpoint mask, at a declared fill, or at a saved
        # theta — no draw anywhere, so nothing for `seed` to influence
        start, start_identity, start_scores = _gate_start(
            name,
            spec,
            load_tensors,
            group=group,
            groups=groups,
            width=width,
            coords=coords,
            load_table=load_table,
        )
        gate = Gate(
            width,
            group=group,
            groups=groups,
            axis=spec.axis if isinstance(spec.axis, str) else None,
            forward=spec.forward,
            parametrization=_gate_parametrization(spec),
            init=start,
            init_identity=start_identity,
            temperature=_concrete_float(spec.temperature),
            stretch=spec.stretch,
            k_schedule=_concrete_k_schedule(
                name,
                spec,
                # a pooled schedule counts the POOL's units, which only the
                # link (`link_budget_pools`) can check once every member exists
                units=None
                if isinstance(spec.pool, str)
                else math.prod(gate_param_shape(group, groups, width)),
            ),
            stop_grad_shift=spec.stop_grad_shift is True,
            pool=spec.pool if isinstance(spec.pool, str) else None,
            dead=spec.dead,
        )
        gate.init_scores = start_scores
        return gate
    raise ProtocolError(
        "P2",
        f"featurizer {name!r} of kind {kind!r} needs a file_path — this engine "
        "does not fit it from data at run start",
    )


def _concrete_k_schedule(
    name: str, spec: FeaturizerSpec, *, units: int | None
) -> dict[str, Any] | None:
    """A budget gate's resolved ``k_schedule`` (§2.5), checked against the unit
    count the site gives the gate — a budget above the units names units the
    gate does not have (``units`` is ``None`` for a pooled gate, whose count is
    the pool's and is checked at the link) — and refused when a sweep reaches
    the build, like an unresolved temperature: a different cut is a different
    fit."""
    if spec.k_schedule is None:
        return None
    out: dict[str, Any] = {}
    for key, value in spec.k_schedule.items():
        if key in ("kind", "of"):
            out[key] = value
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: k_schedule.{key} must be one integer by the "
                f"time the stage is built, got {value!r}",
            )
        if units is not None and value > units:
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: k_schedule.{key}={value} but the gate has "
                f"{units} units — a budget keeps at most every unit (§2.5)",
            )
        out[key] = value
    return out


def _checked_k_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """The schedule as the gate keeps it: kind, ``k`` or the bounds, and the
    eval cut — every value a non-negative integer, the bounds ordered,
    ``log_uniform`` from 1, a sampled kind naming its ``eval``."""
    kind = schedule.get("kind")
    if kind not in ("fixed", "uniform", "log_uniform"):
        raise ValueError(f"unknown k_schedule kind {kind!r}")
    out: dict[str, Any] = {"kind": kind}
    if "of" in schedule:
        if schedule["of"] not in ("patched", "kept"):
            raise ValueError(
                f"k_schedule.of is 'patched' or 'kept', got {schedule['of']!r}"
            )
        out["of"] = schedule["of"]
    keys = ("k",) if kind == "fixed" else ("low", "high")
    for key in (*keys, "eval"):
        if key in schedule:
            value = schedule[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"k_schedule.{key} is a non-negative integer, got {value!r}"
                )
            out[key] = value
        elif key != "eval":
            raise ValueError(f"a {kind} k_schedule needs {key!r}")
    if kind != "fixed":
        if out["low"] > out["high"]:
            raise ValueError("k_schedule bounds are ordered")
        if kind == "log_uniform" and out["low"] < 1:
            raise ValueError("a log_uniform k_schedule starts at 1 or above")
        if "eval" not in out:
            raise ValueError(
                f"a {kind} k_schedule names 'eval', the cut the hard mask is read at"
            )
    return out


def _concrete_float(value: Any) -> float | None:
    """A resolved numeric field, or ``None`` when unauthored. Anything else —
    an unresolved sweep reaching the build — is refused rather than silently
    read as the default: a quietly different β is a different fit."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ProtocolError(
        "P2",
        f"a gate's temperature must be one number by the time the stage is built, "
        f"got {value!r} — a swept value is resolved per point before the build",
    )


def _concrete_top_k(spec: FeaturizerSpec) -> int | None:
    """A loaded gate's resolved ``top_k`` (§2.5), or ``None`` when unauthored;
    a sweep reaching the build is refused like an unresolved temperature —
    a different cut is a different mask."""
    if spec.top_k is None:
        return None
    if isinstance(spec.top_k, int) and not isinstance(spec.top_k, bool):
        return spec.top_k
    raise ProtocolError(
        "P2", f"gate top_k did not resolve to an integer (got {spec.top_k!r})"
    )


def _gate_parametrization(spec: FeaturizerSpec) -> str:
    """A gate spec's theta→mask map, ``sigmoid`` when unauthored (§2.5)."""
    return spec.parametrization if isinstance(spec.parametrization, str) else "sigmoid"


def _gate_start(
    name: str,
    spec: FeaturizerSpec,
    load_tensors: Any,
    *,
    group: str | None,
    groups: tuple[int, int] | None,
    width: int,
    coords: Mapping[str, Any] | None,
    load_table: Any = None,
) -> tuple[float | torch.Tensor | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Where a gate fit starts (§2.5 ``init``), and what a bundle records
    about it: ``(None, None, None)`` for the midpoint mask, ``(fill, None,
    None)`` for a declared mask value, a saved ``theta`` with the ``init_*``
    provenance keys — the :func:`_init_basis` contract, for a gate — or a
    theta read off a score table (:func:`_scores_start`) with the table's
    digest as ``init_digest`` and the third member the record
    ``fit_diagnostics`` keeps of it.

    A saved start has to be a theta **of this gate**: same group and map, same
    parametrization (``θ > 0`` and ``θ > ½`` split a start as differently as
    they split a mask), the unit count this layout has — the same
    :func:`_check_entry_identity` a loaded gate passes, read over the entry's
    record and the header identity. It must carry its own ``produced_by``,
    since that digest is how the fit record names its start."""
    if spec.init is None:
        return None, None, None
    if "fill" in spec.init:
        fill = spec.init["fill"]
        if isinstance(fill, bool) or not isinstance(fill, (int, float)):
            raise ProtocolError(
                "P2",
                f"featurizer {name!r}: init.fill is unresolved ({fill!r}) — a "
                "swept fill is a coordinate, resolved per point before the build",
            )
        return float(fill), None, None
    if "from_scores" in spec.init:
        return _scores_start(
            name,
            spec,
            spec.init["from_scores"],
            load_table,
            group=group,
            groups=groups,
            width=width,
        )
    (slot,) = FEATURIZER_SLOTS["gate"]
    init_path = str(spec.init["file_path"])
    want, implicit = entry_selection(spec.init.get("entry"), coords, name)
    what = f"featurizer {name!r} init ({init_path})"
    point = load_tensors(init_path).point(slot, want, what=what, implicit=implicit)
    _check_entry_identity({**point.identity, **point.record}, spec, what, groups=groups)
    theta = point.tensor(slot)
    expected = math.prod(gate_param_shape(group, groups, width))
    if theta.numel() != expected:
        raise ProtocolError(
            "P2",
            f"{what}: the saved theta holds {theta.numel()} parameters but the "
            f"gate here has {expected} units — a start is a theta of this gate, "
            "at the layout it is fitted at",
        )
    produced_by = point.identity.get("produced_by")
    if produced_by is None:
        raise ProtocolError(
            "P2",
            f"{what}: the selected entry carries no 'produced_by' — the fit "
            "record names the start it began from by that digest, so a theta "
            "without one cannot seed a fit (§2.5)",
        )
    start = theta.detach().to("cpu", torch.float32).contiguous()
    identity = {
        "init_produced_by": produced_by,
        "init_trained_on": point.identity.get("trained_on"),
        "init_digest": hashlib.sha256(start.numpy().tobytes()).hexdigest(),
    }
    return start, identity, None


def gate_poles(
    parametrization: str, stretch: tuple[float, float] | None
) -> tuple[float, float]:
    """``(dropped, kept)`` — the two values of ``theta`` a decisive start is
    written on under one θ→mask map (§2.5): ``(0, 1)`` under ``clamp``, whose
    parameter lives on the unit interval and splits at ½; one unit either
    side of the hard threshold under ``sigmoid`` (``∓1`` around 0) and
    ``hard_concrete`` (around ``logit((½−γ)/(ζ−γ))``, the same arithmetic
    :meth:`Gate.hard_threshold` uses). The convention ``analysis.random_mask``
    writes its controls on, stated once so a ranking start and a size-matched
    control are the same kind of object."""
    if parametrization == "clamp":
        return (0.0, 1.0)
    threshold = 0.0
    if parametrization == "hard_concrete":
        lo, hi = HARD_CONCRETE_STRETCH if stretch is None else stretch
        threshold = hard_concrete_threshold((float(lo), float(hi)))
    return (threshold - 1.0, threshold + 1.0)


def _scores_start(
    name: str,
    spec: FeaturizerSpec,
    scores: Mapping[str, Any],
    load_table: Any,
    *,
    group: str | None,
    groups: tuple[int, int] | None,
    width: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """A gate's start read off a per-unit score table (§2.5
    ``init.from_scores``; rule 32 is what is checked here).

    The table is a saved metric table — a list of row objects — filtered by
    ``where`` (every named column equal to its literal), then read by its
    ``unit`` column(s) and ``value`` column. After the filter it has to name
    **every unit of this gate exactly once**: a unit index is a position in
    ``theta`` — ``[0, units)`` for a one-axis theta, an ``(i, j)`` pair for a
    two-axis one (``expert_neuron``'s expert table) — and a table that skips
    or repeats one would seed a mask nobody authored. Under ``keep`` the top
    ``keep`` units by score (ties by unit index, so the start is a function
    of the table alone) go on the kept pole of the map and the rest on the
    dropped one (:func:`gate_poles`): a decisive start, and the attribution-
    or magnitude-pruning baseline when the gate is not trained. Under
    ``scale`` theta is ``midpoint + scale · z`` with ``z`` the population
    z-score of the values (a constant column is all-midpoint), clipped to the
    unit interval under ``clamp`` — the midpoint being the θ whose mask is ½,
    so a zero score is exactly the untouched start.

    The table's bytes enter the bundle identity as ``init_digest``; the
    canonical form carries them as ``init.from_scores.content_digest``. A
    score table has no ``ArtifactIdentity`` of its own — it is a JSON table,
    not a tensor bundle — so no ``init_produced_by`` is stamped."""
    if load_table is None:
        raise ProtocolError(
            "P2",
            f"featurizer {name!r}: init.from_scores needs a table loader and this "
            "build has none — the executor resolves the table through the run's "
            "artifact store",
        )
    what = f"featurizer {name!r} init.from_scores ({scores['file_path']})"
    rows, raw = load_table(str(scores["file_path"]))
    where = scores.get("where") or {}
    selected = [
        row for row in rows if all(row.get(col) == lit for col, lit in where.items())
    ]
    shape = gate_param_shape(group, groups, width)
    unit_columns = scores["unit"]
    unit_columns = (
        [unit_columns] if isinstance(unit_columns, str) else list(unit_columns)
    )
    where_path = f"featurizers.{name}.init.from_scores"
    if len(unit_columns) != len(shape):
        raise ValidationError(
            32,
            f"{what}: the gate's theta has {len(shape)} axis/axes "
            f"{list(shape)} but 'unit' names {len(unit_columns)} column(s) "
            f"{unit_columns} — one unit column per axis",
            path=f"{where_path}.unit",
        )
    value_column = str(scores["value"])
    values = torch.full(shape, float("nan"), dtype=torch.float32)
    seen: set[tuple[int, ...]] = set()
    for row in selected:
        try:
            index = tuple(int(row[col]) for col in unit_columns)
            value = float(row[value_column])
        except (KeyError, TypeError, ValueError) as err:
            raise ValidationError(
                32,
                f"{what}: a row lacks an integer unit under {unit_columns} or a "
                f"number under {value_column!r}: {row!r} ({err})",
                path=where_path,
            ) from None
        if not all(0 <= i < n for i, n in zip(index, shape)):
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} is outside the gate's theta {list(shape)}",
                path=f"{where_path}.unit",
            )
        if index in seen:
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} is named twice — narrow the table "
                "with 'where' so each unit has one score",
                path=where_path,
            )
        if math.isnan(value):
            raise ValidationError(
                32,
                f"{what}: unit {list(index)} has no score (NaN)",
                path=f"{where_path}.value",
            )
        seen.add(index)
        values[index] = value
    units = math.prod(shape)
    if len(seen) != units:
        raise ValidationError(
            32,
            f"{what}: the table names {len(seen)} of the gate's {units} units "
            + (f"after where={dict(where)} " if where else "")
            + "— a start needs a score for every unit",
            path=where_path,
        )
    parametrization = _gate_parametrization(spec)
    dropped, kept = gate_poles(parametrization, spec.stretch)
    record: dict[str, Any] = {
        "file_path": str(scores["file_path"]),
        "units": units,
        **({"where": dict(where)} if where else {}),
    }
    flat = values.flatten()
    if "keep" in scores:
        keep = scores["keep"]
        if isinstance(keep, bool) or not isinstance(keep, int):
            raise ProtocolError(
                "P2",
                f"{what}: keep is unresolved ({keep!r}) — a swept keep is a "
                "coordinate, resolved per point before the build",
            )
        if keep > units:
            raise ValidationError(
                32,
                f"{what}: keep={keep} exceeds the gate's {units} units",
                path=f"{where_path}.keep",
            )
        # descending by score, ascending by index on ties: a stable sort on
        # the negated scores keeps the index order among equals
        order = torch.sort(-flat, stable=True).indices[:keep]
        theta = torch.full((units,), dropped, dtype=torch.float32)
        theta[order] = kept
        record.update(
            {"keep": keep, "kept_units": sorted(int(i) for i in order.tolist())}
        )
    else:
        scale = scores["scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ProtocolError(
                "P2",
                f"{what}: scale is unresolved ({scale!r}) — a swept scale is a "
                "coordinate, resolved per point before the build",
            )
        std = float(flat.std(unbiased=False))
        z = (flat - flat.mean()) / std if std > 0.0 else torch.zeros_like(flat)
        midpoint = 0.5 if parametrization == "clamp" else (dropped + kept) / 2.0
        theta = midpoint + float(scale) * z
        if parametrization == "clamp":
            theta = theta.clamp(0.0, 1.0)
        record["scale"] = float(scale)
    identity = {"init_digest": hashlib.sha256(raw).hexdigest()}
    return theta.reshape(shape).contiguous(), identity, record


def _init_basis(
    name: str,
    spec: FeaturizerSpec,
    load_tensors: Any,
    *,
    width: int,
    k: int,
    coords: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The first ``k`` columns of the basis a ``subspace`` spec's ``init``
    names, and what a fit seeded from them records about that start (§2.5,
    §8): the basis's own provenance, the component indices taken, and a
    digest of the seeding matrix itself.

    The basis is fitted at one site of one model, so it has to be as wide as
    the site here and hold at least ``k`` components; the identity fields
    that say *which* site and model were checked at load
    (``loader._check_loaded_featurizers``), where the header is readable
    without the tensor. What only the tensor can show is checked here: the
    shape, and that the columns taken are an orthonormal frame — the
    parametrization installs them as its base *verbatim* (:class:`Subspace`),
    so a basis that is not orthonormal would make every weight the fit ever
    produces non-orthonormal, silently. The selected entry's ``produced_by``
    is required too: it is how the fit record names its start, and a swept
    bundle stamps it per entry, so only the selecting point can insist."""
    assert spec.init is not None
    init_path = str(spec.init["file_path"])
    want, implicit = entry_selection(spec.init.get("entry"), coords, name)
    what = f"featurizer {name!r} init ({init_path})"
    point = load_tensors(init_path).point("weight", want, what=what, implicit=implicit)
    basis = point.tensor("weight")
    if basis.ndim != 2 or int(basis.shape[0]) != width:
        raise ProtocolError(
            "P2",
            f"{what}: the basis has shape {tuple(basis.shape)} but the site here "
            f"is {width} wide — a starting subspace lives in the activation "
            "space the fit is trained in, so the basis must be (width, ≥ k)",
        )
    if int(basis.shape[1]) < k:
        raise ProtocolError(
            "P2",
            f"{what}: the basis holds {int(basis.shape[1])} components but the "
            f"fit needs k={k} of them — a rank-k fit starts from the first k "
            "columns, so the basis must hold at least that many",
        )
    columns = basis[:, :k].detach().to("cpu", torch.float32).contiguous()
    deviation = orthonormality_deviation(columns)
    if deviation > ORTHONORMAL_TOLERANCE:
        raise ProtocolError(
            "P2",
            f"{what}: the first {k} columns are not orthonormal (max |PᵀP − I| = "
            f"{deviation:.3g}, tolerance {ORTHONORMAL_TOLERANCE:g}) — a "
            "subspace's start is installed as the base of an orthogonal "
            "parametrization, which only stays orthonormal if the base is",
        )
    produced_by = point.identity.get("produced_by")
    if produced_by is None:
        raise ProtocolError(
            "P2",
            f"{what}: the selected entry carries no 'produced_by' — the fit "
            "record names the basis it started from by that digest, so a "
            "basis without one cannot seed a fit (§2.5)",
        )
    identity = {
        "init_produced_by": produced_by,
        "init_trained_on": point.identity.get("trained_on"),
        "init_components": list(range(k)),
        "init_digest": hashlib.sha256(columns.numpy().tobytes()).hexdigest(),
    }
    return columns, identity
