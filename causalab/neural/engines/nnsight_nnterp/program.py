"""One forward group as data: the program a trace body executes.

nnsight ships a traced block to NDIF as source plus every *name* the block
reads, each pickled whole. A block that reads ``self`` therefore ships the
executor — the document, the bundle, every earlier read, the trained
stages, the caches — and a block that mutates a client object mutates the
server's copy. So the executor never appears inside a trace. Before the
trace it plans the group into a :class:`GroupProgram`: a frozen record
holding everything the forward needs **as data** — the encoded inputs, the
operations in forward order with their resolved sites and interior
addresses, each write's positions, operands and featurizer stacks, each
read's positions — and the block (:mod:`.landers`) is module-level functions
over that record and the model. Locally the same program runs in the same
functions; there is one code path.

What travels is small by construction: token ids, position tables, the
operand slices a write consumes (rows × positions × width, already
gathered), and envoys, which nnsight pickles as module paths the server
resolves against its own model.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Mapping

import torch

from causalab.neural.engines.nnsight_nnterp.sources import SourceAddress
from causalab.neural.shared.executor_base import RaggedValue, TapKey, tap_key
from causalab.neural.shared.featurizers import FeaturizerStack
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.plan import COMPONENT_RANK
from causalab.protocol.schema import PositionSpec, WriteSpec

__all__ = [
    "PATTERN",
    "Entries",
    "FirePlan",
    "FlowPlan",
    "GroupProgram",
    "Op",
    "Order",
    "ReadPlan",
    "StepPlan",
    "WritePlan",
    "needs_eager",
    "schedule",
]

PATTERN = "attention_probs"

#: Which entries a write at one site carries: ``(write name, spec, site)``.
Entries = tuple[tuple[str, WriteSpec, ResolvedSite], ...]


class Order(enum.IntEnum):
    """What one operation does, valued in the order operations at one depth
    run: a ``block_mid`` write-back lands before the target's own writes,
    and writes before reads, so a read at a written address sees the write."""

    WRITEBACK = 0
    WRITE = 1
    READ = 2


@dataclasses.dataclass(frozen=True)
class FlowPlan:
    """How a read's gathered rows become the operand a later group of the
    same session consumes, without leaving the server: the feature tail of
    the read (``site``'s head slice, ``stack``, ``dims``)."""

    site: ResolvedSite
    stack: FeaturizerStack
    dims: Any


@dataclasses.dataclass(frozen=True)
class ReadPlan:
    """One prompt-frame read, reduced inside the forward.

    ``positions`` is every row's positions, resolved before the trace — the
    block gathers the captured contract tensor at them and saves the slice,
    never the tensor; ``None`` is a tap with no contract form (the attention
    pattern), saved whole. ``project`` (the head envoy, for an ``lm_head``
    read served from ``ln_final``) and ``derive`` (the site of a derived
    component) run over the gathered rows in the block, because both call
    the model's own modules. ``flow`` is set when a later group of the same
    session consumes this read."""

    rname: str
    positions: list[list[int]] | None
    project: Any = None
    derive: ResolvedSite | None = None
    flow: FlowPlan | None = None


@dataclasses.dataclass(frozen=True)
class FirePlan:
    """One per-fire read: its position axis is the kernel's fire index,
    whose length only the forward knows — ``whole`` (``pos: all``) or one
    integer ``index``, resolved against the count in the block."""

    rname: str
    whole: bool
    index: int | None


@dataclasses.dataclass(frozen=True)
class WritePlan:
    """Every write at one address, with what the write math needs as data.

    ``positions`` and ``stacks`` are keyed by write name. ``operands`` holds
    every operand resolved on the client — a read's stored value, a
    featurizer slot, a params tensor — by the name the payload spells;
    ``read_operands`` is which of the payload's names are reads, and a read
    operand absent from ``operands`` is one an earlier group of the same
    session produced (it is looked up in the session's flow).
    ``positioned`` says a read operand has a position axis (what a landing
    policy's width check holds it to); ``operand_routing`` is the routing
    table a routed-interior operand was read beside. ``fire_index`` is the
    fire each write of a per-fire address targets."""

    entries: Entries
    positions: Mapping[str, list[list[int]]]
    stacks: Mapping[str, FeaturizerStack]
    operands: Mapping[str, "torch.Tensor | float | RaggedValue"]
    read_operands: frozenset[str]
    positioned: Mapping[str, bool]
    operand_routing: Mapping[str, torch.Tensor]
    code: Mapping[str, Any] | None = None
    fire_index: Mapping[str, int] = dataclasses.field(default_factory=dict)

    @property
    def members(self) -> list[str]:
        return [ename for ename, _, _ in self.entries]


@dataclasses.dataclass(frozen=True)
class Op:
    """One operation of a group's schedule, sorted by ``(depth, order)``.
    ``address`` is ``None`` for a module boundary; ``key`` identifies the
    tensor the operation touches (``tap_key`` over the site and the
    address — two interior taps may share a module, side and shape while
    meaning different ops inside its forward). A write carries its
    :class:`WritePlan`, a read every plan served from its one capture."""

    depth: tuple[int, int]
    order: Order
    site: ResolvedSite
    address: SourceAddress | None
    key: TapKey
    write: WritePlan | None = None
    reads: tuple["ReadPlan | FirePlan", ...] = ()


@dataclasses.dataclass(frozen=True)
class StepPlan:
    """One continuation read: the tap collected per decode step (``key``),
    its position spec — resolved in the block against the continuation the
    decode produced — and the head envoy when the read is ``lm_head``
    projected from kept ``ln_final`` steps."""

    rname: str
    key: TapKey
    spec: PositionSpec
    project: Any = None


@dataclasses.dataclass(frozen=True)
class GroupProgram:
    """One ``(model, input role)`` forward group, ready to run.

    ``depth`` is the decode depth (0: a plain trace); ``step_ops`` the taps
    walked per decode step and ``steps`` the continuation reads served from
    them; ``rows`` / ``field`` are the role's dataset rows, shipped only
    when a continuation position names a ``variable``. ``needs_eager`` asks
    the block to switch the model to eager attention around its forward.
    ``grad`` runs the forward with gradients and keeps every saved value on
    its device with its graph; otherwise values are detached, and moved to
    the CPU in the block when ``offload`` (a forward in another process
    downloads exactly what it saves)."""

    label: str
    model_key: str
    inputs: Mapping[str, torch.Tensor]
    position_ids: torch.Tensor
    ops: tuple[Op, ...]
    step_ops: tuple[Op, ...] = ()
    steps: tuple[StepPlan, ...] = ()
    depth: int = 0
    needs_eager: bool = False
    grad: bool = False
    offload: bool = False
    rows: tuple[Mapping[str, Any], ...] | None = None
    field: str | None = None
    #: the embedding envoy, whose input pins a decode body to its forward
    embedding: Any = None

    @property
    def batch_size(self) -> int:
        return int(self.inputs["input_ids"].shape[0])


def _depth(
    order: Order, site: ResolvedSite, address: SourceAddress | None
) -> tuple[int, int]:
    """Where in the forward one operation on ``site`` is issued.

    A site's own depth (its layer and the plan's component rank), with two
    operations issued elsewhere: a write-back lands at its declared target's
    rank, and a *read* of the attention pattern on a tree with no interior
    address materializes on the mixer's returned weights — at
    ``attention_output``'s depth, once the mixer has run.
    """
    if order is Order.WRITEBACK:
        assert site.writeback is not None
        return site.depth[0], COMPONENT_RANK[site.writeback.component]
    if order is Order.READ and site.component == PATTERN and address is None:
        return site.depth[0], COMPONENT_RANK["attention_output"]
    return site.depth


def schedule(
    captures: list[
        tuple[ResolvedSite, SourceAddress | None, tuple["ReadPlan | FirePlan", ...]]
    ],
    writes: list[tuple[ResolvedSite, SourceAddress | None, WritePlan]],
) -> tuple[Op, ...]:
    """Every operation of one group in forward order: writes at their depth,
    a ``block_mid`` write-back at its target's depth ahead of the target's
    own writes, reads after the writes at their depth — one read operation
    per distinct capture, carrying every plan served from it."""
    ops = [
        Op(
            _depth(Order.WRITE, site, address),
            Order.WRITE,
            site,
            address,
            tap_key(site, address),
            write=plan,
        )
        for site, address, plan in writes
    ]
    ops += [
        Op(
            _depth(Order.WRITEBACK, site, None),
            Order.WRITEBACK,
            site,
            None,
            tap_key(site, address),
        )
        for site, address, _ in writes
        if site.writeback is not None
    ]
    by_key: dict[TapKey, Op] = {}
    for site, address, plans in captures:
        key = tap_key(site, address)
        if key in by_key:
            by_key[key] = dataclasses.replace(
                by_key[key], reads=by_key[key].reads + plans
            )
        else:
            by_key[key] = Op(
                _depth(Order.READ, site, address),
                Order.READ,
                site,
                address,
                key,
                reads=plans,
            )
    ops += by_key.values()
    ops.sort(key=lambda op: (op.depth, op.order))
    return tuple(ops)


def needs_eager(ops: tuple[Op, ...]) -> bool:
    """Whether the group touches anything that exists only under eager
    attention: the pattern off the mixer's returned weights, or an address
    that requires it."""
    return any(
        op.site.component == PATTERN
        or (op.address is not None and "attn_eager" in op.address.requires)
        for op in ops
    )
