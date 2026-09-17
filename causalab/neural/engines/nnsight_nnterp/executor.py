"""Point-protocol execution over one nnsight trace per forward group.

:class:`NnterpExecutor` inherits the lazy forward groups, position
resolution, featurizer stacks and class-ordered write math of
:class:`~causalab.neural.shared.executor_base.ExecutorBase`; what it owns is
the landing. Each ``(model, input role)`` group is **one trace** over every
row of the role: a write clones the live value, runs the shared write math
on its contract form and assigns the result back; a read saves the contract
tensor and is finalized (gathered, projected, featurized) after the trace
returns.

Two landings, one schedule. A **module boundary** — every tap of kind
``in`` / ``out`` the bundle's adapter declares — lands on the envoy's
``input`` / ``output``. An **interior** — the attention function's slots,
the DeltaNet kernel boundary and its per-chunk state, the routed-experts
interior — lands on an op of the anchor envoy's ``.source``, navigated by
the address table in :mod:`.sources` (``(family tree, component)`` → op path,
handle, row bookkeeping). The two share the write math, the contract
conversion and the operation loop; only how the tensor is reached differs.

Operations are issued in the model's forward order — nnsight parks each
request until the model produces the value and refuses one the model has
already run past — which is each site's depth: its layer and the plan's
component rank within it (:data:`~causalab.protocol.plan.COMPONENT_RANK`).
The rank band *is* the in-forward op order the ``.source`` interiors demand:
on one op, ``.inputs`` (q/k, the kernel's arguments) is requested before a
``.source`` drill (the scores, the state) before ``.output`` (z, the kernel's
return); on the experts source the sort's permutation is requested before
anything it sorted for. At one depth, a ``block_mid`` write-back lands
before the target's own writes, writes before reads, so a read at a written
address sees the write (:class:`Order`). Every anchor of the group is
instrumented (``_ = envoy.source``) before the trace opens, because
instrumenting rewrites the forward and must happen before it runs.

Per-chunk components (the DeltaNet state) fire once per kernel chunk: a read
collects every fire through ``tracer.iter`` and its position axis is the
chunk index; a write targets one fire per member, one ``tracer.iter[k]``
body each, and a fire past the last is refused rather than silently never
run. What has no prefill address — the per-token recurrent faces the
chunked kernel never materializes — is refused by name before any trace,
naming the reference engine.

A group with a continuation read (§2.3 ``generated``) runs as one
``model.generate`` trace instead: the schedule above binds the prefill
(occurrence 0 of every location — the whole of "writes are prefill-only"),
and the decode steps are walked afterwards with ``tracer.iter``
(:meth:`NnterpExecutor._decode`). Module boundaries and the attention
function's stackable slots are read per step through their prompt-frame
landings; the DeltaNet state through the recurrent kernel's own address
(:data:`~causalab.neural.engines.nnsight_nnterp.sources.GENERATED_ADDRESSES`);
every other interior is refused by name, because decode dispatches
different kernels than prefill and a prefill address is no evidence the
tensor exists per step.
"""

from __future__ import annotations

import enum
import functools
import math
import operator
from typing import Any, NamedTuple

import torch

from causalab.neural.engines.nnsight_nnterp.adapter import matched_family
from causalab.neural.engines.nnsight_nnterp.sources import (
    ADDRESSES,
    GENERATED_ADDRESSES,
    AddressResolutionError,
    SourceAddress,
    match_op,
)
from causalab.neural.shared.attention_backend import eager_attention
from causalab.neural.shared.encoding import (
    EncodedBatch,
    continuation_frame,
    continuation_widths,
    resolve_steps,
)
from causalab.neural.shared.executor_base import (
    ExecutorBase,
    TapKey,
    refuse_unstackable,
    tap_key,
)
from causalab.neural.shared.fires import FireTally, GroupFires, check_fires, group_label
from causalab.neural.shared.head import HEAD, HEAD_INPUT, head_module
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.neural.shared.loading import torch_module
from causalab.neural.shared.mechanisms import operand_names
from causalab.neural.shared.sites import ResolvedSite, _probe_grouped_mm, resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import COMPONENT_RANK, generated_budget
from causalab.protocol.schema import PositionSpec, ReadSpec, SiteSpec, WriteSpec

__all__ = ["NnterpExecutor"]

#: Tap kinds with no module boundary: served through an address of
#: :data:`~causalab.neural.engines.nnsight_nnterp.sources.ADDRESSES`, refused
#: by name where the tree has none.
_INTERIOR_KINDS = frozenset({"interface", "delta", "experts", "interior"})

#: The per-token DeltaNet faces: prefill runs only the chunked kernel, which
#: never materializes them (the recurrent kernel exists only in decode).
_NOT_IN_PREFILL = frozenset({"delta_kv_mem", "delta_state_update", "delta_state"})

_PATTERN = "attention_probs"

#: Which entries a write at one site carries: ``(write name, spec, site)``.
Entries = list[tuple[str, WriteSpec, ResolvedSite]]

#: One trace's navigated ops and handle values, keyed
#: ``(id(anchor), path)`` and ``(id(anchor), path, handle)``: a ``.source``
#: drill and a handle request happen once per op per trace — two components
#: on one op (q/k on the interface's inputs, gate/up on ``proj_out_0``, the
#: three kernel arguments) share the request, and a write updates the
#: handle's entry so a later read at the same op sees the written value.
Memo = dict[tuple[Any, ...], Any]


class Order(enum.IntEnum):
    """What one operation does, valued in the order operations at one depth
    run: a ``block_mid`` write-back lands before the target's own writes,
    and writes before reads, so a read at a written address sees the write."""

    WRITEBACK = 0
    WRITE = 1
    READ = 2


class Op(NamedTuple):
    """One operation of a group's schedule, sorted by ``(depth, order)``.
    ``address`` is ``None`` for a module boundary; ``key`` identifies the
    tensor the operation touches (``tap_key`` over the site and the
    address — two interior taps may share a module, side and shape while
    meaning different ops inside its forward)."""

    depth: tuple[int, int]
    order: Order
    site: ResolvedSite
    entries: Entries
    address: SourceAddress | None
    key: TapKey


class StepTap(NamedTuple):
    """One continuation read: the site it names, the site collected per
    decode step (``ln_final`` for an ``lm_head`` read, projected afterwards)
    and that site's address, ``None`` for a module boundary."""

    rname: str
    read: ReadSpec
    site: ResolvedSite
    capture: ResolvedSite
    address: SourceAddress | None


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
    if order is Order.READ and site.component == _PATTERN and address is None:
        return site.depth[0], COMPONENT_RANK["attention_output"]
    return site.depth


def _schedule(
    captures: list[tuple[ResolvedSite, SourceAddress | None]],
    writes: list[tuple[ResolvedSite, SourceAddress | None, Entries]],
) -> list[Op]:
    """Every operation of one group in forward order: writes at their depth,
    a ``block_mid`` write-back at its target's depth ahead of the target's
    own writes, reads after the writes at their depth, one read per distinct
    capture."""
    ops = [
        Op(
            _depth(Order.WRITE, site, address),
            Order.WRITE,
            site,
            entries,
            address,
            tap_key(site, address),
        )
        for site, address, entries in writes
    ]
    ops += [
        Op(
            _depth(Order.WRITEBACK, site, None),
            Order.WRITEBACK,
            site,
            [],
            None,
            tap_key(site, address),
        )
        for site, address, _ in writes
        if site.writeback is not None
    ]
    seen: set[TapKey] = set()
    for site, address in captures:
        key = tap_key(site, address)
        if key not in seen:
            seen.add(key)
            ops.append(
                Op(
                    _depth(Order.READ, site, address),
                    Order.READ,
                    site,
                    [],
                    address,
                    key,
                )
            )
    ops.sort(key=lambda op: (op.depth, op.order))
    return ops


def _select(value: Any, select: tuple[int | str, ...]) -> Any:
    """``value[i][j]…`` for an address' ``select``."""
    return functools.reduce(operator.getitem, select, value)


def _with_element(container: Any, select: tuple[int | str, ...], new: Any) -> Any:
    """``container`` with the element at ``select`` replaced — tuples and
    dicts rebuilt, never filled in place."""
    if not select:
        return new
    head, rest = select[0], select[1:]
    if isinstance(container, dict):
        return {**container, head: _with_element(container[head], rest, new)}
    assert isinstance(head, int)
    items = list(container)
    items[head] = _with_element(items[head], rest, new)
    return type(container)(items) if isinstance(container, tuple) else items


def _anchored(spec: PositionSpec) -> bool:
    """Whether ``spec`` needs text to resolve — a span, a variable, a column
    or an anchor — which a fire axis (the kernel's chunk index) has none of."""
    return any(
        getattr(spec, field) is not None
        for field in ("span", "variable", "column", "scope", "relative_to")
    )


def _needs_eager(ops: list[Op]) -> bool:
    """Whether the group touches anything that exists only under eager
    attention: the pattern off the mixer's returned weights, or an address
    that requires it."""
    return any(
        op.site.component == _PATTERN
        or (op.address is not None and "attn_eager" in op.address.requires)
        for op in ops
    )


def _save(value: Any) -> Any:
    """``nnsight.save`` — the extra's package, imported where it is needed."""
    import nnsight

    return nnsight.save(value)


def _native_row_width(site: ResolvedSite) -> int:
    """The packed width of one token-major row of an expert-rows value —
    every declared axis behind the position, fused split included (a fused
    capture's native row is top_k·splits·d; the split is selected later, in
    ``to_contract``) — read off the axes, not off ``shape.width`` (the
    contract width, which a fused shape is narrower than). An integral tap
    has no feature axis, so its row is the top-k itself."""
    widths = [
        axis.width
        for axis in site.shape.axes
        if axis.kind in ("topk", "fused", "feature")
    ]
    assert widths  # expert_rows implies at least a top-k axis
    return math.prod(widths)


def _top_k(site: ResolvedSite) -> int:
    return next(axis.width for axis in site.shape.axes if axis.kind == "topk")


def _fire_native(value: torch.Tensor, site: ResolvedSite) -> torch.Tensor:
    """One fire's state as the declared one-chunk native, ``(b, 1, heads,
    d_k·d_v)``, from the kernel's ``(b, heads, d_k, d_v)``."""
    heads = site.shape.head_space
    assert heads is not None
    return value.reshape(value.shape[0], 1, heads, -1)


class NnterpExecutor(ExecutorBase):
    """Execute one concrete document against one standardized bundle.

    ``batch_rows`` is not honoured: each ``(model, input role)`` group is one
    trace over every row of the role, never a sequence of row windows.
    """

    @functools.cached_property
    def _tree(self) -> str:
        """The registry family whose tree the loaded model is — the address
        table's first key; a tree no family detects has no interior."""
        matched = matched_family(torch_module(self.bundle.model))
        return matched.family if matched is not None else "nnterp_standard"

    def _address(self, what: str, site: ResolvedSite) -> SourceAddress | None:
        """The interior address of ``site``, ``None`` for a module boundary,
        a refusal by name for an interior this tree has no address for."""
        if site.kind not in _INTERIOR_KINDS and site.interface_slot is None:
            return None
        address = ADDRESSES.get((self._tree, site.component))
        if address is None and site.component == _PATTERN and site.kind == "out":
            return None  # read off the mixer's returned weights
        if address is None:
            if site.component in _NOT_IN_PREFILL:
                why = (
                    "a per-token recurrent quantity the chunked kernel this "
                    "engine traces never materializes in prefill (the recurrent "
                    "kernel runs only in decode) — the reference engine serves "
                    "it by stepping the recurrent kernel"
                )
            else:
                why = (
                    f"a {site.kind!r} tap with no interior address on the "
                    f"{self._tree!r} tree of this engine "
                    "(neural/engines/nnsight_nnterp/sources.py) — the reference "
                    "engine serves it"
                )
            raise ProtocolError(
                "P4",
                f"{what} addresses {site.component!r}, which has no module "
                f"boundary and is {why} (neural/engines/pytorch_hooks).",
                reason="component_unavailable",
            )
        if "experts_grouped" in address.requires:
            refusal = _probe_grouped_mm(self.bundle, site.component, site.layer)
            if refusal is not None:
                raise ProtocolError("P4", refusal, reason="component_unavailable")
        return address

    def _write_groups(
        self, write_names: tuple[str, ...]
    ) -> list[tuple[ResolvedSite, SourceAddress | None, Entries]]:
        """This group's writes by the tensor they land on: the shared
        resolution and policy check, regrouped by ``tap_key`` over the site
        *and* its interior address."""
        groups: dict[TapKey, tuple[ResolvedSite, SourceAddress | None, Entries]] = {}
        for _, entries in self._resolve_write_addresses(write_names).values():
            for ename, write, site in entries:
                address = self._address(f"write {ename!r}", site)
                groups.setdefault(tap_key(site, address), (site, address, []))[
                    2
                ].append((ename, write, site))
        return list(groups.values())

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        # operands first — the acyclic model graph is the schedule skeleton
        write_names: tuple[str, ...] = ()
        if model != "original":
            im = self.doc.intervened_models[model]
            write_names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
            for ename in write_names:
                for operand in operand_names(self.doc.writes[ename].do.payload):
                    if operand in self.doc.reads:
                        self.read_value(operand)

        reads: list[tuple[str, ReadSpec]] = []
        decode: list[tuple[str, ReadSpec]] = []
        for rname, read in self.doc.reads.items():
            if str(read.model) == model and str(read.input) == input_role:
                frame = decode if generated_budget(self.doc, read.pos) else reads
                frame.append((rname, read))
        depth = max(
            (generated_budget(self.doc, read.pos) for _, read in decode), default=0
        )
        steps = self._step_taps(decode)
        read_taps = self._read_taps(model, input_role, reads)
        captures = {
            rname: (tap.capture, self._address(f"read {rname!r}", tap.capture))
            for rname, tap in read_taps.items()
        }
        writes = self._write_groups(write_names)

        batch = self._batch(input_role)
        batch_size = int(batch.input_ids.shape[0])
        tally = FireTally()
        for _, _, entries in writes:
            tally.declare([ename for ename, _, _ in entries], 1)
        ops = _schedule(list(captures.values()), writes)
        step_ops = _schedule([(tap.capture, tap.address) for tap in steps], [])
        inputs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        # instrument every anchor before its forward runs (the first .source
        # on a module rewrites the forward); a bare access outside the trace
        anchors = {
            id(op.site.module): op.site.module
            for op in ops + step_ops
            if op.address is not None
        }
        for anchor in anchors.values():
            _ = anchor.source

        saves: dict[TapKey, Any] = {}
        routing: dict[TapKey, torch.Tensor] = {}
        memo: Memo = {}
        decoded: list[tuple[dict[TapKey, Any], torch.Tensor]] = []

        def body(tracer: Any) -> None:
            pending: dict[TapKey, torch.Tensor] = {}
            for op in ops:
                if op.order is Order.READ and op.address is not None:
                    saves[op.key] = self._land_source_read(
                        op, tracer, memo, batch_size, routing
                    )
                elif op.order is Order.READ:
                    saves[op.key] = _save(self._capture(op.site, batch_size))
                elif op.order is Order.WRITE and op.address is not None:
                    self._land_source_write(op, tracer, memo, input_role, batch, tally)
                elif op.order is Order.WRITE:
                    delta = self._land(op.site, op.entries, input_role, batch)
                    tally.fired([ename for ename, _, _ in op.entries])
                    if delta is not None:
                        pending[op.key] = delta
                else:
                    self._land_writeback(op.site, pending.pop(op.key))
            if depth:
                decoded.append(self._decode(tracer, step_ops, depth, batch_size))

        grad = torch.enable_grad() if self.grad_enabled else torch.no_grad()
        with (
            grad,
            torch_kernel_path(torch_module(self.bundle.model)),
            eager_attention(
                self.bundle.model,
                self.applied_requirements,
                needed=_needs_eager(ops + step_ops),
            ),
        ):
            # Either run method traces only as the `with` expression on the
            # `with` line itself (nnsight parses the call site); a call bound
            # to a name first just runs the model.
            if depth:
                # depth+1 forwards give every generated position its
                # activations, the last token's included (the extra draw is
                # never consumed — the reference engine's own decode loop,
                # §2.3); eos_token_id=None keeps decoding past an eos exactly
                # as that loop does, and the widths cut the frame afterwards
                with self.bundle.model.generate(
                    inputs, max_new_tokens=depth + 1, do_sample=False, eos_token_id=None
                ) as tracer:
                    body(tracer)
            else:
                with self.bundle.model.trace(
                    inputs, position_ids=batch.position_ids()
                ) as tracer:
                    body(tracer)
        check_fires(group_label(model, input_role), tally)
        fires = GroupFires()
        fires.fold(tally)
        record = fires.record()
        if record:
            self.fires[(model, input_role)] = record

        for rname, read in reads:
            tap = read_taps[rname]
            capture, address = captures[rname]
            key = tap_key(capture, address)
            if address is not None and address.fires == "per_chunk":
                self._read_values[rname] = self._finalize_per_fire_read(
                    rname, read, tap.site, saves[key], batch, input_role
                )
                continue
            self._read_values[rname] = self._finalize_read(
                rname,
                read,
                tap.site,
                saves[key],
                batch,
                input_role,
                project=tap.project,
                expert_idx=routing.get(key),
            )
        if depth:
            self._finalize_decode(model, input_role, batch, depth, steps, *decoded[0])
        self._groups_run.add((model, input_role))

    # ------------------------------------------------------------------ #
    # the generated frame: step-anchored reads under model.generate
    # ------------------------------------------------------------------ #

    def _step_taps(self, decode: list[tuple[str, ReadSpec]]) -> list[StepTap]:
        """Each continuation read's tap: an ``lm_head`` read captures
        ``ln_final`` per step and is projected at its addressed steps (the
        reference engine's own trick, so both serve the same value); a read
        whose steps do not stack is refused first
        (:func:`refuse_unstackable`)."""
        taps = []
        for rname, read in decode:
            site = resolve_site(self.bundle, self.doc.sites[str(read.site)])
            refuse_unstackable(rname, site)
            capture = (
                resolve_site(self.bundle, SiteSpec(component=HEAD_INPUT))
                if site.component == HEAD
                else site
            )
            address = self._step_address(rname, capture)
            taps.append(StepTap(rname, read, site, capture, address))
        return taps

    def _step_address(self, rname: str, site: ResolvedSite) -> SourceAddress | None:
        """The address a continuation read is served through per decode
        step: ``None`` for a module boundary; the prompt-frame address for an
        attention-function slot (the same call runs in decode); a verified
        decode address for what decode dispatches differently; a refusal by
        name for any other interior."""
        if site.kind not in _INTERIOR_KINDS and site.interface_slot is None:
            return None
        address = GENERATED_ADDRESSES.get((self._tree, site.component))
        if address is None and site.kind == "interface":
            address = self._address(f"read {rname!r}", site)
        if address is None:
            raise ProtocolError(
                "P4",
                f"component {site.component!r} has no generated-frame address "
                "in the nnsight_nnterp engine's tables "
                "(neural/engines/nnsight_nnterp/sources.py): the decode path "
                "dispatches different kernels than prefill, so an interior "
                "tensor is only served per step once its decode address is "
                "verified. Read it in the prompt frame.",
            )
        return address

    def _decode(
        self, tracer: Any, step_ops: list[Op], depth: int, batch_size: int
    ) -> tuple[dict[TapKey, Any], torch.Tensor]:
        """Every continuation tap's contract tensor per decode step (a saved
        list per tap), and the generated ids.

        ``tracer.iter`` counts forward passes and step 0 is the prefill, so
        bodies ``1 … depth`` are the forwards consuming generated tokens
        ``0 … depth-1``. Each body reads the embedding's input first: a
        location that fires every forward pins the body to that forward, so
        an op with no prefill occurrence (the recurrent kernel) is read at
        its step rather than one occurrence early — without it the loop
        outruns such an op and is cut short. Ops are navigated fresh each
        step (one memo per body): a request binds to the current occurrence.
        ``tracer.result`` is the last request — it consumes the whole run.
        """
        sinks = {op.key: _save([]) for op in step_ops}
        embedding = resolve_site(self.bundle, SiteSpec(component="embeddings")).module
        for _ in tracer.iter[1 : depth + 1]:
            _ = embedding.input
            memo: Memo = {}
            for op in step_ops:
                if op.address is None:
                    value = self._capture(op.site, batch_size)
                else:
                    native = self._present_native(op.site, op.address, memo)
                    if op.address.fires == "per_step":
                        native = _fire_native(native, op.site)
                    value = to_contract(native, op.site.shape, batch_size=batch_size)
                sinks[op.key].append(value)
        return sinks, _save(tracer.result)

    def _eos_ids(self) -> tuple[int, ...]:
        """What ends a row's continuation: the generation config's eos ids,
        else the tokenizer's — the reference engine's order."""
        config = getattr(torch_module(self.bundle.model), "generation_config", None)
        ids = getattr(config, "eos_token_id", None)
        if ids is None:
            ids = self.bundle.tokenizer.eos_token_id
        if ids is None:
            return ()
        return tuple(dict.fromkeys([ids] if isinstance(ids, int) else ids))

    def _finalize_decode(
        self,
        model: str,
        input_role: str,
        batch: EncodedBatch,
        depth: int,
        steps: list[StepTap],
        sinks: dict[TapKey, Any],
        ids: torch.Tensor,
    ) -> None:
        """Build the continuation frame the decode produced and finalize its
        reads against it: a row's width is the count before its first eos,
        positions resolve to decode steps, and an ``lm_head`` read is
        projected from the kept ``ln_final`` steps."""
        prompt_len = int(batch.input_ids.shape[1])
        generated = ids[:, prompt_len : prompt_len + depth].detach().cpu()
        continuation = continuation_frame(
            self.bundle.tokenizer,
            generated,
            continuation_widths(generated, self._eos_ids()),
        )
        self._continuations[(model, input_role)] = continuation
        rows = self.role_rows[input_role]
        field = self.role_fields[input_role]
        for rname, read, site, capture, address in steps:
            frame = torch.cat(list(sinks[tap_key(capture, address)]), dim=1)
            per_row = [
                resolve_steps(
                    self._spec(read.pos),
                    continuation,
                    row,
                    dataset_row=rows[row],
                    field=field,
                )
                for row in range(len(rows))
            ]
            self._read_steps[rname] = per_row
            self._read_values[rname] = self._finalize_read(
                rname,
                read,
                site,
                frame,
                batch,
                input_role,
                per_row=per_row,
                project=head_module(self.bundle) if capture is not site else None,
            )

    # ------------------------------------------------------------------ #
    # module boundaries
    # ------------------------------------------------------------------ #

    @staticmethod
    def _capture(site: ResolvedSite, batch_size: int) -> torch.Tensor:
        """One site's live tensor in the contract shape."""
        envoy = site.module
        if site.kind == "out":
            native = tap_tensor(envoy.output, site.tuple_index)
        else:
            native = envoy.input
        return to_contract(native, site.shape, batch_size=batch_size)

    def _rewrite(
        self,
        native: torch.Tensor,
        site: ResolvedSite,
        entries: Entries,
        input_role: str,
        batch: EncodedBatch,
        *,
        per_row: list[list[int]] | None = None,
        routing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``native`` with every write at ``site`` applied: a clone, the
        shared write math on its contract form, the native form back. A site
        naming an expert keeps the result only where ``routing`` names that
        expert — the write math runs over the whole contract."""
        native = native.clone()
        batch_size = int(batch.input_ids.shape[0])
        contract = to_contract(native, site.shape, batch_size=batch_size)
        original = contract.clone() if site.expert is not None else None
        self._apply_writes_to_contract(
            entries, input_role, batch, contract, per_row=per_row, routing=routing
        )
        if original is not None:
            assert routing is not None
            per_slot = contract.shape[-1] // routing.shape[-1]
            mask = (
                (routing == site.expert).unsqueeze(-1).expand(*routing.shape, per_slot)
            )
            contract = torch.where(mask.reshape(contract.shape), contract, original)
        return from_contract(contract, site.shape, batch_size=batch_size, native=native)

    def _land(
        self,
        site: ResolvedSite,
        entries: Entries,
        input_role: str,
        batch: EncodedBatch,
    ) -> torch.Tensor | None:
        """Apply every write at one module boundary and assign the result
        back. An input tap with a declared write-back returns its delta for
        the operation loop to land at the target."""
        if site.component == _PATTERN:
            # the value the attention function consumes, which the mixer's
            # returned weights only *report*: reached through the softmax op
            # of an addressed tree, nowhere on this one
            raise ProtocolError(
                "P4",
                f"a write at {_PATTERN!r} lands on the attention function's "
                f"softmax, which has no interior address on the {self._tree!r} "
                "tree of this engine (neural/engines/nnsight_nnterp/sources.py)",
                reason="component_unavailable",
            )
        envoy = site.module
        if site.kind == "out":
            payload = envoy.output
            original = tap_tensor(payload, site.tuple_index)
            new = self._rewrite(original, site, entries, input_role, batch)
            envoy.output = rebuild_payload(payload, site.tuple_index, new)
            return None
        original = envoy.input
        new = self._rewrite(original, site, entries, input_role, batch)
        envoy.input = new
        return new - original if site.writeback is not None else None

    @staticmethod
    def _land_writeback(site: ResolvedSite, delta: torch.Tensor) -> None:
        """Add a deferred input rewrite to its declared target's output."""
        writeback = site.writeback
        assert writeback is not None
        payload = writeback.module.output
        native = tap_tensor(payload, writeback.tuple_index)
        writeback.module.output = rebuild_payload(
            payload, writeback.tuple_index, native + delta
        )

    # ------------------------------------------------------------------ #
    # interiors: navigation
    # ------------------------------------------------------------------ #

    def _drill(self, source: Any, pattern: str, site: ResolvedSite) -> Any:
        """One matched op on one ``.source``, with the refusal made legible."""

        def line_of(name: str) -> str:
            op = getattr(source, name)
            text, line = getattr(op, "text", None), getattr(op, "line", None)
            if not isinstance(text, str) or not isinstance(line, int):
                return ""
            return text.splitlines()[line - 1]

        try:
            return getattr(source, match_op(pattern, source.names, line_of))
        except AddressResolutionError as error:
            import transformers

            raise ProtocolError(
                "P4",
                f"addressing {site.component!r} at layer {site.layer} of "
                f"{self.bundle.key!r} (transformers "
                f"{transformers.__version__}): {error}",
            ) from error

    def _op(self, site: ResolvedSite, path: tuple[str, ...], memo: Memo) -> Any:
        """The op ``path`` names from the anchor's ``.source``, each prefix
        drilled once per trace."""
        key = (id(site.module), path)
        if key not in memo:
            parent = (
                site.module.source
                if len(path) == 1
                else self._op(site, path[:-1], memo).source
            )
            memo[key] = self._drill(parent, path[-1], site)
        return memo[key]

    def _handle(
        self, site: ResolvedSite, path: tuple[str, ...], handle: str, memo: Memo
    ) -> Any:
        """The live value of one op's handle, requested once per trace."""
        key = (id(site.module), path, handle)
        if key not in memo:
            memo[key] = getattr(self._op(site, path, memo), handle)
        return memo[key]

    def _perm(
        self, site: ResolvedSite, address: SourceAddress, memo: Memo
    ) -> torch.Tensor:
        """The kernel's sorted-row → token-major permutation, off the very
        sort the kernel ran."""
        assert address.align is not None
        return self._handle(site, address.align, "output", memo)[1]

    def _routing(self, site: ResolvedSite, memo: Memo, batch_size: int) -> torch.Tensor:
        """The routing table the experts anchor was called with,
        ``(tokens, top_k)`` → contract ``(batch, position, top_k)`` — requested
        at the anchor's entry, before its interior."""
        key = (id(site.module), (), "inputs")
        if key not in memo:
            memo[key] = site.module.inputs
        idx = memo[key][0][1]
        return idx.reshape(batch_size, -1, idx.shape[-1])

    def _present_native(
        self, site: ResolvedSite, address: SourceAddress, memo: Memo
    ) -> Any:
        """The value an address names, as the declared shape describes it —
        semantic order. An ``align`` address' rows are in the kernel's
        expert-sorted order: un-sort them (a gather, so the result is a copy);
        ``expert_rows`` re-packs ``(batch·position·top_k, …)`` rows into the
        declared 2-D native ``(batch·position, top_k·…)``."""
        perm = self._perm(site, address, memo) if address.align is not None else None
        value = _select(
            self._handle(site, address.path, address.handle, memo), address.select
        )
        if address.derive == "argsort_perm":
            value = torch.argsort(value)
        if perm is not None:
            value = value[torch.argsort(perm)]
        if address.expert_rows:
            value = value.reshape(-1, _native_row_width(site))
        return value

    def _fire_ops(
        self, site: ResolvedSite, address: SourceAddress, memo: Memo
    ) -> tuple[Any, int]:
        """The value op of a per-fire address and its fire count — the
        length of the loop's own ``range(...)``, requested before the loop."""
        assert address.trip is not None
        count = len(self._handle(site, address.trip, "output", memo))
        return self._op(site, address.path, memo), count

    # ------------------------------------------------------------------ #
    # interiors: landing
    # ------------------------------------------------------------------ #

    def _land_source_read(
        self,
        op: Op,
        tracer: Any,
        memo: Memo,
        batch_size: int,
        routing: dict[TapKey, torch.Tensor],
    ) -> Any:
        """One interior read's contract tensor, saved — or, per fire, the
        saved list of every fire's tensor."""
        site, address = op.site, op.address
        assert address is not None
        if address.expert_rows:
            routing[op.key] = _save(self._routing(site, memo, batch_size))
        if address.fires == "per_chunk":
            value_op, count = self._fire_ops(site, address, memo)
            sink = _save([])
            for _ in tracer.iter[:count]:
                sink.append(value_op.output)
            return sink
        native = self._present_native(site, address, memo)
        return _save(to_contract(native, site.shape, batch_size=batch_size))

    def _land_source_write(
        self,
        op: Op,
        tracer: Any,
        memo: Memo,
        input_role: str,
        batch: EncodedBatch,
        tally: FireTally,
    ) -> None:
        """Apply every write at one interior address and assign the result
        back through the op's handle (``op.output = …`` / ``op.inputs = …``,
        the container rebuilt around the new value)."""
        site, address, entries = op.site, op.address, op.entries
        assert address is not None
        if address.fires == "per_chunk":
            self._land_per_fire_writes(op, tracer, memo, input_role, batch, tally)
            return
        batch_size = int(batch.input_ids.shape[0])
        idx = self._routing(site, memo, batch_size) if address.expert_rows else None
        perm = self._perm(site, address, memo) if address.align is not None else None
        new = self._rewrite(
            self._present_native(site, address, memo),
            site,
            entries,
            input_role,
            batch,
            routing=idx,
        )
        if address.expert_rows:
            # back to one row per (token, slot), then to the kernel's own
            # order — the exact inverse of _present_native
            new = new.reshape(-1, _native_row_width(site) // _top_k(site))
            if perm is not None:
                new = new[perm]
        key = (id(site.module), address.path, address.handle)
        memo[key] = _with_element(
            self._handle(site, address.path, address.handle, memo), address.select, new
        )
        setattr(self._op(site, address.path, memo), address.handle, memo[key])
        tally.fired([ename for ename, _, _ in entries])

    def _land_per_fire_writes(
        self,
        op: Op,
        tracer: Any,
        memo: Memo,
        input_role: str,
        batch: EncodedBatch,
        tally: FireTally,
    ) -> None:
        """Land writes on specific fires — one ``tracer.iter[k]`` body per
        targeted fire, each applying the members that named it. A body
        bound past the last fire would never run (nnsight keeps the fires it
        reached and drops the rest of the block), so a fire the count does
        not cover is refused before any body is bound."""
        site, address, entries = op.site, op.address, op.entries
        assert address is not None
        batch_size = int(batch.input_ids.shape[0])
        value_op, count = self._fire_ops(site, address, memo)
        fires = self._write_fire_indices(site, entries, count)
        for k in sorted(set(fires.values())):
            members = [entry for entry in entries if fires[entry[0]] == k]
            for _ in tracer.iter[k]:
                value = value_op.output
                new = self._rewrite(
                    _fire_native(value, site),
                    site,
                    members,
                    input_role,
                    batch,
                    per_row=[[0]] * batch_size,
                )
                value_op.output = new.reshape(value.shape)
                tally.fired([ename for ename, _, _ in members], step=k)

    def _write_fire_indices(
        self, site: ResolvedSite, entries: Entries, count: int
    ) -> dict[str, int]:
        """Which of the ``count`` fires each write at this address targets:
        one plain integer index per write, a negative one counting from the
        last fire exactly as a read's does (:meth:`_fire_positions`).
        Anything anchored refuses — a chunk axis has no text — and so does an
        index the count does not cover."""
        fires: dict[str, int] = {}
        for ename, write, _site in entries:
            spec = self._spec(write.pos)
            index = spec.index if isinstance(spec.index, int) else None
            if index is None or _anchored(spec):
                raise ProtocolError(
                    "P4",
                    f"write {ename!r} targets {site.component!r}, whose "
                    f"position axis is the kernel's fire index "
                    f"({site.shape.describe()}): only a plain integer index "
                    "resolves there — a chunk axis has no text for an anchor "
                    "or a span to resolve against.",
                )
            if not -count <= index < count:
                raise ProtocolError(
                    "P2",
                    f"write {ename!r} addresses fire {index} of "
                    f"{site.component!r}, but the kernel fired {count} times on "
                    "this batch — a tracer.iter body past the last fire never "
                    "runs, so this is refused rather than silently skipped",
                )
            fires[ename] = index % count
        return fires

    def _finalize_per_fire_read(
        self,
        rname: str,
        read: ReadSpec,
        site: ResolvedSite,
        sink: Any,
        batch: EncodedBatch,
        input_role: str,
    ) -> torch.Tensor:
        """One per-fire read's value: the fires stacked on the declared
        position axis, then the shared gather/featurize path with positions
        resolved against the fire count."""
        fires = list(sink)
        native = torch.cat([_fire_native(fire, site) for fire in fires], dim=1)
        raw = to_contract(native, site.shape, batch_size=int(batch.input_ids.shape[0]))
        per_row = self._fire_positions(rname, read.pos, len(fires), raw.shape[0])
        return self._finalize_read(
            rname, read, site, raw, batch, input_role, per_row=per_row
        )

    def _fire_positions(
        self, rname: str, pos: Any, n_fires: int, n_rows: int
    ) -> list[list[int]]:
        """Positions on a fire axis: ``all``, or one integer index (negative
        counts from the last fire). Anything anchored refuses — there is no
        text on a chunk axis."""
        spec = self._spec(pos)
        anchored = _anchored(spec)
        if not anchored and getattr(spec, "all", None) is True:
            return [list(range(n_fires))] * n_rows
        index = spec.index if isinstance(spec.index, int) else None
        if anchored or index is None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} addresses positions on a per-fire component, "
                "whose position axis is the kernel's chunk index: only "
                '"all" or a plain integer index resolves there — text '
                "anchors and spans have nothing to resolve against.",
            )
        if not -n_fires <= index < n_fires:
            raise ProtocolError(
                "P2",
                f"read {rname!r} addresses fire {index}, but the kernel fired "
                f"{n_fires} times on this batch",
            )
        return [[index % n_fires]] * n_rows
