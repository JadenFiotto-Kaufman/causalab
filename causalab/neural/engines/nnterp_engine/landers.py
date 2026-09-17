"""The block side of the engine: what runs inside a trace.

Everything here is a module-level function over the model and a frozen
:class:`~causalab.neural.engines.nnterp_engine.program.GroupProgram` — no
executor, no ``self``. That is what makes a trace body shippable: nnsight
pickles every name a block reads, so :func:`run_program`'s two ``with``
bodies read only ``model``, ``tracer``, ``program``, ``flow``, ``nnsight``
and :func:`execute`, and the payload is the program (kilobytes) rather than
the executor (everything the run has touched). These functions are
``causalab``'s, which is installed where the block runs.

The NDIF rules the structure enforces:

* **one saved container, bound at block level.** ``out = nnsight.save({})``
  is the only channel back; the landers fill its slots with plain
  assignments. (A ``nnsight.save`` into a dict slot marks a value no block
  variable names, and a real server returns nothing for it.)
* **nothing on the client is mutated.** Fires are appended to
  ``out["fired"]`` and replayed into the client's tally afterwards; the
  expert alignment's mismatch counts land in ``out["mismatch"]``.
* **reduce before saving.** A read is gathered at its positions in the
  block — and projected through the head or derived through the mixer's
  projection there, since both need the model's weights — so the download
  is ``rows × positions × width``, not the contract tensor. Values are
  detached, and moved to the CPU when the program says the forward is in
  another process; a gradient-enabled program keeps device tensors and
  their graph, flowing values included — inside a fit
  (:mod:`.fit`) the graph of a step spans its traces, and ``backward()``
  runs between them.
* **the eager switch happens in the block**, on the module the model's
  envoy resolves to where the block runs, before any operation, and is
  reversed in a ``finally`` once the forward has finished or been stopped:
  a config mutated on the client is not in the payload. That module must be
  the served model itself, so a block handed a ``meta`` copy (a sandboxed
  NDIF deployment) refuses by name first.
* **the forward runs no further than it is read.** A plain forward is
  stopped after its group's last operation (``tracer.stop()``); a generate
  consumes its whole run, and a gradient-enabled forward runs whole.
* **operands stay on the server.** Inside a session, a read a later
  consumer needs — a later group's write, a fit's objective — is finished
  into ``flow`` (its feature tail applied from the program's stack) and
  looked up there.
* **a remote run is checked before it is submitted.** Every function here
  that takes ``remote`` first holds the server's ``causalab`` and ``nnterp``
  to this client's (:mod:`.versions`).

Two landings, one schedule. A **module boundary** lands on the envoy's
``input`` / ``output``. An **interior** lands on an op of the anchor envoy's
``.source``, navigated by the address table in :mod:`.sources` through one
:class:`Navigation` per trace. Operations are issued in the program's order,
the model's forward order — nnsight parks each request until the model
produces the value and refuses one the model has already run past.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from typing import Any, Callable

import torch

from causalab.neural.engines.nnterp_engine.program import (
    Entries,
    FirePlan,
    GroupProgram,
    Op,
    Order,
    ReadPlan,
    SlotRef,
    StackRef,
    WritePlan,
)
from causalab.neural.engines.nnterp_engine.versions import ensure_server_matches
from causalab.neural.engines.nnterp_engine.sources import (
    AddressResolutionError,
    SourceAddress,
    match_op,
)
from causalab.neural.shared.attention_backend import (
    restore_attention,
    switch_to_eager,
)
from causalab.neural.shared.encoding import (
    continuation_frame,
    continuation_widths,
    resolve_steps,
)
from causalab.neural.shared.executor_base import (
    RaggedValue,
    RowWindow,
    TapKey,
    WriteServices,
    _derive,
    apply_writes_to_contract,
    gather_rows,
    read_features,
    read_operand,
)
from causalab.neural.shared.featurizers import FeaturizerStack
from causalab.neural.shared.layout import (
    from_contract,
    rebuild_payload,
    tap_tensor,
    to_contract,
)
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.errors import ProtocolError

__all__ = [
    "Navigation",
    "execute",
    "fire_ops",
    "present_native",
    "refuse_meta",
    "routing",
    "run_program",
    "run_session",
]


@dataclasses.dataclass
class Navigation:
    """One trace's navigated ops and handle values, keyed
    ``(id(anchor), path)`` and ``(id(anchor), path, handle)``: a ``.source``
    drill and a handle request happen once per op per trace — two components
    on one op (q/k on the interface's inputs, gate/up on ``proj_out_0``, the
    three kernel arguments) share the request, and a write updates the
    handle's entry so a later read at the same op sees the written value.
    ``model_key`` names the model in a navigation refusal."""

    model_key: str
    memo: dict[tuple[Any, ...], Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------- #
# running a program
# ---------------------------------------------------------------------- #


def run_program(
    model: Any,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    *,
    remote: bool | str = False,
) -> dict[str, Any]:
    """Run one group's forward and hand back what its block saved.

    One function for every tier. Locally ``remote`` is ``False`` and the
    block runs in this process; ``remote=True`` (or ``"local"``, nnsight's
    serialize → deserialize → execute dry run) ships the block as its own
    job; inside :func:`run_session` it is called on the server with
    ``remote`` unset, because the session already is the job.

    Either run method traces only as the ``with`` expression on the ``with``
    line itself (nnsight parses the call site); a call bound to a name first
    just runs the model.
    """
    import nnsight

    ensure_server_matches(remote)
    with torch.set_grad_enabled(program.grad):
        if program.depth:
            # depth+1 forwards give every generated position its
            # activations, the last token's included (the extra draw is
            # never consumed — the reference engine's own decode loop,
            # §2.3); eos_token_id=None keeps decoding past an eos exactly
            # as that loop does, and the widths cut the frame afterwards
            with model.generate(
                program.inputs,
                max_new_tokens=program.depth + 1,
                do_sample=False,
                eos_token_id=None,
                remote=remote,
            ) as tracer:
                out = nnsight.save({})
                execute(model, tracer, program, flow, out)
        else:
            # the cache is on only where a decode consumes it — the reference
            # engine's rule (``use_cache=depth > 0``). A prompt-only forward
            # handed a ``DynamicCache`` (HF's default) attends over the
            # re-laid-out keys and values ``DynamicCache.update`` returns,
            # whose matmul rounds an ulp away from the cacheless forward's;
            # a gradient-enabled one would also keep graph-attached copies
            # of every layer's keys and values
            with model.trace(
                program.inputs,
                position_ids=program.position_ids,
                use_cache=False,
                remote=remote,
            ) as tracer:
                out = nnsight.save({})
                execute(model, tracer, program, flow, out)
    return out


def run_session(
    model: Any, programs: tuple[GroupProgram, ...], *, remote: bool | str
) -> dict[str, dict[str, Any]]:
    """Run a point's groups, in order, as **one** job: one queue wait, one
    payload, one download. ``flow`` lives in the session, so a read a later
    group writes with never leaves the server; what comes back is each
    group's saved container, by the group's label."""
    import nnsight

    ensure_server_matches(remote)
    with model.session(remote=remote):
        results = nnsight.save({})
        flow = {}
        for program in programs:
            results[program.label] = run_program(model, program, flow)
    return results


def execute(
    model: Any,
    tracer: Any,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
) -> None:
    """The body of one trace: the eager switch, every operation in forward
    order, the decode walk or the early stop, and the switch put back once
    the forward is over. This is the last statement of its block: a plain
    forward is stopped after its last operation, which ends the block."""
    out.update(reads={}, routing={}, fired=[], mismatch={}, steps={})
    module = model._module
    refuse_meta(module, program)
    previous = switch_to_eager(module) if program.needs_eager else None
    try:
        if previous is not None:
            out["attn_eager"] = True
        nav = Navigation(program.model_key)
        pending: dict[TapKey, torch.Tensor] = {}
        for op in program.ops:
            if op.order is Order.READ:
                _land_read(op, tracer, nav, program, flow, out)
            elif op.order is Order.WRITE and op.address is not None:
                _land_source_write(op, tracer, nav, program, flow, out)
            elif op.order is Order.WRITE:
                delta = _land(op, program, flow, out)
                if delta is not None:
                    pending[op.key] = delta
            else:
                _land_writeback(op.site, pending.pop(op.key))
        if program.depth:
            _decode(model, tracer, program, out)
        out["mismatch"] = {
            key: (_offload(missing, program), per_example)
            for key, (missing, per_example) in out["mismatch"].items()
        }
        if program.depth:
            pass  # ``tracer.result`` consumed the whole run
        elif not program.grad:
            # nothing past the last operation is read, so the forward ends
            # here: a read at layer 5 of 80 runs 6 layers. The stop unwinds
            # this function — the ``finally`` below puts the attention back
            # while the forward is parked — and ends the trace's block, whose
            # saved container is already filled
            tracer.stop()
        elif previous is not None:
            # a gradient-enabled forward runs whole: a stop makes whatever the
            # block still holds after this function unreachable. Park until
            # the forward is over — the eager implementation must outlive
            # every attention layer still to run
            _ = model.output
    finally:
        if previous is not None:
            restore_attention(module, previous)


def refuse_meta(module: torch.nn.Module, program: GroupProgram) -> None:
    """Refuse a block that was handed a weight-free model.

    The block works on the model its envoy resolves to where it runs: it
    flips that model's attention implementation, calls its head and its
    mixer projections, and reads its generation config. A sandboxed
    (untrusted) NDIF deployment runs the block in a runner process against a
    ``meta`` copy while the forward runs on the host, so the switch would
    flip the wrong model and a projection would call meta weights."""
    parameter = next(module.parameters(), None)
    if parameter is not None and parameter.is_meta:
        raise ProtocolError(
            "P4",
            f"group {program.label!r} of {program.model_key!r} was handed a "
            "model whose parameters are on 'meta' where its block runs. The "
            "nnterp engine needs a trusted, in-process NDIF "
            "deployment, where the block runs against the served model "
            "itself: a sandboxed (untrusted) deployment runs it in a runner "
            "process against a weight-free copy, where the eager-attention switch "
            "would flip the wrong model and a head or mixer projection would "
            "call meta weights.",
        )


def _offload(value: torch.Tensor, program: GroupProgram) -> torch.Tensor:
    """A value on its way into the saved container: as the forward made it
    under ``grad`` (training needs the graph and the device), else detached
    — and on the CPU when the container is a download."""
    if program.grad:
        return value
    value = value.detach()
    return value.cpu() if program.offload else value


def _flat(value: "torch.Tensor | RaggedValue") -> torch.Tensor:
    """The tensor of a gathered value; a ragged one's widths are its
    positions', which the client already holds."""
    return value.flat if isinstance(value, RaggedValue) else value


def _over(
    value: "torch.Tensor | RaggedValue", fn: Callable[[torch.Tensor], torch.Tensor]
) -> "torch.Tensor | RaggedValue":
    if isinstance(value, RaggedValue):
        return RaggedValue(flat=fn(value.flat), widths=value.widths)
    return fn(value)


def _stack_on(
    stack: "FeaturizerStack | StackRef", device: torch.device
) -> FeaturizerStack:
    """``stack`` with its stages where the activation is. A no-op in one
    process (the stack was built on the bundle's device); on a server the
    shipped stages arrive on the CPU and the activation is wherever the
    server put the model — a device the client cannot know. A stack still
    named (:class:`StackRef`) belongs to a fit, whose body binds it to its
    stage table before any forward (:func:`.fit.bind_stages`)."""
    if isinstance(stack, StackRef):
        raise ProtocolError(
            "P4",
            f"a program names the featurizer stack {list(stack.names)} instead "
            "of carrying it: by-name stacks resolve against a fit's stage "
            "table (nnterp_engine/fit.py) and this forward runs outside one",
        )
    for stage in stack.stages:
        if any(p.device != device for p in stage.parameters()) or any(
            b.device != device for b in stage.buffers()
        ):
            stage.to(device)
    return stack


# ---------------------------------------------------------------------- #
# reads
# ---------------------------------------------------------------------- #


def _capture(site: ResolvedSite, batch_size: int) -> torch.Tensor:
    """One module-boundary site's live tensor in the contract shape."""
    envoy = site.module
    if site.kind == "out":
        native = tap_tensor(envoy.output, site.tuple_index)
    else:
        native = envoy.input
    return to_contract(native, site.shape, batch_size=batch_size)


def _land_read(
    op: Op,
    tracer: Any,
    nav: Navigation,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
) -> None:
    """Capture one tap and reduce every read served from it."""
    site, address = op.site, op.address
    batch_size = program.batch_size
    idx = None
    if address is None:
        contract = _capture(site, batch_size)
    else:
        if address.expert_rows:
            idx = routing(site, nav, batch_size)
        if address.fires == "per_chunk":
            # every fire, stacked on the declared position axis
            value_op, count = fire_ops(site, address, nav)
            fires = []
            for _ in tracer.iter[:count]:
                fires.append(_fire_native(value_op.output, site))
            native = torch.cat(fires, dim=1)
        else:
            native = present_native(site, address, nav)
        contract = to_contract(native, site.shape, batch_size=batch_size)
    for plan in op.reads:
        if isinstance(plan, FirePlan):
            positions = _fire_positions(plan, contract.shape[1], contract.shape[0])
            out["reads"][plan.rname] = _offload(
                _flat(gather_rows(contract, positions)), program
            )
        else:
            _reduce(plan, contract, idx, program, flow, out)


def _reduce(
    plan: ReadPlan,
    contract: torch.Tensor,
    idx: torch.Tensor | None,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
) -> None:
    """One read's slice of a captured tensor into the saved container: the
    gather at its positions, then what needs the model's modules (the head's
    projection, a derived component). The rest of the read — the head's
    slice, the featurizer stack, ``dims`` — needs only the document, and is
    the client's (``ExecutorBase._finalize_read(pregathered=True)``)."""
    if plan.positions is None:  # no contract form: the whole native tensor
        out["reads"][plan.rname] = _offload(contract, program)
        return
    gathered = gather_rows(contract, plan.positions)
    if plan.project is not None:
        gathered = _over(gathered, plan.project)
    if plan.derive is not None:
        gathered = _over(
            gathered, functools.partial(_derive_as, plan.derive, plan.rname)
        )
    out["reads"][plan.rname] = _offload(_flat(gathered), program)
    if idx is not None:
        out["routing"][plan.rname] = _offload(
            _flat(gather_rows(idx, plan.positions)), program
        )
    if plan.flow is not None:
        # the read's feature tail, where its consumer is: under ``grad`` with
        # its graph (a fit's write operand, its objective), else built under
        # ``no_grad`` and detached
        rows = _flat(gathered)
        value = read_features(
            rows,
            plan.flow.site,
            _stack_on(plan.flow.stack, rows.device),
            plan.flow.dims,
            grad_enabled=program.grad,
        )
        flow[plan.rname] = value if program.grad else value.detach()


def _derive_as(site: ResolvedSite, rname: str, value: torch.Tensor) -> torch.Tensor:
    return _derive(site, value, rname)


def _fire_positions(plan: FirePlan, n_fires: int, n_rows: int) -> list[list[int]]:
    """Positions on a fire axis, against the count the forward produced:
    ``all``, or one integer index (negative counts from the last fire)."""
    if plan.index is None:
        return [list(range(n_fires))] * n_rows
    if not -n_fires <= plan.index < n_fires:
        raise ProtocolError(
            "P2",
            f"read {plan.rname!r} addresses fire {plan.index}, but the kernel "
            f"fired {n_fires} times on this batch",
        )
    return [[plan.index % n_fires]] * n_rows


# ---------------------------------------------------------------------- #
# writes
# ---------------------------------------------------------------------- #


def _services(
    write: WritePlan,
    flow: dict[str, torch.Tensor],
    device: torch.device,
    out: dict[str, Any],
) -> WriteServices:
    """The write math's services over one write plan's tables — the data
    twin of ``ExecutorBase._write_services``, with operands moved to the
    device of the tensor being written (the only device a block can know)."""

    def lookup(
        value: Any, *, rows: RowWindow | None = None, ragged: Any = None
    ) -> torch.Tensor | float:
        if not isinstance(value, str):
            return float(value)
        if value in write.read_operands:
            stored = write.operands[value] if value in write.operands else flow[value]
            return read_operand(
                value,
                stored,
                device=device,
                rows=rows,
                ragged=ragged,
                positioned=ragged is not None and write.positioned[value],
            )
        if value not in write.operands or isinstance(write.operands[value], SlotRef):
            # a slot still named belongs to a fit's stage table, as a stack does
            raise ProtocolError("P2", f"operand {value!r} did not resolve at run time")
        return write.operands[value]

    def routing_of(value: Any, rows: RowWindow | None) -> torch.Tensor | None:
        if not isinstance(value, str) or value not in write.operand_routing:
            return None
        table = write.operand_routing[value].to(device)
        return table if rows is None or rows.whole else table[rows.index]

    return WriteServices(
        positions_of=lambda ename, _write: write.positions[ename],
        lookup=lookup,
        stack_of=lambda ename, _write, _site: _stack_on(write.stacks[ename], device),
        routing_of=routing_of,
        reads=write.read_operands,
        code=write.code,
        mismatches=out["mismatch"],
    )


def _rewrite(
    native: torch.Tensor,
    site: ResolvedSite,
    write: WritePlan,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
    *,
    entries: Entries | None = None,
    per_row: list[list[int]] | None = None,
    routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """``native`` with every write at ``site`` applied: a clone, the shared
    write math on its contract form, the native form back. A site naming an
    expert keeps the result only where ``routing`` names that expert — the
    write math runs over the whole contract. ``entries`` narrows the plan to
    the members one fire carries."""
    native = native.clone()
    batch_size = program.batch_size
    contract = to_contract(native, site.shape, batch_size=batch_size)
    original = contract.clone() if site.expert is not None else None
    apply_writes_to_contract(
        write.entries if entries is None else entries,
        contract,
        _services(write, flow, contract.device, out),
        per_row=per_row,
        routing=routing,
    )
    if original is not None:
        assert routing is not None
        per_slot = contract.shape[-1] // routing.shape[-1]
        mask = (routing == site.expert).unsqueeze(-1).expand(*routing.shape, per_slot)
        contract = torch.where(mask.reshape(contract.shape), contract, original)
    return from_contract(contract, site.shape, batch_size=batch_size, native=native)


def _land(
    op: Op, program: GroupProgram, flow: dict[str, torch.Tensor], out: dict[str, Any]
) -> torch.Tensor | None:
    """Apply every write at one module boundary and assign the result back.
    An input tap with a declared write-back returns its delta for the
    operation loop to land at the target."""
    site, write = op.site, op.write
    assert write is not None
    envoy = site.module
    delta = None
    if site.kind == "out":
        payload = envoy.output
        original = tap_tensor(payload, site.tuple_index)
        new = _rewrite(original, site, write, program, flow, out)
        envoy.output = rebuild_payload(payload, site.tuple_index, new)
    else:
        original = envoy.input
        new = _rewrite(original, site, write, program, flow, out)
        envoy.input = new
        if site.writeback is not None:
            delta = new - original
    out["fired"].append((write.members, None))
    return delta


def _land_writeback(site: ResolvedSite, delta: torch.Tensor) -> None:
    """Add a deferred input rewrite to its declared target's output."""
    writeback = site.writeback
    assert writeback is not None
    payload = writeback.module.output
    native = tap_tensor(payload, writeback.tuple_index)
    writeback.module.output = rebuild_payload(
        payload, writeback.tuple_index, native + delta
    )


def _land_source_write(
    op: Op,
    tracer: Any,
    nav: Navigation,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
) -> None:
    """Apply every write at one interior address and assign the result
    back through the op's handle (``op.output = …`` / ``op.inputs = …``,
    the container rebuilt around the new value)."""
    site, address, write = op.site, op.address, op.write
    assert address is not None and write is not None
    if address.fires == "per_chunk":
        _land_per_fire_writes(op, tracer, nav, program, flow, out)
        return
    idx = routing(site, nav, program.batch_size) if address.expert_rows else None
    perm = _perm(site, address, nav) if address.align is not None else None
    new = _rewrite(
        present_native(site, address, nav),
        site,
        write,
        program,
        flow,
        out,
        routing=idx,
    )
    if address.expert_rows:
        # back to one row per (token, slot), then to the kernel's own
        # order — the exact inverse of present_native
        new = new.reshape(-1, _native_row_width(site) // _top_k(site))
        if perm is not None:
            new = new[perm]
    key = (id(site.module), address.path, address.handle)
    nav.memo[key] = _with_element(
        _handle(site, address.path, address.handle, nav), address.select, new
    )
    setattr(_op(site, address.path, nav), address.handle, nav.memo[key])
    out["fired"].append((write.members, None))


def _land_per_fire_writes(
    op: Op,
    tracer: Any,
    nav: Navigation,
    program: GroupProgram,
    flow: dict[str, torch.Tensor],
    out: dict[str, Any],
) -> None:
    """Land writes on specific fires — one ``tracer.iter[k]`` body per
    targeted fire, each applying the members that named it. A body
    bound past the last fire would never run (nnsight keeps the fires it
    reached and drops the rest of the block), so a fire the count does
    not cover is refused before any body is bound."""
    site, address, write = op.site, op.address, op.write
    assert address is not None and write is not None
    value_op, count = fire_ops(site, address, nav)
    fires = _fire_targets(site, write, count)
    for k in sorted(set(fires.values())):
        members = tuple(entry for entry in write.entries if fires[entry[0]] == k)
        for _ in tracer.iter[k]:
            value = value_op.output
            new = _rewrite(
                _fire_native(value, site),
                site,
                write,
                program,
                flow,
                out,
                entries=members,
                per_row=[[0]] * program.batch_size,
            )
            value_op.output = new.reshape(value.shape)
            out["fired"].append(([ename for ename, _, _ in members], k))


def _fire_targets(site: ResolvedSite, write: WritePlan, count: int) -> dict[str, int]:
    """Which of the ``count`` fires each write at this address targets: its
    planned integer index, a negative one counting from the last fire
    exactly as a read's does. An index the count does not cover refuses."""
    fires: dict[str, int] = {}
    for ename, index in write.fire_index.items():
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


# ---------------------------------------------------------------------- #
# interiors: navigation
# ---------------------------------------------------------------------- #


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


def _drill(source: Any, pattern: str, site: ResolvedSite, nav: Navigation) -> Any:
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
            f"{nav.model_key!r} (transformers "
            f"{transformers.__version__}): {error}",
        ) from error


def _op(site: ResolvedSite, path: tuple[str, ...], nav: Navigation) -> Any:
    """The op ``path`` names from the anchor's ``.source``, each prefix
    drilled once per trace."""
    key = (id(site.module), path)
    if key not in nav.memo:
        parent = (
            site.module.source if len(path) == 1 else _op(site, path[:-1], nav).source
        )
        nav.memo[key] = _drill(parent, path[-1], site, nav)
    return nav.memo[key]


def _handle(
    site: ResolvedSite, path: tuple[str, ...], handle: str, nav: Navigation
) -> Any:
    """The live value of one op's handle, requested once per trace."""
    key = (id(site.module), path, handle)
    if key not in nav.memo:
        nav.memo[key] = getattr(_op(site, path, nav), handle)
    return nav.memo[key]


def _perm(site: ResolvedSite, address: SourceAddress, nav: Navigation) -> torch.Tensor:
    """The kernel's sorted-row → token-major permutation, off the very
    sort the kernel ran."""
    assert address.align is not None
    return _handle(site, address.align, "output", nav)[1]


def routing(site: ResolvedSite, nav: Navigation, batch_size: int) -> torch.Tensor:
    """The routing table the experts anchor was called with,
    ``(tokens, top_k)`` → contract ``(batch, position, top_k)`` — requested
    at the anchor's entry, before its interior."""
    key = (id(site.module), (), "inputs")
    if key not in nav.memo:
        nav.memo[key] = site.module.inputs
    idx = nav.memo[key][0][1]
    return idx.reshape(batch_size, -1, idx.shape[-1])


def present_native(site: ResolvedSite, address: SourceAddress, nav: Navigation) -> Any:
    """The value an address names, as the declared shape describes it —
    semantic order. An ``align`` address' rows are in the kernel's
    expert-sorted order: un-sort them (a gather, so the result is a copy);
    ``expert_rows`` re-packs ``(batch·position·top_k, …)`` rows into the
    declared 2-D native ``(batch·position, top_k·…)``."""
    perm = _perm(site, address, nav) if address.align is not None else None
    value = _select(_handle(site, address.path, address.handle, nav), address.select)
    if address.derive == "argsort_perm":
        value = torch.argsort(value)
    if perm is not None:
        value = value[torch.argsort(perm)]
    if address.expert_rows:
        value = value.reshape(-1, _native_row_width(site))
    return value


def fire_ops(
    site: ResolvedSite, address: SourceAddress, nav: Navigation
) -> tuple[Any, int]:
    """The value op of a per-fire address and its fire count — the
    length of the loop's own ``range(...)``, requested before the loop."""
    assert address.trip is not None
    count = len(_handle(site, address.trip, "output", nav))
    return _op(site, address.path, nav), count


# ---------------------------------------------------------------------- #
# the generated frame
# ---------------------------------------------------------------------- #


def _eos_ids(model: Any) -> tuple[int, ...]:
    """What ends a row's continuation: the generation config's eos ids,
    else the tokenizer's — the reference engine's order, read off the model
    that decoded."""
    config = getattr(model._module, "generation_config", None)
    ids = getattr(config, "eos_token_id", None)
    if ids is None:
        ids = model.tokenizer.eos_token_id
    if ids is None:
        return ()
    return tuple(dict.fromkeys([ids] if isinstance(ids, int) else ids))


def _decode(
    model: Any, tracer: Any, program: GroupProgram, out: dict[str, Any]
) -> None:
    """Walk the decode steps, then reduce every continuation read against
    the continuation the decode produced.

    ``tracer.iter`` counts forward passes and step 0 is the prefill, so
    bodies ``1 … depth`` are the forwards consuming generated tokens
    ``0 … depth-1``. Each body reads the embedding's input first: a
    location that fires every forward pins the body to that forward, so
    an op with no prefill occurrence (the recurrent kernel) is read at
    its step rather than one occurrence early — without it the loop
    outruns such an op and is cut short. Ops are navigated fresh each
    step (one :class:`Navigation` per body): a request binds to the current
    occurrence. ``tracer.result`` is the last request — it consumes the
    whole run.

    The continuation frame is built here, where the decode ran: a row's
    width is the count before its first eos, positions resolve to decode
    steps, each read is gathered at its steps, and an ``lm_head`` read is
    projected from the kept ``ln_final`` steps. The ids, the widths and
    each read's steps go back with the values, so the client rebuilds the
    same frame without the activations.
    """
    depth, batch_size = program.depth, program.batch_size
    sinks: dict[TapKey, list[torch.Tensor]] = {op.key: [] for op in program.step_ops}
    embedding = program.embedding
    for _ in tracer.iter[1 : depth + 1]:
        _ = embedding.input
        nav = Navigation(program.model_key)
        for op in program.step_ops:
            if op.address is None:
                value = _capture(op.site, batch_size)
            else:
                native = present_native(op.site, op.address, nav)
                if op.address.fires == "per_step":
                    native = _fire_native(native, op.site)
                value = to_contract(native, op.site.shape, batch_size=batch_size)
            sinks[op.key].append(value)
    ids = tracer.result
    prompt_len = int(program.inputs["input_ids"].shape[1])
    generated = ids[:, prompt_len : prompt_len + depth].detach().cpu()
    widths = continuation_widths(generated, _eos_ids(model))
    continuation = continuation_frame(model.tokenizer, generated, widths)
    for step in program.steps:
        frame = torch.cat(sinks[step.key], dim=1)
        per_row = [
            resolve_steps(
                step.spec,
                continuation,
                row,
                dataset_row=None if program.rows is None else program.rows[row],
                field=program.field,
            )
            for row in range(batch_size)
        ]
        gathered = gather_rows(frame, per_row)
        if step.project is not None:
            gathered = _over(gathered, step.project)
        out["reads"][step.rname] = _offload(_flat(gathered), program)
        out["steps"][step.rname] = per_row
    out["generated"] = generated
    out["widths"] = widths
