"""Point-protocol execution over one nnsight trace per forward group.

:class:`NnterpExecutor` inherits the lazy forward groups, position
resolution, featurizer stacks and class-ordered write math of
:class:`~causalab.neural.shared.executor_base.ExecutorBase`. What it owns is
the two ends of a forward group, and nothing in between:

* **before the trace — planning** (:meth:`NnterpExecutor._plan`): resolve
  the group's reads and writes to sites and interior addresses, refuse what
  has none, resolve every position, look every operand up, build every
  featurizer stack, and freeze the result as a
  :class:`~causalab.neural.engines.nnsight_nnterp.program.GroupProgram`;
* **after the trace — finalization** (:meth:`NnterpExecutor._finalize`):
  replay the fires the block reported and check them, and finish each read
  from the slice the block gathered (the featurizer stack and ``dims``, the
  part that needs only the document).

The trace itself is :func:`~causalab.neural.engines.nnsight_nnterp.landers.run_program`,
module-level functions over the program. **No trace body in this package
reads ``self``**: nnsight ships every name a block reads, whole, so an
executor inside a block would travel to NDIF with the document, the bundle,
every earlier read and every cache (measured: a 50 MB payload against
11 KB). The same program runs the same functions locally, so there is one
code path — ``_run_group`` reads top to bottom as plan → program → run →
finalize whatever ``remote`` is.

Operations are issued in the model's forward order — nnsight parks each
request until the model produces the value and refuses one the model has
already run past — which is each site's depth: its layer and the plan's
component rank within it (:data:`~causalab.protocol.plan.COMPONENT_RANK`).
The rank band *is* the in-forward op order the ``.source`` interiors demand:
on one op, ``.inputs`` (q/k, the kernel's arguments) is requested before a
``.source`` drill (the scores, the state) before ``.output`` (z, the kernel's
return); on the experts source the sort's permutation is requested before
anything it sorted for. Every anchor of the group is instrumented
(``_ = envoy.source``) before the trace opens, because instrumenting rewrites
the forward and must happen before it runs.

Per-chunk components (the DeltaNet state) fire once per kernel chunk: a read
collects every fire and its position axis is the chunk index; a write
targets one fire per member, and a fire past the last is refused rather
than silently never run. What has no prefill address — the per-token
recurrent faces the chunked kernel never materializes — is refused by name
before any trace, naming the reference engine.

A group with a continuation read (§2.3 ``generated``) runs as one
``model.generate`` trace instead: the schedule binds the prefill
(occurrence 0 of every location — the whole of "writes are prefill-only"),
and the decode steps are walked afterwards. Module boundaries and the
attention function's stackable slots are read per step through their
prompt-frame landings; the DeltaNet state through the recurrent kernel's own
address
(:data:`~causalab.neural.engines.nnsight_nnterp.sources.GENERATED_ADDRESSES`);
every other interior is refused by name, because decode dispatches
different kernels than prefill and a prefill address is no evidence the
tensor exists per step.

**Remote mode** (``remote=True``, a host URL, or ``"local"`` — nnsight's
in-process dry run of the serialize → deserialize → execute path) runs the
forwards on NDIF against a weight-free bundle. :meth:`NnterpExecutor.run_all`
then runs the whole point as **one session**: every group is planned on the
client, the groups run in dependency order inside one
``model.session(remote=…)``, and a read a later group writes with flows
between the traces on the server (its feature tail applied there from the
shipped stack) instead of round-tripping through the client. A point with an
operand that cannot be finished on the server — a ragged read, the ragged
``expert:`` face, a routed-interior read (its routing table travels with
it), a state or per-fire read, a continuation read — falls back to one job
per group with the operand shipped by value; so does a lazy
:meth:`~causalab.neural.shared.executor_base.ExecutorBase.read_value`
before ``run_all``. Remote mode refuses what cannot cross: a
gradient-enabled executor (saved values come back detached) and a
``pytorch_fn`` write (its code is the caller's, not the server's).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from typing import Any

import torch

from causalab.neural.engines.nnsight_nnterp.adapter import matched_family
from causalab.neural.engines.nnsight_nnterp.landers import run_program, run_session
from causalab.neural.engines.nnsight_nnterp.program import (
    PATTERN,
    FirePlan,
    FlowPlan,
    GroupProgram,
    ReadPlan,
    StepPlan,
    WritePlan,
    needs_eager,
    schedule,
)
from causalab.neural.engines.nnsight_nnterp.sources import (
    ADDRESSES,
    GENERATED_ADDRESSES,
    SourceAddress,
)
from causalab.neural.shared.encoding import EncodedBatch, continuation_frame
from causalab.neural.shared.executor_base import (
    ExecutorBase,
    RaggedValue,
    TapKey,
    refuse_unstackable,
    tap_key,
    whole_native_tensor,
)
from causalab.neural.shared.fires import FireTally, GroupFires, check_fires, group_label
from causalab.neural.shared.head import HEAD, HEAD_INPUT, head_module
from causalab.neural.shared.kernels import torch_kernel_path
from causalab.neural.shared.loading import torch_module
from causalab.neural.shared.mechanisms import operand_names
from causalab.neural.shared.sites import ResolvedSite, _probe_grouped_mm, resolve_site
from causalab.protocol.errors import ProtocolError
from causalab.protocol.plan import generated_budget
from causalab.protocol.schema import PositionSpec, ReadSpec, SiteSpec, WriteSpec

__all__ = ["NnterpExecutor"]

#: Tap kinds with no module boundary: served through an address of
#: :data:`~causalab.neural.engines.nnsight_nnterp.sources.ADDRESSES`, refused
#: by name where the tree has none.
_INTERIOR_KINDS = frozenset({"interface", "delta", "experts", "interior"})

#: The per-token DeltaNet faces: prefill runs only the chunked kernel, which
#: never materializes them (the recurrent kernel exists only in decode).
_NOT_IN_PREFILL = frozenset({"delta_kv_mem", "delta_state_update", "delta_state"})

Group = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class _Read:
    """One read of a planned group, as finalization needs it: the site it
    names and the positions the block gathered at — ``None`` where only the
    forward knows them (a fire axis, the continuation frame) or there are
    none (a tap with no contract form)."""

    rname: str
    read: ReadSpec
    site: ResolvedSite
    per_row: list[list[int]] | None


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    """One planned group: the program that ships, and what stays on the
    client to finish it — the fire declarations, the reads, the batch.
    ``unflowable`` names the reads a later group consumes that cannot be
    finished on the server."""

    group: Group
    program: GroupProgram
    batch: EncodedBatch
    tally: FireTally
    reads: tuple[_Read, ...]
    steps: tuple[_Read, ...]
    unflowable: tuple[str, ...]


def _anchored(spec: PositionSpec) -> bool:
    """Whether ``spec`` needs text to resolve — a span, a variable, a column
    or an anchor — which a fire axis (the kernel's chunk index) has none of."""
    return any(
        getattr(spec, field) is not None
        for field in ("span", "variable", "column", "scope", "relative_to")
    )


class NnterpExecutor(ExecutorBase):
    """Execute one concrete document against one standardized bundle.

    ``batch_rows`` is not honoured: each ``(model, input role)`` group is one
    trace over every row of the role, never a sequence of row windows.

    ``remote`` is where the forwards run: ``False`` in this process,
    ``True`` (or a host URL) on NDIF, ``"local"`` through nnsight's
    in-process dry run of the remote path. It defaults to the bundle's own
    (a weight-free bundle can only run remotely).
    """

    def __init__(
        self, *args: Any, remote: bool | str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.remote: bool | str = (
            getattr(self.bundle, "remote", False) if remote is None else remote
        )
        if self.remote and self.grad_enabled:
            raise ProtocolError(
                "P4",
                "a gradient-enabled executor cannot run remotely: what a "
                "remote forward saves comes back detached, on the CPU, so no "
                "gradient reaches a trained parameter — train against a "
                "locally loaded bundle",
            )

    @functools.cached_property
    def _tree(self) -> str:
        """The registry family whose tree the loaded model is — the address
        table's first key; a tree no family detects has no interior."""
        matched = matched_family(torch_module(self.bundle.model))
        return matched.family if matched is not None else "nnterp_standard"

    # ------------------------------------------------------------------ #
    # running: plan → program → run → finalize
    # ------------------------------------------------------------------ #

    def _run_group(self, model: str, input_role: str) -> None:
        if (model, input_role) in self._groups_run:
            return
        # operands first — the acyclic model graph is the schedule skeleton
        for operand in self._operand_reads(model):
            self.read_value(operand)
        plan = self._plan(model, input_role)
        with self._kernel_path():
            out = run_program(self.bundle.model, plan.program, {}, remote=self.remote)
        self._finalize(plan, out)

    def run_all(self) -> None:
        """Run every group the document implies. Locally that is the lazy
        per-group run; remotely the whole point is one session
        (:func:`~causalab.neural.engines.nnsight_nnterp.landers.run_session`):
        every group planned here, run in dependency order there, operands
        flowing between the traces on the server, reads finalized here
        afterwards. A point with an operand the server cannot finish runs
        one job per group instead."""
        if not self.remote:
            super().run_all()
            return
        self.check_write_widths()
        self.check_answer_forms()
        self.check_scoring()
        self.check_edit_groups()
        order = self._group_order()
        flowing = frozenset(
            operand
            for model, _ in order
            for operand in self._operand_reads(model)
            if operand not in self._read_values
        )
        plans = [self._plan(*group, flowing=flowing) for group in order]
        if any(plan.unflowable for plan in plans):
            for group in order:
                self._run_group(*group)
            return
        with self._kernel_path():
            results = run_session(
                self.bundle.model,
                tuple(plan.program for plan in plans),
                remote=self.remote,
            )
        for plan in plans:
            self._finalize(plan, results[plan.program.label])

    def _kernel_path(self) -> Any:
        """The torch path for the DeltaNet kernel globals around a forward
        in this process (``shared/kernels.py``); a weight-free bundle runs no
        forward here, so there is nothing to bind."""
        if getattr(self.bundle, "remote", False):
            return contextlib.nullcontext()
        return torch_kernel_path(torch_module(self.bundle.model))

    def _writes_of(self, model: str) -> tuple[str, ...]:
        if model == "original":
            return ()
        writes = self.doc.intervened_models[model].writes
        return tuple(writes) if isinstance(writes, tuple) else ()

    def _operand_reads(self, model: str) -> list[str]:
        """The reads ``model``'s writes consume, in write order."""
        return [
            operand
            for ename in self._writes_of(model)
            for operand in operand_names(self.doc.writes[ename].do.payload)
            if operand in self.doc.reads
        ]

    def _group_order(self) -> list[Group]:
        """The point's groups still to run, each after the groups whose
        reads its writes consume — a depth-first walk over the operands,
        originals first."""
        groups = dict.fromkeys(
            (str(read.model), str(read.input)) for read in self.doc.reads.values()
        )
        order: list[Group] = []
        seen: set[Group] = set(self._groups_run)

        def visit(group: Group) -> None:
            if group in seen:
                return
            seen.add(group)
            for operand in self._operand_reads(group[0]):
                read = self.doc.reads[operand]
                visit((str(read.model), str(read.input)))
            order.append(group)

        for group in sorted(groups, key=lambda group: group[0] != "original"):
            visit(group)
        return order

    # ------------------------------------------------------------------ #
    # before the trace: planning
    # ------------------------------------------------------------------ #

    def _plan(
        self, model: str, input_role: str, *, flowing: frozenset[str] = frozenset()
    ) -> GroupPlan:
        """Everything one group's forward needs, resolved and frozen.
        ``flowing`` names the reads a later group of the same session
        consumes; each gets the plan that finishes it on the server."""
        prompt: list[tuple[str, ReadSpec]] = []
        decode: list[tuple[str, ReadSpec]] = []
        for rname, read in self.doc.reads.items():
            if str(read.model) == model and str(read.input) == input_role:
                frame = decode if generated_budget(self.doc, read.pos) else prompt
                frame.append((rname, read))
        depth = max(
            (generated_budget(self.doc, read.pos) for _, read in decode), default=0
        )
        batch = self._batch(input_role)

        reads: list[_Read] = []
        captures = []
        unflowable: list[str] = []
        for rname, tap in self._read_taps(model, input_role, prompt).items():
            read = self.doc.reads[rname]
            address = self._address(f"read {rname!r}", tap.capture)
            plan, per_row = self._read_plan(
                rname, read, tap, address, batch, input_role, flows=rname in flowing
            )
            if rname in flowing and (
                not isinstance(plan, ReadPlan) or plan.flow is None
            ):
                unflowable.append(rname)
            reads.append(_Read(rname, read, tap.site, per_row))
            captures.append((tap.capture, address, (plan,)))
        unflowable += [rname for rname, _ in decode if rname in flowing]

        tally = FireTally()
        writes = []
        for site, address, entries in self._write_groups(self._writes_of(model)):
            plan = self._write_plan(site, address, entries, batch, input_role)
            tally.declare(plan.members, 1)
            writes.append((site, address, plan))

        steps, step_plans, step_captures = self._step_plans(decode)
        ops = schedule(captures, writes)
        step_ops = schedule(step_captures, [])
        # instrument every anchor before its forward runs (the first .source
        # on a module rewrites the forward); a bare access outside the trace
        anchors = {
            id(op.site.module): op.site.module
            for op in ops + step_ops
            if op.address is not None
        }
        for anchor in anchors.values():
            _ = anchor.source

        variable = any(plan.spec.variable is not None for plan in step_plans)
        program = GroupProgram(
            label=group_label(model, input_role),
            model_key=str(self.bundle.key),
            inputs={
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
            },
            position_ids=batch.position_ids(),
            ops=ops,
            step_ops=step_ops,
            steps=step_plans,
            depth=depth,
            needs_eager=needs_eager(ops + step_ops),
            grad=self.grad_enabled,
            offload=bool(self.remote),
            rows=tuple(self.role_rows[input_role]) if variable else None,
            field=self.role_fields[input_role],
            embedding=(
                resolve_site(self.bundle, SiteSpec(component="embeddings")).module
                if depth
                else None
            ),
        )
        return GroupPlan(
            group=(model, input_role),
            program=program,
            batch=batch,
            tally=tally,
            reads=tuple(reads),
            steps=steps,
            unflowable=tuple(unflowable),
        )

    def _address(self, what: str, site: ResolvedSite) -> SourceAddress | None:
        """The interior address of ``site``, ``None`` for a module boundary,
        a refusal by name for an interior this tree has no address for."""
        if site.kind not in _INTERIOR_KINDS and site.interface_slot is None:
            return None
        address = ADDRESSES.get((self._tree, site.component))
        if address is None and site.component == PATTERN and site.kind == "out":
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

    def _read_plan(
        self,
        rname: str,
        read: ReadSpec,
        tap: Any,
        address: SourceAddress | None,
        batch: EncodedBatch,
        input_role: str,
        *,
        flows: bool,
    ) -> tuple[ReadPlan | FirePlan, list[list[int]] | None]:
        """How the block reduces one prompt-frame read, and the positions it
        gathers at. A read a later group consumes (``flows``) also gets the
        plan that finishes it on the server — when it can be: a dense,
        positioned read off no routing table."""
        site = tap.site
        if address is not None and address.fires == "per_chunk":
            return self._fire_plan(rname, read.pos), None
        if not site.shape.has_contract_form:
            return ReadPlan(rname, None), None
        per_row = self._positions(read.pos, batch, input_role, cell=rname)
        flowable = (
            site.expert is None
            and not site.shape.state_axes
            and not (address is not None and address.expert_rows)
            and len({len(row) for row in per_row}) == 1
        )
        plan = ReadPlan(
            rname,
            per_row,
            project=tap.project,
            derive=site if site.derivation is not None else None,
            flow=(
                FlowPlan(site, self._read_stack(read, site), read.dims)
                if flows and flowable
                else None
            ),
        )
        return plan, per_row

    def _fire_plan(self, rname: str, pos: Any) -> FirePlan:
        """Positions on a fire axis: ``all``, or one integer index (negative
        counts from the last fire, resolved against the count in the block).
        Anything anchored refuses — there is no text on a chunk axis."""
        spec = self._spec(pos)
        anchored = _anchored(spec)
        if not anchored and getattr(spec, "all", None) is True:
            return FirePlan(rname, True, None)
        index = spec.index if isinstance(spec.index, int) else None
        if anchored or index is None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} addresses positions on a per-fire component, "
                "whose position axis is the kernel's chunk index: only "
                '"all" or a plain integer index resolves there — text '
                "anchors and spans have nothing to resolve against.",
            )
        return FirePlan(rname, False, index)

    def _write_groups(
        self, write_names: tuple[str, ...]
    ) -> list[tuple[ResolvedSite, SourceAddress | None, list[Any]]]:
        """This group's writes by the tensor they land on: the shared
        resolution and policy check, regrouped by ``tap_key`` over the site
        *and* its interior address."""
        groups: dict[TapKey, tuple[ResolvedSite, SourceAddress | None, list[Any]]] = {}
        for _, entries in self._resolve_write_addresses(write_names).values():
            for ename, write, site in entries:
                address = self._address(f"write {ename!r}", site)
                if address is None and site.component == PATTERN:
                    # the value the attention function consumes, which the
                    # mixer's returned weights only *report*: reached through
                    # the softmax op of an addressed tree, nowhere on this one
                    raise ProtocolError(
                        "P4",
                        f"a write at {PATTERN!r} lands on the attention "
                        "function's softmax, which has no interior address on "
                        f"the {self._tree!r} tree of this engine "
                        "(neural/engines/nnsight_nnterp/sources.py)",
                        reason="component_unavailable",
                    )
                groups.setdefault(tap_key(site, address), (site, address, []))[
                    2
                ].append((ename, write, site))
        return list(groups.values())

    def _write_plan(
        self,
        site: ResolvedSite,
        address: SourceAddress | None,
        entries: list[tuple[str, WriteSpec, ResolvedSite]],
        batch: EncodedBatch,
        input_role: str,
    ) -> WritePlan:
        """Every write at one address with its positions, operands and
        stacks looked up — the write math's services as tables
        (``ExecutorBase._write_services`` is the same services as bound
        methods). A read operand not yet run is left to the session's flow."""
        per_fire = address is not None and address.fires == "per_chunk"
        positions: dict[str, list[list[int]]] = {}
        operands: dict[str, Any] = {}
        read_operands: set[str] = set()
        routing: dict[str, torch.Tensor] = {}
        code = None
        for ename, write, entry_site in entries:
            if str(write.do.mechanism) == "pytorch_fn":
                if self.remote:
                    raise ProtocolError(
                        "P4",
                        f"write {ename!r} applies 'pytorch_fn', whose code is "
                        "the caller's own module: a remote forward cannot "
                        "import it — run the point against a locally loaded "
                        "bundle",
                    )
                code = self.doc.code
            if not entry_site.shape.has_contract_form:
                # the tap's own refusals (positions, featurizer, dims) come
                # before anything is built for it, as the write math orders them
                whole_native_tensor(ename, write, None, entry_site)
            elif not per_fire:
                positions[ename] = self._positions(write.pos, batch, input_role)
            for name in operand_names(write.do.payload):
                if name in self.doc.reads:
                    read_operands.add(name)
                    if name in self._read_values:
                        operands[name] = self._read_values[name]
                    if name in self._read_routing:
                        routing[name] = self._read_routing[name]
                elif name not in operands:
                    resolved = self._client_operand(name)
                    if resolved is not None:
                        operands[name] = resolved
        return WritePlan(
            entries=tuple(entries),
            positions=positions,
            stacks={
                ename: self._read_stack(write, entry_site)
                for ename, write, entry_site in entries
            },
            operands=operands,
            read_operands=frozenset(read_operands),
            positioned={name: self._positioned(name) for name in read_operands},
            operand_routing=routing,
            code=code,
            fire_index=self._write_fire_indices(site, entries) if per_fire else {},
        )

    def _client_operand(self, name: str) -> "torch.Tensor | float | None":
        """A featurizer slot or a params entry, resolved here where the
        artifacts are; ``None`` for a payload string that names no operand
        (a mechanism option such as ``gaussian``'s ``axis``) — the block
        refuses it as unresolved if a mechanism ever asks."""
        try:
            return self._operand_lookup(name)
        except ProtocolError as error:
            if "did not resolve at run time" in str(error):
                return None
            raise

    def _write_fire_indices(
        self, site: ResolvedSite, entries: list[tuple[str, WriteSpec, ResolvedSite]]
    ) -> dict[str, int]:
        """The fire each write at a per-fire address targets: one plain
        integer index per write, a negative one counting from the last fire
        exactly as a read's does. Anything anchored refuses — a chunk axis
        has no text; an index the fire count does not cover refuses in the
        block, where the count is known."""
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
            fires[ename] = index
        return fires

    def _step_plans(self, decode: list[tuple[str, ReadSpec]]) -> tuple[Any, Any, Any]:
        """Each continuation read's tap: an ``lm_head`` read captures
        ``ln_final`` per step and is projected at its addressed steps (the
        reference engine's own trick, so both serve the same value); a read
        whose steps do not stack is refused first
        (:func:`refuse_unstackable`)."""
        reads, plans, captures = [], [], []
        for rname, read in decode:
            site = resolve_site(self.bundle, self.doc.sites[str(read.site)])
            refuse_unstackable(rname, site)
            projected = site.component == HEAD
            capture = (
                resolve_site(self.bundle, SiteSpec(component=HEAD_INPUT))
                if projected
                else site
            )
            address = self._step_address(rname, capture)
            reads.append(_Read(rname, read, site, None))
            plans.append(
                StepPlan(
                    rname,
                    tap_key(capture, address),
                    self._spec(read.pos),
                    project=head_module(self.bundle) if projected else None,
                )
            )
            captures.append((capture, address, ()))
        return tuple(reads), tuple(plans), captures

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

    # ------------------------------------------------------------------ #
    # after the trace: finalization
    # ------------------------------------------------------------------ #

    def _finalize(self, plan: GroupPlan, out: dict[str, Any]) -> None:
        """Finish one group from what its block saved: the receipt's stamps,
        the fire check, then each read from its gathered slice."""
        model, input_role = plan.group
        if out.get("attn_eager"):
            self.applied_requirements.add("attn_eager")
        for members, step in out["fired"]:
            plan.tally.fired(members, step=step)
        check_fires(plan.program.label, plan.tally)
        fires = GroupFires()
        fires.fold(plan.tally)
        record = fires.record()
        if record:
            self.fires[plan.group] = record
        for key, counts in out["mismatch"].items():
            # re-inserted at the end: the last write of a row wins
            self._routing_mismatch_pending.pop(key, None)
            self._routing_mismatch_pending[key] = counts

        for entry in plan.reads:
            self._read_values[entry.rname] = self._finish(
                entry, entry.per_row, plan, out
            )
        if plan.steps:
            self._continuations[plan.group] = continuation_frame(
                self.bundle.tokenizer, out["generated"], tuple(out["widths"])
            )
            for entry in plan.steps:
                per_row = out["steps"][entry.rname]
                self._read_steps[entry.rname] = per_row
                self._read_values[entry.rname] = self._finish(entry, per_row, plan, out)
        self._groups_run.add(plan.group)

    def _finish(
        self,
        entry: _Read,
        per_row: list[list[int]] | None,
        plan: GroupPlan,
        out: dict[str, Any],
    ) -> "torch.Tensor | RaggedValue":
        """One read's value from the slice the block gathered: the shared
        finalization past the gather (``pregathered``) — the head's slice,
        the featurizer stack, ``dims``."""
        value = out["reads"][entry.rname]
        routing = out["routing"].get(entry.rname)
        if per_row is None and entry.site.shape.has_contract_form:
            # a fire axis: the block resolved the fires and gathered them
            per_row = [list(range(value.shape[1]))] * int(value.shape[0])
        if per_row is not None and len({len(row) for row in per_row}) != 1:
            widths = tuple(len(row) for row in per_row)
            value = RaggedValue(flat=value, widths=widths)
            if routing is not None:
                routing = RaggedValue(flat=routing, widths=widths)
        return self._finalize_read(
            entry.rname,
            entry.read,
            entry.site,
            value,
            plan.batch,
            plan.group[1],
            per_row=per_row,
            expert_idx=routing,
            pregathered=True,
        )
