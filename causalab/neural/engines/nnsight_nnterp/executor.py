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
every earlier read and every cache (measured: a 50 MB payload against the
program's ~13.5 KB). The same program runs the same functions locally, so there is one
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
in-process dry run of the serialize → deserialize path, over a loaded
bundle) runs the forwards on NDIF against a weight-free bundle.
:meth:`NnterpExecutor.run_all` plans every group on the client, once, before
any job is spent, and runs the point as **one session**: the groups in
dependency order inside one ``model.session(remote=…)``, a read a later
group writes with flowing between the traces on the server (its feature tail
applied there from the shipped stack) instead of round-tripping through the
client. An operand the server cannot finish — a ragged read, the ragged
``expert:`` face, a routed-interior read (its routing table travels with
it), a state or per-fire read, a continuation read — comes home first: the
session is cut in front of the group that consumes it, which ships the
operand by value, and the operands around it still flow. A lazy
:meth:`~causalab.neural.shared.executor_base.ExecutorBase.read_value` before
``run_all`` runs one job per group. A plain forward stops after its group's
last operation, so a shallow read does not pay for the layers above it.
Remote mode refuses what cannot cross: a gradient-enabled executor (saved
values come back detached), a ``pytorch_fn`` write (its code is the
caller's, not the server's), and a weight-free bundle asked to run in this
process.

What remote mode requires of the server, and what it does not promise:

* **a trusted, in-process NDIF deployment with the same ``causalab``
  installed.** The block is ``causalab``'s own functions, imported where it
  runs, and it works on the served model itself — it flips that model's
  attention implementation and calls its head and mixer projections. A
  sandboxed (untrusted) deployment runs the block against a ``meta`` copy;
  the block refuses there by name
  (:func:`~causalab.neural.engines.nnsight_nnterp.landers.execute`). A
  version skew between the two ``causalab`` installs is a skew between the
  plan and the code that reads it.
* **a hard kill mid-block can leave a shared deployment on eager
  attention.** The switch is reversed in a ``finally``, which covers an
  exception and an early stop; a worker killed between the switch and the
  restore (a walltime, an OOM kill) runs no ``finally``, and the next
  request on that replica runs eager until something switches it back.
* **featurizer stages ship by value, per session.** A write's stack (and a
  flowing read's) is pickled into the payload: a full-rank rotation at
  hidden 8192 is 8192² fp32 entries, ~268 MB, in every session that uses
  it. The program without stages is ~13.5 KB, beside the ~85 KB of nnterp
  that a remote ``StandardizedTransformer`` registers for by-value pickling.
* **bit-identity with the local path holds for CPU fp32.** A deployment
  that runs requests under bf16 autocast computes the block's own
  arithmetic (the featurizer tail, the write math, the projections) in
  bf16, and agrees with a local fp32 run to bf16's tolerance, not to the
  bit.
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
    client to finish it — the fire declarations, the reads, the batch."""

    group: Group
    program: GroupProgram
    batch: EncodedBatch
    tally: FireTally
    reads: tuple[_Read, ...]
    steps: tuple[_Read, ...]


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
    in-process dry run of the remote path. ``None`` inherits the bundle's
    own. The two combinations with a bundle:

    * a **weight-free** bundle runs on NDIF only. ``False`` and ``"local"``
      are refused: both run the forward in this process, where nnsight
      would dispatch the whole checkpoint to serve it.
    * a **loaded** bundle runs anywhere. Under ``True`` or a host URL its
      weights sit idle — the block names the model by key and NDIF runs its
      own copy — which is what lets one bundle serve a local reference and a
      remote run side by side.
    """

    def __init__(
        self, *args: Any, remote: bool | str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        weight_free = bool(getattr(self.bundle, "remote", False))
        self.remote: bool | str = weight_free if remote is None else remote
        if weight_free and (not self.remote or self.remote == "local"):
            raise ProtocolError(
                "P4",
                f"the bundle for {self.bundle.key!r} is weight-free (loaded "
                f"with remote=True) and cannot run with remote={self.remote!r}: "
                "that forward runs in this process, where nnsight would "
                "dispatch the whole checkpoint to serve it. Run it on NDIF "
                "(remote=True or a host URL, or leave remote unset to inherit "
                "the bundle's), or load the bundle with its weights.",
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
            out = run_program(
                self.bundle.model, self._bound(plan.program), {}, remote=self.remote
            )
        self._finalize(plan, out)

    def run_all(self) -> None:
        """Run every group the document implies. Locally that is the lazy
        per-group run. Remotely every group is planned here, once, before
        any job is spent, and the groups run in dependency order as sessions
        (:func:`~causalab.neural.engines.nnsight_nnterp.landers.run_session`)
        — one where every operand can flow between the traces on the server,
        which is the usual point. An operand the server cannot finish
        (:meth:`_flowable`) has to come home first, so the session is cut in
        front of the group that consumes it and that group's program is
        bound to the finished value; the operands around it still flow."""
        if not self.remote:
            super().run_all()
            return
        self._preflight()
        segments = self._segments(self._group_order())
        plans = [
            [self._plan(*group, flowing=flowing) for group in groups]
            for groups, flowing in segments
        ]
        for segment in plans:
            programs = tuple(self._bound(plan.program) for plan in segment)
            with self._kernel_path():
                results = run_session(self.bundle.model, programs, remote=self.remote)
            for plan in segment:
                self._finalize(plan, results[plan.program.label])

    def _segments(self, order: list[Group]) -> list[tuple[list[Group], frozenset[str]]]:
        """``order`` cut into sessions, each with the reads that flow inside
        it: a group that consumes an unflowable read of the running session
        opens the next one."""
        segments: list[tuple[list[Group], set[str]]] = [([], set())]
        produced: set[str] = set()
        for group in order:
            local = [name for name in self._operand_reads(group[0]) if name in produced]
            if any(not self._flowable(name) for name in local):
                segments.append(([], set()))
                produced, local = set(), []
            segments[-1][0].append(group)
            segments[-1][1].update(local)
            produced |= {
                rname
                for rname, read in self.doc.reads.items()
                if (str(read.model), str(read.input)) == group
            }
        # a point with nothing left to run is no session, not an empty job
        return [(groups, frozenset(flowing)) for groups, flowing in segments if groups]

    def _flowable(self, rname: str) -> bool:
        """Whether the server can finish read ``rname`` into a session's
        flow: a dense, positioned prompt-frame read off no routing table.
        A ragged read, the ragged ``expert:`` face, a routed-interior read
        (its routing table travels with it), a state or per-fire read, a
        whole-native read and a continuation read are finished here."""
        read = self.doc.reads[rname]
        if generated_budget(self.doc, read.pos):
            return False
        model, input_role = str(read.model), str(read.input)
        tap = self._read_taps(model, input_role, [(rname, read)])[rname]
        site = tap.site
        address = self._address(f"read {rname!r}", tap.capture)
        if (
            not site.shape.has_contract_form
            or site.expert is not None
            or site.shape.state_axes
            or (
                address is not None and (address.expert_rows or address.fires != "once")
            )
        ):
            return False
        per_row = self._positions(
            read.pos, self._batch(input_role), input_role, cell=rname
        )
        return len({len(row) for row in per_row}) == 1

    def _bound(self, program: GroupProgram) -> GroupProgram:
        """``program`` with every read operand this client holds shipped by
        value — what an earlier job brought home. A read operand left out is
        one the running session's flow supplies."""

        def bound(op: Any) -> Any:
            write = op.write
            if write is None:
                return op
            held = [
                name
                for name in sorted(write.read_operands)
                if name in self._read_values
            ]
            if not held:
                return op
            return dataclasses.replace(
                op,
                write=dataclasses.replace(
                    write,
                    operands={
                        **write.operands,
                        **{name: self._read_values[name] for name in held},
                    },
                    operand_routing={
                        name: self._read_routing[name]
                        for name in held
                        if name in self._read_routing
                    },
                ),
            )

        return dataclasses.replace(program, ops=tuple(bound(op) for op in program.ops))

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
        """Everything one group's forward needs but the values of its read
        operands (:meth:`_bound`), resolved and frozen. ``flowing`` names the
        reads a later group of the same session consumes; each gets the plan
        that finishes it on the server."""
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
        for rname, tap in self._read_taps(model, input_role, prompt).items():
            read = self.doc.reads[rname]
            address = self._address(f"read {rname!r}", tap.capture)
            plan, per_row = self._read_plan(
                rname, read, tap, address, batch, input_role, flows=rname in flowing
            )
            reads.append(_Read(rname, read, tap.site, per_row))
            captures.append((tap.capture, address, (plan,)))

        tally = FireTally()
        writes = []
        for site, address, entries in self._write_groups(self._writes_of(model)):
            plan = self._write_plan(site, address, entries, batch, input_role)
            tally.declare(plan.members, 1)
            writes.append((site, address, plan))

        steps, step_plans, step_captures = self._step_plans(decode)
        ops = schedule(captures, writes)
        step_ops = schedule(step_captures, [])
        # build every anchor's `.source` before the trace opens: the block
        # navigates ops by name, and the first access parses the forward. On
        # a loaded tree that access also rewrites the forward, which must
        # precede the run; on a weight-free tree only the parse happens here
        # — the server instruments its own module when the block drills.
        # Under the kernel path: the rewritten forward runs over a snapshot
        # of its module's globals taken at this access, so the DeltaNet
        # kernel globals it will call are the ones bound right now
        anchors = {
            id(op.site.module): op.site.module
            for op in ops + step_ops
            if op.address is not None
        }
        with self._kernel_path():
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
        gathers at. A read a later group of the session consumes (``flows``,
        which :meth:`_flowable` admitted) also gets the plan that finishes
        it on the server."""
        site = tap.site
        if address is not None and address.fires == "per_chunk":
            return self._fire_plan(rname, read.pos), None
        if not site.shape.has_contract_form:
            # the tap's own refusals (positions, featurizer, dims) before the
            # forward, as a write's are: remotely the whole (rows, heads,
            # query, key) tensor would otherwise be a job and a download
            # spent on a read ``_finalize_read`` then refuses
            whole_native_tensor(rname, read, None, site)
            return ReadPlan(rname, None), None
        per_row = self._positions(read.pos, batch, input_role, cell=rname)
        plan = ReadPlan(
            rname,
            per_row,
            project=tap.project,
            derive=site if site.derivation is not None else None,
            flow=(
                FlowPlan(site, self._read_stack(read, site), read.dims)
                if flows
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
            return FirePlan(rname, None)
        index = spec.index if isinstance(spec.index, int) else None
        if anchored or index is None:
            raise ProtocolError(
                "P4",
                f"read {rname!r} addresses positions on a per-fire component, "
                "whose position axis is the kernel's chunk index: only "
                '"all" or a plain integer index resolves there — text '
                "anchors and spans have nothing to resolve against.",
            )
        return FirePlan(rname, index)

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
        """Every write at one address with its positions, artifact operands
        and stacks looked up — the write math's services as tables
        (``ExecutorBase._write_services`` is the same services as bound
        methods). A read operand's value is not planned: :meth:`_bound`
        ships the ones the client holds when the program runs, and the
        session's flow supplies the rest."""
        per_fire = address is not None and address.fires == "per_chunk"
        positions: dict[str, list[list[int]]] = {}
        operands: dict[str, Any] = {}
        read_operands: set[str] = set()
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
                elif name not in operands:
                    # a featurizer slot or a params entry, resolved here where
                    # the artifacts are. A payload string that names neither is
                    # a mechanism option (``gaussian``'s ``axis``): the block
                    # refuses it as unresolved if a mechanism ever asks
                    resolved = self._artifact_operand(name)
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
            operand_routing={},
            code=code,
            fire_index=self._write_fire_indices(site, entries) if per_fire else {},
        )

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
                "in the nnterp engine's tables "
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
