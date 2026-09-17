"""§2.2 ``draw``: the drawn roles of a fit — one member per row, redrawn
every epoch — and the row slicing a minibatch is cut with."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from causalab.neural.shared.encoding import EncodedBatch, encode
from causalab.neural.shared.executor_base import ExecutorBase
from causalab.neural.shared.services import input_roles
from causalab.protocol.errors import ProtocolError
from causalab.protocol.schema import DataRole, Document

if TYPE_CHECKING:
    from causalab.neural.shared.training.fit import ExecutorFactory, Fit

__all__ = ["Drawn", "slice_rows"]


def slice_rows(
    role_rows: Mapping[str, list[dict[str, Any]]], indices: list[int]
) -> dict[str, list[dict[str, Any]]]:
    return {role: [rows[i] for i in indices] for role, rows in role_rows.items()}


class Drawn:
    """§2.2 ``draw``: the drawn roles of one fit — every member of every row
    encoded once into one expanded frame per role, so a minibatch under any
    draw is a **selection** of that frame (the same padded width, and a
    cohort's members encode the same texts, so they still concatenate — by
    construction), and the per-epoch redraw that rebuilds the fit's
    minibatch executors with one member per row.

    The draw runs on its own stream: the document seed hashed with the word
    ``draw`` (:func:`_stream_seed`), not the raw seed the batch order and the
    mask samples share — both of those call ``torch.rand`` too, and a member
    drawn from the same numbers as a gate's first resample would be
    correlated with the mask noise.

    Refusals at prepare, two codes: a row with no members, an ``eval`` past a
    row's count and a per-member sibling of another length are the *data*'s
    shape (P2); prepared encodings and ``segments`` are things this engine
    cannot combine with a re-encoded role (P4).

    The minibatch executors are the engine's: ``executor_factory``
    (:class:`~causalab.neural.shared.training.fit.ExecutorFactory`) builds
    each one, told it is ``drawn``. The shapes *are* epoch-invariant (every
    minibatch is a ``select`` of one frame); it is the per-epoch rebuild of
    the executors that anything keyed by executor cannot follow, which
    :meth:`redraw` tells the fit (``Fit.minibatches_rebuilt``)."""

    def __init__(
        self,
        doc: Document,
        executor: ExecutorBase,
        roles: Mapping[str, DataRole],
        seed: int,
        executor_factory: "ExecutorFactory",
    ) -> None:
        self.doc = doc
        self.executor = executor
        self.executor_factory = executor_factory
        self.roles = dict(roles)
        self.rng = torch.Generator().manual_seed(_stream_seed(seed, "draw"))
        self.trace: dict[str, list[list[int]]] = {role: [] for role in roles}
        self.members: dict[str, list[list[Any]]] = {}
        self.offsets: dict[str, list[int]] = {}
        self.expanded: dict[str, EncodedBatch] = {}
        for role, spec in roles.items():
            column = spec.draw_column
            assert column is not None
            members: list[list[Any]] = []
            for i, row in enumerate(executor.role_rows[role]):
                values = row.get(column)
                if not isinstance(values, list) or not values:
                    raise ProtocolError(
                        "P2",
                        f"data.{role}: row {i} has no non-empty list in column "
                        f"{column!r} to draw from",
                    )
                if spec.eval_member >= len(values):
                    raise ProtocolError(
                        "P2",
                        f"data.{role}.draw.eval = {spec.eval_member} but row {i} "
                        f"holds {len(values)} member(s) in {column!r}",
                    )
                members.append(list(values))
                # the per-member siblings are checked here, once, with the other
                # data-shape refusals — not on every epoch's redraw
                _check_member_siblings(row, column, len(values), role=role, index=i)
            offsets, total = [], 0
            for values in members:
                offsets.append(total)
                total += len(values)
            self.members[role] = members
            self.offsets[role] = offsets
            self.expanded[role] = encode(
                executor.bundle.tokenizer,
                [str(t) for values in members for t in values],
                device=executor.bundle.device,
            )

    @classmethod
    def of(
        cls,
        doc: Document,
        executor: ExecutorBase,
        seed: int,
        executor_factory: "ExecutorFactory",
    ) -> "Drawn | None":
        roles = {
            role: spec
            for role, spec in input_roles(doc).items()
            if spec.draw is not None and role in executor.role_rows
        }
        if not roles:
            return None
        if doc.segments is not None:
            raise ProtocolError(
                "P4",
                "data.*.draw: a drawn role is encoded as plain text; it does not "
                "combine with `segments` (a declared frame) in this engine",
            )
        for role, spec in roles.items():
            # every index of the collapsed lists reads the drawn member, so the
            # field the rest of the document reads must be the one the fit
            # uses: `resolve_roles` hands the engine `spec.resolved_field`, and
            # the minibatch executors take the executor's fields as they are.
            # An epoch-invariant — checked here once, like the siblings, not
            # on every redraw
            assert executor.role_fields[role] == spec.resolved_field, (
                f"data.{role}: the executor reads {executor.role_fields[role]!r}, "
                f"the draw {spec.resolved_field!r}"
            )
            # prepared token sequences own their frame (`executor_base._batch`
            # dispatches to `encode_prepared` on a `<column>_encoding` sibling);
            # a drawn role is re-encoded from text, so the fit would run on
            # other tokens than the point's reads and `train.eval` — refused,
            # for the same reason `segments` is
            prepared = f"{spec.draw_column}_encoding"
            if any(prepared in row for row in executor.role_rows[role]):
                raise ProtocolError(
                    "P4",
                    f"data.{role}.draw: prepared inputs ({prepared!r}) own their "
                    "frame; this engine re-encodes a drawn role from text every "
                    "epoch, so the fit would train on other tokens than the "
                    "point reads — drop the encoding sibling or the draw",
                )
        return cls(doc, executor, roles, seed, executor_factory)

    def record(self) -> dict[str, dict[str, Any]]:
        """§2.2: what ``fit_diagnostics.json`` carries under ``draws`` — per
        drawn role the kind, ``eval`` and the member each row took, one list
        per epoch. Built here, from the roles that drew, so a role the executor
        does not carry gets no entry."""
        return {
            role: {
                "kind": str(spec.draw["kind"]),  # type: ignore[index]  # `of` filtered on draw
                "eval": spec.eval_member,
                "members": [list(epoch) for epoch in self.trace[role]],
            }
            for role, spec in self.roles.items()
        }

    def draw(self) -> dict[str, list[int]]:
        """One member index per row per drawn role, uniform over the row's
        members, from this fit's own generator."""
        out: dict[str, list[int]] = {}
        for role, members in self.members.items():
            u = torch.rand(len(members), generator=self.rng)
            out[role] = [
                min(int(u[i].item() * len(values)), len(values) - 1)
                for i, values in enumerate(members)
            ]
        return out

    def minibatches(self) -> list[ExecutorBase]:
        """One epoch's minibatch executors over the partition and frames
        :meth:`bind` set: a fresh draw, the drawn rows, and per batch a
        selection of the expanded frame. The factory is told the executors
        are ``drawn``: a forward store is not to be consulted by them at all —
        a source forward over a drawn role is constant for no two epochs, and
        the store is per executor, so the base role's forwards are recomputed
        with it (narrowing the bypass to the drawn role is a follow-up)."""
        picks = self.draw()
        for role, chosen in picks.items():
            self.trace[role].append(list(chosen))
        executor = self.executor
        role_rows: dict[str, list[dict[str, Any]]] = dict(executor.role_rows)
        for role, spec in self.roles.items():
            column = spec.draw_column
            assert column is not None
            role_rows[role] = [
                _drawn_row(row, column, picks[role][i])
                for i, row in enumerate(executor.role_rows[role])
            ]
        out: list[ExecutorBase] = []
        for indices in self.batches:
            selected = {
                role: (
                    self.expanded[role].select(
                        [self.offsets[role][i] + picks[role][i] for i in indices]
                    )
                    if role in self.roles
                    else frame.select(indices)
                )
                for role, frame in self.frames.items()
            }
            out.append(
                # the factory hands every inner executor the outer one's
                # fields as they are (`role_fields[role]` is the drawn
                # member's — `of` checks it once, at prepare) and its stage
                # cache, shared, not per executor: `prepare_fit` built the
                # optimizer's groups over these stages, so a rebuilt
                # minibatch's `stage(name)` is a cache hit on the tensors
                # being stepped — a fresh cache would re-initialise every
                # featurizer each epoch while the optimizer stepped the
                # originals (the identity test diverges at the first
                # checkpoint)
                self.executor_factory(
                    self.doc,
                    executor,
                    role_rows=slice_rows(role_rows, indices),
                    grad_enabled=True,
                    rows=tuple(indices),
                    batches=selected,
                    drawn=True,
                )
            )
        return out

    def bind(
        self, batches: Sequence[list[int]], frames: Mapping[str, EncodedBatch]
    ) -> None:
        """The fit's minibatch partition and its per-role frames — what every
        epoch's :meth:`minibatches` selects from; known once the frames are
        encoded, and bound before the first draw is taken, so no reader of
        ``fit.drawn`` can find them unset."""
        self.batches = list(batches)
        self.frames = dict(frames)

    def redraw(self, fit: "Fit") -> None:
        """A new member per row for a new epoch: the fit's minibatch
        executors are rebuilt over this draw, and whatever the engine keyed
        by executor falls with them (``Fit.minibatches_rebuilt``)."""
        fit.minibatch_executors = self.minibatches()
        fit.minibatches_rebuilt()


#: The per-member siblings of a list column (§2.2): the prompt-variable table
#: (``encoding.variable_value`` reads ``<column>_variables``) and the prepared
#: token sequences (``prepared.encoding_field``: ``<column>_encoding``; a drawn
#: role refuses those in :meth:`Drawn.of`, so neither consumer below ever
#: sees one — it is censused here as the convention's second member). Named,
#: not a prefix: another ``<column>_…`` column is the author's own and is
#: left alone.
_MEMBER_SIBLINGS: tuple[str, ...] = ("_variables", "_encoding")


def _member_siblings(row: Mapping[str, Any], column: str) -> list[str]:
    return [column + suffix for suffix in _MEMBER_SIBLINGS if column + suffix in row]


def _check_member_siblings(
    row: Mapping[str, Any], column: str, count: int, *, role: str, index: int
) -> None:
    """A per-member sibling holds one entry per member: a shorter or longer
    list is refused rather than passed through half-collapsed — the document
    would read member 0 of it against member ``m`` of the text. (The
    fixed-member path tolerates a short ``_variables`` table by falling back
    to the plain column; under a draw there is no one member to fall back
    to.) A prepare-time check: it depends on the row, never on the draw."""
    for key in _member_siblings(row, column):
        value = row[key]
        if isinstance(value, list) and len(value) != count:
            raise ProtocolError(
                "P2",
                f"data.{role}: row {index} holds {len(value)} entries in {key!r} for "
                f"the {count} member(s) of {column!r} — a per-member sibling holds "
                "one entry per member",
            )


def _drawn_row(row: Mapping[str, Any], column: str, member: int) -> dict[str, Any]:
    """The row a minibatch executor reads for a drawn role. The list column
    and its per-member siblings (:data:`_MEMBER_SIBLINGS`) hold the drawn
    member at **every** index, so ``<column>[eval]`` and ``<column>[0]`` alike
    read it inside the fit (``eval`` need not be 0). Every other column — a
    ``<column>_…`` list of the author's own included — is untouched. Pure
    rewriting: the shape was checked at prepare (:func:`_check_member_siblings`),
    so every list here has one entry per member and its own length is the
    count."""
    out = dict(row)
    for key in (column, *_member_siblings(row, column)):
        value = row[key]
        if isinstance(value, list):
            out[key] = [value[member]] * len(value)
    return out


def _stream_seed(seed: int, purpose: str) -> int:
    """A generator seed for one purpose derived from the document seed, so
    two purposes that both call ``torch.rand`` do not consume the same
    numbers; stable across processes (sha256, not ``hash``)."""
    digest = hashlib.sha256(f"{int(seed)}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)
