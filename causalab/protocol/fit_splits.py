"""A fit's declared splits are endpoint-disjoint across tables (§2.2, §5 rule 22).

Rule 22's three resolver-side refusals are each a property of **one table**:
every table declares its splits, a bare ref may not name a partitioned table,
and the splits of one table share no endpoint. A fit consumes rows in two
roles — the training rows its ``data`` refs select and the held-out rows
``train.eval.split`` names — and when both are splits of one table the third
refusal already makes them disjoint. When they are **two different tables**
nothing did: ``weekdays/train`` and ``weekdays/test`` asserted their
relationship in their names and nowhere a reader could reach, which is exactly
the two-file layout §2.2 disparages. This is the fourth refusal, the same
property generalized from one table to the tables one fit names.

Two placements, one function. The ``validate --data`` pass calls it where the
other row-level checks live (:func:`~causalab.protocol.loader.check_data_columns`),
and :mod:`causalab.neural.shared.execution` calls it once per point before an
executor exists — so ``run``, a workflow step and every engine's ``execute``
reach it before any forward. It lives in its own module rather than in
``resolve.py`` or ``loader.py`` because those are in every hashed script step's
import closure, and a byte changed there moves a workflow pin.

What is **not** refused: the same ref named for both roles. A deliberate
train-equals-test ablation is spelled by naming one split twice, where the
document shows it (``resolve._check_splits_are_disjoint``); two refs that
quietly coincide are the leak, one ref written twice is the declaration.

Torch-free and stdlib-only beyond the protocol package.
"""

from __future__ import annotations

from typing import Any

from causalab.protocol.errors import ValidationError
from causalab.protocol.resolve import DatasetResolver, endpoints, split_dataset_ref
from causalab.protocol.schema import Document

__all__ = ["HELD_OUT_ROLE", "check_fit_splits", "fit_roles"]

#: The path of the held-out role — the one ``train`` field that names a
#: dataset ref. There is no third role on a ``train`` block today
#: (``checkpoint.file_path`` is an artifact path, not a table); one that
#: arrives joins :func:`fit_roles` and is checked by the same loop.
HELD_OUT_ROLE = "train.eval.split"


def fit_roles(doc: Document) -> tuple[list[tuple[str, str]], str | None]:
    """The refs a fit consumes, by role: ``(path, ref)`` for every training
    role (``data.base``, ``data.counterfactual``, ``data.counterfactual[j]``,
    the spelling :func:`~causalab.protocol.loader.check_data_columns` reports
    under) and the held-out ref, or ``None`` when the fit declares no ``eval``.

    A value that is still a sweep wrapper is skipped: the check runs on
    expanded points, where every ref is one string.
    """
    training: list[tuple[str, str]] = []
    for name, value in doc.data.items():
        roles = value if isinstance(value, tuple) else (value,)
        for index, role in enumerate(roles):
            path = (
                f"data.{name}[{index}]" if isinstance(value, tuple) else f"data.{name}"
            )
            if isinstance(role.dataset, str):
                training.append((path, role.dataset))
    held_out: str | None = None
    if doc.train is not None and doc.train.eval is not None:
        split = doc.train.eval.get("split")
        if isinstance(split, str):
            held_out = split
    return training, held_out


def check_fit_splits(doc: Document, datasets: DatasetResolver) -> None:
    """Refuse a fit whose training rows and held-out rows share a prompt at
    either endpoint, unless both roles name the very same ref [V22].

    An endpoint is what :func:`~causalab.protocol.resolve.endpoints` says: the
    row's ``input`` and every ``counterfactual_inputs`` string — the prompts a
    row puts in front of the model, on both sides of the pair. The leak that
    matters is a training base reappearing as a held-out counterfactual, which
    reports a training score under a held-out name.

    A document with no ``train`` block, or a fit with no ``eval``, has one role
    and nothing to compare; it returns without touching the resolver.
    """
    if doc.train is None:
        return
    training, held_out = fit_roles(doc)
    if held_out is None:
        return
    rows_by_ref: dict[str, list[dict[str, Any]]] = {}

    def rows(ref: str) -> list[dict[str, Any]]:
        if ref not in rows_by_ref:
            rows_by_ref[ref] = datasets.rows(ref)
        return rows_by_ref[ref]

    held_out_endpoints: set[str] | None = None
    for path, ref in training:
        if ref == held_out:
            continue  # one ref, twice: the visible train-equals-test ablation
        if held_out_endpoints is None:
            held_out_endpoints = set()
            for row in rows(held_out):
                held_out_endpoints |= endpoints(row)
        for row in rows(ref):
            shared = sorted(endpoints(row) & held_out_endpoints)
            if shared:
                base, _ = split_dataset_ref(ref)
                raise ValidationError(
                    22,
                    f"fit splits leak across tables: the prompt {shared[0]!r} "
                    f"appears in both {ref!r} (the training rows, {path}) and "
                    f"{held_out!r} (the held-out rows, {HELD_OUT_ROLE}). A fit's "
                    f"splits must be endpoint-disjoint across the tables it names "
                    f"(§2.2) — select two splits of one table ({base}#train and "
                    f"{base}#test of one endpoint-disjoint table), or "
                    f"name the same ref for both roles to spell a deliberate "
                    f"train-equals-test ablation where it is visible",
                    path=HELD_OUT_ROLE,
                )
