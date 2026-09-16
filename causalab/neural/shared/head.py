"""The vocabulary head where the document reads it (spec §4, "Elision").

An ``lm_head`` read at named positions asks for ``lm_head(ln_final)`` at
those positions and nowhere else. The model's own forward runs the head over
every position of every row — ``[rows·seq, d_model] × [d_model, vocab]`` —
and a tap at the head then gathers a column or two out of the result. 📐 On
the A3B (vocab 248 320, 13-token rows) that projection is the largest tensor
of a training step, its forward and backward GEMMs run at 13× the rows the
read wants, and the gather's backward is a vocabulary-wide zero-fill and
accumulate per step (``fill_ [96, 13, 248320]`` 1.1 ms, ``add_`` 3.0 ms);
an eval pass over 900 rows materializes 5.8 GB of logits to read 900 rows
of them.

So such a read taps ``ln_final`` — the head's input — gathers the rows it
names, and runs the head module over the gathered ``[rows, width, d_model]``:
the same module, the same weights and dtype, the same ``F.linear``. Each
logit is one dot product over the same ``d_model`` entries in the same
order; only the GEMM's ``M`` changes, so the value is the model's to the
bit on the CPU, and expected so on CUDA (the parity script under ``_gpu/``
is what says so for a given cuBLAS). A forward on which nothing reads or
writes the head then runs without it — the engine swaps the module out for
the call (``executor._without_head``) — which is the elision spec §4 allows
past the deepest tap, done for the one module past every block.

What keeps the head as an ordinary tap is decided here, once per read and
from the document alone (:func:`projects_head`):

* a read of the **whole sequence** (``pos: all``) — the head as the model
  runs it, over every position;
* a **continuation** read (``generated``) — the decode's own path: the
  prefill's logits pick the first token, and the generate tail already
  projects kept ``ln_final`` steps (``executor._finalize_generated``);
* any read in a group that **decodes** — the prefill's logits are consumed;
* a read in an intervened model that **writes at the head** — the read must
  see the written logits, and a tap below the head cannot;
* a read a **gradient flows through** (:func:`resolve_read_taps`'s
  ``differentiable``: a grad-enabled executor's read of the model it trains).
  📐 The forward is the model's to the bit on the H100 at every workflow
  shape (M ∈ {42, 96, 900}, bf16 and fp32), but the head's *backward* GEMM —
  ``grad[M, vocab] × W[vocab, d_model]`` at ``M = rows`` instead of
  ``rows·seq`` with zero rows — is a different problem for cuBLAS over
  ``K = 248 320``, and a different split-K moves the fp32 accumulation
  order: on the A3B the bf16 gradient bits moved, an early stop flipped, and
  the fit diverged (measured on one H100, 2026-09-15). So a training read keeps the
  head as the model runs it and its gradient is bit-identical; every no-grad
  pass — an eval, a scoring pass, a locate / control / harvest / ablate
  forward, a fit's fit-constant source groups — projects.
  :data:`ENV_PROJECT_UNDER_GRAD` set to ``1`` projects under grad too, as a
  documented bf16-level change to the gradient (the forward is exact).

A featurizer or ``dims`` on the read changes nothing: both apply to the
gathered value, after the projection (``ExecutorBase._finalize_read``). A
``save`` of the read writes the same tensor either way — the value is
position-resolved before it is saved.

The same predicate drives the campaign store's tap union
(``execution._tap_union``), so a shared pass captures ``ln_final`` for such
a read and every later point finds it under that key: the store holds
``[rows, seq, d_model]`` where it held ``[rows, seq, vocab]``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Iterable

import torch

from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.plan import generated_budget
from causalab.protocol.schema import Document, PositionSpec, ReadSpec, SiteSpec

__all__ = [
    "ENV_PROJECT_UNDER_GRAD",
    "HEAD",
    "HEAD_INPUT",
    "ReadTap",
    "capture_spec",
    "head_module",
    "projects_head",
    "resolve_read_taps",
    "taps_head",
]

#: The component the vocabulary projection is read at.
HEAD = "lm_head"
#: The component that is the head's input — what a projecting read taps.
HEAD_INPUT = "ln_final"
#: Set to ``1`` to project a read a gradient flows through as well — the
#: head's backward then runs at ``M = rows`` and its bf16 gradient may differ
#: from the model's in the last bit (module docstring). Unset: exact training.
ENV_PROJECT_UNDER_GRAD = "CAUSALAB_PROJECT_HEAD_UNDER_GRAD"


def projects_under_grad() -> bool:
    """Whether the environment asks for the projection under grad too."""
    return os.environ.get(ENV_PROJECT_UNDER_GRAD, "") == "1"


def _group_reads(doc: Document, model: str, input_role: str) -> list[ReadSpec]:
    return [
        read
        for read in doc.reads.values()
        if str(read.model) == model and str(read.input) == input_role
    ]


def _writes_at_head(doc: Document, model: str) -> bool:
    if model == "original":
        return False
    writes = doc.intervened_models[model].writes
    if not isinstance(writes, tuple):
        return True  # an unexpanded write set: decide nothing, keep the head
    return any(
        doc.sites[str(doc.writes[ename].site)].component == HEAD for ename in writes
    )


def projects_head(doc: Document, model: str, input_role: str, rname: str) -> bool:
    """Whether read ``rname`` — a read of group ``(model, input_role)`` — is
    served by projecting the gathered ``ln_final`` rows through the head
    rather than by tapping the head (module docstring). ``False`` for every
    read that does not tap ``lm_head``."""
    read = doc.reads[rname]
    if doc.sites[str(read.site)].component != HEAD:
        return False
    spec = doc.positions[read.pos] if isinstance(read.pos, str) else read.pos
    if not isinstance(spec, PositionSpec) or spec.all is not None:
        return False
    if any(
        generated_budget(doc, other.pos) is not None
        for other in _group_reads(doc, model, input_role)
    ):
        return False
    return not _writes_at_head(doc, model)


def capture_spec(doc: Document, model: str, input_role: str, rname: str) -> SiteSpec:
    """The site read ``rname`` **captures** in its group's forward: its own,
    or ``ln_final`` when it projects the head itself."""
    if projects_head(doc, model, input_role, rname):
        return SiteSpec(component=HEAD_INPUT)
    return doc.sites[str(doc.reads[rname].site)]


def head_module(bundle: Any) -> Any:
    """The head module of ``bundle``'s model, as the site resolver finds it."""
    return resolve_site(bundle, SiteSpec(component=HEAD)).module


@dataclasses.dataclass(frozen=True)
class ReadTap:
    """What one read captures and how the captured slice becomes its value:
    ``site`` is the site the read names (its shape, its width, what a
    featurizer is built for), ``capture`` the site the forward taps for it,
    and ``project`` — the head module, when the two differ — runs over the
    gathered rows before anything else (``_finalize_read(project=)``)."""

    site: ResolvedSite
    capture: ResolvedSite
    project: Callable[[torch.Tensor], torch.Tensor] | None = None


def resolve_read_taps(
    bundle: Any,
    doc: Document,
    model: str,
    input_role: str,
    reads: Iterable[tuple[str, ReadSpec]],
    *,
    differentiable: bool = False,
) -> dict[str, ReadTap]:
    """Resolve the prompt-frame reads of one group to their taps: each read's
    own site, and — for a read :func:`projects_head` admits — ``ln_final``
    as the capture with the head module as the projection. The head is
    resolved once for the group.

    ``differentiable`` says a gradient flows through this group's reads (a
    grad-enabled executor reading the model it trains): such a group keeps
    the head as the model runs it, so the backward GEMM is the one the
    model's own forward would have paid and the gradient is bit-identical
    (module docstring) — unless :data:`ENV_PROJECT_UNDER_GRAD` asks
    otherwise."""
    project = not differentiable or projects_under_grad()
    out: dict[str, ReadTap] = {}
    head: Any = None
    norm: ResolvedSite | None = None
    for rname, read in reads:
        site = resolve_site(bundle, doc.sites[str(read.site)])
        if not project or not projects_head(doc, model, input_role, rname):
            out[rname] = ReadTap(site=site, capture=site)
            continue
        if head is None:
            head = head_module(bundle)
            norm = resolve_site(bundle, SiteSpec(component=HEAD_INPUT))
        assert norm is not None
        out[rname] = ReadTap(site=site, capture=norm, project=head)
    return out


def taps_head(sites: Iterable[ResolvedSite]) -> bool:
    """Whether any of ``sites`` is the head — a tap or a write there means
    the forward has to run it."""
    return any(site.component == HEAD for site in sites)
