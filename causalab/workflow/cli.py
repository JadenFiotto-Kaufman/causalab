"""The CLI verbs for **workflow** documents (docs/workflow_protocol.md §9).

Split out of ``protocol/cli.py`` so the protocol package carries no workflow
code: someone who wants only the intervention protocol imports only that.
Dispatch between the two document types is :mod:`causalab.cli`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

from causalab.protocol.engine import DEFAULT_ENGINE
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.loader import check_data_columns

__all__ = ["main"]


def _pin_summary(pins: Any) -> str:
    """``3 documents, 2 scripts, 1 dataset`` — the census by category, for
    the `pin`, `run` and `explain` lines."""
    parts = [
        f"{len(entries)} {category if len(entries) != 1 else category[:-1]}"
        for category, entries in pins.items()
        if entries
    ]
    return ", ".join(parts) if parts else "nothing to pin"


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    from causalab.protocol.compile import read_document
    from causalab.protocol.loader import apply_overrides as _apply
    from causalab.protocol.loader import load_text as _load_text
    from causalab.cli import register_model_key, wants_hf_registration
    from causalab.workflow.document import load_workflow

    if args.verb == "run":
        if getattr(args, "dtype", None) is not None:
            print(
                "refused: --dtype sets model.dtype on one intervention "
                "specification; a workflow's steps each declare their own "
                "realization — set it in the step's document, or with that "
                "step's own `set` block",
                file=sys.stderr,
            )
            return 1
        if args.points is not None:
            print(
                "refused: --points shards a single document's expanded "
                "campaign; a workflow schedules whole steps — shard the "
                "inner document runs instead",
                file=sys.stderr,
            )
            return 1

    if wants_hf_registration(args):
        # `run` touches models anyway, and `--register-from-hf` is the author
        # asking for it: pre-register **every inner** model key BEFORE loading,
        # because canonicalization derives widths from the registry. A workflow
        # names several documents, so registering only the outer one would
        # pre-flight nothing — which is why the three A3B runs' hand-rolled
        # wrappers had to be workflow-aware too.
        raw_wf = _apply(dict(_load_text(args.document)), dict(args.parsed_set))
        for doc_path, overrides in _inner_documents(
            raw_wf, args.document.resolve().parent, _load_text
        ):
            try:
                # the compiler's own read prefix (IM spec §9), so the
                # key this pre-registers is the one the compile will read
                inner = read_document(doc_path, doc_path.parent, overrides)
            except ProtocolError:
                continue
            register_model_key(dict(inner.raw))

    loaded = load_workflow(
        args.document.resolve(),
        env,
        overrides=dict(args.parsed_set),
        # `pin` exists to replace a stale section, so it is the one verb the
        # section cannot refuse (§7); every other verb holds the document to it
        hold_pins=args.verb != "pin",
    )
    if args.verb == "validate":
        if getattr(args, "data", False):
            for name in loaded.order:
                inner = loaded.inner.get(name)
                if inner is not None:
                    check_data_columns(inner, env)
        print(f"OK: {args.document} — {len(loaded.document.steps)} steps")
        return 0
    if args.verb == "pin":
        # the census the load just made, written into the file as its last
        # section (§7) — under `--set` the census describes the overridden
        # closure, not the file's, so there is nothing honest to stamp
        if args.parsed_set:
            print(
                "refused: pin stamps what the document as written touches; "
                "--set describes another closure. Pin without --set, or write "
                "the override into the document first",
                file=sys.stderr,
            )
            return 1
        from causalab.workflow.pins import stamp_pins

        stamp_pins(args.document, loaded.pins)
        print(f"pinned {args.document} — {_pin_summary(loaded.pins)}")
        return 0
    if args.verb == "digest":
        # the identities `--resume` compares, one per step in schedule order
        # (§7): a script, behavioral, decision, conditional or fanned-out step's
        # entry digest; an unfanned protocol step's inner document digest.
        # There is no whole-workflow digest to print — nothing compares one.
        for name in loaded.order:
            identity = loaded.step_digests.get(name) or loaded.inner_digests[name]
            print(f"{name}  {identity}")
        return 0
    if args.verb == "explain":
        print(f"schedule  {len(loaded.levels)} levels")
        for i, level in enumerate(loaded.levels):
            print(f"  level {i}: {', '.join(level)}")
        _explain_steps(loaded, "  ")
        if loaded.nondeterministic:
            # §7: explain names the steps that make a run unreplayable, so the
            # gap is visible before anyone trusts a rerun
            print(
                "not replayable: "
                + ", ".join(loaded.nondeterministic)
                + " (is_deterministic: false)"
            )
        if loaded.unchecked_paths:
            # rule 4: an absolute path is not existence-checked at load,
            # because validation and execution routinely run on different hosts
            print("unchecked absolute paths (verified at run time):")
            for item in loaded.unchecked_paths:
                print(f"  {item}")
        # §7: a pinned document was just held to its section (rule 21) — say
        # so; an unpinned one says how it gets pinned
        if loaded.document.pins is not None:
            print(f"pins      checked — {_pin_summary(loaded.pins)}")
        else:
            print(
                f"pins      none — the first `run` stamps {_pin_summary(loaded.pins)}"
                " into the document (or `causalab pin <wf>`)"
            )
        return 0
    # run — engines are optional, lazily-imported extras; --engine picks
    # the list, choose_engine routes per protocol step
    from causalab.cli import load_engines
    from causalab.workflow import run_workflow

    if loaded.document.pins is None:
        # §7: the first run of an unpinned workflow stamps it, so the next
        # load holds the document to exactly the closure this run consumed.
        # Under `--set` the census is the overridden closure, not the file's:
        # nothing honest to write, and the run proceeds unpinned
        if args.parsed_set:
            print(
                f"note: {args.document} is not pinned and --set is in effect, so "
                "this run stamps nothing — run once without --set, or `causalab "
                "pin` it",
                file=sys.stderr,
            )
        else:
            from causalab.workflow.pins import stamp_pins

            stamp_pins(args.document, loaded.pins)
            print(f"pinned {args.document} — {_pin_summary(loaded.pins)}")

    result = run_workflow(
        loaded,
        env,
        args.out,
        load_engines(
            getattr(args, "engine", None) or DEFAULT_ENGINE,
            args.device,
            cuda_graphs=getattr(args, "cuda_graphs", False),
            batch_rows=getattr(args, "batch_rows", None),
            fit_rows=getattr(args, "fit_rows", None),
        ),
        resume=getattr(args, "resume", False),
        reuse_nondeterministic=getattr(args, "reuse_nondeterministic", False),
    )
    for name, entry in sorted(result.manifest["steps"].items()):
        files = ", ".join(entry.get("files", ()))
        print(f"{entry.get('status', 'completed')} {name}: {files}")
    print(f"manifest {result.run_root / 'workflow.json'}")
    return 0


def _inner_documents(
    raw_wf: Any,
    workflow_dir: Path,
    load_text: Any,
    *,
    outer_set: Any = None,
    seen: tuple[Path, ...] = (),
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Every intervention specification a workflow names, with the ``set``
    the compile will read — through nested ``workflow`` steps too (§2.10),
    whose ``set`` is the nested form laid over the inner step's own. Malformed
    shapes and loops are left to ``load_workflow`` to refuse properly."""
    steps_raw = raw_wf.get("steps", {}) if isinstance(raw_wf, dict) else {}
    if not isinstance(steps_raw, dict):
        return
    laid_over = outer_set if isinstance(outer_set, dict) else {}
    for name, step_raw in steps_raw.items():
        if not isinstance(step_raw, dict):
            continue
        document = step_raw.get("document")
        if not isinstance(document, str):
            continue
        doc_path = (workflow_dir / document).resolve()
        if not doc_path.is_file():
            continue
        authored = step_raw.get("set", {}) or {}
        authored = dict(authored) if isinstance(authored, dict) else {}
        kind = step_raw.get("type")
        if kind in ("intervention_protocol", "behavioral"):
            laid = laid_over.get(str(name), {})
            yield doc_path, {**authored, **(laid if isinstance(laid, dict) else {})}
        elif kind == "workflow" and doc_path not in seen:
            try:
                inner_raw = dict(load_text(doc_path))
            except ProtocolError:
                continue
            yield from _inner_documents(
                inner_raw,
                doc_path.parent,
                load_text,
                outer_set=authored,
                seen=(*seen, doc_path),
            )


def _explain_steps(loaded: Any, indent: str) -> None:
    """One line per step of the derived order (§9); a nested workflow's steps
    (§2.10) under their ``workflow`` step's own line, indented two more
    spaces and in their own order — so depth reads as indentation."""
    from causalab.workflow import fan_out, nested
    from causalab.workflow.document import (
        BehavioralStep,
        ConditionalStep,
        DecisionStep,
        ProtocolStep,
        WorkflowStep,
    )

    shown: set[str] = set()
    for name in loaded.order:
        head = name.split(nested.SEPARATOR, 1)[0]
        container = loaded.document.steps.get(head)
        if isinstance(container, WorkflowStep):
            if head not in shown:
                shown.add(head)
                print(
                    f"{indent}{nested.describe(head, container, loaded.nested[head])}"
                )
                _explain_steps(loaded.nested[head], indent + "  ")
            continue
        step = loaded.document.steps[name]
        # §2.9: a fanned-out step reports its width and join; each child
        # its shard — the schedule above already lists them as steps
        shard = getattr(step, "shard", None)
        fanned = (
            f" — {fan_out.describe(step, loaded.children[name])}"
            if isinstance(step, (ProtocolStep, BehavioralStep))
            and step.fan_out is not None
            else ""
        )
        sharded = (
            f" — shard {shard['index']} of {shard['of']}: "
            f"{len(shard['points'])} point(s)"
            if isinstance(shard, dict)
            else ""
        )
        if isinstance(step, ProtocolStep):
            inner = loaded.inner[name]
            kind = loaded.inner_digest_kind[name]
            print(
                f"{indent}{name}: intervention_protocol {step.document} — "
                f"{len(inner.expansion.points)} point(s), "
                f"{kind} digest {loaded.inner_digests[name][:16]}…"
                f"{fanned}{sharded}"
            )
        elif isinstance(step, BehavioralStep):
            inner = loaded.inner[name]
            print(
                f"{indent}{name}: behavioral {step.document} — "
                f"{len(inner.expansion.points)} point(s), "
                f"decoding {step.decoding['mode']}, split {step.split}, "
                f"step digest {loaded.step_digests[name][:16]}…"
                f"{fanned}{sharded}"
            )
        elif isinstance(step, DecisionStep):
            print(
                f"{indent}{name}: decision over {step.values.target} — rule on "
                f"{', '.join(sorted(step.rule))}, "
                f"step digest {loaded.step_digests[name][:16]}…"
            )
        elif isinstance(step, ConditionalStep):
            comparator = next(c for c in ("eq", "ne", "in") if c in step.predicate)
            print(
                f"{indent}{name}: conditional on "
                f"{step.predicate['decision']['step']}.{step.predicate['field']} "
                f"{comparator} {step.predicate[comparator]!r} -> "
                f"on_true [{', '.join(step.on_true)}] / "
                f"on_false [{', '.join(step.on_false)}] (scope {step.scope})"
            )
        else:
            marks = []
            if not step.is_deterministic:
                marks.append("non-deterministic")
            if step.runtime and step.runtime.get("isolate"):
                marks.append("isolated")
            suffix = f" [{', '.join(marks)}]" if marks else ""
            print(
                f"{indent}{name}: script {step.script} -> "
                f"{', '.join(sorted(d.file for d in step.outputs.values()))}"
                f"{suffix}"
            )
