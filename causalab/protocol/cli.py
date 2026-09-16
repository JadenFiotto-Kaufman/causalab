"""The CLI verbs for an **intervention specification** (spec §9), ``dry-run``
included.

Every verb compiles through :func:`causalab.protocol.compile.compile_protocol`
against a resolution environment built by :mod:`causalab.cli`, which also owns
argument parsing and the dispatch between document types. This module
therefore links against nothing in the workflow layer.

``run`` needs an execution engine; the reference engine
(:mod:`causalab.neural.engines.pytorch_hooks`) is imported lazily so the pure verbs stay
torch-free.

The run itself is :func:`causalab.protocol.run.run_protocol` — a public Python
function, and the primitive. What stays here is argument parsing, what
gets printed, and the exit code.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from causalab.protocol.compile import CompiledProtocol, compile_protocol
from causalab.protocol.dry_run import DryRunReport, Refusal, dry_run
from causalab.protocol.engine import DEFAULT_ENGINE
from causalab.protocol.errors import ProtocolError, ValidationError, ValidationErrors
from causalab.protocol.loader import check_data_columns
from causalab.protocol.plan import plan_point
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.run import run_protocol
from causalab.protocol.schema import MODEL_DTYPE_DEFAULT
from causalab.protocol.sweep import coordinate_label

__all__ = ["main"]


def _compile(args: argparse.Namespace, env: ResolutionEnv) -> CompiledProtocol:
    """The one compiler, with the CLI's inputs: the file, its own directory
    for relative references, ``--set``, the environment's two resolvers, and
    no engine (the pure verbs never load one; ``run`` routes afterwards)."""
    from causalab.protocol.sweep import DEFAULT_POINT_CAP

    return compile_protocol(
        args.document,
        args.document.parent,
        dict(args.parsed_set),
        env.datasets,
        env.artifacts,
        None,
        point_cap=args.max_points if args.max_points is not None else DEFAULT_POINT_CAP,
        model_info=env.model_info,
    )


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    """Run one verb against an **intervention specification**.

    Dispatch between document types lives in :mod:`causalab.cli`, so this module
    — and the whole ``protocol/`` package — links against nothing in the
    workflow layer. That is what lets someone use the intervention protocol on
    its own."""
    try:
        from causalab.cli import ensure_model_registered, wants_hf_registration

        if args.verb == "dry-run":
            # before the registration hook: a dry run never fetches a config
            return _dry_run(args, env)
        if wants_hf_registration(args):
            ensure_model_registered(args)
        compiled = _compile(args, env)
        if args.verb == "validate":
            if args.data:
                check_data_columns(compiled, env)
            n = len(compiled.points.points)
            print(
                f"OK: {args.document} — {n} point{'s' if n != 1 else ''}, "
                f"digest {compiled.digests.document[:16]}…"
            )
            return 0
        if args.verb == "digest":
            print(compiled.digests.document)
            return 0
        if args.verb == "explain":
            _explain(compiled)
            _explain_engine(compiled, getattr(args, "engine", None))
            return 0
        # run — engines are optional, lazily-imported extras so the pure
        # verbs stay torch-free; --engine picks the list, choose_engine routes
        from causalab.cli import load_engines

        result = run_protocol(
            compiled,
            env,
            load_engines(
                getattr(args, "engine", None) or DEFAULT_ENGINE,
                args.device,
                cuda_graphs=getattr(args, "cuda_graphs", False),
                batch_rows=getattr(args, "batch_rows", None),
                fit_rows=getattr(args, "fit_rows", None),
            ),
            args.out,
            points=args.points,
        )
        for manifest_path, disk_path in sorted(result.files.items()):
            print(f"saved {manifest_path} -> {disk_path}")
        if result.cells:
            # the denominator is data (§4.1): how many cells measured, and
            # which were excluded and why — read from the result, not kept
            # by the campaign
            print(f"cells {result.denominator.render()}")
        return 0
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


def _dry_run(args: argparse.Namespace, env: ResolutionEnv) -> int:
    """``dry-run``: everything a run decides before weights load, resolved and
    reported (:mod:`causalab.protocol.dry_run`).

    Exit ``0`` when the document compiles and every fact is resolved or
    explicitly undecided, with no shortfall for the requested engine(s); ``1``
    on any refusal — a compile refusal (printed as ``refused: …`` with its
    reason-coded record, exactly as ``validate`` refuses), a shortfall for a
    pinned ``--engine`` (under ``auto``, only when no candidate serves), or a
    ``--data`` refusal. Engines are built only when ``--engine`` is given, on
    the CPU, and constructing one loads no weights.

    ``--register-from-hf`` is refused rather than inherited: the flag's one
    effect is a config fetch, and a dry run's contract is that it never
    touches the network — an unregistered ``model.key`` is the registry's
    ``V4`` refusal.
    """
    if getattr(args, "register_from_hf", False):
        raise ProtocolError(
            "P4",
            "--register-from-hf does not apply to dry-run: a dry run resolves "
            "the model from the registry alone and never fetches a config. An "
            "unregistered model.key is refused [V4]; register its static entry "
            "(causalab.protocol.registry.register_model), or pre-flight with "
            "'validate --register-from-hf'",
        )
    try:
        compiled = _compile(args, env)
    except ProtocolError as err:
        # the compile's refusal, plus the record: the rule's slug, the field
        # and the reason code — the reason is what the rendered text lacks
        print(f"refused: {err}", file=sys.stderr)
        each = err.errors if isinstance(err, ValidationErrors) else (err,)
        for violation in each:
            print(f"  {Refusal.from_error(violation).render()}", file=sys.stderr)
        return 1
    choice = getattr(args, "engine", None)
    engines: list[Any] = []
    if choice is not None:
        from causalab.cli import load_engines

        engines = load_engines(choice, "cpu")
    report = dry_run(
        compiled,
        env,
        engines=engines,
        shard_size=getattr(args, "shard_size", None),
        overrides=dict(args.parsed_set),
        check_data=bool(getattr(args, "data", False)),
    )
    _print_dry_run(report, args.document)
    for refusal in report.refusals:
        print(f"refused: {refusal.message}", file=sys.stderr)
        print(f"  {refusal.render()}", file=sys.stderr)
    return 0 if report.ok else 1


def _print_dry_run(report: DryRunReport, document: Any) -> None:
    """The report, one block per fact, ending with the ``undecided`` line —
    so a run's tokenizer-time refusal is never mistaken for a green."""
    print(f"dry-run   {document}")
    print(f"digest    {report.composition.digest}")
    if report.composition.title:
        print(f"title     {report.composition.title}")
    if report.composition.overrides:
        applied = ", ".join(f"{k}={v}" for k, v in report.composition.overrides.items())
        print(f"overrides {applied}")
    model = report.model
    realization = f"{model.key}@{model.revision} {model.dtype}"
    if model.quantization is not None:
        realization += f" + {model.quantization}"
    print(f"model     {realization}")
    pattern = (
        "declares no layer pattern"
        if model.layer_pattern is None
        else "layer pattern "
        + ", ".join(
            f"{model.layer_pattern.count(s)} {s}"
            for s in sorted(set(model.layer_pattern))
        )
    )
    print(
        f"  {model.num_layers} layers, hidden {model.hidden_size}, "
        f"{model.num_heads} heads ({model.num_kv_heads} kv) x {model.head_dim}, "
        f"vocab {model.vocab_size}, family {model.family or 'unknown'}; {pattern}"
    )
    print("data")
    for entry in report.data:
        roles = ", ".join(entry.roles) or "(no role)"
        print(
            f"  {entry.ref} ({roles}): digest {entry.digest[:16]}… "
            f"{len(entry.columns)} columns"
        )
    if report.points.axes:
        axes = ", ".join(f"{axis} ({n} values)" for axis, n in report.points.axes)
        print(f"axes      {axes}")
    print(f"points    {report.points.n}")
    print(
        f"forwards  {report.forwards.per_point} per point, "
        f"{report.forwards.campaign} interned"
    )
    if report.shards.count is None:
        print(f"shards    {report.shards.n_points} points; pass --shard-size N to plan")
    else:
        n = report.shards.n_points
        print(
            f"shards    {report.shards.count} of at most {report.shards.shard_size} "
            f"points ({n} point{'s' if n != 1 else ''})"
        )
    print(f"requires  {list(report.capabilities) or 'nothing beyond a forward pass'}")
    for engine in report.engines:
        if engine.serves:
            print(f"engine    {engine.name}: serves")
        else:
            assert engine.shortfall is not None
            print(
                f"engine    {engine.name}: {engine.shortfall.kind} {engine.shortfall.message}"
            )
    print("sites")
    for site in report.sites:
        where = site.component
        if site.layers:
            layers = (
                f"layer {site.layers[0]}"
                if len(site.layers) == 1
                else f"layers {site.layers[0]}..{site.layers[-1]} ({len(site.layers)})"
            )
            where += f" {layers}"
        if site.head is not None:
            where += f" head {site.head}"
        if site.expert is not None:
            where += f" expert {site.expert}"
        if site.stream is not None:
            where += f" [{site.stream}]"
        print(f"  {site.name}: {where}: {site.status}")
        if site.refusal is not None:
            print(f"    {site.refusal.message}")
            print(f"    {site.refusal.render()}")
            continue
        heads = (
            "no head axis"
            if site.head_space is None
            else f"head space {site.head_space}"
        )
        print(f"    shape {site.shape}, width {site.width}, {heads}")
        writes = (
            f"read-only ({site.why})"
            if site.writes is None
            else "writes " + ", ".join(site.writes)
        )
        print(f"    reads {', '.join(site.reads)}; {writes}")
        for why in site.undecided:
            print(f"    undecided: {why}")
    if report.inventory is not None:
        streams = ", ".join(
            f"{report.inventory.count(s)} {s}"
            for s in sorted({layer.stream for layer in report.inventory.layers})
        )
        print(
            f"inventory {len(report.inventory.layers)} layers ({streams}); "
            f"layerless {', '.join(report.inventory.layerless)}"
        )
    else:
        print("inventory undecided (see below)")
    print("readouts")
    for read in report.readouts:
        metrics = ", ".join(read.metrics) or "(saved or operand only)"
        print(
            f"  {read.name}: {read.model} on {read.input} at {read.site} -> {metrics}"
        )
    print("save")
    for out in report.outputs:
        kind = f" [{out.kind}]" if out.kind else ""
        print(f"  {out.value} ({out.binding}) -> {out.file_path}{kind}")
    for diagnostic in report.diagnostics:
        print(f"diagnostic {diagnostic.kind}: {diagnostic.message}")
    for refusal in report.refusals:
        print(f"refusal   {refusal.message}")
        print(f"  {refusal.render()}")
    for item in report.undecided:
        print(f"  {item.topic}: {item.detail}")
    print(
        "undecided (decided when the run encodes its inputs): "
        + ", ".join(report.undecided_topics)
    )


def _explain_engine(compiled: CompiledProtocol, choice: str | None) -> None:
    """Print which engine ``choose_engine`` would pick, or the §8 refusal.

    ``explain`` printed ``requires`` and stopped there, so routing could not be
    pre-flighted at all — and routing is exactly what is not obvious on a model
    where one family of components is hooks-only and another is nnsight-only.
    The refusal is the *more* useful answer of the two, so it is printed rather
    than raised.

    Opt-in because engines are heavy: without ``--engine`` nothing here loads,
    and the pure verbs stay torch-free (``test_load_is_torch_free``). The
    import is inside ``main`` for the same reason ``run``'s is — ``protocol/``
    never links against an engine at module scope.
    """
    if choice is None:
        return
    from causalab.cli import load_engines
    from causalab.protocol.engine import choose_engine

    engines = load_engines(choice, "cpu")
    try:
        print(
            f"engine    {choose_engine(list(compiled.point_documents), engines).name}"
        )
    except ValidationError as err:
        print(f"engine    refused: {err}")


def _explain(compiled: CompiledProtocol) -> None:
    doc = compiled.point_documents[0]
    axes = compiled.points.axes
    print(f"digest    {compiled.digests.document}")
    if doc.title:
        print(f"title     {doc.title}")
    model = doc.model
    realization = f"{model.key}@{model.revision} {model.dtype or MODEL_DTYPE_DEFAULT}"
    if model.quantization is not None:
        realization += f" + {model.quantization.scheme} ({model.quantization.method})"
    print(f"model     {realization}")
    if axes:
        print(
            f"axes      {', '.join(f'{a.id} ({len(a.values)} values)' for a in axes)}"
        )
    print(f"points    {len(compiled.points.points)}")
    # the compiler's required-capability set (§8), read from the registry rows
    print(
        f"requires  {sorted(compiled.capabilities) or 'nothing beyond a forward pass'}"
    )
    # the derived record of a path block (§3.2): the policy, the receivers in
    # the order they are injected (one joint pass), and the restorer boundary
    # in forward order — none of which the canonical form spells out
    path = compiled.lowered.get("path_patching")
    if path:
        receivers = ", ".join(path["receivers"])
        print(
            f"path      {path['restoration']}: {path['sender']} -> {receivers} "
            f"(harvest {path['harvest']!r}, inject {path['inject']!r})"
        )
        boundary = " < ".join(
            f"{name}={component}@{layer}"
            for layer, component, name in path["restorers"]
        )
        print(f"  restorers {boundary or '(none: adjacent layers)'}")
    plan = plan_point(doc)
    print(f"forwards  {plan.num_forwards} per point")
    for group in plan.groups:
        taps = ", ".join(t.read for t in group.taps) or "(no reads — operands only)"
        print(f"  {group.model} on {group.input}: {taps}")
        if group.decode_depth:
            # print what the decode obliges, so the bill of a document is
            # readable before it runs — the mechanism stays the engine's
            print(f"    decode {group.decode_depth} tokens (greedy)")
            for item in group.materialize:
                needs = (
                    "distribution per addressed position"
                    if item.needs_distribution
                    else "no distribution — ids and activations only"
                )
                print(f"    {item.read} at {item.site}: {needs}")
    print("save")
    for entry in doc.save:
        binding = (
            f"model={entry.model}, input={entry.input}"
            if entry.site is None
            else f"site={entry.site}"
        )
        print(f"  {entry.value} ({binding}) -> {entry.file_path}")
    if axes:
        first = compiled.points.points[0]
        print(
            f"first point {coordinate_label(first.coords)} digest "
            f"{compiled.digests.points[0][:16]}…"
        )


if __name__ == "__main__":
    raise SystemExit(main())
