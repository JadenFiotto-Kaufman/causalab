"""The ``causalab`` CLI: ``run · validate · explain · dry-run · digest · migrate``.

One entry point over **two document types**. Argument parsing and the
resolution environment are shared; the verbs themselves are not:

* an **intervention specification** → :mod:`causalab.protocol.cli`
* a **workflow** document → :mod:`causalab.workflow.cli`

Dispatch is on the document's ``steps`` section (workflow spec §1). Keeping it
here is what lets ``protocol/`` carry no workflow code and ``workflow/`` depend
on ``protocol/`` one way only — so the intervention protocol is usable on its
own, which is the point of having two packages.

``run`` needs an execution engine; the reference engine
(:mod:`causalab.neural.engines.pytorch_hooks`) is imported lazily by whichever half
needs it, so the pure verbs stay torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from causalab.protocol.engine import DEFAULT_ENGINE, ENGINE_CHOICES
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import PRECISION_DTYPES

from causalab.tasks import TASKS_ROOT

__all__ = ["ensure_model_registered", "load_engines", "main", "register_model_key"]


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set takes path=value, got {item!r}")
        dotted, _, raw_value = item.partition("=")
        try:
            overrides[dotted] = json.loads(raw_value)
        except json.JSONDecodeError:
            overrides[dotted] = raw_value  # a bare word is a string
    return overrides


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """``--set`` overrides plus the ``--dtype`` shorthand, which is one of
    them: dtype belongs to the document, so the only way to change it from
    the command line is the way every other field changes (§9)."""
    overrides = _parse_set(args.set)
    dtype = getattr(args, "dtype", None)
    if dtype is None:
        return overrides
    already = overrides.get("model.dtype")
    if already is not None and already != dtype:
        raise SystemExit(
            f"--dtype {dtype} contradicts --set model.dtype={already} — "
            "they set the same field"
        )
    overrides["model.dtype"] = dtype
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causalab",
        description="Intervention specifications and workflows: run, validate, "
        "explain, dry-run, digest, pin, migrate (docs/intervention_protocol.md, "
        "docs/workflow_protocol.md).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)
    migrate = sub.add_parser(
        "migrate",
        help="rewrite earlier-version intervention specifications (v1's flat "
        "sections, v2's scalar site 'layer'), a workflow's dotted "
        "sites.<name>.layer ids, and the fenced JSON examples in markdown "
        "files, as the current protocol_version, in place",
    )
    migrate.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="JSON documents or markdown files (a YAML document is refused: "
        "regroup it by hand); a document already at the current version is "
        "left as it is",
    )
    migrate.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any file would change",
    )
    for verb, help_text in (
        ("run", "validate, expand, plan, execute, stamp"),
        ("validate", "the §5 load-error checklist"),
        (
            "explain",
            "models, forward plan, point count, requires, digest, save products",
        ),
        (
            "dry-run",
            "resolve everything a run decides before weights load, and report it",
        ),
        ("digest", "the campaign digest"),
        (
            "pin",
            "stamp a workflow's `pins` section — the digests of every document, "
            "script, table, code module and file it touches — into the "
            "workflow file (workflow documents only; `run` stamps an unpinned "
            "workflow on its first run)",
        ),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("document", type=Path, help="a protocol JSON (or YAML) file")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
        p.add_argument(
            "--data-root",
            type=Path,
            default=TASKS_ROOT,
            help="where dataset refs resolve (`<root>/<ref>.json`). Defaults to "
            "the task packages themselves, so a shipped document resolves "
            "with no flag: `<task>/data/<variant>#<split>` names the table "
            "the task ships under causalab/tasks/<task>/data/",
        )
        p.add_argument("--artifacts-root", type=Path, default=Path("."))
        p.add_argument(
            "--max-points",
            type=int,
            default=None,
            help="override the sweep point cap (§5.14)",
        )
        p.add_argument(
            "--register-from-hf",
            action="store_true",
            help="resolve an unregistered model key from its HF config before "
            "loading, instead of refusing [V4]. Opt-in: without it the pure "
            "verbs stay registry-only, so a digest never depends on the "
            "network. 'run' always does this — the flag is for the verbs that "
            "otherwise never touch a model",
        )
        if verb == "validate":
            p.add_argument(
                "--data", action="store_true", help="also check column references"
            )
        if verb == "explain":
            p.add_argument(
                "--engine",
                choices=ENGINE_CHOICES,
                default=None,
                help="also route the document and print which engine would "
                "serve it (or the §8 refusal). Loads engines, so `explain` "
                "without it stays torch-free",
            )
        if verb == "dry-run":
            p.add_argument(
                "--engine",
                choices=ENGINE_CHOICES,
                default=None,
                help="also ask check_engine, per candidate engine, what it would "
                "refuse — reported as a capability_shortfall, not raised; a "
                "pinned engine's shortfall exits 1, under 'auto' only no "
                "candidate serving does. Builds engines (no weights), so "
                "`dry-run` without it stays torch-free",
            )
            p.add_argument(
                "--data",
                action="store_true",
                help="also check column and prompt-variable references and the "
                "declared row roles at every point (the `validate --data` pass); "
                "a refusal is reported and exits 1",
            )
            p.add_argument(
                "--shard-size",
                type=_positive_int,
                default=None,
                metavar="N",
                help="plan `--points` shards of at most N points and report how "
                "many the campaign needs (ceil(points / N))",
            )
        if verb == "run":
            p.add_argument(
                "--cuda-graphs",
                action="store_true",
                help="use CUDA replay for supported pytorch_hooks workloads",
            )
            p.add_argument(
                "--out",
                type=Path,
                required=True,
                help="run output directory; for a workflow, the ROOT under "
                "which the document's own output_dir is created (§1.1)",
            )
            p.add_argument(
                "--resume",
                action="store_true",
                help="skip a step whose outputs exist with a matching stamped "
                "digest (workflow documents only)",
            )
            p.add_argument(
                "--reuse-nondeterministic",
                action="store_true",
                help="with --resume, also reuse steps declaring "
                "is_deterministic: false",
            )
            p.add_argument(
                "--device",
                default="cpu",
                help="torch device string for the reference engine "
                "(cpu, cuda, cuda:1, mps)",
            )
            p.add_argument(
                "--engine",
                choices=ENGINE_CHOICES,
                default=DEFAULT_ENGINE,
                help="execution engine: 'auto' (default) is every installed "
                "engine with the reference FIRST, routed by choose_engine "
                "(§8); name one to pin it. Routing is §8's own answer, and "
                "list order is preference, so anything the reference serves "
                "behaves exactly as a pinned 'pytorch_hooks' would — while a "
                "document only the nnsight engine can serve now runs instead "
                "of refusing by name",
            )
            p.add_argument(
                "--dtype",
                choices=PRECISION_DTYPES,
                default=None,
                help="shorthand for --set model.dtype=… — precision is a "
                "document fact (§2.1), so an override enters the digest and "
                "the record never lies about what produced the numbers",
            )
            p.add_argument(
                "--points",
                default=None,
                metavar="START:STOP",
                help="execute only this half-open point-index range of the "
                "expanded campaign — the seam external schedulers shard on "
                "(document runs only; digests and stamps are unaffected)",
            )
            p.add_argument(
                "--batch-rows",
                type=_positive_int,
                default=None,
                metavar="N",
                help="reference engine: run a forward group over more than N "
                "rows as several forwards of at most N rows each, captures "
                "concatenated in row order (§8, execution scale). Execution "
                "only — the numbers equal the single-forward run up to dtype "
                "rounding, and digests and stamps are unaffected. Bounds "
                "no-grad forwards, including train.eval passes; a training "
                "minibatch keeps its train.batch.pairs rows",
            )
            p.add_argument(
                "--fit-rows",
                type=_positive_int,
                default=None,
                metavar="N",
                help="reference engine: bound how many rows one grad forward "
                "of a fit covers — the members of a fit cohort are packed into "
                "forwards of at most N rows each, a member's own minibatch "
                "(train.batch.pairs rows) is never split, and without the flag "
                "the bound is measured on the cohort's first step from the "
                "device's free memory and recorded as "
                "execution.fit_rows_resolved (§8, execution scale). Execution "
                "only — digests and stamps are unaffected, and the receipt "
                "records the bound as execution.fit_rows",
            )
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive row count, got {text}")
    return value


def _env(args: argparse.Namespace) -> ResolutionEnv:
    return ResolutionEnv(
        # the shipped task tables stay reachable behind any --data-root
        datasets=FileDatasets(root=args.data_root, fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=args.artifacts_root),
    )


def load_engines(
    choice: str,
    device: str,
    *,
    cuda_graphs: bool = False,
    batch_rows: int | None = None,
    fit_rows: int | None = None,
) -> list[Any]:
    """Build the ``run`` verb's engine list — lazily, so the pure verbs stay
    torch-free (importlib keeps the layering honest: ``protocol/`` never
    links against an execution engine).

    ``auto`` is every installed engine with the reference first — list order
    is routing preference (§8), so pytorch_hooks serves what it can and the
    nnsight engine picks up what it refuses. A missing optional engine is
    only an error when named explicitly.

    ``batch_rows`` is the reference engine's microbatch bound (``--batch-rows``)
    and ``fit_rows`` its rows-per-grad-forward bound for a fit (``--fit-rows``);
    the nnsight engine runs each group as one batch, has no grad path, and is
    built without either."""
    import importlib

    engines: list[Any] = []
    if choice in ("pytorch_hooks", "auto"):
        try:
            hooks = importlib.import_module("causalab.neural.engines.pytorch_hooks")
        except ModuleNotFoundError as err:
            raise ProtocolError(
                "P2",
                f"no execution engine available ({err}) — 'run' needs the "
                "reference engine causalab.neural.engines.pytorch_hooks",
            ) from err
        # `fit_rows` only when set: the engine's default is None already, and
        # passing it explicitly would make the kwarg part of the constructor
        # contract for every stand-in
        extra = {"fit_rows": fit_rows} if fit_rows is not None else {}
        if cuda_graphs:
            extra["cuda_graphs"] = True
        engines.append(
            hooks.PytorchHooksEngine(device=device, batch_rows=batch_rows, **extra)
        )
    if choice in ("nnsight", "auto"):
        try:
            tracing = importlib.import_module("causalab.neural.engines.nnsight_tracing")
        except ModuleNotFoundError as err:
            if choice == "nnsight":
                raise ProtocolError(
                    "P2",
                    f"the nnsight engine is not installed ({err}) — install "
                    "the 'nnsight' extra (pip install 'causalab[nnsight]')",
                ) from err
        else:
            engines.append(tracing.NnsightEngine(device=device))
    return engines


def wants_hf_registration(args: argparse.Namespace) -> bool:
    """Whether this invocation may resolve an unregistered key over the network.

    ``run`` touches the model anyway. For the pure verbs it is the author
    saying so with ``--register-from-hf``: pre-flighting a document on an
    unregistered model was impossible without it, and the documented
    workaround — validate against a *similar* registered model — produces a
    **false** refusal (`[V4] layer 36 out of range for the 36-layer model
    'Qwen/Qwen3-4B-Instruct-2507'` on a valid 40-layer document). All three A3B
    protocol runs wrote the same nine-line wrapper instead.
    """
    return args.verb == "run" or bool(getattr(args, "register_from_hf", False))


def ensure_model_registered(args: argparse.Namespace) -> None:
    """Resolve an unregistered model key from its HF config and register it
    before canonicalization.

    Called for ``run`` unconditionally and for the pure verbs only under
    ``--register-from-hf``, so the invariant survives: without the flag a
    digest never depends on the network."""
    from causalab.protocol.compile import read_document

    # read through the compiler's own prefix, so `--set model.key=…` is applied
    # and the key read here is the one the compile will read
    try:
        authored = read_document(
            args.document, args.document.resolve().parent, dict(args.parsed_set)
        )
    except ProtocolError:
        return  # a malformed document refuses properly in the real compile
    register_model_key(dict(authored.raw))


def register_model_key(raw: dict[str, Any]) -> None:
    from causalab.protocol.registry import (
        get_model_info,
        model_info_from_hf_config,
        register_model,
    )

    model = raw.get("model", {})
    key = model.get("key") if isinstance(model, dict) else None
    if not isinstance(key, str):
        return
    try:
        get_model_info(key)
    except ProtocolError:
        from transformers import AutoConfig

        revision = model.get("revision", "main") if isinstance(model, dict) else "main"
        config = AutoConfig.from_pretrained(key, revision=revision)
        register_model(model_info_from_hf_config(key, config))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, build the environment, and dispatch on document type."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.verb == "migrate":
        # a rewrite of files, not a compile: no environment, no dispatch
        from causalab.protocol.migrate import main as migrate_main

        return migrate_main(args)
    if getattr(args, "engine", None) == "nnsight":
        # fail closed: both bounds are the reference engine's, and a pinned
        # nnsight run would drop them silently while the receipt said null
        if getattr(args, "batch_rows", None) is not None:
            parser.error(
                "--batch-rows bounds only the reference engine, and --engine "
                "nnsight pins an engine that runs every group as one batch, so "
                "the two flags cannot be combined"
            )
        if getattr(args, "fit_rows", None) is not None:
            parser.error(
                "--fit-rows bounds only the reference engine's grad forwards, "
                "and --engine nnsight pins an engine with no grad path, so the "
                "two flags cannot be combined"
            )
    args.parsed_set = _overrides(args)
    env = _env(args)
    try:
        from causalab.protocol.loader import load_text
        from causalab.workflow.document import is_workflow

        if is_workflow(load_text(args.document)):
            if args.verb == "dry-run":
                print(
                    "refused: dry-run is per intervention specification; the "
                    "workflow has no dry run of its own — validate or explain "
                    "the workflow, and dry-run its documents one by one",
                    file=sys.stderr,
                )
                return 1
            from causalab.workflow import cli as workflow_cli

            return workflow_cli.main(args, env)
        if args.verb == "pin":
            print(
                "refused: pins are a workflow's — an intervention specification "
                "carries no pins section and is pinned by the workflow that "
                "runs it (workflow spec §7). Wrap the document in a workflow "
                "step and pin that",
                file=sys.stderr,
            )
            return 1
        if getattr(args, "resume", False):
            print(
                "refused: --resume is a workflow flag — it reuses a published "
                "step whose identity and files still match, and an intervention "
                "specification run has no step boundaries to resume at (IM spec "
                "§9). Wrap the document in a workflow step, or shard it with "
                "--points",
                file=sys.stderr,
            )
            return 1
        from causalab.protocol import cli as protocol_cli

        return protocol_cli.main(args, env)
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
