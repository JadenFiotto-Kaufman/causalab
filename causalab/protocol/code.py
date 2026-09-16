"""In-document code references, identified by content (spec §2.8.1).

A ``pytorch_fn`` write runs *user code* inside the intervention. Until this
module existed the document said only ``"torch.relu"`` — a name — so the
function's body, its arguments, the files it opened, the environment it read
and the row convention it assumed were all outside the digest. Two runs of
demonstrably different interventions could therefore carry one protocol
identity. ``ROME``'s corruption function is the case: it took its noise scale
from an externally selected file and assumed an 11-row batch (one clean row
followed by ten corrupted), and neither fact reached the digest.

A ``code`` entry closes that by declaring, in the document:

* ``locator`` — the importable dotted path to the function;
* ``args`` — its typed JSON arguments, passed as keywords;
* ``data_inputs`` — the files it may read, by name, content-digested at load;
* ``env_inputs`` — the environment variables it is allowed to read;
* ``row_roles`` — what the rows of the batch it receives *are*, in order.

and by stamping ``source_sha256`` into the canonical form (§6), so editing the
code moves the document digest.

**What is hashed, and why it is the module.** ``source_sha256`` is the sha256
of the defining module's bytes — the same quantity a workflow script step
hashes (:func:`source_sha256` is what ``workflow.document.check_script``
calls). Not the function's own source segment, for two reasons: a function's
behaviour depends on its module's helpers, constants and imports, so a
segment hash reports "unchanged" for an edit that changes every output; and a
function need not *have* a source segment — the arithmetic golden's
``apply_target_<n>`` are built by a closure factory at import time and appear
in no ``ast.FunctionDef``. The module always exists and always has bytes.

**And where the module boundary is the end of it.** Behaviour depends on
imports too, but what a hashed module *imports* divides in two. The
``causalab`` package's own modules are **runtime identity**: every step record
carries the package's ``tree_digest`` (:func:`causalab.provenance.runtime_identity`)
and ``--resume`` compares it before reusing anything, so an edit to
``causalab.io.step_record.aggregate`` already re-runs every step, and putting
those bytes into the *document* identity as well only made every edit to the
protocol core move every workflow digest in the repository (the re-pin churn
of 2026-09). Code **beside** the hashed module — a ``{"path": …}`` script's
sibling helper, a user package's sibling on ``sys.path`` — is covered by
nothing else, so *that* is what the declared import closure names: a manifest
``closure: {relative_path: sha256}`` and one hash over it, ``closure_sha256``,
beside ``source_sha256`` (the defining module alone, a hash a human can
``sha256sum``). Both enter the identity only when the manifest is non-empty;
a module inside the package, or one installed under site-packages or the
stdlib, declares no closure at all (:func:`import_closure` with
``repository=False``, workflow spec §4.2).

What the closure is, exactly: the modules the defining module's source
*declares* — top-level and function-local imports alike (a lazy import is
static text and a real run-time dependence) — resolved by filesystem probe
against the module's own root, never against ``sys.path`` at large, and never
inside the ``causalab`` package (:func:`package_root`). Anything that resolves
elsewhere (site-packages, the stdlib, ``torch``, ``numpy``, ``pandas``) is
third-party and **excluded**: a dependency's version is runtime identity,
recorded by :func:`causalab.provenance.runtime_identity` and the lockfile,
never by the document. An ``if TYPE_CHECKING:`` block never executes and is
excluded; a parent package's ``__init__`` is executed by Python but not
declared, and is excluded unless ``include_parents`` is set. Dynamic imports —
``importlib.import_module``, a module ``__getattr__`` — are invisible to a
static read, the same boundary the declaration checker below states. The
repository-wide walk (``repository=True``, the default) is the layering tool
the test suite uses to prove a module reaches no engine and no numerics.

**Nothing here imports the referenced module.** Resolution walks the package
tree on disk: :func:`importlib.util.find_spec` for the top-level name (which
imports nothing) and then a filesystem probe per dotted part. ``validate``,
``explain`` and ``digest`` stay torch-free, which is the property that lets
the hash reach the digest at all — the same argument
``workflow.document.check_script`` makes. The closure walk keeps it: every
member is parsed with ``ast.parse`` from its bytes, never imported.

**Static checks only.** The undeclared-read refusals below read the function's
AST. They catch the reads a document can be *written* to declare — a literal
environment name, a literal path — and they say nothing about a read routed
through a variable. That is deliberate: this is a declaration checker, not a
sandbox. A function whose reads are all undetectable is not refused; it is
simply undeclared, which is what the digest and the review are for.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib.util
import site
import sysconfig
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causalab.protocol.errors import ValidationError

__all__ = [
    "CODE_RULE",
    "READER_CALLS",
    "ROW_ROLE_RULE",
    "ResolvedCode",
    "closure_sha256",
    "function_def",
    "import_closure",
    "is_installed_module",
    "repo_root",
    "resolve_locator",
    "resolve_locator_or_refuse",
    "signature_problems",
    "source_root",
    "source_sha256",
    "undeclared_reads",
]

#: §5 rule 24 — a code reference agrees with the source it names. Every
#: refusal raised out of this module carries it, so a test pins the rule and
#: not the sentence.
CODE_RULE: int = 24

#: §5 rule 25 — declared row roles match the resolved data. Needs the tables,
#: so it lives in the ``validate --data`` pass (like rules 4's column half and
#: 20), and is raised from :mod:`causalab.protocol.loader`.
ROW_ROLE_RULE: int = 25

#: Attribute calls whose first positional argument, when it is a string
#: literal, is a path being read. Deliberately short: every entry is a call
#: whose literal-string first argument is a filename in every library that
#: spells it this way, so the check cannot fire on valid work that reads
#: nothing. A read routed through a variable is not in this set and is not
#: refused — see the module docstring.
READER_CALLS: frozenset[str] = frozenset(
    {
        "load",
        "loadtxt",
        "read_bytes",
        "read_csv",
        "read_json",
        "read_parquet",
        "read_text",
        "open",
    }
)

#: Keyword arguments the runtime supplies to a referenced function when the
#: declaration asks for them, so the signature check does not demand that the
#: author also list them under ``args``.
#:
#: Only ``row_roles``. It is the one declaration a function cannot act on
#: without being *told*: the alternative is inferring roles from physical
#: batch positions, which is the half of the ROME case above this exists to
#: remove.
#: ``data_inputs`` and ``env_inputs`` are the other kind of declaration —
#: identity plus an allowlist. The function opens its own file and reads its
#: own variable, and what the declaration adds is that the file is
#: content-digested into the protocol and that an undeclared one is refused.
RUNTIME_KEYWORDS: tuple[str, ...] = ("row_roles",)


class CodeResolutionError(Exception):
    """A locator does not name Python source. Translated to the caller's
    rule number (a :class:`~causalab.protocol.errors.ValidationError`) at the
    boundary — this module knows nothing about the checklist."""


@dataclasses.dataclass(frozen=True)
class ResolvedCode:
    """Where a locator landed: the defining module, its file, and the dotted
    attribute path inside it (empty when the locator *is* a module)."""

    module: str
    path: Path
    attr: tuple[str, ...]


def source_sha256(path: Path) -> str:
    """The sha256 of a Python source file's bytes.

    One function, two callers: a workflow script step (§4.2) and a protocol
    code reference (§2.8.1) hash the same quantity the same way, so a module
    that is both cannot get two identities.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_root() -> Path:
    """The directory the ``causalab`` package sits in.

    One thing resolves against it: the repository-wide import-closure walk
    (:func:`import_closure` with ``repository=True``), the layering tool, which
    only a checkout runs. The identity walk (``repository=False``) never
    resolves here, and neither does a workflow document's ``path`` reference —
    that is relative to the document (workflow spec §3), because an installed
    wheel has no repository root to offer.
    """
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """The ``causalab`` package directory — the tree whose bytes are the
    ``tree_digest`` every step record carries (:func:`causalab.provenance.runtime_identity`)
    and ``--resume`` compares. A module under it is runtime identity and is
    excluded from every declared import closure that enters a digest."""
    return Path(__file__).resolve().parents[1]


@functools.cache
def _installation_roots() -> tuple[Path, ...]:
    """Where installed Python lives on this interpreter: the stdlib and every
    site-packages directory (``sysconfig`` and ``site`` agree on the venv's,
    the base interpreter's and the user's)."""
    paths = sysconfig.get_paths()
    roots = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    roots.update(site.getsitepackages())
    roots.add(site.getusersitepackages())
    return tuple(sorted(Path(root).resolve() for root in roots))


def is_installed_module(path: Path) -> bool:
    """Whether the module at ``path`` is *installed* — lives under the stdlib
    or a site-packages directory — as opposed to being repository code or a
    user's own code on ``sys.path``.

    The one predicate behind "a hashed module that itself lives outside the
    repository declares no closure; its imports are runtime identity". A
    ``code`` locator (§2.8.1) or a ``{"module": …}`` script step (workflow
    spec §4.2) may legitimately name a stdlib or third-party file — that file
    is still ``source_sha256``, because the document named it — but walking
    *its* imports with the stdlib or site-packages as the own root would admit
    everything reachable there, which is exactly the third-party code the spec
    excludes. The test is by installation directory rather than by
    ``repo_root``, for two reasons: a project venv conventionally sits *inside*
    the repository (``<repo>/.venv``), so "under the repo root" would still walk
    ``numpy``; and a user's own module outside the repository — a session-local
    package, a test's ``tmp_path`` — is the author's code, whose imported
    siblings :func:`import_closure` admits through the module's own root. Both
    callers ask here rather than each spelling the test.
    """
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in _installation_roots())


def source_root(resolved: ResolvedCode) -> Path:
    """The directory ``resolved.module``'s dotted name was resolved from — the
    root a sibling ``import`` inside that module resolves against."""
    depth = len(resolved.module.split("."))
    if resolved.path.name != "__init__.py":
        depth -= 1
    return resolved.path.resolve().parents[depth]


def closure_sha256(closure: Mapping[str, str]) -> str:
    """One hash over a closure manifest: ``sha256`` of the newline-joined
    ``"<path> <sha256>"`` lines, sorted by path. Empty manifest, one fixed
    value; the loaders write neither key for an empty manifest, so a module
    with no sibling imports carries no closure fields at all."""
    lines = "\n".join(f"{path} {sha}" for path, sha in sorted(closure.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class _ImportRef:
    """One ``import`` statement, as the walk needs it: the dotted module (``""``
    for ``from . import x``), the relative level, and the names a ``from``
    form binds (empty for a plain ``import``)."""

    module: str
    level: int
    names: tuple[str, ...]


#: Parsed import statements per source sha256. Content-addressed, so a file
#: rewritten in place — which the digest-moving tests do many times a second —
#: can never be served a stale parse; and a load that hashes the same seven
#: thousand lines of ``protocol/`` for its second script step parses nothing.
_IMPORTS_BY_SHA: dict[str, tuple[_ImportRef, ...]] = {}


def import_closure(
    path: Path,
    *,
    root: Path | None = None,
    include_parents: bool = False,
    repository: bool = True,
) -> dict[str, str]:
    """The declared import closure of the module at ``path``, as a manifest
    ``{relative_path: sha256}`` sorted by path — every module the module
    reaches transitively through its ``import`` statements, **not** counting
    the module itself (its own hash is :func:`source_sha256`, right beside
    this in every record).

    ``root`` is the directory the module's own dotted name resolves from
    (:func:`source_root`; the file's directory for a ``{"path": …}`` script).
    A name is probed there by filesystem shape alone — ``x/y.py`` or
    ``x/y/__init__.py`` — exactly as :func:`resolve_locator` probes, and never
    through ``sys.path``: a module that lives anywhere else is third-party and
    stays out. Manifest keys are relative to the root a member was found
    under, so moving a workflow tree moves no digest.

    ``repository`` selects which of two walks this is. The default, the
    **layering** walk, probes :func:`repo_root` after ``root`` and admits every
    repository module: it is how the test suite proves a module reaches no
    engine and no numerics, and it enters no digest. ``repository=False`` is
    the **identity** walk the loaders use (workflow spec §4.2, IM spec
    §2.8.1): only ``root`` is probed, and a member under :func:`package_root`
    is dropped, because the package's bytes are runtime identity — the
    ``tree_digest`` every step record carries and ``--resume`` compares — so a
    document naming them would only move on every edit to the package. What
    remains is the code beside the hashed module that nothing else covers: a
    ``{"path": …}`` script's sibling helpers, a user package's siblings on
    ``sys.path``. A hashed module that itself lives in an installation
    directory declares no closure at all (:func:`is_installed_module`), because
    with ``root`` at the stdlib or site-packages every module reachable there
    would be admitted.

    ``include_parents`` also admits every parent package ``__init__.py``
    Python executes on the way to an imported name — the *execution* closure.
    Off by default: a parent is executed, not declared, and admitting it would
    make ``causalab/protocol/__init__.py``'s eager fan-out part of every
    script's identity. The switch exists so the acceptance test can pin the
    choice either way rather than leave it implicit.

    Nothing is imported. Each member is read, hashed and parsed with
    ``ast.parse``; a member that does not parse contributes its bytes and no
    edges.
    """
    start = path.resolve()
    own = (root if root is not None else start.parent).resolve()
    if repository:
        roots = [own] if own == repo_root() else [own, repo_root()]
    else:
        roots = [own]
    package = package_root()
    manifest: dict[str, str] = {}
    seen: set[Path] = {start}
    pending: list[tuple[Path, Path]] = [(start, own)]
    while pending:
        file, file_root = pending.pop()
        data = file.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if file != start:
            manifest[file.relative_to(file_root).as_posix()] = sha
        refs = _IMPORTS_BY_SHA.get(sha)
        if refs is None:
            refs = _import_refs(data, file)
            _IMPORTS_BY_SHA[sha] = refs
        for ref in refs:
            for target, target_root in _resolve_ref(
                ref, file, file_root, roots, include_parents
            ):
                if not repository and target.is_relative_to(package):
                    continue  # runtime identity: the package's own bytes
                if target not in seen:
                    seen.add(target)
                    pending.append((target, target_root))
    return dict(sorted(manifest.items()))


def _import_refs(data: bytes, file: Path) -> tuple[_ImportRef, ...]:
    """Every import statement in ``data`` outside ``if TYPE_CHECKING:`` bodies,
    function-local ones included."""
    try:
        tree = ast.parse(data, filename=str(file))
    except (SyntaxError, ValueError):
        return ()
    out: list[_ImportRef] = []
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)  # the run-time branch still counts
            continue
        if isinstance(node, ast.Import):
            out.extend(_ImportRef(alias.name, 0, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append(
                _ImportRef(
                    node.module or "",
                    node.level,
                    tuple(alias.name for alias in node.names),
                )
            )
        stack.extend(ast.iter_child_nodes(node))
    return tuple(out)


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` — a block the
    interpreter never runs, so nothing in it is a dependence."""
    return _attribute_tail(test) == "TYPE_CHECKING"


def _resolve_ref(
    ref: _ImportRef,
    file: Path,
    file_root: Path,
    roots: Sequence[Path],
    include_parents: bool,
) -> list[tuple[Path, Path]]:
    """The files one import statement declares, each with the root it was
    found under; empty when the name resolves under no root (third-party)."""
    if ref.level:
        # relative: anchored in the importing module's own package (a
        # module's directory; for an ``__init__.py`` the package it defines)
        package = file.parent
        for _ in range(ref.level - 1):
            package = package.parent
        try:
            base = list(package.relative_to(file_root).parts)
        except ValueError:
            return []
        base += ref.module.split(".") if ref.module else []
        candidates: list[tuple[list[str], Path]] = [(base, file_root)]
    else:
        base = ref.module.split(".")
        candidates = [(base, root) for root in roots]

    for parts, root in candidates:
        found = _probe(root, parts)
        if found is None:
            continue
        out: list[tuple[Path, Path]] = []
        if ref.names and found.name == "__init__.py":
            # ``from pkg import name``: a submodule when one exists on disk,
            # otherwise an attribute the package's ``__init__`` defines
            for name in ref.names:
                child = _probe(root, [*parts, name]) if name != "*" else None
                out.append((child if child is not None else found, root))
        else:
            out.append((found, root))
        if include_parents:
            out.extend(
                (init, root)
                for depth in range(1, len(parts) + 1)
                if (init := root.joinpath(*parts[:depth], "__init__.py")).is_file()
            )
        return out
    return []


def _probe(root: Path, parts: Sequence[str]) -> Path | None:
    """``root/a/b/c.py`` or ``root/a/b/c/__init__.py`` for ``a.b.c``, or
    ``None`` — the shape :func:`_child_module` probes, from a fixed root."""
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    current = root
    for index, part in enumerate(parts):
        package_init = current / part / "__init__.py"
        if package_init.is_file():
            current = current / part
            continue
        module_file = current / f"{part}.py"
        if module_file.is_file() and index == len(parts) - 1:
            return module_file
        return None
    return current / "__init__.py"


def resolve_locator(locator: str) -> ResolvedCode:
    """Resolve a dotted locator to its defining module's file, **without
    importing it**.

    ``find_spec`` on a *submodule* imports its parent, and on
    ``pkg.mod.attr`` it would import ``pkg.mod`` itself — the module holding
    the user's torch code. So only the top-level name goes through
    ``find_spec`` (which imports nothing), and every further dotted part is a
    filesystem probe against the package's search locations. The walk stops at
    the first part that is not a file on disk; what remains is the attribute
    path.
    """
    parts = locator.split(".")
    if not all(part.isidentifier() for part in parts):
        raise CodeResolutionError(f"{locator!r} is not a dotted importable name")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError) as err:
        # a missing or unimportable top-level package raises rather than
        # returning None
        raise CodeResolutionError(f"{parts[0]!r} does not import: {err}") from err
    if spec is None:
        raise CodeResolutionError(f"no module named {parts[0]!r}")
    origin = spec.origin
    search = [Path(root) for root in (spec.submodule_search_locations or [])]
    consumed = 1
    while consumed < len(parts) and search:
        found = _child_module(search, parts[consumed])
        if found is None:
            break
        origin, search = str(found[0]), found[1]
        consumed += 1
    if origin is None or not origin.endswith(".py"):
        raise CodeResolutionError(
            f"{locator!r} resolves to {origin or 'a namespace package'}, which is "
            "not Python source — a code reference is identified by its source "
            "bytes, so there has to be some"
        )
    return ResolvedCode(
        module=".".join(parts[:consumed]),
        path=Path(origin),
        attr=tuple(parts[consumed:]),
    )


def _child_module(search: Sequence[Path], name: str) -> tuple[Path, list[Path]] | None:
    """``(origin, next search locations)`` for a child module found on disk."""
    for root in search:
        package_init = root / name / "__init__.py"
        if package_init.is_file():
            return package_init, [root / name]
        module_file = root / f"{name}.py"
        if module_file.is_file():
            return module_file, []
    return None


def function_def(
    resolved: ResolvedCode,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``def`` the locator names, or ``None`` when the attribute is not a
    literal function in the module's source.

    ``None`` is the honest answer for a C function (``torch.relu``), an
    attribute built by a factory at import time, or a re-export. It is not a
    refusal: it is the boundary of what a static read can see, and the checks
    that need a signature simply do not run.
    """
    if not resolved.attr:
        return None
    try:
        tree = ast.parse(resolved.path.read_bytes(), filename=str(resolved.path))
    except (SyntaxError, ValueError):
        return None
    body: Iterable[ast.stmt] = tree.body
    for index, name in enumerate(resolved.attr):
        last = index == len(resolved.attr) - 1
        match = next(
            (
                node
                for node in body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            return None
        if last:
            return match if not isinstance(match, ast.ClassDef) else None
        if not isinstance(match, ast.ClassDef):
            return None
        body = match.body
    return None


def signature_problems(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    args: Mapping[str, Any],
    supplied: Sequence[str] = (),
) -> list[str]:
    """Why the declared ``args`` do not fit the function's signature.

    The first positional parameter is the tensor the mechanism writes, so it
    is never declared. ``supplied`` names the keywords the runtime provides
    (``row_roles``, ``data_inputs``) because the declaration asked for them.
    A ``**kwargs`` function accepts any name, so only the *missing* half of
    the check runs against it.
    """
    positional = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    keyword_only = [a.arg for a in fn.args.kwonlyargs]
    tensor = positional[:1]
    accepts_any = fn.args.kwarg is not None
    known = set(positional[1:]) | set(keyword_only)

    problems: list[str] = []
    if not tensor:
        problems.append(
            f"{fn.name}() takes no positional parameter — the mechanism passes "
            "the feature slice as the first argument"
        )
    if not accepts_any:
        for name in sorted(set(args) - known):
            problems.append(
                f"declared argument {name!r} is not a parameter of {fn.name}() "
                f"(it takes {sorted(known) or 'no keyword arguments'})"
            )
        for name in sorted(set(supplied) - known):
            problems.append(
                f"the declaration asks the runtime for {name!r}, but {fn.name}() "
                "has no such parameter"
            )

    required = _required_parameters(fn, skip=len(tensor))
    for name in sorted(required - set(args) - set(supplied)):
        problems.append(
            f"{fn.name}() requires {name!r} and the declaration does not give it — "
            "an argument that is not declared is not in the digest"
        )
    return problems


def _required_parameters(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, *, skip: int
) -> set[str]:
    positional = [*fn.args.posonlyargs, *fn.args.args][skip:]
    defaults = fn.args.defaults
    without_default = (
        positional[: len(positional) - len(defaults)] if defaults else positional
    )
    required = {a.arg for a in without_default}
    required |= {
        a.arg
        for a, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
        if default is None
    }
    return required


def undeclared_reads(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    env_inputs: Iterable[str],
    data_inputs: Iterable[str],
) -> list[str]:
    """Statically detectable reads the declaration does not cover.

    Two kinds, both literal: an environment variable named by a string
    constant (``os.getenv("X")``, ``os.environ["X"]``,
    ``os.environ.get("X")``) and a file named by a string constant passed
    first to one of :data:`READER_CALLS` or to ``open``. Anything routed
    through a variable is invisible here and is not refused — see the module
    docstring.
    """
    allowed_env = set(env_inputs)
    allowed_files = set(data_inputs)
    problems: list[str] = []

    for node in ast.walk(fn):
        name = _env_name(node)
        if name is not None and name not in allowed_env:
            problems.append(
                f"reads environment variable {name!r}, which the declaration's "
                "'env_inputs' does not allow"
            )
        path = _read_path(node)
        if path is not None and path not in allowed_files:
            problems.append(
                f"reads the file {path!r}, which the declaration's 'data_inputs' "
                "does not name"
            )
    return sorted(dict.fromkeys(problems))


def _literal_str(node: ast.expr | None) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _env_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and _attribute_tail(node.value) == "environ":
        return _literal_str(node.slice)
    if isinstance(node, ast.Call):
        tail = _attribute_tail(node.func)
        if tail == "getenv" and node.args:
            return _literal_str(node.args[0])
        if (
            tail == "get"
            and isinstance(node.func, ast.Attribute)
            and _attribute_tail(node.func.value) == "environ"
            and node.args
        ):
            return _literal_str(node.args[0])
    return None


def _read_path(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open" and node.args:
        return _literal_str(node.args[0])
    if not isinstance(func, ast.Attribute) or func.attr not in READER_CALLS:
        return None
    # ``Path("scale.json").read_text()`` — the literal is on the receiver, and
    # the reading call itself takes no arguments at all
    receiver = func.value
    if (
        isinstance(receiver, ast.Call)
        and _attribute_tail(receiver.func) in ("Path", "PurePath", "PosixPath")
        and receiver.args
    ):
        literal = _literal_str(receiver.args[0])
        if literal is not None:
            return literal
    return _literal_str(node.args[0]) if node.args else None


def _attribute_tail(node: ast.AST) -> str | None:
    """``os.environ`` and a bare ``environ`` both answer ``"environ"``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def resolve_locator_or_refuse(locator: str, *, path: str) -> ResolvedCode:
    """:func:`resolve_locator`, with the failure spelled as §5 rule 24.

    Canonicalization and validation both need the resolved source and both
    have to refuse the same way, so the translation lives here rather than
    being written out twice.
    """
    try:
        return resolve_locator(locator)
    except CodeResolutionError as err:
        raise ValidationError(CODE_RULE, str(err), path=path) from err
