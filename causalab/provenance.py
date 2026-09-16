"""What is actually installed and running — the runtime's own identity.

A run can produce numerically excellent results *on a package that is not the
requested branch* and have no way to notice. The stub this module replaces is
`code_commit()` — a `git rev-parse --short HEAD` that returned the string
``"unknown"`` on failure and was stamped into every output identity, so a run
with no resolvable provenance recorded a value that reads like a value.

Three properties make this different from the stub.

**Requested and resolved are separate fields.** "The branch I asked for" and
"the commit that is installed" are different facts, and conflating them is
precisely how a run can report a branch it did not execute. The split comes from
PEP 610's ``direct_url.json`` in the installed distribution's metadata.

**The metadata is read without importing the package.**
``importlib.metadata`` reads the ``.dist-info`` directory; it does not execute
the code it describes. A check that imports the package under test can only
ever describe *an* installed copy, and on a path shadowing or a stale
``site-packages`` entry it describes the wrong one.

**There is no "unknown".** A field is either a fact or absent, and absence is
itself a fact with a reason:

* ``resolved_revision is None`` means *this install records no revision* — a
  published wheel from an index genuinely has none. That is knowledge, not
  ignorance.
* a *failure* to determine the source kind, or to read the files that will
  execute, raises :class:`ProvenanceError`. A broken environment is not
  something to summarize as a string.

What always exists is :attr:`RuntimeIdentity.tree_digest`: a deterministic
digest over the bytes that will actually run. Every source kind has one, so
every run has *an* identity even when it has no revision — which is why
replacing ``code_commit`` does not make an index-installed causalab unrunnable.

Torch-free, on purpose: attestation has to be answerable before any weight
loads (the attestation check at model load refuses there), and this module is
imported by the pure layer.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Literal

__all__ = [
    "ModuleLocation",
    "ProvenanceError",
    "RuntimeIdentity",
    "SOURCE_KINDS",
    "SourceKind",
    "runtime_identity",
]

#: How the running copy got here. Four kinds, and the mapping from PEP 610 is
#: stated rather than inferred (see :func:`_source`):
#:
#: * ``git`` — installed from a VCS URL; the revision is recorded by the
#:   installer and needs no local checkout to read;
#: * ``editable`` — a source tree on ``sys.path``; the revision and the dirty
#:   state come from that tree;
#: * ``sdist`` — built from a source tree or archive, so there was a build step
#:   over sources rather than a published wheel;
#: * ``wheel`` — a built artifact, from an index or a local ``.whl``.
SourceKind = Literal["git", "editable", "sdist", "wheel"]
SOURCE_KINDS: tuple[SourceKind, ...] = ("git", "editable", "sdist", "wheel")

#: What a distribution *ships* and the runtime executes or loads: code,
#: extension modules, the typing marker, and the data files causalab reads
#: beside its code (task tables, method and workflow configs, docs). An
#: allowlist rather than a denylist, because a denylist describes "every file
#: present" and rots the first time a run writes something new under the
#: package; the allowlist describes the claim the digest makes.
_SHIPPED_SUFFIXES = frozenset(
    {".py", ".pyi", ".so", ".pyd", ".json", ".yaml", ".yml", ".md"}
)
_SHIPPED_NAMES = frozenset({"py.typed"})

#: Never hashed into a tree digest, whatever they contain. Caches are a property
#: of the interpreter that ran, not of the code that will run; build residue
#: (``*.egg-info``) is metadata about the code; and ``outputs`` is a runtime
#: output directory a task run materializes *inside its own package*
#: (``.gitignore``: ``/causalab/tasks/**/outputs/``). A run that wrote there
#: would otherwise change the digest of the very package it attests.
_IGNORED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "outputs",
    }
)
#: The other runtime output shapes ``.gitignore`` names under ``causalab/tasks``
#: — ``*results``, ``*datasets``, ``*logs`` — are directory-name *suffixes*.
_IGNORED_DIR_SUFFIXES = ("results", "datasets", "logs", ".egg-info")


class ProvenanceError(RuntimeError):
    """The runtime cannot describe itself.

    Raised rather than summarized. Every caller of this module is asking the
    question "may I trust a number this run produces", and a placeholder answer
    to that question is worse than no answer — it is the ``"unknown"`` this
    module exists to remove.
    """


@dataclasses.dataclass(frozen=True)
class ModuleLocation:
    """One subpackage of the running distribution, and the digest of its bytes.

    Present so that a tree-digest mismatch can be *localized*: two installs
    differing in one module say so, instead of differing in a single 64-hexit
    number with no way to narrow it.
    """

    name: str
    #: Relative to the package root, POSIX-separated, so the value is the same
    #: on every platform.
    path: str
    digest: str
    files: int


@dataclasses.dataclass(frozen=True)
class RuntimeIdentity:
    """What is installed, where it came from, and what will execute."""

    distribution: str
    #: The installed distribution's version — a *package* version, kept
    #: deliberately separate from :attr:`resolved_revision`. Conflating them is
    #: how one version string comes to describe several different trees; the
    #: build-time stamp makes this one name a revision, and it stays its own
    #: field regardless.
    version: str
    source_kind: SourceKind
    #: Absolute path of the package root whose bytes will run.
    location: str
    #: The URL the install came from (PEP 610 ``url``), or ``None`` for an
    #: install from an index, which records no origin.
    origin: str | None
    #: What was *asked for* — a branch, a tag, a ref. ``None`` when the install
    #: records no request, which an index install does not.
    requested_revision: str | None
    #: What is *installed*. ``None`` only when the install records no revision
    #: and no source tree can be consulted.
    resolved_revision: str | None
    #: PEP 610 ``archive_info.hashes``, for an install from an archive.
    archive_digest: str | None
    #: Deterministic digest over every file that will execute. Always present.
    tree_digest: str
    #: Whether the source tree this install reads has uncommitted changes
    #: **under the package root** — the same subtree :attr:`tree_digest`
    #: covers, so the two fields answer the same question. An edited README
    #: changes no byte that will run and does not make an install dirty.
    #:
    #: For an install with **no** source tree — a wheel or sdist from an index,
    #: or a VCS install whose bytes the installer copied out of the clone — this
    #: is ``False``, and that is a statement rather than a default: nothing
    #: local is being read, so nothing local can have diverged. A wheel *built*
    #: from a dirty tree cannot be detected here at all, which is what the
    #: build-time stamp is for.
    #:
    #: ``None`` when there *is* a source tree but git could not answer for it
    #: (not a checkout, no ``git`` on PATH, a stalled status). That is the one
    #: place this module says "unknown", and it says so as an absence rather
    #: than as ``False``: a failed question must not report cleaner than it
    #: knows, and :attr:`attested` treats it as not attested.
    dirty: bool | None
    modules: tuple[ModuleLocation, ...]
    dependencies: tuple[tuple[str, str], ...]

    @property
    def short_revision(self) -> str:
        """A short content identity that always exists.

        The resolved revision when there is one, else the tree digest. Both
        identify content; neither is a placeholder. This is what replaces
        ``code_commit()``'s return value in the output identity, so that field
        keeps its shape — a short hex string — while losing its ability to say
        ``"unknown"``.
        """
        return (self.resolved_revision or self.tree_digest)[:12]

    @property
    def attested(self) -> bool:
        """Whether this install can name the revision it is running.

        Requires ``dirty`` to be *known* false: an unanswered dirty state is not
        a clean one. Not a judgement about *which* revision — that comparison
        is the attestation check's.
        """
        return self.resolved_revision is not None and self.dirty is False

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form, for the run receipt.

        Sorted and free of timestamps, hostnames and absolute paths *except*
        ``location``, which the receipt's `observed` section carries and its
        `verification` section does not — the shard-identical requirement lands
        on the latter.
        """
        return {
            "distribution": self.distribution,
            "version": self.version,
            "source_kind": self.source_kind,
            "location": self.location,
            "origin": self.origin,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "archive_digest": self.archive_digest,
            "tree_digest": self.tree_digest,
            "dirty": self.dirty,
            "modules": [dataclasses.asdict(m) for m in self.modules],
            "dependencies": dict(self.dependencies),
        }


# --------------------------------------------------------------------------- #
# reading the install, without importing it
# --------------------------------------------------------------------------- #


def _distribution(name: str) -> Any:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return distribution(name)
    except PackageNotFoundError as error:
        raise ProvenanceError(
            f"{name!r} is not an installed distribution, so its provenance "
            "cannot be read. An import-path install (a bare `sys.path` entry, "
            "a `PYTHONPATH` shim) has no metadata to attest — install it, "
            "editable is enough"
        ) from error


def _direct_url(dist: Any) -> dict[str, Any] | None:
    """PEP 610's record of where this install came from.

    Read as *metadata text*: this is the whole reason the check does not import
    the package. Absent for an install from an index, which is a fact about the
    install and not a failure.
    """
    try:
        raw = dist.read_text("direct_url.json")
    except (OSError, KeyError):  # pragma: no cover - unreadable dist-info
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProvenanceError(
            f"{dist.metadata['Name']}'s direct_url.json is not valid JSON, so "
            "the install cannot say where it came from"
        ) from error
    return parsed if isinstance(parsed, dict) else None


def _url_path(url: str) -> Path | None:
    """The local path a ``file://`` URL names, or ``None`` for a remote URL."""
    if not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    return Path(unquote(urlparse(url).path))


def _source(dist: Any, direct: dict[str, Any] | None) -> tuple[SourceKind, Path | None]:
    """``(source_kind, source_tree)`` — the mapping from PEP 610, stated.

    ``source_tree`` is the local directory this install came from, or ``None``
    when there is none. Only an editable install *reads* it; for the other
    kinds it says where the bytes were copied from and nothing more.
    """
    if direct is None:
        # No origin recorded: an install from an index. The artifact is a wheel
        # when the dist-info says so, and otherwise an sdist that was built here.
        try:
            wheel = dist.read_text("WHEEL")
        except (OSError, KeyError):  # pragma: no cover
            wheel = None
        return ("wheel" if wheel else "sdist"), None

    url = direct.get("url")
    if not isinstance(url, str):
        raise ProvenanceError(
            "direct_url.json carries no 'url', so the install records an "
            "origin it cannot name (PEP 610 requires one)"
        )
    if isinstance(direct.get("vcs_info"), dict):
        # A VCS install records its own commit, and the installer *copied* the
        # bytes out of the clone — so the local tree, if the URL is a local
        # clone, is not what runs. It is returned so the identity can say where
        # the install came from; nothing about the running copy is read from it
        # (see `_package_root` and `_revision_and_dirt`).
        return "git", _url_path(url)
    if isinstance(direct.get("dir_info"), dict):
        tree = _url_path(url)
        if direct["dir_info"].get("editable"):
            return "editable", tree
        # built from a source tree: a build step over sources, not a published
        # artifact — which is what `sdist` names here
        return "sdist", tree
    if isinstance(direct.get("archive_info"), dict):
        # An archive is a file, not a tree: there is no directory whose git
        # state could describe this install, even when the archive is local.
        return ("wheel" if url.endswith(".whl") else "sdist"), None
    raise ProvenanceError(
        f"direct_url.json for {url!r} carries none of vcs_info / dir_info / "
        "archive_info, so PEP 610 cannot say what kind of install this is"
    )


def _archive_digest(direct: dict[str, Any] | None) -> str | None:
    if not direct:
        return None
    info = direct.get("archive_info")
    if not isinstance(info, dict):
        return None
    hashes = info.get("hashes")
    if isinstance(hashes, dict) and hashes:
        # The strongest digest on offer, not the alphabetically first: a record
        # carrying both md5 and sha256 must not identify the artifact by md5
        # because ``"md5" < "sha256"``. sha256 leads because it is the hash
        # every installer records and the one the rest of this module uses.
        for preferred in _ARCHIVE_HASH_PREFERENCE:
            if preferred in hashes:
                return f"{preferred}:{hashes[preferred]}"
        name, value = sorted(hashes.items())[0]
        return f"{name}:{value}"
    legacy = info.get("hash")  # the pre-`hashes` spelling, still emitted
    return str(legacy) if legacy else None


_ARCHIVE_HASH_PREFERENCE = ("sha256", "sha512", "sha384", "blake2b", "sha1", "md5")


def _module_name(distribution: str) -> str:
    """The import package a distribution name conventionally installs."""
    return distribution.replace("-", "_")


def _package_root(dist: Any, name: str, kind: SourceKind, tree: Path | None) -> Path:
    """Where the bytes that will execute actually live.

    For an editable install that is the source tree, not ``site-packages`` —
    which is the distinction a tree digest has to get right, because an editable
    install's ``site-packages`` holds a link and no code. For every *other*
    kind the installer copied files out of the tree (a ``git+file://`` clone, a
    ``pip install .`` directory), so the tree is where the revision came from
    and hashing it would describe a copy that will not run — and would move
    the identity every time someone edits the clone afterwards.
    """
    module = _module_name(name)
    if kind == "editable" and tree is not None:
        candidate = tree / module
        if candidate.is_dir():
            return candidate
    located = dist.locate_file(module)
    root = Path(str(located))
    if root.is_dir():
        return root
    raise ProvenanceError(
        f"cannot find the {module!r} package under {root} — the install's "
        "metadata and its files disagree, so there is nothing to hash"
    )


# --------------------------------------------------------------------------- #
# hashing what will run
# --------------------------------------------------------------------------- #


def _is_ignored_dir(name: str) -> bool:
    return name in _IGNORED_DIRS or name.endswith(_IGNORED_DIR_SUFFIXES)


def _files(root: Path) -> list[Path]:
    """Every file that ships, sorted: what executes or loads, and nothing a run
    wrote under the package afterwards.

    Membership is by the allowlist (:data:`_SHIPPED_SUFFIXES`,
    :data:`_SHIPPED_NAMES`) and the excluded directories, so the same tree
    digests the same before and after a task run materializes its datasets,
    results or outputs beside the task's code.
    """
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in _SHIPPED_SUFFIXES and path.name not in _SHIPPED_NAMES:
            continue
        if any(_is_ignored_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        out.append(path)
    return out


#: ``(relative POSIX path, content digest)`` — one file, hashed once.
FileDigest = tuple[str, bytes]


def _file_digests(root: Path, paths: Iterable[Path]) -> list[FileDigest]:
    """Each file under ``root`` hashed once, sorted by its relative POSIX path.

    Sorted *here*, on the string, rather than inherited from the caller's
    iteration order: ``PurePath`` ordering compares path components and is
    case-normalized on Windows, so a digest that inherited it would differ
    between platforms for an identical tree. The string order is the same
    everywhere, which is what makes the tree digest platform-independent
    rather than incidentally so.
    """
    out = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).digest())
        for path in paths
    ]
    out.sort(key=lambda entry: entry[0])
    return out


def _fold(entries: Iterable[FileDigest]) -> tuple[str, int]:
    """``(digest, count)`` over pre-hashed files.

    Path *and* content are folded in, so a renamed file changes the digest: a
    tree digest that only covered bytes would call two different layouts
    identical. The tree digest and each module digest are this same fold over
    subsets of the same entries, so the relationship between them is by
    construction rather than by re-reading every file.
    """
    digest = hashlib.sha256()
    count = 0
    for relative, content in entries:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
        count += 1
    return digest.hexdigest(), count


def _digest_of(root: Path, paths: Iterable[Path]) -> tuple[str, int]:
    """``(digest, count)`` over ``paths``, relative to ``root``: hash then fold."""
    return _fold(_file_digests(root, paths))


def _modules(root: Path, entries: Iterable[FileDigest]) -> tuple[ModuleLocation, ...]:
    """One entry per top-level member of the package, so a mismatch localizes."""
    groups: dict[str, list[FileDigest]] = {}
    for relative, content in entries:
        groups.setdefault(relative.split("/", 1)[0], []).append((relative, content))
    out: list[ModuleLocation] = []
    for member, members in sorted(groups.items()):
        digest, count = _fold(members)
        out.append(
            ModuleLocation(
                name=f"{root.name}.{member[:-3] if member.endswith('.py') else member}",
                path=member,
                digest=digest,
                files=count,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# git, for the trees that have one
# --------------------------------------------------------------------------- #


def _git(tree: Path, *args: str) -> str | None:
    """A git command in ``tree``, or ``None`` when git cannot answer.

    ``None`` rather than a raise: a source tree that is not a checkout is an
    ordinary situation (an unpacked sdist), and it makes the *revision* absent
    rather than the provenance unreadable. A timeout is the same kind of
    non-answer: this runs on the startup path of every request, and git
    stalling on a lock or a slow filesystem must not hang the run.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(tree), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return result.stdout.strip()


_GIT_TIMEOUT_S = 30


def _revision_and_dirt(
    kind: SourceKind, tree: Path | None, direct: dict[str, Any] | None, name: str
) -> tuple[str | None, bool | None]:
    """``(resolved_revision, dirty)`` for one install.

    A VCS install's revision is the installer's own record — authoritative,
    readable with no checkout, and the *only* fact: the installer copied the
    bytes out of the clone, so a record with no ``commit_id`` yields ``None``
    rather than the clone's current ``HEAD``, which names a tree that is not
    running, and ``dirty`` is ``False`` for the same reason.

    Every other kind with a source tree asks git. ``dirty`` is scoped to the
    package subtree — the same subtree the tree digest covers — so the two
    fields answer the same question; a repo-wide status would call an install
    dirty for an edited README, which changes no byte that will run and would
    make the attestation check refuse on it. When git cannot answer, ``dirty``
    is ``None``: a
    failed question must not report cleaner than it knows.
    """
    if kind == "git":
        recorded = None
        if direct and isinstance(direct.get("vcs_info"), dict):
            commit = direct["vcs_info"].get("commit_id")
            recorded = str(commit) if commit else None
        return recorded, False
    if tree is None:
        return None, False
    revision = _git(tree, "rev-parse", "HEAD")
    status = _git(tree, "status", "--porcelain", "--", str(tree / _module_name(name)))
    return revision, None if status is None else bool(status)


def _requested_revision(direct: dict[str, Any] | None) -> str | None:
    if not direct:
        return None
    info = direct.get("vcs_info")
    if isinstance(info, dict):
        requested = info.get("requested_revision")
        return str(requested) if requested else None
    return None


def _dependencies(dist: Any) -> tuple[tuple[str, str], ...]:
    """Installed versions of everything this distribution requires.

    Names only, resolved against the live environment — the *requirement* text
    is in the metadata already, and what a run needs recorded is what is
    actually importable beside it.
    """
    from importlib.metadata import PackageNotFoundError, version

    import re

    out: dict[str, str] = {}
    for requirement in dist.requires or ():
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", str(requirement))
        if not match:
            continue
        name = match.group(1)
        if name in out:
            continue
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue  # an extra's dependency, not installed here
    return tuple(sorted(out.items()))


# --------------------------------------------------------------------------- #
# the one entry point
# --------------------------------------------------------------------------- #


@functools.cache
def runtime_identity(distribution: str = "causalab") -> RuntimeIdentity:
    """What is installed and running, as data.

    Cached: the answer is a property of the process, and a run that consulted it
    twice and got two answers would be recording something other than what it
    executed. Call ``runtime_identity.cache_clear()`` in a test that changes the
    tree underneath it.
    """
    dist = _distribution(distribution)
    direct = _direct_url(dist)
    kind, tree = _source(dist, direct)
    root = _package_root(dist, distribution, kind, tree)
    files = _files(root)
    if not files:
        raise ProvenanceError(
            f"the {distribution!r} package at {root} contains no files, so "
            "there is nothing that could execute"
        )
    entries = _file_digests(root, files)
    tree_digest, _ = _fold(entries)
    revision, dirty = _revision_and_dirt(kind, tree, direct, distribution)
    return RuntimeIdentity(
        distribution=distribution,
        version=dist.version,
        source_kind=kind,
        location=str(root),
        origin=str(direct["url"]) if direct else None,
        requested_revision=_requested_revision(direct),
        resolved_revision=revision,
        archive_digest=_archive_digest(direct),
        tree_digest=tree_digest,
        dirty=dirty,
        modules=_modules(root, entries),
        dependencies=_dependencies(dist),
    )
