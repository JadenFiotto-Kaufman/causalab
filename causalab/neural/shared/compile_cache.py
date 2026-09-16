"""Where compiled kernels live across jobs: one shared root, namespaced by the
toolchain that compiled them (``CAUSALAB_COMPILE_CACHE``).

A run on a Gated DeltaNet model compiles kernels before its first forward:
Triton builds FLA's delta-rule kernels, TileLang builds the Hopper backward
kernels (several seconds each), and a run that compiles the forward with
``torch.compile`` adds Inductor's artifacts. Each compiler keeps its own
on-disk cache under the user's home directory by default, so a fresh machine,
a fresh container or another user pays the whole compile again — 20–30 s on a
six-step Qwen3.6-35B-A3B workflow — while a shared filesystem the jobs
already have could hold the artifacts once.

Setting ``CAUSALAB_COMPILE_CACHE=/shared/path`` points every compiler a run
uses at that root. What keeps concurrent jobs from clobbering each other is
the layout, not locking: artifacts land under a **signature** of the
toolchain that produced them — the versions of torch, its CUDA runtime,
triton, tilelang, flash-linear-attention and transformers, the CPython ABI
tag, and the GPU's name and compute capability — so two jobs share a
directory exactly when they would produce interchangeable artifacts, and two
environments that would not never see each other's files::

    <root>/<signature>/toolchain.json      what the signature stands for
    <root>/<signature>/triton/             TRITON_CACHE_DIR
    <root>/<signature>/tilelang/           TILELANG_CACHE_DIR
    <root>/<signature>/inductor[-<policy>] TORCHINDUCTOR_CACHE_DIR

The CPython ABI is in the signature because Triton's cache holds more than
device code: its CUDA backend compiles a C launcher module into the same
directory, keyed on the launcher *source* and the platform
(``triton/runtime/build.py::compile_module_from_src``, ``platform_key`` =
machine, system, architecture) — not on the interpreter that will import it.
Inductor's FX graph key carries ``sys.version``; Triton's does not, so the
namespace has to.

**Concurrent writers.** Within one directory each compiler publishes an
artifact by writing a temporary file and renaming it into place — Triton's
``FileCacheManager.put`` (a ``tmp.pid_*`` directory, then ``os.replace``),
TileLang 0.1.14's ``KernelCache._atomic_write`` (a ``.<name>.<pid>_<uuid>.tmp``
sibling, ``fsync``, ``os.replace``), Inductor's ``codecache.write_atomic``
(a ``.<pid>.<tid>.tmp`` sibling, ``rename``) — and reads a key only once its
files exist. A rename is atomic per directory on POSIX filesystems, NFS
included, so a reader sees a whole artifact or none; NFS attribute caching
can make a reader briefly miss a file another node just published, and a miss
recompiles the same bytes, which is benign. The manifest here is written the
same way, once, and a truncated one (an unclean node shutdown between the
write and the writeback) is replaced on the next run.

**The root is opt-in.** With the variable unset or empty, nothing changes:
every compiler keeps its own default cache, and no machine is opted in by
accident. ``CAUSALAB_COMPILE_CACHE=<path>`` names the root; the directory may
already exist (an operator's ``mkdir -m 2770 <path>`` on a volume several
users mount) or be created here on first use. A root whose mode is
group-writable is *shared* (:func:`is_shared`): the directories created there
are made group-writable and setgid and take the root's group, and the process
umask is widened once to allow group writes so the compilers' own per-kernel
directories are too (:func:`_prepare_directories` says why each part is
needed and what it costs). Any other root is personal and left to the umask.

**Who is on the other side of the root.** The artifacts are cubins and
shared objects every participating job's compiler *loads into its process*,
and none of the three caches is a trust boundary — so a shared root is a
statement that everyone in its group may run code in everyone else's jobs.
Keep a root writable by no one outside that group; a world-writable root is
warned about, shared or personal.

Three things are deliberately **not** here. A captured CUDA graph cannot be
serialized, so every process still captures its own (``cuda_graphs.py``).
FLA's autotune "cache" (``FLA_CACHE_MODE``) reads configuration files FLA
ships per GPU; nothing a run generates, so nothing to share. And the model
weights have their own cache (the Hugging Face hub's).

All three variables are read by their compilers when a kernel is first
compiled, not at import (Triton's ``knobs.cache.dir`` per cache manager,
TileLang's ``EnvVar`` descriptor, Inductor's ``cache_dir()`` per call), so
:func:`configure` may run any time before the first forward. The loaders call
it as a model lands on its device rather than the CLI at startup, because the
library is imported at least as often as it is run from the command line and
the device is only known once a model is placed; the setting is process-wide,
so the last call wins, and a process spanning unlike GPUs labels one
namespace with the other's manifest (the compilers' own keys still carry the
architecture, so artifacts are not confused — only the label is).

While the root is set it is the one setting: an explicit ``TRITON_CACHE_DIR``,
``TILELANG_CACHE_DIR`` or ``TORCHINDUCTOR_CACHE_DIR`` in the environment is
overridden, with a warning, rather than honoured. The specific variable does
not win over the general one here because the point of the root is that the
three compilers' artifacts sit in *one* namespace: a Triton directory kept
outside it would be unsigned, and its launcher module would be exactly the
ABI hazard above. An operator who wants one compiler elsewhere runs without
a root (``CAUSALAB_COMPILE_CACHE=``). A root that cannot be created or written
is a warning too, not a failed model load: the compilers keep their own
defaults.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import stat
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ENV",
    "CacheLayout",
    "Toolchain",
    "configure",
    "is_shared",
    "layout",
]

#: the environment variable naming the shared root: a path, or unset or the
#: empty string to leave every compiler on its own default cache
ENV = "CAUSALAB_COMPILE_CACHE"

#: the compilers' own variables, by the layout attribute that fills them
VARIABLES: Mapping[str, str] = {
    "triton": "TRITON_CACHE_DIR",
    "tilelang": "TILELANG_CACHE_DIR",
    "inductor": "TORCHINDUCTOR_CACHE_DIR",
}

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """What decides whether two jobs' compiled artifacts are interchangeable.
    Versions are the installed distributions' (``None`` when a package is
    absent — a run without the FLA extra compiles nothing of FLA's);
    ``python`` is the CPython ABI tag (``SOABI``, e.g. ``cpython-310-x86_64-
    linux-gnu``) the compiled launcher modules are built against; the GPU is
    the device's name and compute capability."""

    torch: str
    cuda: str | None
    triton: str | None
    tilelang: str | None
    fla: str | None
    transformers: str | None
    python: str
    gpu: str
    capability: str

    @classmethod
    def detect(cls, device: Any) -> "Toolchain | None":
        """The toolchain of ``device``, or ``None`` off CUDA — the compilers
        this module points at build for CUDA devices only."""
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return cls(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            triton=_version("triton"),
            tilelang=_version("tilelang"),
            fla=_version("flash-linear-attention"),
            transformers=_version("transformers"),
            python=python_abi(),
            gpu=torch.cuda.get_device_name(index),
            capability=f"{major}.{minor}",
        )

    def signature(self) -> str:
        """A short content hash of the fields: the directory name two jobs
        with this toolchain share. ``None`` is encoded as JSON ``null``, its
        own value — never the string that spells it."""
        text = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def python_abi() -> str:
    """The running interpreter's ABI tag, or its version where the build
    reports no tag."""
    return sysconfig.get_config_var("SOABI") or "python-%d.%d" % sys.version_info[:2]


@dataclasses.dataclass(frozen=True)
class CacheLayout:
    """The directories one toolchain's artifacts live in: ``home`` is
    ``<root>/<signature>``, and the three compilers' directories sit under
    it — derived, so no layout can name a subdirectory outside its home."""

    root: Path
    signature: str
    #: the Inductor directory's name: ``inductor``, or ``inductor-<policy>``
    inductor_name: str = "inductor"

    @property
    def home(self) -> Path:
        return self.root / self.signature

    @property
    def triton(self) -> Path:
        return self.home / "triton"

    @property
    def tilelang(self) -> Path:
        return self.home / "tilelang"

    @property
    def inductor(self) -> Path:
        return self.home / self.inductor_name

    @property
    def manifest(self) -> Path:
        return self.home / "toolchain.json"

    def directories(self) -> tuple[Path, ...]:
        return (self.triton, self.tilelang, self.inductor)

    def environment(self) -> dict[str, str]:
        return {
            variable: str(getattr(self, attribute))
            for attribute, variable in VARIABLES.items()
        }


def layout(
    root: str | Path, toolchain: Toolchain, *, inductor_policy: str | None = None
) -> CacheLayout:
    """The layout for ``toolchain`` under ``root``. ``inductor_policy`` names
    a compilation policy whose lowerings Inductor's own key does not see (a
    compile's set of retained eager operators, say); it gets its own
    Inductor directory beside the plain one, named by a readable slug of the
    policy plus a hash of its exact text, so two policies that read alike
    never share one."""
    name = "inductor"
    if inductor_policy:
        digest = hashlib.sha256(inductor_policy.encode()).hexdigest()[:8]
        name = f"inductor-{_slug(inductor_policy)}-{digest}"
    return CacheLayout(
        root=Path(root).expanduser(),
        signature=toolchain.signature(),
        inductor_name=name,
    )


def configure(
    device: Any,
    *,
    root: str | Path | None = None,
    inductor_policy: str | None = None,
    toolchain: Toolchain | None = None,
) -> CacheLayout | None:
    """Point the compilers at the shared root, when there is one.

    ``root`` defaults to :data:`ENV`; with the variable unset or empty there
    is no root. With no root, or off CUDA, nothing changes and ``None`` is
    returned.
    Otherwise the toolchain of ``device`` is detected (or taken from
    ``toolchain``), its directories are created (group-writable, setgid and
    in the root's group when the root is shared, see
    :func:`_prepare_directories`), the manifest is written
    if absent or empty, the three compiler variables are set, and the layout
    is returned. A root that cannot be created or written is logged as a
    warning and leaves the compilers on their own defaults (``None``).
    Idempotent: a second call with the same root and toolchain sets the same
    values again.
    """
    if root is None:
        root = os.environ.get(ENV) or None  # unset or "" runs without a root
    if root is None:
        return None
    if toolchain is None:
        toolchain = Toolchain.detect(device)
    if toolchain is None:
        return None
    chosen = layout(root, toolchain, inductor_policy=inductor_policy)
    try:
        _prepare_directories(chosen)
    except OSError as err:
        _log.warning(
            "compile cache root %s is not usable (%s); the compilers keep their "
            "own cache directories",
            chosen.root,
            err,
        )
        return None
    try:
        _write_manifest(chosen.manifest, dataclasses.asdict(toolchain))
    except OSError as err:
        # the manifest is documentation for a human; the caches are usable
        # without it (a home another user created, whose bits refuse us)
        _log.warning("compile cache manifest %s not written (%s)", chosen.manifest, err)
    for variable, value in chosen.environment().items():
        previous = os.environ.get(variable)
        if previous is not None and previous != value:
            level = (
                logging.INFO
                if _is_compilers_own_default(variable, previous)
                else logging.WARNING
            )
            _log.log(
                level,
                "compile cache root %s overrides %s=%s",
                chosen.root,
                variable,
                previous,
            )
        os.environ[variable] = value
    _log.info("compile cache: %s", chosen.home)
    return chosen


#: the mode a group-shared cache directory gets: group-writable, setgid so
#: what anyone creates inside stays the group's, closed to others
GROUP_SHARED_MODE = 0o2770


def is_shared(bits: int) -> bool:
    """Whether a root with these permission bits is one several users write into: its
    own mode says so — the group-write bit an operator set (``mkdir -m
    2770``). Read off the root, not off the caller's account: a user whose
    primary group happens to be the volume's would otherwise take a personal
    path on a shared root and leave ``0o755`` directories the next user cannot
    complete."""
    return bool(bits & stat.S_IWGRP)


def _prepare_directories(chosen: CacheLayout) -> None:
    """Create the root (if needed) and the layout's directories.

    **A shared root is shared across users.** A root whose mode is
    group-writable (:func:`is_shared`) is one several users write into. Every
    directory this module creates there is made
    :data:`GROUP_SHARED_MODE`, and — because the compilers create their own
    per-kernel directories under the process umask, which by default denies
    the group write — the process umask is widened once to allow group writes
    (:func:`_allow_group_writes`). Without that, a second user could read the
    first user's kernels but not complete a kernel directory the first user's
    killed job left partial, or compile the same missing kernel at the same
    time: both end in ``EACCES`` inside the compiler. The umask change is
    process-wide and deliberate — every file the job writes afterwards is
    group-writable, which on a shared volume is the point — and it is logged.

    **A personal root** is left to the umask: a directory created under a root
    that already existed copies the root's bits, a root created here follows
    the umask as it did.

    Whether *this* process created a directory is what ``mkdir`` without
    ``exist_ok`` answers atomically; an ``exists()`` check first would not —
    two jobs racing on one root (on NFS, a negative dentry another node's
    creation has not yet invalidated widens the race to seconds) would both
    believe they created it, and the loser would ``chmod`` a directory it
    does not own. A directory another job created keeps that job's bits, and
    a ``chmod`` refused anyway is not a reason to give the cache up. The
    root's bits are read once, after it exists; a world-writable root is
    warned about on every path, a shared one most of all — anyone who can
    create ``<root>/<signature>`` first supplies the artifacts every job
    with that toolchain loads.

    Directories created under a shared root also take the root's *group*
    (:func:`_adopt_group`): a setgid root hands it down on its own, but a
    root that is merely group-writable gives each new directory its
    creator's primary group — on a system with per-user primary groups, one
    nobody else is in — and the ``2770`` applied next would then lock every
    other user out of the first user's ``<signature>``.
    """
    root = chosen.root
    try:
        root.mkdir(parents=True)
        created_root = True
    except FileExistsError:
        created_root = False
    root_stat = root.stat()
    bits = stat.S_IMODE(root_stat.st_mode)
    if bits & stat.S_IWOTH:
        _log.warning(
            "compile cache root %s is world-writable; anyone who can write "
            "to it can run code in every job that reads it",
            root,
        )
    shared = is_shared(bits)
    if shared:
        bits = GROUP_SHARED_MODE
    for directory in (chosen.home, *chosen.directories()):
        try:
            directory.mkdir(parents=True)
        except FileExistsError:
            continue  # another job created it, and its bits are that job's
        if shared:
            _adopt_group(directory, root_stat.st_gid)
        if shared or not created_root:
            # after the chown: POSIX permits chown to clear setgid
            _chmod_quietly(directory, bits)
    if shared:
        # only once the tree exists: a root that refuses us above raises out
        # of here and the cache stays off, so the process-wide umask change
        # is never paid for a cache the run did not get
        _allow_group_writes(root)


def _adopt_group(path: Path, gid: int) -> None:
    """Give a directory just created under a shared root the root's group,
    where the root's setgid bit has not already done so. Group ownership can
    be handed to any group the caller belongs to; a refusal means the caller
    is not in the root's group, and then the other users of the root will
    not be able to write under this directory — worth a warning, not a
    failed model load."""
    if path.stat().st_gid == gid:
        return
    try:
        os.chown(path, -1, gid)
    except OSError as err:
        _log.warning(
            "compile cache: %s could not take its root's group %d (%s); other "
            "users of the root may be unable to write under it",
            path,
            gid,
            err,
        )


def _chmod_quietly(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError as err:  # not ours to chmod: it keeps its creator's bits
        _log.debug("compile cache: %s keeps its created mode (%s)", path, err)


def _allow_group_writes(root: Path) -> None:
    """Clear the group-write bit from the process umask, once, so the
    compilers' own kernel directories under a group-shared root come out
    group-writable. ``os.umask`` sets and returns in one call, so reading the
    current value means setting one: the probe is deliberately *tighter* than
    any plausible caller's, so the two-syscall window can only narrow what
    another thread creates, never widen it."""
    previous = os.umask(0o077)
    wanted = previous & ~0o020
    os.umask(wanted)
    if wanted != previous:
        _log.info(
            "compile cache root %s is group-shared; process umask %03o -> %03o "
            "so kernel directories are group-writable",
            root,
            previous,
            wanted,
        )


def _is_compilers_own_default(variable: str, value: str) -> bool:
    """Whether ``value`` is not an operator's setting but the compiler's own
    default written back into the environment: Inductor's ``cache_dir()``
    does that on first use, so a process that touched Inductor before the
    loader shows a populated variable nobody set."""
    if variable != "TORCHINDUCTOR_CACHE_DIR":
        return False
    try:
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
    except Exception:  # torch absent or older: treat as explicit
        return False
    try:
        return value == default_cache_dir()
    except Exception:
        return False


def _version(distribution: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-") or "policy"


def _write_manifest(path: Path, content: Mapping[str, Any]) -> None:
    """Write ``content`` as JSON at ``path`` unless a non-empty file is already
    there — through a temporary file, ``fsync`` and a rename, so a reader
    never sees a partial file, two writers racing leave one whole copy, and
    an unclean shutdown between the write and the writeback (a zero-length
    file at ``path``) is repaired by the next run."""
    if path.exists() and path.stat().st_size > 0:
        return
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".toolchain-", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w") as fh:
            json.dump(content, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates 0600; the manifest is the one file whose job is to
        # tell the next person what the signature stands for, so it takes its
        # directory's bits, masked to a file's — the root's when the tree
        # copied them, the umask's when the tree was created here
        os.chmod(temporary, stat.S_IMODE(path.parent.stat().st_mode) & 0o666)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
