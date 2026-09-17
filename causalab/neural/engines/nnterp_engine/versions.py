"""The version guard of a remote run: the server runs the code this client
planned for.

A block of this engine is ``causalab``'s own module-level functions (and
``nnterp``'s accessors), imported **where the block runs**, reading a program
planned **here**. A skew between the two installs is a skew between the plan
and the code that reads it — a renamed field, a changed default — and nothing
downstream would name it. So before any job is submitted the server's
environment is looked up (``nnsight.ndif.get_remote_env``, which nnsight
caches per host) and the two installs are compared by strict string equality,
package by package.

Once per ``(process, host)``: a host that matched is remembered here and not
asked again. ``remote="local"`` — nnsight's in-process dry run — has no
server, and ``remote=False`` no submission; neither is checked.

``causalab``'s own version is **static** in ``pyproject.toml`` (``0.0.1``):
it does not move with the code, so its half of the guard passes between two
installs at any two commits. Making it dynamic is a repository-wide decision
and is not taken here; what catches a skew today is the other three
(``nnterp``, ``nnsight``, ``torch``).
"""

from __future__ import annotations

import importlib.metadata

from causalab.protocol.errors import ProtocolError

__all__ = ["GUARDED", "ensure_server_matches"]

#: The packages whose code decides what a block does where it runs, by import
#: name — how a server's ``/env`` keys its packages. ``causalab`` and
#: ``nnterp`` are what the block imports; ``nnsight`` is what compiles and
#: runs it; ``torch`` is what its arithmetic is, and a fit's claim to be the
#: local fit **to the bit** rests on the two sides agreeing on it —
#: ``randperm``'s stream and ``manual_seed``'s algorithm are its.
GUARDED = ("causalab", "nnterp", "nnsight", "torch")

#: The resolved hosts whose environment matched, this process.
_MATCHED: set[str] = set()


def ensure_server_matches(remote: bool | str) -> None:
    """Refuse (P4) a remote submission to a server one of whose
    :data:`GUARDED` packages is not this client's, or is absent, or which
    does not answer for its environment at all; nothing is submitted first. ``remote`` is what the run method takes: ``True`` for the
    configured host, a host URL, or ``False`` / ``"local"`` (no server)."""
    if not remote or remote == "local":
        return
    import nnsight.ndif as ndif

    host = ndif.resolve_host(remote if isinstance(remote, str) else None)
    if host in _MATCHED:
        return
    try:
        served = ndif.get_remote_env(host)["packages"]
    except RuntimeError as err:
        raise ProtocolError(
            "P4",
            f"remote run refused before any job was submitted: the NDIF "
            f"server at {host} did not answer for its environment ({err}). "
            "Every remote run of this engine compares the server's installed "
            "packages with this client's first (module docstring), so a host "
            "that is unreachable, or too old to serve /env, is refused here "
            "rather than at the first job.",
        ) from err
    for name in GUARDED:
        here = importlib.metadata.version(name)
        there = served.get(name)
        if there == here:
            continue
        found = (
            f"does not have {name!r} installed (its /env lists no such package)"
            if there is None
            else f"has {name} {there!r}"
        )
        raise ProtocolError(
            "P4",
            f"remote run refused before any job was submitted: the NDIF server "
            f"at {host} {found}; this client has {name} {here!r}. The block is "
            f"{name}'s own functions imported where it runs, reading a program "
            "planned here, so the two installs must be the same version. "
            f"Install {name}=={here} on the server, install the server's "
            "version here, or point `remote` at a matching deployment.",
        )
    _MATCHED.add(host)
